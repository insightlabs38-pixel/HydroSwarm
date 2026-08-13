"""Capability diagnostic Sections 16-17: classical-prior train/serve parity
audit (Part A) and classical-prior ablation (Part B).

Part A: for each of the 3 governed training topology families
(golden-reference, branched-loop, loop-grid), generate 5 fresh, non-locked
scenarios and compute the real MODEL-INPUT `classical_prior`
(`hydroswarm.training.corpus.model_input_classical_prior`, the exact
algorithm Stage-F training tensors were built with) alongside the
`hydroswarm.classical.signature_policy.resolve_signature_mode` verdict for
that topology's real content hash. Then repeats the same construction for
one deliberately UNSEEN dev-only topology (`data/topologies/coastal-branch.
inp`, `scripts/generate_cycle_b_corpus.py`'s own `DEV_OOD_TOPOLOGY`) and
measures how different its resulting prior's shape is from both a uniform
distribution and the average governed-topology prior shape.

Part B: for 10 of the 15 governed-topology scenarios above, calls
`HydraulicFeatureBuilder.build(..., classical_prior=<override>)` and
`pipeline.model(built.batch)` DIRECTLY (bypassing `analyze()`'s automatic
prior computation -- same pattern as `train_serve_parity_full.py`'s
`_run_one`) under 4 classical_prior conditions to measure how load-bearing
the prior feature is for the neural head's own source-node prediction.

No locked-test access: only fresh WNTRScenarioGenerator-generated scenarios
with seeds outside every existing frozen/train/calibration/validation seed
set used elsewhere in this repo (2026081x is already used by
train_serve_parity_full.py/temporal_evidence_ablation.py; this uses
2026081_5xx, a visibly distinct base).
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.classical.signature_policy import resolve_signature_mode  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder  # noqa: E402
from hydroswarm.runtime.paths import resolve_v4_bundle_dir  # noqa: E402
from hydroswarm.runtime.v4_defaults import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.corpus import (  # noqa: E402
    build_feature_context,
    build_sensor_series,
    model_input_classical_prior,
    resolve_model_input_signature_library,
)

from generate_cycle_b_corpus import DEV_OOD_TOPOLOGY, TRAIN_TOPOLOGIES  # noqa: E402

GOVERNED_FAMILIES = ["golden-reference", "branched-loop", "loop-grid"]
# Distinct base from every seed family already used elsewhere in this
# session's diagnostic scripts (2026081300+, 2026081x00+): 20260815_00..14,
# 15 distinct seeds total, 5 per governed family (disjoint slices).
ALL_PARITY_SEEDS = [20260815_00 + i for i in range(15)]
SEEDS_BY_FAMILY = {
    family: ALL_PARITY_SEEDS[index * 5 : (index + 1) * 5] for index, family in enumerate(GOVERNED_FAMILIES)
}
COASTAL_SEED = 20260815_98


def _entropy(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-(arr * np.log(arr)).sum())


def _js_vs_uniform(prior: dict[str, float]) -> dict[str, float]:
    nodes = sorted(prior)
    p = np.asarray([max(prior[n], 0.0) for n in nodes], dtype=np.float64)
    total = p.sum()
    p = (p / total) if total > 0 else np.full_like(p, 1.0 / len(p))
    u = np.full_like(p, 1.0 / len(p))
    js = float(jensenshannon(p, u, base=2.0) ** 2)  # squared JS distance = JS divergence
    js_distance = float(jensenshannon(p, u, base=2.0))
    kl = float(np.sum(np.where(p > 0, p * np.log2(p / u), 0.0)))
    return {"js_divergence_bits_squared_distance": js, "js_distance_bits": js_distance, "kl_to_uniform_bits": kl, "n_nodes": len(nodes)}


def _sorted_shape_resampled(prior: dict[str, float], n_bins: int = 20) -> np.ndarray:
    """Node-count-independent 'shape' representation: sort probabilities
    descending, treat rank/N as the x-axis, linearly resample onto a fixed
    grid of n_bins points, renormalize to sum to 1. This lets priors over
    DIFFERENT junction counts be compared as shapes (peaked vs flat) --
    NOT as node-identity-aligned distributions, which is impossible across
    different topologies. Documented caveat: this discards which node is
    favored entirely; it only compares "how concentrated is the prior."
    """

    values = np.sort(np.asarray(list(prior.values()), dtype=np.float64))[::-1]
    total = values.sum()
    values = (values / total) if total > 0 else np.full_like(values, 1.0 / len(values))
    n = len(values)
    if n == 1:
        resampled = np.full(n_bins, 1.0)
    else:
        x_original = np.linspace(0.0, 1.0, n)
        x_target = np.linspace(0.0, 1.0, n_bins)
        resampled = np.interp(x_target, x_original, values)
    resampled_total = resampled.sum()
    return resampled / resampled_total if resampled_total > 0 else np.full(n_bins, 1.0 / n_bins)


def _network_for_family(family: str) -> Any:
    if family == "golden-reference":
        return build_wntr_network()
    loaders = dict(TRAIN_TOPOLOGIES)
    return loaders[family]()


def _generate_scenario(network: Any, seed: int, family: str, split: DatasetSplit) -> Any:
    generator = WNTRScenarioGenerator()
    config = ScenarioGenerationConfig(
        seed=seed,
        network_id=family,
        network_family=family,
        split=split,
        stage=CurriculumStage.OPERATIONAL,
        event_type=EventType.CONTAMINATION,
        pipe_outage_probability=0.0,
    )
    return generator.generate(network, config)


def _part_a_one(seed: int, family: str, split: DatasetSplit, network: Any | None = None) -> dict[str, Any]:
    if network is None:
        network = _network_for_family(family)
    scenario = _generate_scenario(network, seed, family, split)
    context = build_feature_context(network)
    series = build_sensor_series(scenario, context)
    junctions = tuple(sorted(network.junction_name_list))
    topology_hash = network_sha256(network)
    mode_direct = resolve_signature_mode(topology_hash)
    library, reference_timestamps, mode_via_resolver = resolve_model_input_signature_library(
        topology_hash, junctions, network
    )
    prior = model_input_classical_prior(library, junctions, series, reference_timestamps)
    return {
        "seed": seed,
        "topology_family": family,
        "split": split.value if hasattr(split, "value") else str(split),
        "scenario_id": str(scenario.manifest.scenario_id),
        "network_sha256": topology_hash,
        "n_junctions": len(junctions),
        "resolve_signature_mode_direct": mode_direct,
        "resolve_model_input_signature_library_mode": mode_via_resolver,
        "modes_agree": mode_direct == mode_via_resolver,
        "classical_prior": prior,
        "prior_entropy_bits": _entropy(list(prior.values())) / np.log(2),
        "prior_max_prob": max(prior.values()) if prior else None,
        "js_vs_uniform": _js_vs_uniform(prior),
        "source_node_truth": scenario.manifest.incident.source_nodes[0],
    }


def _part_b_one(seed: int, family: str, split: DatasetSplit, pipeline: Any) -> dict[str, Any]:
    network = _network_for_family(family)
    scenario = _generate_scenario(network, seed, family, split)
    context = build_feature_context(network)
    series = build_sensor_series(scenario, context)
    junctions = tuple(sorted(network.junction_name_list))
    topology_hash = network_sha256(network)
    library, reference_timestamps, _mode = resolve_model_input_signature_library(topology_hash, junctions, network)
    real_prior = model_input_classical_prior(library, junctions, series, reference_timestamps)
    uniform_prior = {node: 1.0 / len(junctions) for node in junctions}
    zeroed_prior = {node: 0.0 for node in junctions}
    truth = scenario.manifest.incident.source_nodes[0]

    builder = HydraulicFeatureBuilder(
        node_normalization=pipeline.feature_builder.node_normalization,
        edge_normalization=pipeline.feature_builder.edge_normalization,
    )
    window_steps = len(scenario.timestamps_seconds)

    conditions: dict[str, dict[str, Any]] = {}
    for condition_name, prior in [
        ("a_real_governed_prior", real_prior),
        ("b_uniform_prior", uniform_prior),
        ("c_oracle_prior_identical_to_a_for_governed_topology", real_prior),
        ("d_zeroed_prior", zeroed_prior),
    ]:
        try:
            built = builder.build(
                network, context.graph, context.state, series, classical_prior=prior, window_steps=window_steps
            )
            with torch.no_grad():
                pipeline.model.eval()
                output = pipeline.model(built.batch)
            logits = output["source_node_logits"][0, : len(built.node_ids)]
            probs = torch.softmax(logits, dim=-1).numpy()
            belief = dict(zip(built.node_ids, map(float, probs), strict=True))
            top1 = localization_top_k(belief, truth, k=1)
            top3 = localization_top_k(belief, truth, k=3)
            mrr = mean_reciprocal_rank([belief], [truth])
            conditions[condition_name] = {
                "ok": True,
                "top1": top1,
                "top3": top3,
                "reciprocal_rank": mrr,
                "true_source_probability": belief.get(truth, 0.0),
                "max_belief_probability": max(belief.values()) if belief else None,
            }
        except Exception as exc:  # noqa: BLE001 -- deliberately captured, reported
            conditions[condition_name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "seed": seed,
        "topology_family": family,
        "scenario_id": str(scenario.manifest.scenario_id),
        "source_node_truth": truth,
        "conditions": conditions,
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    # ---------------- Part A: parity + coastal-branch comparison ----------------
    part_a_results: list[dict[str, Any]] = []
    errors_a: list[dict[str, Any]] = []
    for family in GOVERNED_FAMILIES:
        for seed in SEEDS_BY_FAMILY[family]:
            try:
                result = _part_a_one(seed, family, DatasetSplit.VALIDATION)
                part_a_results.append(result)
            except Exception as exc:  # noqa: BLE001
                errors_a.append({"seed": seed, "family": family, "error": f"{type(exc).__name__}: {exc}"})

    coastal_family, coastal_loader = DEV_OOD_TOPOLOGY
    coastal_network = coastal_loader()
    try:
        coastal_result = _part_a_one(COASTAL_SEED, coastal_family, DatasetSplit.DEVELOPMENT_HOLDOUT, network=coastal_network)
    except Exception as exc:  # noqa: BLE001
        coastal_result = {"error": f"{type(exc).__name__}: {exc}"}

    governed_ok = [r for r in part_a_results if "classical_prior" in r]
    all_modes_governed = all(r["resolve_signature_mode_direct"] == "GOVERNED_KNOWN_NETWORK" for r in governed_ok)
    coastal_mode = coastal_result.get("resolve_signature_mode_direct")
    coastal_is_runtime_generated = coastal_mode == "RUNTIME_GENERATED_IMPORTED_NETWORK"

    # Shape comparison: coastal prior vs AVERAGE governed-prior shape,
    # both resampled to a fixed 20-bin sorted-descending-probability grid
    # (node-count-independent, but discards node identity -- see
    # _sorted_shape_resampled's own docstring for the honest caveat).
    shape_comparison: dict[str, Any] | None = None
    if governed_ok and "classical_prior" in coastal_result:
        governed_shapes = np.stack([_sorted_shape_resampled(r["classical_prior"]) for r in governed_ok])
        average_governed_shape = governed_shapes.mean(axis=0)
        average_governed_shape = average_governed_shape / average_governed_shape.sum()
        coastal_shape = _sorted_shape_resampled(coastal_result["classical_prior"])
        js_shape = float(jensenshannon(coastal_shape, average_governed_shape, base=2.0) ** 2)
        uniform_shape = np.full(20, 1.0 / 20)
        js_governed_avg_vs_uniform_shape = float(jensenshannon(average_governed_shape, uniform_shape, base=2.0) ** 2)
        shape_comparison = {
            "method": "sorted-descending-probability, resampled to 20 bins via linear interpolation, renormalized",
            "caveat": (
                "This compares CONCENTRATION SHAPE only (how peaked vs flat each prior is), not node-identity "
                "alignment -- coastal-branch (6 junctions) and the governed families (4/7/8 junctions) have no "
                "shared node space, so a node-aligned JS divergence is not meaningful across topologies. This "
                "resampling is a best-effort proxy, not a rigorous distributional equivalence claim."
            ),
            "coastal_shape_resampled": coastal_shape.tolist(),
            "average_governed_shape_resampled": average_governed_shape.tolist(),
            "js_divergence_coastal_vs_average_governed_shape": js_shape,
            "js_divergence_average_governed_shape_vs_uniform": js_governed_avg_vs_uniform_shape,
            "js_divergence_coastal_vs_uniform_raw": coastal_result.get("js_vs_uniform", {}).get(
                "js_divergence_bits_squared_distance"
            ),
            "interpretation": (
                "If js_divergence_coastal_vs_average_governed_shape is small relative to "
                "js_divergence_coastal_vs_uniform_raw / js_divergence_average_governed_shape_vs_uniform, the "
                "runtime-generated (coastal) prior has a SIMILARLY-SHAPED (similarly peaked/flat) distribution "
                "to the governed priors, not a degenerate/collapsed-to-uniform one, despite using the "
                "RUNTIME_GENERATED_IMPORTED_NETWORK signature path rather than a committed training artifact."
            ),
        }

    # ---------------- Part B: prior ablation (10 of the 15 Part-A scenarios) ----------------
    factory = V4PipelineFactory(resolve_v4_bundle_dir())
    pipeline = factory(None, str(ROOT / "data" / "frozen" / "golden_network.inp"))

    part_b_scenarios = [(r["seed"], r["topology_family"]) for r in governed_ok][:10]
    part_b_results: list[dict[str, Any]] = []
    errors_b: list[dict[str, Any]] = []
    for seed, family in part_b_scenarios:
        try:
            part_b_results.append(_part_b_one(seed, family, DatasetSplit.VALIDATION, pipeline))
        except Exception as exc:  # noqa: BLE001
            errors_b.append({"seed": seed, "family": family, "error": f"{type(exc).__name__}: {exc}"})

    condition_names = [
        "a_real_governed_prior",
        "b_uniform_prior",
        "c_oracle_prior_identical_to_a_for_governed_topology",
        "d_zeroed_prior",
    ]
    condition_summary: dict[str, Any] = {}
    zeroed_prior_safe = True
    for name in condition_names:
        ok_records = [r["conditions"][name] for r in part_b_results if r["conditions"].get(name, {}).get("ok")]
        error_records = [r["conditions"][name] for r in part_b_results if not r["conditions"].get(name, {}).get("ok", False)]
        if name == "d_zeroed_prior" and error_records:
            zeroed_prior_safe = False
        condition_summary[name] = {
            "n": len(part_b_results),
            "n_ok": len(ok_records),
            "n_errors": len(error_records),
            "errors": error_records,
            "top1": (sum(r["top1"] for r in ok_records) / len(ok_records)) if ok_records else None,
            "top3": (sum(r["top3"] for r in ok_records) / len(ok_records)) if ok_records else None,
            "mrr": (sum(r["reciprocal_rank"] for r in ok_records) / len(ok_records)) if ok_records else None,
            "mean_true_source_probability": (
                (sum(r["true_source_probability"] for r in ok_records) / len(ok_records)) if ok_records else None
            ),
        }

    real_top1 = condition_summary["a_real_governed_prior"]["top1"]
    uniform_top1 = condition_summary["b_uniform_prior"]["top1"]
    zeroed_top1 = condition_summary["d_zeroed_prior"]["top1"]
    prior_is_load_bearing = (
        real_top1 is not None
        and uniform_top1 is not None
        and (real_top1 - uniform_top1) > 0.15
    )

    locked_after = locked_test_opened(ROOT)
    report = {
        "schema_version": 1,
        "sections": "16_17_classical_prior_parity_and_ablation",
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "part_a_parity": {
            "governed_families": GOVERNED_FAMILIES,
            "seeds_by_family": SEEDS_BY_FAMILY,
            "n_scenarios": len(part_a_results) + len(errors_a),
            "n_ok": len(part_a_results),
            "n_errors": len(errors_a),
            "errors": errors_a,
            "all_governed_scenarios_resolve_governed_known_network": all_modes_governed,
            "all_resolver_modes_agree_with_direct_resolve_signature_mode": all(
                r["modes_agree"] for r in governed_ok
            ),
            "results": part_a_results,
            "coastal_branch_unseen_topology": {
                "seed": COASTAL_SEED,
                "result": coastal_result,
                "resolves_to_runtime_generated_imported_network": coastal_is_runtime_generated,
            },
            "shape_comparison_coastal_vs_governed_average": shape_comparison,
        },
        "part_b_ablation": {
            "n_scenarios_used": len(part_b_scenarios),
            "n_errors_building_scenarios": len(errors_b),
            "errors_building_scenarios": errors_b,
            "conditions": condition_names,
            "condition_notes": {
                "a_real_governed_prior": "Baseline: real governed classical_prior computed via model_input_classical_prior.",
                "b_uniform_prior": "Uniform 1/N prior over junctions -- tests whether the model needs a peaked prior at all.",
                "c_oracle_prior_identical_to_a_for_governed_topology": (
                    "Identical to condition (a) for these governed topologies -- there is no separate 'oracle' "
                    "prior available beyond the real governed one for a topology already in the governed set. "
                    "Reported explicitly as a duplicate rather than padded with a fabricated distinct condition, "
                    "per the diagnostic protocol's rule-zero prohibition on invented numbers."
                ),
                "d_zeroed_prior": "All-zero classical_prior tensor -- tests whether the forward pass tolerates a degenerate all-zero prior feature.",
            },
            "zeroed_prior_forward_pass_safe": zeroed_prior_safe,
            "per_scenario_results": part_b_results,
            "condition_summary": condition_summary,
            "prior_load_bearing_verdict": {
                "real_top1": real_top1,
                "uniform_top1": uniform_top1,
                "zeroed_top1": zeroed_top1,
                "real_minus_uniform_top1": (real_top1 - uniform_top1) if (real_top1 is not None and uniform_top1 is not None) else None,
                "classical_prior_is_load_bearing_for_neural_head": prior_is_load_bearing,
                "interpretation": (
                    "A large drop from real to uniform/zeroed prior means the model's source_node prediction is "
                    "substantially anchored to the classical_prior FEATURE itself (not purely learned from raw "
                    "hydraulic/temporal features), which would make any classical-prior train/serve mismatch "
                    "(Section E hypothesis) a first-order contributor to LIVE's localization gap. A small/no "
                    "drop means the classical prior is a secondary signal for this head."
                ),
            },
        },
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "classical-prior-parity.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_governed_scenarios_resolve_governed_known_network": all_modes_governed,
        "coastal_resolves_to_runtime_generated_imported_network": coastal_is_runtime_generated,
        "js_divergence_coastal_vs_uniform": coastal_result.get("js_vs_uniform"),
        "condition_summary_top1": {k: v["top1"] for k, v in condition_summary.items()},
        "zeroed_prior_forward_pass_safe": zeroed_prior_safe,
        "classical_prior_is_load_bearing_for_neural_head": prior_is_load_bearing,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
