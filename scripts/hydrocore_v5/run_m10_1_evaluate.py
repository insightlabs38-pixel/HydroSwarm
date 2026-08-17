"""Milestone 10.1: OOD/fusion validation -- fresh development-only
population generation and inference across the M10 protocol's frozen
conditions, for the frozen M9 HydroCore-S predictor.

Frozen protocol: docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md Sections
3/4/5/6. FROZEN-CHECKPOINT EVALUATION ONLY: no training, no tuning, no
calibration refit. Reuses UNMODIFIED: run_m7_topology._infer/_classical_belief
(model-independent-except-for-model-arg inference plumbing, already
extensively tested), hydroswarm.inference.fusion.fuse_source_probabilities
(unchanged), hydroswarm.inference.ood.OODDetector (unchanged).

Four comparator arms per row (Section 2 of the protocol):
  A: deterministic OODDetector.evaluate() combined score (classical-only)
  B: neural S + frozen calibration, ood_category NOT surfaced (today's
     production default)
  C: neural ood_category_head surfaced (evaluation-only; NOT wired to any
     live control decision -- Section 9 of the protocol)
  D: existing fuse_source_probabilities dynamic-trust fusion (already live)

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-population-policy.json
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-canonical-results.jsonl
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-manifest.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.classical.state_estimation import EstimatedHydraulicState, StateResidualReport  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage as ScenarioCurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.inference.fusion import TrustFeatures, fuse_source_probabilities  # noqa: E402
from hydroswarm.inference.ood import OODDetector  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import scenario_to_prefix_example, truncate_causal_prefix  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402

import m10_common as m10  # noqa: E402
from run_m7_topology import _classical_belief, _infer  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG, _build_libraries, _library_for  # noqa: E402

REPEATS_PER_SOURCE = 3
#: EARLY-representative and MATURE-representative depths only -- a
#: deliberately reduced grid (vs M9's full 7-depth grid) to keep this
#: characterization milestone's runtime bounded; disclosed explicitly in
#: the manifest, not silently narrowed.
DEPTHS = (2, 12)

#: Section 5 of the protocol: explicit, disclosed perturbation parameters
#: for the two non-identity conditions. Frozen before execution.
CONDITION_OVERRIDES: dict[str, dict[str, Any]] = {
    "IN_DISTRIBUTION": {},
    "SENSOR_DROPOUT": {"missing_probability": 0.35, "communication_outage_probability": 0.15},
    "SEVERITY_SHIFT": {"strength_bins": (2.0, 3.0, 4.0)},
}

#: Deliberately-disclosed stub: no HydraulicStateEstimator reconciliation is
#: performed in this milestone (out of scope -- see M10 protocol Section 5),
#: so OODDetector.evaluate()'s demand_shift channel always reads its
#: documented `.get("demand", 0.0)` fallback rather than a real reconciled
#: mismatch score. latent_distance likewise reads 0.0 (no v5 latent
#: reference has ever been fit -- OODReference() default, honest, not
#: fabricated).
_STUB_HYDRAULIC_STATE = EstimatedHydraulicState(
    timestamp_seconds=0, pressure_m={}, demand_m3s={}, flow_m3s={}, velocity_mps={},
    tank_level_m={}, pump_open={}, valve_open={}, zone_demand_multipliers={},
    residuals=StateResidualReport(
        pressure_rmse_m=0.0, demand_rmse_m3s=0.0, flow_rmse_m3s=0.0,
        missing_values_imputed=0, reconciled_pumps=(), reconciled_valves=(), mismatch_scores={},
    ),
)


def _load_s_checkpoint(seed: int) -> HydroCore:
    record = m10.canonical_s_checkpoint(seed)
    model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(record["canonical_export_path"], device="cpu"), strict=True)
    model.eval()
    return model


def _generate_condition_scenarios(family: str, loader: Any, condition: str, condition_index: int) -> list[tuple[Any, Any, Any, dict[str, Any]]]:
    junctions = m10.full_junction_list(family, loader)
    seed_base = m10.m10_1_seed_base(family, condition, condition_index)
    generator = WNTRScenarioGenerator()
    overrides = CONDITION_OVERRIDES[condition]
    out: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    for source_index, source in enumerate(junctions):
        for repeat in range(REPEATS_PER_SOURCE):
            seed = seed_base + source_index * m10.M10_1_SOURCE_STRIDE + repeat
            network = loader()
            config = ScenarioGenerationConfig(
                seed=seed, network_id=family, network_family=family,
                split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=ScenarioCurriculumStage.OPERATIONAL,
                event_type=EventType.CONTAMINATION, source_node=source,
                sensor_count=min(len(junctions), 4), pipe_outage_probability=0.0,
                **overrides,
            )
            scenario, randomized_network = generator.generate_with_network(network, config)
            context = build_feature_context(randomized_network)
            covariates = {
                "family": family, "condition": condition, "source_node": source, "source_index": source_index,
                "repeat": repeat, "generator_seed": seed, "known_family": family in m10.TRAINED_FAMILIES,
                "scenario_id": str(scenario.manifest.scenario_id), "network_hash": scenario.manifest.network_sha256,
                "relative_strength": scenario.manifest.incident.relative_strength,
            }
            out.append((scenario, randomized_network, context, covariates))
    return out


def main() -> None:
    m10.M10_1_DIR.mkdir(parents=True, exist_ok=True)
    branch = m10.current_branch()
    assert branch == m10.FROZEN_BRANCH
    locked_before = m10.assert_locked_test_closed()
    start = time.time()

    print("building classical libraries (M9.0a machinery, unmodified)...", flush=True)
    libraries = _build_libraries()

    print("generating M10.1 development-only OOD population...", flush=True)
    pools: dict[tuple[str, str], list[tuple[Any, Any, Any, dict[str, Any]]]] = {}
    for family in m10.ALL_FAMILIES:
        loader = m10.ALL_FAMILY_LOADERS[family]
        for condition_index, condition in enumerate(m10.M10_1_CONDITIONS):
            pools[(family, condition)] = _generate_condition_scenarios(family, loader, condition, condition_index)
    total_scenarios = sum(len(v) for v in pools.values())
    print(f"generated {total_scenarios} scenarios across {len(m10.ALL_FAMILIES)} families x {len(m10.M10_1_CONDITIONS)} conditions", flush=True)

    ood_detector = OODDetector()
    rows: list[dict[str, Any]] = []
    for seed in m10.SEEDS:
        print(f"loading S checkpoint seed {seed}...", flush=True)
        model = _load_s_checkpoint(seed)
        for family in m10.ALL_FAMILIES:
            library = _library_for(libraries, family, "ARM_B2" if family != "golden-reference" else "ARM_B")
            for condition in m10.M10_1_CONDITIONS:
                for scenario, network, context, covariates in pools[(family, condition)]:
                    full_series = build_sensor_series(scenario, context)
                    reference_timestamps = full_series[0].timestamps_seconds if full_series else ()
                    truth_node = scenario.manifest.incident.source_nodes[0]
                    for depth in DEPTHS:
                        series = [truncate_causal_prefix(item, depth) for item in full_series]
                        result = _infer(model, network, context, series, library, reference_timestamps)
                        node_ids = result["node_ids"]
                        if truth_node not in node_ids:
                            continue
                        truth_index = node_ids.index(truth_node)
                        classical_belief, residual_raw = _classical_belief(library, node_ids, series, reference_timestamps)

                        # _infer (above) already ran the model but only
                        # returns source_node_logits/evidence_sufficiency;
                        # ood_category_logits needs a second extraction on
                        # the SAME inputs -- matching _infer's own
                        # HydraulicFeatureBuilder construction exactly, not a
                        # different one.
                        from hydroswarm.preprocessing import HydraulicFeatureBuilder
                        from run_m7_topology import model_input_classical_prior
                        classical_prior = model_input_classical_prior(library, list(library.node_ids), series, reference_timestamps)
                        window_steps = max(len(item.timestamps_seconds) for item in series)
                        built_batch = HydraulicFeatureBuilder().build(
                            network, context.graph, context.state, series,
                            classical_prior=classical_prior, window_steps=window_steps,
                        )
                        with torch.no_grad():
                            output = model(built_batch.batch)
                        ood_category_logits = output["ood_category_logits"][0].detach().numpy()
                        ood_category_probs = torch.softmax(output["ood_category_logits"][0], dim=-1).detach().numpy()

                        healthy = float(np.mean([item.health[-1] for item in series])) if series else 1.0
                        missing = float(np.mean([float(item.missing[-1]) for item in series])) if series else 0.0

                        # Comparator A: deterministic OODDetector (Section 2).
                        components, level = ood_detector.evaluate(
                            node_count=len(node_ids), network_hash=covariates["network_hash"],
                            state=_STUB_HYDRAULIC_STATE, sensor_series=series,
                            latent=None, neural_probabilities=np.asarray(result["neural_probs"]),
                        )

                        # Comparator D: existing dynamic-trust fusion (Section 2).
                        mask = np.ones(len(node_ids), dtype=bool)
                        classical_array = np.asarray(classical_belief, dtype=np.float64)
                        classical_array = classical_array / max(float(classical_array.sum()), 1e-9)
                        features = TrustFeatures(
                            healthy_sensor_fraction=float(np.clip(healthy, 0.0, 1.0)),
                            missing_rate=float(np.clip(missing, 0.0, 1.0)),
                            normalized_residual=0.0, hydraulic_uncertainty=0.0,
                            neural_entropy=float(-np.sum(np.clip(result["neural_probs"], 1e-9, None) * np.log2(np.clip(result["neural_probs"], 1e-9, None)))) / max(np.log2(len(node_ids)), 1e-9),
                            classical_entropy=float(-np.sum(np.clip(classical_array, 1e-9, None) * np.log2(np.clip(classical_array, 1e-9, None)))) / max(np.log2(len(node_ids)), 1e-9),
                            ood_score=components.combined,
                        )
                        fused, diag = fuse_source_probabilities(result["neural_logits"], classical_array, mask, features)

                        rows.append({
                            "seed": seed, "family": family, "condition": condition, "known_family": covariates["known_family"],
                            "depth": depth, "depth_bucket": m10.depth_bucket_of(depth), "scenario_id": covariates["scenario_id"],
                            "source_index": covariates["source_index"], "repeat": covariates["repeat"],
                            "truth_index": truth_index, "node_ids": node_ids,
                            "neural_probs": result["neural_probs"], "classical_probs": classical_array.tolist(),
                            "fused_probs": fused.tolist(), "fusion_trust": diag.classical_trust, "fusion_disagreement_js": diag.disagreement_js,
                            "ood_combined": components.combined, "ood_level": level.value,
                            "ood_network_novelty": components.network_novelty, "ood_energy": components.energy,
                            "ood_category_probs": ood_category_probs.tolist(), "ood_category_argmax": int(np.argmax(ood_category_probs)),
                            "evidence_sufficiency": result["evidence_sufficiency"], "healthy_sensor_fraction": healthy, "missing_rate": missing,
                            "relative_strength": covariates["relative_strength"],
                            "runtime_condition": result["condition"],
                        })

    locked_after = m10.assert_locked_test_closed()
    with (m10.M10_1_DIR / "m10-1-canonical-results.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    manifest = {
        "kind": "M10_1_EVALUATION_MANIFEST", "milestone": "M10.1", "branch": branch, "commit": m10.current_commit(),
        "seeds": list(m10.SEEDS), "families": list(m10.ALL_FAMILIES), "trained_families": list(m10.TRAINED_FAMILIES),
        "conditions": list(m10.M10_1_CONDITIONS), "condition_overrides": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in CONDITION_OVERRIDES.items()},
        "depths": list(DEPTHS), "repeats_per_source": REPEATS_PER_SOURCE, "n_rows": len(rows), "n_scenarios": total_scenarios,
        "seed_base": m10.M10_1_SEED_BASE, "source_stride": m10.M10_1_SOURCE_STRIDE,
        "demand_shift_channel_disclosure": "OODDetector.evaluate()'s demand_shift always reads the documented .get('demand', 0.0) fallback -- no HydraulicStateEstimator reconciliation performed in M10.1 (out of scope, disclosed not fabricated).",
        "latent_distance_channel_disclosure": "always 0.0 -- no v5 latent reference (OODReference.latent_center) has ever been fit; OODDetector's own default-empty-tuple graceful degradation, not a bug.",
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "wall_seconds": time.time() - start, "environment": m10.environment_info(),
    }
    (m10.M10_1_DIR / "m10-1-manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    population_policy = {
        "kind": "M10_1_POPULATION_POLICY", "seed_namespace_role": "ood_development_m10_1",
        "seed_base": m10.M10_1_SEED_BASE, "source_stride": m10.M10_1_SOURCE_STRIDE,
        "disjoint_from_all_prior_milestones": True, "uses_locked_final_test": False, "uses_locked_topology_test": False,
        "split_discipline": "physical scenarios split (by source_index/seed) before any causal-prefix truncation; derived depth-truncated variants of one scenario share the same scenario_id and remain together",
    }
    (m10.M10_1_DIR / "m10-1-population-policy.json").write_text(json.dumps(population_policy, indent=2, default=str) + "\n")

    print(f"M10.1 evaluate complete: {len(rows)} rows, {total_scenarios} scenarios, {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
