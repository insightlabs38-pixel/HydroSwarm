"""Capability diagnostic Section 23: topology generalization decomposition.

Runs the REAL, unmodified, frozen production pipeline
(`hydroswarm.runtime.v4_defaults.V4PipelineFactory`, the exact class
`hydroswarm.api.app` uses) against four conditions that isolate structural
distance from the golden-reference training topology from mere hydraulic
configuration randomization:

  (A) pristine golden-reference -- the exact in-memory construction
      (`hydroswarm.simulation.network.build_wntr_network`) used by this
      repo's established diagnostic scripts (train_serve_parity_full.py,
      temporal_evidence_ablation.py) and by Cycle B corpus generation for
      the golden-reference family.
  (B) randomized physical configuration of golden-reference -- SAME
      generator/topology, but `ScenarioGenerationConfig` carries non-zero
      roughness_variation_fraction/tank_level_variation_fraction/demand_regimes,
      and (unlike A) the REAL randomized WNTR model returned by
      `generate_with_network` (not the pristine network) is fed to
      `pipeline.analyze()`, so both the simulated evidence AND the network
      object's own structural hash reflect the randomization -- this is
      the condition that actually exercises the calibration/topology-hash
      identity path a real "same family, different physical state" served
      network would.
  (C) another governed family (branched-loop) -- parsed from its
      canonical .inp file exactly like Cycle B corpus generation did.
  (D) coastal-branch -- a development-transfer topology never in any
      training/calibration split, parsed from its canonical .inp file.

For each condition, 8 fresh, non-locked, seed-distinct scenarios are run
through the real `pipeline.analyze()`, and neural/classical/fused top1,
top3, and MRR are read directly off `IncidentAnalysisResult`. Real
structural descriptors (node/link counts, degree distribution, cycle
rank) are computed via `networkx` directly on `context.graph` (an
`nx.MultiDiGraph`, per `hydroswarm.training.corpus.build_feature_context`).

No production code is modified. No locked-test path is read.
"""

from __future__ import annotations

import json
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.runtime.paths import resolve_v4_bundle_dir  # noqa: E402
from hydroswarm.runtime.v4_defaults import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402

import wntr  # noqa: E402

# Distinct, non-colliding seed base: every other diagnostic script this
# session uses 20260813xx (train/temporal parity) or 20260899 (reserved,
# protocol.json, confirmation holdout). This section uses 20260821xx.
TOPOLOGY_SEEDS = [20260821_00 + i for i in range(8)]

NETWORK_PATHS = {
    "golden-reference": ROOT / "data" / "frozen" / "golden_network.inp",
    "branched-loop": ROOT / "data" / "topology-transfer" / "branched-loop.inp",
    "loop-grid": ROOT / "data" / "topologies" / "loop-grid.inp",
    "coastal-branch": ROOT / "data" / "topologies" / "coastal-branch.inp",
}


def _rank_metrics(belief: dict[str, float] | None, truth: str) -> dict[str, Any]:
    if not belief or sum(belief.values()) <= 0:
        return {"top1": None, "top3": None, "mrr": None, "true_source_probability": None}
    return {
        "top1": localization_top_k(belief, truth, k=1),
        "top3": localization_top_k(belief, truth, k=3),
        "mrr": mean_reciprocal_rank([belief], [truth]),
        "true_source_probability": belief.get(truth, 0.0),
    }


def _structural_descriptors(graph: nx.MultiDiGraph, network: Any) -> dict[str, Any]:
    undirected = graph.to_undirected()
    degrees = [degree for _node, degree in graph.degree()]
    n_nodes = graph.number_of_nodes()
    n_edges_directed = graph.number_of_edges()
    n_components = nx.number_connected_components(undirected)
    # Cycle rank (first Betti number) of the undirected simple projection:
    # E - N + C. MultiDiGraph.to_undirected() collapses parallel/anti-
    # parallel directed edges into single undirected edges by default in
    # networkx, which is the correct simple-graph convention for a real
    # loop count (parallel pipes between the same two nodes would
    # otherwise double-count as extra "loops" that are not really
    # independent hydraulic loops in this network's topology).
    n_edges_undirected = undirected.number_of_edges()
    cycle_rank = n_edges_undirected - n_nodes + n_components
    return {
        "junction_count": len(getattr(network, "junction_name_list", [])),
        "link_count": len(getattr(network, "link_name_list", [])),
        "graph_node_count": n_nodes,
        "graph_edge_count_directed": n_edges_directed,
        "graph_edge_count_undirected": n_edges_undirected,
        "connected_components": n_components,
        "cycle_rank": cycle_rank,
        "degree_mean": statistics.fmean(degrees) if degrees else None,
        "degree_stdev": statistics.pstdev(degrees) if len(degrees) > 1 else 0.0,
        "degree_min": min(degrees) if degrees else None,
        "degree_max": max(degrees) if degrees else None,
    }


def _load_network(condition: str) -> Any:
    if condition == "golden-reference":
        return build_wntr_network()
    return wntr.network.WaterNetworkModel(str(NETWORK_PATHS[condition]))


def _run_condition_A_pristine(pipeline: Any, generator: WNTRScenarioGenerator) -> dict[str, Any]:
    network = build_wntr_network()
    context = build_feature_context(network)
    descriptors = _structural_descriptors(context.graph, network)
    per_scenario = []
    for seed in TOPOLOGY_SEEDS:
        config = ScenarioGenerationConfig(
            seed=seed, network_id="golden-reference", network_family="golden-reference",
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
        )
        scenario = generator.generate(network, config)
        truth = scenario.manifest.incident.source_nodes[0]
        series = build_sensor_series(scenario, context)
        try:
            result = pipeline.analyze(uuid.uuid4(), network, series)
            per_scenario.append({
                "seed": seed, "network_sha256": network_sha256(network),
                "neural": _rank_metrics(dict(result.neural_belief) if result.neural_belief else None, truth),
                "classical": _rank_metrics(dict(result.classical_belief), truth),
                "fused": _rank_metrics(dict(result.fused_belief), truth),
                "calibrated": result.calibrated,
                "ood_level": result.ood_level.value,
            })
        except Exception as exc:  # noqa: BLE001
            per_scenario.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    return {"structural_descriptors": descriptors, "per_scenario": per_scenario}


def _run_condition_B_randomized_config(pipeline: Any, generator: WNTRScenarioGenerator) -> dict[str, Any]:
    pristine_network = build_wntr_network()
    pristine_context = build_feature_context(pristine_network)
    descriptors = _structural_descriptors(pristine_context.graph, pristine_network)
    per_scenario = []
    for seed in TOPOLOGY_SEEDS:
        config = ScenarioGenerationConfig(
            seed=seed, network_id="golden-reference", network_family="golden-reference",
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
            roughness_variation_fraction=0.30, tank_level_variation_fraction=0.30,
            demand_regimes=(0.7, 1.0, 1.3),
        )
        scenario, randomized_model = generator.generate_with_network(pristine_network, config)
        truth = scenario.manifest.incident.source_nodes[0]
        # core-issues.txt repair item 4 (cited in scenarios.py's
        # generate_with_network docstring): a corpus/eval builder must
        # construct the feature context from THIS randomized model, not
        # the pristine network -- otherwise the physical configuration
        # change this condition exists to test would be silently
        # discarded before it ever reaches the pipeline.
        randomized_context = build_feature_context(randomized_model)
        series = build_sensor_series(scenario, randomized_context)
        try:
            result = pipeline.analyze(uuid.uuid4(), randomized_model, series)
            per_scenario.append({
                "seed": seed, "network_sha256": network_sha256(randomized_model),
                "neural": _rank_metrics(dict(result.neural_belief) if result.neural_belief else None, truth),
                "classical": _rank_metrics(dict(result.classical_belief), truth),
                "fused": _rank_metrics(dict(result.fused_belief), truth),
                "calibrated": result.calibrated,
                "ood_level": result.ood_level.value,
            })
        except Exception as exc:  # noqa: BLE001
            per_scenario.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "structural_descriptors": descriptors,
        "structural_descriptors_note": (
            "Identical to condition A -- roughness/tank-level/demand randomization changes hydraulic "
            "PARAMETER VALUES on existing links/nodes, never graph connectivity (no nodes/links added or "
            "removed), so node/link counts and degree sequence are structurally invariant under this "
            "condition by construction; only network_sha256 (which folds in roughness) differs per scenario."
        ),
        "per_scenario": per_scenario,
    }


def _run_condition_pristine_other_family(
    pipeline_factory: V4PipelineFactory, generator: WNTRScenarioGenerator, family: str, network_path: Path
) -> dict[str, Any]:
    network = wntr.network.WaterNetworkModel(str(network_path))
    pipeline = pipeline_factory(None, network_path)
    context = build_feature_context(network)
    descriptors = _structural_descriptors(context.graph, network)
    per_scenario = []
    for seed in TOPOLOGY_SEEDS:
        config = ScenarioGenerationConfig(
            seed=seed, network_id=family, network_family=family,
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
        )
        scenario = generator.generate(network, config)
        truth = scenario.manifest.incident.source_nodes[0]
        series = build_sensor_series(scenario, context)
        try:
            result = pipeline.analyze(uuid.uuid4(), network, series)
            per_scenario.append({
                "seed": seed, "network_sha256": network_sha256(network),
                "neural": _rank_metrics(dict(result.neural_belief) if result.neural_belief else None, truth),
                "classical": _rank_metrics(dict(result.classical_belief), truth),
                "fused": _rank_metrics(dict(result.fused_belief), truth),
                "calibrated": result.calibrated,
                "ood_level": result.ood_level.value,
            })
        except Exception as exc:  # noqa: BLE001
            per_scenario.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
    return {"structural_descriptors": descriptors, "per_scenario": per_scenario}


def _aggregate(per_scenario: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in per_scenario if "error" not in r]
    out: dict[str, Any] = {"n": len(per_scenario), "n_ok": len(ok), "n_errors": len(per_scenario) - len(ok)}
    for arm in ("neural", "classical", "fused"):
        vals = [r[arm] for r in ok if r.get(arm, {}).get("top1") is not None]
        out[arm] = {
            "n_with_metrics": len(vals),
            "top1": (sum(v["top1"] for v in vals) / len(vals)) if vals else None,
            "top3": (sum(v["top3"] for v in vals) / len(vals)) if vals else None,
            "mrr": (sum(v["mrr"] for v in vals) / len(vals)) if vals else None,
            "mean_true_source_probability": (
                sum(v["true_source_probability"] for v in vals) / len(vals)
            ) if vals else None,
        }
    out["calibrated_rate"] = (
        sum(1 for r in ok if r.get("calibrated")) / len(ok)
    ) if ok else None
    return out


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"
    locked_before = locked_test_opened(ROOT)

    factory = V4PipelineFactory(resolve_v4_bundle_dir())
    golden_pipeline = factory(None, NETWORK_PATHS["golden-reference"])
    generator = WNTRScenarioGenerator()

    conditions: dict[str, Any] = {}
    conditions["A_pristine_golden_reference"] = _run_condition_A_pristine(golden_pipeline, generator)
    conditions["B_randomized_physical_config_golden_reference"] = _run_condition_B_randomized_config(
        golden_pipeline, generator
    )
    conditions["C_another_governed_family_branched_loop"] = _run_condition_pristine_other_family(
        factory, generator, "branched-loop", NETWORK_PATHS["branched-loop"]
    )
    conditions["D_development_transfer_coastal_branch"] = _run_condition_pristine_other_family(
        factory, generator, "coastal-branch", NETWORK_PATHS["coastal-branch"]
    )

    summary = {}
    golden_descriptors = conditions["A_pristine_golden_reference"]["structural_descriptors"]
    for key, payload in conditions.items():
        agg = _aggregate(payload["per_scenario"])
        descriptors = payload["structural_descriptors"]
        node_count_diff = abs(descriptors["graph_node_count"] - golden_descriptors["graph_node_count"])
        degree_mean_diff = (
            abs(descriptors["degree_mean"] - golden_descriptors["degree_mean"])
            if descriptors["degree_mean"] is not None and golden_descriptors["degree_mean"] is not None
            else None
        )
        cycle_rank_diff = abs(descriptors["cycle_rank"] - golden_descriptors["cycle_rank"])
        summary[key] = {
            "aggregate": agg,
            "structural_descriptors": descriptors,
            "node_count_diff_from_golden_reference": node_count_diff,
            "degree_mean_diff_from_golden_reference": degree_mean_diff,
            "cycle_rank_diff_from_golden_reference": cycle_rank_diff,
        }

    # Qualitative, data-grounded correlation check: rank conditions by
    # each of (node_count_diff, degree_mean_diff, cycle_rank_diff) versus
    # rank by fused top1, using only the 4 real condition-level means
    # (n=4 -- explicitly too small for a real Spearman correlation
    # p-value; this reports direction/ordering only, honestly labeled).
    ordering_by_fused_top1 = sorted(
        summary, key=lambda key: (summary[key]["aggregate"]["fused"]["top1"] or -1), reverse=True
    )
    ordering_by_node_count_diff = sorted(summary, key=lambda key: summary[key]["node_count_diff_from_golden_reference"])
    ordering_by_cycle_rank_diff = sorted(summary, key=lambda key: summary[key]["cycle_rank_diff_from_golden_reference"])

    report = {
        "schema_version": 1,
        "section": "23_topology_generalization_decomposition",
        "locked_test_opened_before": locked_before,
        "n_scenarios_per_condition": len(TOPOLOGY_SEEDS),
        "seeds": TOPOLOGY_SEEDS,
        "methodology_note": (
            "Conditions A/C/D pass the SAME (pristine, unrandomized) network object used to build "
            "feature context/sensor series directly to pipeline.analyze(), matching this repo's existing "
            "diagnostic convention (train_serve_parity_full.py, temporal_evidence_ablation.py). Condition B "
            "instead passes the REAL randomized WNTR model returned by generate_with_network() (not the "
            "pristine network) so that its structural hash and hydraulic evidence both genuinely reflect "
            "the randomized physical configuration -- otherwise a randomized-but-then-discarded config "
            "would be indistinguishable from condition A at the analyze() call boundary."
        ),
        "conditions": conditions,
        "summary": summary,
        "structural_distance_vs_transfer_quality": {
            "ordering_by_fused_top1_desc": ordering_by_fused_top1,
            "ordering_by_node_count_diff_asc": ordering_by_node_count_diff,
            "ordering_by_cycle_rank_diff_asc": ordering_by_cycle_rank_diff,
            "orderings_agree_node_count": ordering_by_fused_top1 == ordering_by_node_count_diff,
            "orderings_agree_cycle_rank": ordering_by_fused_top1 == ordering_by_cycle_rank_diff,
            "caveat": (
                "n=4 conditions -- far too small for a real correlation statistic (no p-value/coefficient "
                "is computed or claimed). This reports whether the SAME ordinal ranking holds under both "
                "'transfer quality' and each structural-distance proxy, as a qualitative, data-grounded "
                "observation only, per the task's own framing."
            ),
        },
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "topology-analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["structural_distance_vs_transfer_quality"], indent=2, default=str))
    for key, payload in summary.items():
        print(key, "fused_top1=", payload["aggregate"]["fused"]["top1"], "calibrated_rate=", payload["aggregate"]["calibrated_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
