"""Build the per-example failure-mode diagnostic table from the FROZEN
M11.6 locked evidence (branch exp/failure-mode-diagnostics, Phase 2).

READ-ONLY with respect to every frozen artifact it touches:

  - data/locked/m11-6/locked_final_test/scenarios.jsonl   (105 known-family incidents)
  - data/locked/m11-6/locked_topology_test/scenarios.jsonl (20 novel-topology incidents)
  - data/locked/m11-6/topologies/locked-topology-{0,1,2,3}.inp
  - data/frozen/golden_network.inp, data/topology-transfer/branched-loop.inp,
    data/topologies/loop-grid.inp  (canonical known-family structures --
    confirmed by node-count match against dataset-report.json's
    topology_node_counts / m11-6a novelty spec's prior_topologies before use)
  - reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-raw-incidents.jsonl
    (the 125 per-incident prediction/outcome records the locked evaluation
    harness itself already produced)

This script never opens, re-simulates, retrains against, or writes to any
of the above. Its only output is a NEW file under
reports/evaluation/failure-mode-diagnostics/ (an explicitly experimental,
non-release location distinct from every m9-*/m10-*/m11-*/locked path).

Every column in the emitted table is tagged "recorded" (taken verbatim from
a frozen artifact) or "derived" (computed here, e.g. graph centrality) --
see COLUMN_PROVENANCE at the bottom. No column is invented for a field that
does not exist in the source data (e.g. exact realized incident strength/
duration/start-time and sensor node identity are NOT in the locked
scenario records -- only condition-level categorical/range metadata is --
so they are omitted rather than guessed; see the diagnostic plan doc).

Usage: python3 scripts/hydrocore_v5_experimental/failure_mode_diagnostics/build_m11_6_diagnostic_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.hydrocore_v5_experimental.failure_mode_diagnostics import graph_features as gf  # noqa: E402

LOCKED_ROOT = ROOT / "data" / "locked" / "m11-6"
RAW_INCIDENTS = (
    ROOT
    / "reports"
    / "evaluation"
    / "hydrocore-v5"
    / "m11"
    / "m11-6-final"
    / "m11-6-raw-incidents.jsonl"
)
OUTPUT_DIR = ROOT / "reports" / "evaluation" / "failure-mode-diagnostics"

# Canonical known-family .inp files, verified by node-count match against
# dataset-report.json's topology_node_counts (golden-reference=6,
# branched-loop=8, loop-grid=9) and the m11-6a novelty spec's own
# prior_topologies node_count list before being trusted here.
KNOWN_FAMILY_INP = {
    "golden-reference": ROOT / "data" / "frozen" / "golden_network.inp",
    "branched-loop": ROOT / "data" / "topology-transfer" / "branched-loop.inp",
    "loop-grid": ROOT / "data" / "topologies" / "loop-grid.inp",
}
NOVEL_TOPOLOGY_INP = {
    f"locked-topology:{index}": LOCKED_ROOT / "topologies" / f"locked-topology-{index}.inp"
    for index in range(4)
}

EXPECTED_NODE_COUNTS = {"golden-reference": 6, "branched-loop": 8, "loop-grid": 9}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _graph_cache() -> dict[str, tuple[Any, list[str]]]:
    cache: dict[str, tuple[Any, list[str]]] = {}
    for family, path in KNOWN_FAMILY_INP.items():
        graph, reservoirs = gf.load_inp_graph(path)
        if graph.number_of_nodes() != EXPECTED_NODE_COUNTS[family]:
            raise ValueError(
                f"{family}: expected {EXPECTED_NODE_COUNTS[family]} nodes from "
                f"dataset-report.json, got {graph.number_of_nodes()} from {path} "
                "-- refusing to trust an unverified topology file"
            )
        cache[family] = (graph, reservoirs)
    for topology_id, path in NOVEL_TOPOLOGY_INP.items():
        cache[topology_id] = gf.load_inp_graph(path)
    return cache


def build_table() -> list[dict[str, Any]]:
    graphs = _graph_cache()

    # scenarios.jsonl records carry no scenario_id of their own (only
    # scenario_index within their split); m11-6-materialization-manifest.json's
    # own "scenarios" list is the bridge that ties scenario_id to
    # (split, scenario_index), so it is used here purely as a join key --
    # every other field is still read from scenarios.jsonl itself, and
    # seed/source_node are cross-checked below to catch any join-key drift.
    manifest = json.loads((LOCKED_ROOT / "m11-6-materialization-manifest.json").read_text(encoding="utf-8"))
    scenario_id_by_key = {
        (entry["split"], entry["scenario_index"]): entry["scenario_id"] for entry in manifest["scenarios"]
    }
    scenarios_by_id: dict[str, dict[str, Any]] = {}
    for split_name, filename in (
        ("locked_final_test", "locked_final_test"),
        ("locked_topology_test", "locked_topology_test"),
    ):
        for row in _load_jsonl(LOCKED_ROOT / filename / "scenarios.jsonl"):
            scenario_id = scenario_id_by_key[(row["split"], row["scenario_index"])]
            scenarios_by_id[scenario_id] = row

    incidents = _load_jsonl(RAW_INCIDENTS)
    if len(incidents) != 125:
        raise ValueError(f"expected 125 locked incidents (105+20), found {len(incidents)}")

    rows: list[dict[str, Any]] = []
    for incident in incidents:
        scenario = scenarios_by_id.get(incident["scenario_id"])
        if scenario is None:
            raise ValueError(f"incident {incident['scenario_id']} has no matching locked scenario record")
        if scenario["source_node"] != incident["source_node"] or scenario["topology_id"] != incident["topology_id"]:
            raise ValueError(
                f"join-key drift for {incident['scenario_id']}: incident/scenario records disagree"
            )

        is_novel_topology = incident["split"] == "locked_topology_test"
        family_key = incident["topology_id"] if is_novel_topology else incident["network_family"]
        graph, reservoir_ids = graphs[family_key]
        level = gf.graph_level_features(graph, reservoir_ids=reservoir_ids)
        source = gf.source_node_features(graph, incident["source_node"], reservoir_ids=reservoir_ids)

        condition = scenario.get("condition", {})
        generator_config = scenario.get("generator_config", {})

        row: dict[str, Any] = {
            # --- recorded: prediction/outcome (m11-6-raw-incidents.jsonl) ---
            "scenario_id": incident["scenario_id"],
            "split": incident["split"],
            "seen_topology": not is_novel_topology,
            "network_family": incident["network_family"],
            "topology_id": incident["topology_id"],
            "condition_kind": incident["condition_kind"],
            "source_node": incident["source_node"],
            "top1_correct": incident["top1_correct"],
            "top3_correct": incident["top3_correct"],
            "reciprocal_rank": incident["reciprocal_rank"],
            "posterior_entropy": incident["posterior_entropy"],
            "candidate_set_size": incident["candidate_set_size"],
            "calibrated": incident["calibrated"],
            "conformal_truth_coverage": incident["conformal_truth_coverage"],
            "outcome": incident["outcome"],
            "control_action": incident["control_action"],
            "final_status": incident["final_status"],
            "planning_allowed": incident["planning_allowed"],
            "plans_generated": incident["plans_generated"],
            "plans_rejected": incident["plans_rejected"],
            "plans_verified": incident["plans_verified"],
            "samples_taken": incident["samples_taken"],
            "no_safe_plan": incident["no_safe_plan"],
            "approval_attempted": incident["approval_attempted"],
            "human_approved": incident["human_approved"],
            # actionable (task's own semantics, m11-6a-actionability-semantics.json):
            # a successful /approve with decision==VERIFIED. Proxy here from the
            # fields the raw incident record already carries (no re-derivation
            # of a stricter/looser definition than the locked harness itself used).
            "actionable": bool(incident["human_approved"]),
            # --- recorded: incident/condition metadata (locked scenarios.jsonl) ---
            "event_type": scenario.get("event_type"),
            "perturbation_type": condition.get("perturbation_type"),
            "perturbation_level": condition.get("perturbation_level"),
            "sensor_noise_std_config": generator_config.get("sensor_noise_std"),
            "missing_rate": condition.get("missing_rate"),
            "health_fraction": condition.get("health_fraction"),
            "health_mode": condition.get("health_mode"),
            "coverage_condition": condition.get("coverage"),
            "ambiguity_condition": condition.get("ambiguity"),
            "hydraulic_condition": condition.get("hydraulic"),
            "sensor_count_config": generator_config.get("sensor_count"),
            # --- derived: graph-structural (this script, via networkx over
            # the frozen .inp topology; see graph_features.py) ---
            "node_count": level.node_count,
            "edge_count": level.edge_count,
            "graph_density": level.density,
            "graph_diameter": level.diameter,
            "dead_end_count": level.dead_end_count,
            "reservoir_count": level.reservoir_count,
            "source_degree": source.degree if source else None,
            "source_betweenness_centrality": source.betweenness_centrality if source else None,
            "source_closeness_centrality": source.closeness_centrality if source else None,
            "source_normalized_graph_position": source.normalized_graph_position if source else None,
            "source_hops_to_reservoir": source.hops_to_reservoir if source else None,
            "source_hops_to_nearest_dead_end": source.hops_to_nearest_dead_end if source else None,
            "source_is_boundary_node": source.is_boundary_node if source else None,
            "source_eccentricity": source.eccentricity if source else None,
        }
        rows.append(row)
    return rows


COLUMN_PROVENANCE = {
    "recorded_from_m11_6_raw_incidents": [
        "scenario_id", "split", "network_family", "topology_id", "condition_kind",
        "source_node", "top1_correct", "top3_correct", "reciprocal_rank",
        "posterior_entropy", "candidate_set_size", "calibrated",
        "conformal_truth_coverage", "outcome", "control_action", "final_status",
        "planning_allowed", "plans_generated", "plans_rejected", "plans_verified",
        "samples_taken", "no_safe_plan", "approval_attempted", "human_approved",
    ],
    "derived_from_m11_6_raw_incidents": ["seen_topology", "actionable"],
    "recorded_from_locked_scenario_manifests": [
        "event_type", "perturbation_type", "perturbation_level",
        "sensor_noise_std_config", "missing_rate", "health_fraction",
        "health_mode", "coverage_condition", "ambiguity_condition",
        "hydraulic_condition", "sensor_count_config",
    ],
    "derived_via_networkx_from_frozen_inp_topology": [
        "node_count", "edge_count", "graph_density", "graph_diameter",
        "dead_end_count", "reservoir_count", "source_degree",
        "source_betweenness_centrality", "source_closeness_centrality",
        "source_normalized_graph_position", "source_hops_to_reservoir",
        "source_hops_to_nearest_dead_end", "source_is_boundary_node",
        "source_eccentricity",
    ],
    "not_available_in_frozen_data_and_therefore_omitted": [
        "exact realized contamination strength/duration/start-time per incident "
        "(scenarios.jsonl only carries generator_config BIN RANGES, not the "
        "realized per-scenario draw)",
        "sensor node identity/placement (only sensor_count=4, constant across "
        "all 125 incidents, is recorded -- no per-example sensor node IDs, so "
        "source-to-sensor graph distance cannot be computed for this frozen set)",
        "event/evidence-head raw outputs (the locked harness's raw incident "
        "record does not carry them; only aggregate scout/planning counters do)",
    ],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_table()
    table_path = OUTPUT_DIR / "m11-6-diagnostic-table.jsonl"
    with table_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    provenance_path = OUTPUT_DIR / "m11-6-diagnostic-table-provenance.json"
    provenance_path.write_text(json.dumps(COLUMN_PROVENANCE, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {table_path}")
    print(f"Wrote column provenance to {provenance_path}")


if __name__ == "__main__":
    main()
