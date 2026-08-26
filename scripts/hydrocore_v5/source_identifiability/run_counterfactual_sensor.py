#!/usr/bin/env python3
"""Phase 7: bounded counterfactual "one more sensor" analysis.

For the hardest (lowest-identifiability) confirmatory incidents, asks a
purely deterministic question: if ONE additional node were also a sensor,
how much would the true source's separation from its nearest competitor
improve, and would the oracle's Top-1/Top-3 flip? No new EPANET calls are
needed -- `simulate_incident` already returns concentration for every
network node, so this reslices already-computed candidate traces from a
second (cheap) reconstruction pass. No learned sampling policy; this is a
deterministic value-of-one-more-observation calculation only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import common, library, oracle, signatures  # noqa: E402

CONFIRMATORY_PATH = common.OUTPUT_ROOT / "confirmatory" / "confirmatory-identifiability.jsonl"
OUTPUT_DIR = common.OUTPUT_ROOT / "counterfactual-sensor"
OUTPUT_PATH = OUTPUT_DIR / "counterfactual-sensor-results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "counterfactual-sensor-summary.json"

#: "Hard" = bottom tercile of the primary (normalized+correlation)
#: identifiability score across the confirmatory tier -- a data-driven,
#: pre-declared cut computed once here, not tuned per incident.
HARD_TERCILE = 1.0 / 3.0


def _primary_score(record: dict) -> float:
    return record["identifiability"]["normalized"]["correlation"]["identifiability_score"]


def _select_hard_incidents() -> list[dict]:
    records = [json.loads(line) for line in CONFIRMATORY_PATH.read_text().splitlines() if line.strip()]
    scores = sorted(_primary_score(r) for r in records)
    if not scores:
        return []
    cutoff = scores[max(0, int(len(scores) * HARD_TERCILE) - 1)]
    return [r for r in records if _primary_score(r) <= cutoff]


def _row_lookup() -> dict[tuple[str, int], dict]:
    lookup = {}
    for path in (common.LOCKED_FINAL_TEST, common.LOCKED_TOPOLOGY_TEST):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            lookup[(row["split"], row["seed"])] = row
    return lookup


def evaluate_incident(record: dict, row: dict) -> dict:
    bundle = library.build_incident_bundle(row)
    true_source = row["source_node"]
    sensors = list(bundle.incident.sensor_nodes)
    # Restrict candidate extra-sensor placements to junctions, matching the
    # repo's own convention that sensors are always drawn from
    # `junction_name_list` (scenarios.py) -- reservoirs/tanks are fixed
    # boundary conditions, not physically meaningful monitoring points.
    candidate_extra_nodes = [n for n in bundle.incident.junctions if n not in sensors]

    baseline_raw = bundle.signature_set.raw
    baseline_candidates, baseline_matrix = signatures.pairwise_distance_matrix(
        bundle.signature_set.normalized, metric="correlation"
    )
    baseline_ident = signatures.identifiability_metrics(
        baseline_candidates, baseline_matrix, true_source=true_source, noise_floor_distance=bundle.noise_floor_distance
    )
    node_index = {node: i for i, node in enumerate(bundle.all_node_ids)}

    best = None
    per_node_scores = {}
    for extra_node in candidate_extra_nodes:
        column = node_index[extra_node]
        extended = {
            c: np.column_stack([baseline_raw[c], bundle.full_node_traces[c][:, column : column + 1]])
            for c in bundle.incident.junctions
        }
        norm_extended = _normalize_all(extended)
        candidates, matrix = signatures.pairwise_distance_matrix(norm_extended, metric="correlation")
        ident = signatures.identifiability_metrics(
            candidates, matrix, true_source=true_source, noise_floor_distance=bundle.noise_floor_distance
        )
        per_node_scores[extra_node] = ident.identifiability_score
        if best is None or ident.identifiability_score > best[1].identifiability_score:
            best = (extra_node, ident)

    oracle_baseline = oracle.rank_candidates(baseline_raw, baseline_raw[true_source], true_source=true_source)
    oracle_best = None
    if best is not None:
        extra_node, _ = best
        column = node_index[extra_node]
        extended_raw = {
            c: np.column_stack([baseline_raw[c], bundle.full_node_traces[c][:, column : column + 1]])
            for c in bundle.incident.junctions
        }
        extended_observation = np.column_stack(
            [baseline_raw[true_source], bundle.full_node_traces[true_source][:, column : column + 1]]
        )
        oracle_best = oracle.rank_candidates(extended_raw, extended_observation, true_source=true_source)

    return {
        "split": row["split"],
        "seed": row["seed"],
        "source_node": true_source,
        "network_family": row["network_family"],
        "condition_kind": row["condition_kind"],
        "baseline_identifiability_score": baseline_ident.identifiability_score,
        "baseline_nearest_competitor_distance": baseline_ident.nearest_competitor_distance,
        "baseline_oracle_top1": oracle_baseline.top1,
        "candidate_extra_sensor_scores": per_node_scores,
        "best_extra_sensor_node": best[0] if best else None,
        "best_identifiability_score": best[1].identifiability_score if best else None,
        "best_nearest_competitor_distance": best[1].nearest_competitor_distance if best else None,
        "score_improvement": (best[1].identifiability_score - baseline_ident.identifiability_score)
        if best
        else None,
        "oracle_top1_with_best_sensor": oracle_best.top1 if oracle_best else None,
        "oracle_rank_with_best_sensor": oracle_best.true_source_rank if oracle_best else None,
        "oracle_rank_baseline": oracle_baseline.true_source_rank,
        "resolved_ambiguity": bool(best and best[1].identifiability_score > 1.0 >= baseline_ident.identifiability_score),
    }


def _normalize_all(signature_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for candidate, matrix in signature_map.items():
        norm = np.linalg.norm(matrix)
        out[candidate] = matrix / norm if norm > 0 else matrix
    return out


def main() -> None:
    hard = _select_hard_incidents()
    lookup = _row_lookup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    with OUTPUT_PATH.open("w") as handle:
        for record in hard:
            row = lookup[(record["split"], record["seed"])]
            result = evaluate_incident(record, row)
            results.append(result)
            handle.write(json.dumps(result) + "\n")
    n_resolved = sum(1 for r in results if r["resolved_ambiguity"])
    n_oracle_flipped_top1 = sum(
        1
        for r in results
        if r["oracle_top1_with_best_sensor"] == 1.0 and r["baseline_oracle_top1"] == 0.0
    )
    node_frequency: dict[str, int] = {}
    for r in results:
        if r["best_extra_sensor_node"]:
            node_frequency[r["best_extra_sensor_node"]] = node_frequency.get(r["best_extra_sensor_node"], 0) + 1
    summary = {
        "kind": "SOURCE_IDENTIFIABILITY_COUNTERFACTUAL_SENSOR",
        "label": "NON-PROMOTABLE / DIAGNOSTIC ONLY -- deterministic value-of-one-more-observation estimate",
        "n_hard_incidents": len(hard),
        "hard_tercile_definition": "bottom third of normalized+correlation identifiability_score, confirmatory tier",
        "n_resolved_ambiguity_by_best_single_sensor": n_resolved,
        "fraction_resolved": n_resolved / len(hard) if hard else None,
        "n_oracle_top1_flipped_by_best_single_sensor": n_oracle_flipped_top1,
        "mean_score_improvement": float(np.mean([r["score_improvement"] for r in results if r["score_improvement"] is not None]))
        if results
        else None,
        "most_frequently_best_extra_sensor_nodes": sorted(node_frequency.items(), key=lambda kv: -kv[1])[:10],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
