#!/usr/bin/env python3
"""Phase 2-4 (confirmatory tier): build signature libraries, pairwise
distinguishability, and oracle-baseline results for every incident in the
frozen M11.6 locked evaluation, WITHOUT touching the locked evaluation
itself (read-only replay of the scenario *specs* only -- never opens
`m11-6-raw-incidents.jsonl` for anything but the later join step, never
writes into `data/locked/**` or `reports/evaluation/hydrocore-v5/m11/**`).

Usage: python run_build_confirmatory.py [--limit N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import centrality, common, library, oracle, signatures  # noqa: E402

OUTPUT_DIR = common.OUTPUT_ROOT / "confirmatory"
OUTPUT_PATH = OUTPUT_DIR / "confirmatory-identifiability.jsonl"


def _load_rows() -> list[dict]:
    rows = []
    for path, split_label in (
        (common.LOCKED_FINAL_TEST, "locked_final_test"),
        (common.LOCKED_TOPOLOGY_TEST, "locked_topology_test"),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["split"] == split_label
            rows.append(row)
    return rows


def _identifiability_block(sig_set: signatures.SignatureSet, true_source: str, noise_floor: float) -> dict:
    block = {}
    for definition, sig_map, metric_names in (
        ("raw", sig_set.raw, ("rmse", "cosine", "correlation")),
        ("normalized", sig_set.normalized, ("rmse", "cosine", "correlation")),
        ("arrival_order", sig_set.arrival_order, ("arrival_l1",)),
    ):
        block[definition] = {}
        for metric_name in metric_names:
            candidates, matrix = signatures.pairwise_distance_matrix(sig_map, metric=metric_name)
            result = signatures.identifiability_metrics(
                candidates, matrix, true_source=true_source, noise_floor_distance=noise_floor
            )
            block[definition][metric_name] = dataclasses.asdict(result)
    return block


def _structural_block(bundle: library.IncidentBundle) -> dict:
    net = bundle.incident.randomized_network
    graph = common.undirected_graph(net)
    feats = centrality.compute_structural_features(
        graph,
        reservoir_nodes=centrality.reservoir_and_tank_nodes(net),
        sensor_nodes=bundle.incident.sensor_nodes,
    )
    return {node: dataclasses.asdict(f) for node, f in feats.items()}


def build_row(row: dict) -> dict:
    bundle = library.build_incident_bundle(row)
    true_source = row["source_node"]
    identifiability = _identifiability_block(bundle.signature_set, true_source, bundle.noise_floor_distance)
    oracle_clean = oracle.rank_candidates(
        bundle.signature_set.raw,
        bundle.signature_set.raw[true_source],
        true_source=true_source,
    )
    oracle_observed = oracle.rank_candidates(
        bundle.signature_set.raw,
        bundle.observed_sensor_matrix,
        true_source=true_source,
        observation_mask=bundle.observed_mask,
    )
    structural = _structural_block(bundle)
    known = row["network_family"] in common.KNOWN_NETWORK_FAMILIES
    return {
        "split": row["split"],
        "scenario_index": row["scenario_index"],
        "seed": row["seed"],
        "source_node": true_source,
        "network_family": row["network_family"],
        "topology_id": row["topology_id"],
        "condition_kind": row["condition_kind"],
        "known_topology": known,
        "n_candidates": len(bundle.incident.junctions),
        "sensor_nodes": list(bundle.incident.sensor_nodes),
        "candidates": list(bundle.incident.junctions),
        "start_minute": bundle.incident.start_minute,
        "duration_minutes": bundle.incident.duration_minutes,
        "relative_strength": bundle.incident.relative_strength,
        "stress_treatment": bundle.stress_treatment,
        "noise_floor_distance": bundle.noise_floor_distance,
        "identifiability": identifiability,
        "oracle_clean": dataclasses.asdict(oracle_clean) | {"probabilities": None, "residual_rmse": None},
        "oracle_observed": dataclasses.asdict(oracle_observed) | {"probabilities": None, "residual_rmse": None},
        "oracle_observed_full": {
            "probabilities": oracle_observed.probabilities,
            "residual_rmse": oracle_observed.residual_rmse,
        },
        "true_source_structural": structural.get(true_source),
        "structural_by_candidate": {c: structural[c] for c in bundle.incident.junctions if c in structural},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = _load_rows()
    if args.limit:
        rows = rows[: args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = build_row(row)
            handle.write(json.dumps(record) + "\n")
            if (index + 1) % 10 == 0 or index == len(rows) - 1:
                elapsed = time.time() - start
                print(f"[{index + 1}/{len(rows)}] elapsed={elapsed:.1f}s", flush=True)
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
