#!/usr/bin/env python3
"""Phase 2-4/7 (exploratory tier): a larger, self-generated identifiability
corpus on the SAME seven M11.6 networks, for statistical power the 125
confirmatory incidents lack -- no HydroCore-v5 prediction is attached to
any of these (no model is invoked), so this tier answers physical
identifiability / clean-vs-stress questions only, never a HydroCore-v5
outcome comparison. Seeds are freshly drawn and checked disjoint from every
locked M11.6 seed (leakage-risk requirement in the protocol doc).

Usage: python run_build_exploratory.py [--per-network N] [--limit-networks N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import centrality, common, library, oracle, signatures  # noqa: E402

OUTPUT_DIR = common.OUTPUT_ROOT / "exploratory"
OUTPUT_PATH = OUTPUT_DIR / "exploratory-identifiability.jsonl"

SEED_RNG_SEED = 20260826  # today's date at authoring time -- fixed, documented, reproducible.

NETWORKS = [
    ("golden-reference", "locked-final:golden-reference"),
    ("branched-loop", "locked-final:branched-loop"),
    ("loop-grid", "locked-final:loop-grid"),
    ("locked-topology-procedural", "locked-topology:0"),
    ("locked-topology-procedural", "locked-topology:1"),
    ("locked-topology-procedural", "locked-topology:2"),
    ("locked-topology-procedural", "locked-topology:3"),
]

CONDITIONS = [
    ("CLEAN", {}),
    ("MEASUREMENT_NOISE", {"sensor_noise_std": 0.05}),
    ("SENSOR_DROPOUT", {"missing_probability": 0.3}),
]


def _locked_seeds() -> set[int]:
    seeds = set()
    for path in (common.LOCKED_FINAL_TEST, common.LOCKED_TOPOLOGY_TEST):
        for line in path.read_text().splitlines():
            seeds.add(json.loads(line)["seed"])
    return seeds


def _network_sha_and_warm_cache(topology_id: str, network_family: str) -> str:
    """Loads the base network directly (bypassing `load_base_network`'s
    verification, which needs a `network_sha256` to check against -- the
    very value we're computing here), then warms `common._NETWORK_CACHE`
    so every subsequent `load_base_network`/`reconstruct_incident` call for
    this topology re-verifies against a value this function itself derived
    straight from the on-disk `.inp` file."""

    import wntr

    from hydroswarm.data.scenarios import network_sha256

    if topology_id.startswith("locked-topology:"):
        index = topology_id.split(":", 1)[1]
        network = wntr.network.WaterNetworkModel(
            str(common.LOCKED_TOPOLOGIES_DIR / f"locked-topology-{index}.inp")
        )
    else:
        network = common.KNOWN_FAMILY_LOADERS[network_family]()
    common._NETWORK_CACHE[topology_id] = network
    return network_sha256(network)


def _make_rows(per_network: int, networks: list[tuple[str, str]]) -> list[dict]:
    locked_seeds = _locked_seeds()
    rng = np.random.default_rng(SEED_RNG_SEED)
    rows = []
    for network_family, topology_id in networks:
        network_sha = _network_sha_and_warm_cache(topology_id, network_family)
        base = common.load_base_network(
            {"topology_id": topology_id, "network_family": network_family, "network_sha256": network_sha}
        )
        junctions = sorted(base.junction_name_list)
        for condition_kind, config_overrides in CONDITIONS:
            for i in range(per_network):
                source_node = junctions[i % len(junctions)]
                while True:
                    seed = int(rng.integers(2**62, 2**63 - 1))
                    if seed not in locked_seeds:
                        locked_seeds.add(seed)  # also guard against intra-corpus collisions
                        break
                generator_config = {
                    "communication_outage_probability": 0.0,
                    "demand_regimes": [0.8, 1.0, 1.2],
                    "drift_per_hour": 0.0,
                    "duration_bins_min": [30, 60, 120],
                    "frozen_probability": 0.0,
                    "missing_probability": 0.0,
                    "pipe_outage_probability": 0.0,
                    "quantization_step": 0.001,
                    "roughness_variation_fraction": 0.05,
                    "sensor_count": 4,
                    "sensor_noise_std": 0.01 if condition_kind != "CLEAN" else 0.0,
                    "stage": "operational" if condition_kind != "CLEAN" else "clean",
                    "start_time_bins_min": [0, 60, 120, 240],
                    "strength_bins": [0.5, 1.0, 2.0],
                    "tank_level_variation_fraction": 0.1,
                    "unit_mismatch_probability": 0.01,
                }
                generator_config.update(config_overrides)
                rows.append(
                    {
                        "seed": seed,
                        "source_node": source_node,
                        "network_family": network_family,
                        "topology_id": topology_id,
                        "network_sha256": network_sha,
                        "condition_kind": condition_kind,
                        "condition": {"perturbation_type": condition_kind.lower()},
                        "event_type": "contamination",
                        "generator_config": generator_config,
                        "split": "exploratory",
                        "scenario_index": i,
                    }
                )
    return rows


def build_row(row: dict) -> dict:
    bundle = library.build_incident_bundle(row)
    true_source = row["source_node"]
    candidates, matrix = signatures.pairwise_distance_matrix(bundle.signature_set.normalized, metric="correlation")
    ident = signatures.identifiability_metrics(
        candidates, matrix, true_source=true_source, noise_floor_distance=bundle.noise_floor_distance
    )
    oracle_observed = oracle.rank_candidates(
        bundle.signature_set.raw,
        bundle.observed_sensor_matrix,
        true_source=true_source,
        observation_mask=bundle.observed_mask,
    )
    net = bundle.incident.randomized_network
    graph = common.undirected_graph(net)
    feats = centrality.compute_structural_features(
        graph,
        reservoir_nodes=centrality.reservoir_and_tank_nodes(net),
        sensor_nodes=bundle.incident.sensor_nodes,
    )
    known = row["network_family"] in common.KNOWN_NETWORK_FAMILIES
    return {
        "seed": row["seed"],
        "source_node": true_source,
        "network_family": row["network_family"],
        "topology_id": row["topology_id"],
        "condition_kind": row["condition_kind"],
        "known_topology": known,
        "n_candidates": len(bundle.incident.junctions),
        "identifiability_normalized_correlation": dataclasses.asdict(ident),
        "oracle_top1": oracle_observed.top1,
        "oracle_top3": oracle_observed.top3,
        "oracle_mrr": oracle_observed.mrr,
        "oracle_rank": oracle_observed.true_source_rank,
        "oracle_residual_margin": oracle_observed.residual_margin,
        "true_source_structural": dataclasses.asdict(feats[true_source]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-network", type=int, default=25)
    parser.add_argument("--limit-networks", type=int, default=None)
    args = parser.parse_args()

    networks = NETWORKS[: args.limit_networks] if args.limit_networks else NETWORKS
    rows = _make_rows(args.per_network, networks)
    assert len(rows) == len(set(r["seed"] for r in rows)), "seed collision in exploratory corpus"
    assert not (set(r["seed"] for r in rows) & _locked_seeds()), "exploratory seed collides with locked seed"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = build_row(row)
            handle.write(json.dumps(record) + "\n")
            if (index + 1) % 20 == 0 or index == len(rows) - 1:
                print(f"[{index + 1}/{len(rows)}] elapsed={time.time() - start:.1f}s", flush=True)
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
