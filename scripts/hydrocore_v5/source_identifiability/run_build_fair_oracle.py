#!/usr/bin/env python3
"""Build the fair (nuisance-searched) oracle comparator over the confirmatory
M11.6 locked-evaluation replay set (Task 1 audit correction -- see
`docs/evaluation/ORACLE_INFORMATION_AUDIT.md`).

Same read-only replay discipline as `run_build_confirmatory.py`: never
writes into `data/locked/**` or `models/**`, never re-opens or re-scores the
frozen M11.6 evaluation itself. Default scope is the 56 confirmatory
incidents where HydroCore-v5's own recorded Top-1 was wrong (the population
the original "96.4%" figure describes) since the nuisance grid search is
~36x the simulator cost of the original oracle per incident; pass
`--all` to additionally cover the 69 incidents HydroCore-v5 got right, for
the aggregate (not just failure-subset) fair-oracle Top-1 figure.

Usage:
    python run_build_fair_oracle.py [--limit N] [--all] [--grid-subset]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_identifiability import common, fair_oracle, library  # noqa: E402

OUTPUT_DIR = common.OUTPUT_ROOT / "fair-oracle"
OUTPUT_PATH = OUTPUT_DIR / "fair-oracle-results.jsonl"
JOINED_PATH = common.OUTPUT_ROOT / "joined" / "joined-incidents.jsonl"


def _load_scenario_rows() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path, split_label in (
        (common.LOCKED_FINAL_TEST, "locked_final_test"),
        (common.LOCKED_TOPOLOGY_TEST, "locked_topology_test"),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row["split"] == split_label
            rows[int(row["seed"])] = row
    return rows


def _load_hydrocore_outcomes() -> dict[int, dict]:
    outcomes: dict[int, dict] = {}
    for line in JOINED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        outcomes[int(record["seed"])] = record
    return outcomes


def build_row(row: dict, hydrocore_record: dict, *, grid) -> dict:
    bundle = library.build_incident_bundle(row)
    fair = fair_oracle.rank_candidates_fair(bundle, row, grid=grid)
    ranked = fair.result
    search_summary = {
        candidate: {
            "best_start_minute": s.best_point.start_minute,
            "best_duration_minutes": s.best_point.duration_minutes,
            "best_relative_strength": s.best_point.relative_strength,
            "best_residual_rmse": s.best_residual_rmse,
        }
        for candidate, s in fair.per_candidate_search.items()
    }
    true_source = row["source_node"]
    incident = bundle.incident
    true_point_is_best = (
        search_summary[true_source]["best_start_minute"] == incident.start_minute
        and search_summary[true_source]["best_duration_minutes"] == incident.duration_minutes
        and search_summary[true_source]["best_relative_strength"] == incident.relative_strength
    )
    return {
        "seed": row["seed"],
        "split": row["split"],
        "source_node": true_source,
        "network_family": row["network_family"],
        "topology_id": row["topology_id"],
        "condition_kind": row["condition_kind"],
        "n_candidates": len(bundle.incident.junctions),
        "grid_size": fair.grid_size,
        "true_start_minute": incident.start_minute,
        "true_duration_minutes": incident.duration_minutes,
        "true_relative_strength": incident.relative_strength,
        "true_source_recovered_true_nuisance_point": true_point_is_best,
        "fair_oracle_top1": ranked.top1,
        "fair_oracle_top3": ranked.top3,
        "fair_oracle_mrr": ranked.mrr,
        "fair_oracle_true_source_rank": ranked.true_source_rank,
        "fair_oracle_residual_margin": ranked.residual_margin,
        "fair_oracle_probability_margin": ranked.probability_margin,
        "privileged_oracle_top1": hydrocore_record["oracle_top1"],
        "privileged_oracle_true_source_rank": hydrocore_record["oracle_rank"],
        "hydrocore_top1_correct": hydrocore_record["hydrocore_top1_correct"],
        "hydrocore_reciprocal_rank": hydrocore_record["hydrocore_reciprocal_rank"],
        "identifiability_tercile": hydrocore_record["identifiability_tercile"],
        "search_by_candidate": search_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--all", action="store_true", help="cover both HydroCore-v5 failures and successes (default: failures only)"
    )
    args = parser.parse_args()

    scenario_rows = _load_scenario_rows()
    hydrocore_outcomes = _load_hydrocore_outcomes()
    seeds = sorted(scenario_rows)
    if not args.all:
        seeds = [s for s in seeds if not hydrocore_outcomes[s]["hydrocore_top1_correct"]]
    if args.limit:
        seeds = seeds[: args.limit]

    grid = fair_oracle.default_nuisance_grid()
    print(f"scope: {len(seeds)} incidents, grid_size={len(grid)} nuisance points per candidate", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for index, seed in enumerate(seeds):
            record = build_row(scenario_rows[seed], hydrocore_outcomes[seed], grid=grid)
            handle.write(json.dumps(record) + "\n")
            elapsed = time.time() - start
            print(
                f"[{index + 1}/{len(seeds)}] seed={seed} n_candidates={record['n_candidates']} "
                f"fair_top1={record['fair_oracle_top1']} elapsed={elapsed:.1f}s",
                flush=True,
            )
    print(f"wrote {len(seeds)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
