"""Milestone 9.1: evaluate ONE trained/reused (arm, seed) checkpoint against
`development_holdout` and fit+evaluate its independent `B_DEPTH_AWARE`
calibration from its own `calibration`-split rows (frozen protocol
`docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, Sections 3/9/11.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_evaluate.py --arm CURRENT --seed 20260814

Requires `m9-1-runs/{ARM}-seed{SEED}.json` (written by
`run_m9_1_train_arm.py`) to already exist. Idempotent by (arm, seed): reruns
recompute deterministically and overwrite only that (arm, seed)'s own block
in `m9-1-results.json`/`m9-1-calibration.json`, never another arm/seed's.

Writes (merged, never silently overwriting unrelated (arm, seed) entries):
  reports/evaluation/hydrocore-v5/m9-1-results.json
  reports/evaluation/hydrocore-v5/m9-1-calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import m9_1_common as common  # noqa: E402
from m9_0b_calibration_schemes import _quantile, wilson_interval  # noqa: E402


def _agg_top1(rows: list[dict[str, Any]], depths: tuple[int, ...]) -> float:
    subset = [row["metrics"]["top1"] for row in rows if row["depth"] in depths]
    return statistics.fmean(subset) if subset else float("nan")


def _agg_mrr(rows: list[dict[str, Any]]) -> float:
    return statistics.fmean(row["metrics"]["mrr"] for row in rows)


def _summary_over(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return {"n": 0}
    covered = [row["covered"] for row in metric_rows]
    n = len(metric_rows)
    n_covered = sum(covered)
    lower, upper = wilson_interval(n_covered, n)
    return {
        "n": n,
        "coverage": n_covered / n,
        "wilson_95_ci": [lower, upper],
        "mean_set_size": statistics.fmean(row["set_size"] for row in metric_rows),
        "mean_normalized_set_size": statistics.fmean(row["normalized_set_size"] for row in metric_rows),
        "singleton_rate": statistics.fmean(row["set_size"] == 1 for row in metric_rows),
    }


def _by_bucket(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {bucket: _summary_over([r for r in metric_rows if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}


def _group_support(calibrator: Any) -> dict[str, Any]:
    groups = {}
    for key, scores in calibrator.artifact.network_scores.items():
        n = len(scores)
        rank = min(n, max(1, int(np.ceil((n + 1) * (1 - common.ALPHA)))))
        groups[str(key)] = {
            "n": n, "quantile": _quantile(scores, common.ALPHA), "quantile_rank": rank,
            "meets_minimum_group_size": n >= common.MINIMUM_GROUP_SIZE,
        }
    return {"network_groups": groups, "global_n": len(calibrator.artifact.global_scores)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=common.ALL_ARMS)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    arm, seed = args.arm, args.seed

    code_under_test_commit = common.assert_code_under_test_commit()
    locked_before = common.assert_locked_test_closed()

    record = common.load_run_record(arm, seed)
    if arm != "CURRENT" and record.get("instability_flag") is not None:
        raise SystemExit(f"{arm}-seed{seed}: training itself was {record['instability_flag']} -- no checkpoint to evaluate")
    model = common.load_checkpoint_from_record(record)

    print(f"{arm}-seed{seed}: loading pools...", flush=True)
    train_records = common.load_pool("train")
    library = common.fit_library(train_records)
    dev_records = common.load_pool("development_holdout")
    calibration_records = common.load_pool("calibration")

    print(f"{arm}-seed{seed}: evaluating development_holdout ({len(dev_records)} scenarios x {len(common.CAUSAL_PREFIX_DEPTHS)} depths)...", flush=True)
    dev_rows = common.evaluate_split(model, arm, seed, dev_records, "development_holdout", library)

    print(f"{arm}-seed{seed}: evaluating calibration split ({len(calibration_records)} scenarios x {len(common.CAUSAL_PREFIX_DEPTHS)} depths)...", flush=True)
    calibration_rows = common.evaluate_split(model, arm, seed, calibration_records, "calibration", library)

    any_step_limit = any(row["solver_step_limit_exceeded"] for row in dev_rows + calibration_rows)
    all_finite = all(row["forward_finite"] for row in dev_rows + calibration_rows)

    by_incident: dict[int, dict[int, dict[str, Any]]] = {}
    for row in dev_rows:
        by_incident.setdefault(row["scenario_index"], {})[row["depth"]] = row
    primary_metric_per_incident = [
        common.primary_metric_per_incident(depth_rows) for _index, depth_rows in sorted(by_incident.items())
    ]

    aggregates = {
        "early_top1": _agg_top1(dev_rows, common.EARLY_DEPTHS),
        "mid_top1": _agg_top1(dev_rows, common.MID_DEPTHS),
        "mature_top1": _agg_top1(dev_rows, common.MATURE_DEPTHS),
        "overall_mrr": _agg_mrr(dev_rows),
        "primary_metric_mean": statistics.fmean(primary_metric_per_incident),
        "all_finite": all_finite,
        "solver_step_limit_exceeded": any_step_limit,
        "n_dev_incidents": len(dev_records),
    }

    results_entry = {
        "arm": arm, "seed": seed,
        "protocol_frozen_at_commit": common.PROTOCOL_FROZEN_AT_COMMIT,
        "code_under_test_commit": code_under_test_commit,
        "checkpoint_sha256": record.get("checkpoint_sha256") or record["training_summary"]["export_sha256"],
        "aggregates": aggregates,
        "primary_metric_per_incident": primary_metric_per_incident,
        "dev_rows": dev_rows,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": common.assert_locked_test_closed(),
    }
    common.merge_nested_json(common.RESULTS_PATH, arm, str(seed), results_entry)
    print(f"{arm}-seed{seed}: wrote results block ({common.RESULTS_PATH})")

    print(f"{arm}-seed{seed}: fitting B_DEPTH_AWARE calibration...", flush=True)
    calibrator = common.fit_b_depth_aware(calibration_rows, model_hash=f"m9-1-{arm}-seed{seed}")
    group_support = _group_support(calibrator)

    metric_rows = []
    for row in dev_rows:
        scheme_row = common.to_scheme_row(row)
        network_id = f"{common.NETWORK_FAMILY}:{row['depth_bucket']}"
        candidate = calibrator.candidate_set(scheme_row.probabilities, condition=scheme_row.condition, network_id=network_id)
        metric_rows.append({
            "depth_bucket": row["depth_bucket"],
            "covered": row["truth_index"] in candidate,
            "set_size": len(candidate),
            "normalized_set_size": len(candidate) / len(row["probabilities"]),
        })

    calibration_entry = {
        "arm": arm, "seed": seed,
        "alpha": common.ALPHA,
        "minimum_group_size": common.MINIMUM_GROUP_SIZE,
        "group_support": group_support,
        "marginal": _summary_over(metric_rows),
        "by_maturity": _by_bucket(metric_rows),
        "n_calibration_rows": len(calibration_rows),
        "n_development_rows": len(dev_rows),
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": common.assert_locked_test_closed(),
    }
    common.merge_nested_json(common.CALIBRATION_PATH, arm, str(seed), calibration_entry)
    print(f"{arm}-seed{seed}: wrote calibration block ({common.CALIBRATION_PATH})")

    print(json.dumps({"arm": arm, "seed": seed, "aggregates": aggregates, "marginal_coverage": calibration_entry["marginal"].get("coverage")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
