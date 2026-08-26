"""candidate-conditioned-localizer-v1 (EXPERIMENTAL, NON-RELEASE): paired
statistical analysis of `run_experiment.py`'s per-arm evaluation outputs.

Reimplements `exp/graph-structural-encoder-v2`'s `analyze_results.py`
structure (not imported -- that branch is not merged into this one) for a
2-population comparison (known: validation + development_holdout; unseen:
ood-UNSEEN_TOPOLOGY) against A_CONTROL, on the paired examples every arm
evaluates identically (same `scenario_id`, same seed/split -- see
run_experiment.py). Bootstrap convention: 2000 resamples, seed 20260826,
90% percentile interval (matches `exp/graph-structural-encoder-v2` and
`exp/source-identifiability-analysis`'s own `stats_utils.py` family of
conventions).

Produces (all under reports/evaluation/candidate-conditioned-localizer-v1/):
  - metric-table.json / .md
  - centrality-subgroups.json, distance-subgroups.json
  - subgroup-paired-bootstrap.json (low-centrality / long-distance
    significance test -- the primary H2 test)
  - paired-transitions.json (2x2 top1/top3 tables, rank/margin deltas, per
    arm vs A_CONTROL, per population)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "reports" / "evaluation" / "candidate-conditioned-localizer-v1"

ARM_NAMES = ("A_CONTROL", "B_CANDIDATE_CONDITIONED", "C_PHYSICS_INFORMED")
POPULATIONS = ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY")
KNOWN_POPULATIONS = ("validation", "development_holdout")
UNSEEN_POPULATIONS = ("ood-UNSEEN_TOPOLOGY",)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_INTERVAL = 0.90


def load_evaluation(arm: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / f"{arm.lower()}-evaluation.json").read_text(encoding="utf-8"))


def load_rows(arm: str, population: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"{arm.lower()}-{population}-rows.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_bootstrap(control: list[float], experimental: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    assert len(control) == len(experimental)
    if not control:
        return {"observed": None, "ci_low": None, "ci_high": None, "n": 0, "resamples": BOOTSTRAP_RESAMPLES}
    control_arr = np.asarray(control, dtype=np.float64)
    experimental_arr = np.asarray(experimental, dtype=np.float64)
    observed = float(experimental_arr.mean() - control_arr.mean())
    rng = np.random.default_rng(seed)
    n = len(control_arr)
    deltas = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, n, size=n)
        deltas[index] = experimental_arr[sample].mean() - control_arr[sample].mean()
    alpha = (1 - BOOTSTRAP_INTERVAL) / 2
    lower, upper = np.quantile(deltas, [alpha, 1 - alpha])
    return {
        "observed": observed,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "excludes_zero": bool(lower > 0 or upper < 0),
        "n": n,
        "resamples": BOOTSTRAP_RESAMPLES,
    }


# ---------------------------------------------------------------------------
# 1. Metric table
# ---------------------------------------------------------------------------


def build_metric_table(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for population in POPULATIONS:
        table[population] = {}
        for arm in ARM_NAMES:
            pop_summary = evaluations[arm]["populations"].get(population, {})
            table[population][arm] = {
                "n": pop_summary.get("n"),
                "n_localized": pop_summary.get("n_localized"),
                "top1": pop_summary.get("top1"),
                "top3": pop_summary.get("top3"),
                "mrr": pop_summary.get("mrr"),
                "proxy_actionable_rate": pop_summary.get("proxy_actionable_rate"),
                "proxy_abstention_rate": pop_summary.get("proxy_abstention_rate"),
                "proxy_candidate_set_size": pop_summary.get("proxy_candidate_set_size"),
                "proxy_calibrated_coverage": pop_summary.get("proxy_calibrated_coverage"),
                "ood_caution_or_outside_rate": pop_summary.get("ood_caution_or_outside_rate"),
            }
    return table


def metric_table_markdown(table: dict[str, Any]) -> str:
    lines = ["| population | " + " | ".join(ARM_NAMES) + " |", "|---|" + "---|" * len(ARM_NAMES)]
    for population in POPULATIONS:
        cells = []
        for arm in ARM_NAMES:
            row = table[population][arm]
            top1, top3, mrr = row.get("top1"), row.get("top3"), row.get("mrr")
            cells.append(f"{top1:.3f}/{top3:.3f}/{mrr:.3f}" if top1 is not None else "n/a")
        lines.append(f"| {population} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Centrality / distance subgroups
# ---------------------------------------------------------------------------


def _tercile_bounds(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    return ordered[max(0, n // 3 - 1)], ordered[max(0, 2 * n // 3 - 1)]


def _bucket(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value <= high_cut:
        return "mid"
    return "high"


def _pooled_localized(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]], arm: str, key: str) -> list[dict[str, Any]]:
    return [
        row
        for population in POPULATIONS
        for row in rows_by_arm[arm][population]
        if row.get("has_source") and key in row
    ]


def centrality_subgroups(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    # Reference cutoffs from A_CONTROL: centrality is a deterministic
    # function of the true source's position in its own topology, identical
    # across arms for the same physical scenario_id, so using CONTROL's
    # cutoffs for every arm keeps bucket boundaries paired/apples-to-apples.
    reference = _pooled_localized(rows_by_arm, "A_CONTROL", "source_betweenness_centrality")
    low_cut, high_cut = _tercile_bounds([row["source_betweenness_centrality"] for row in reference])
    result: dict[str, Any] = {"tercile_cutoffs": {"low_cut": low_cut, "high_cut": high_cut}, "by_arm": {}}
    for arm in ARM_NAMES:
        pooled = _pooled_localized(rows_by_arm, arm, "source_betweenness_centrality")
        buckets: dict[str, list[dict[str, Any]]] = {"low": [], "mid": [], "high": []}
        for row in pooled:
            buckets[_bucket(row["source_betweenness_centrality"], low_cut, high_cut)].append(row)
        result["by_arm"][arm] = {
            name: {
                "n": len(group),
                "top1": statistics.fmean(row["top1"] for row in group) if group else None,
                "top3": statistics.fmean(row["top3"] for row in group) if group else None,
                "mrr": statistics.fmean(row["reciprocal_rank"] for row in group) if group else None,
            }
            for name, group in buckets.items()
        }
    return result


def distance_subgroups(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    reference = _pooled_localized(rows_by_arm, "A_CONTROL", "source_hop_to_nearest_sensor_normalized")
    median = statistics.median(row["source_hop_to_nearest_sensor_normalized"] for row in reference) if reference else 0.0
    result: dict[str, Any] = {"distance_median_split": median, "by_arm": {}}
    for arm in ARM_NAMES:
        pooled = _pooled_localized(rows_by_arm, arm, "source_hop_to_nearest_sensor_normalized")
        short = [row for row in pooled if row["source_hop_to_nearest_sensor_normalized"] <= median]
        long = [row for row in pooled if row["source_hop_to_nearest_sensor_normalized"] > median]

        def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "n": len(group),
                "top1": statistics.fmean(row["top1"] for row in group) if group else None,
                "top3": statistics.fmean(row["top3"] for row in group) if group else None,
                "mrr": statistics.fmean(row["reciprocal_rank"] for row in group) if group else None,
            }

        result["by_arm"][arm] = {"short_distance": summarize(short), "long_distance": summarize(long)}
    return result


def subgroup_paired_bootstrap(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Paired bootstrap CI (arm vs A_CONTROL, matched by scenario_id) on
    top1 -- restricted to low-centrality and long-distance subgroups --
    the primary H2 significance test: does either arm improve the
    diagnosed hard subgroup, with a CI that excludes zero, not just a
    point estimate."""

    reference_centrality = _pooled_localized(rows_by_arm, "A_CONTROL", "source_betweenness_centrality")
    low_cut, high_cut = _tercile_bounds([row["source_betweenness_centrality"] for row in reference_centrality])
    control_by_id = {row["scenario_id"]: row for row in reference_centrality}
    low_centrality_ids = {sid for sid, row in control_by_id.items() if _bucket(row["source_betweenness_centrality"], low_cut, high_cut) == "low"}

    reference_distance = _pooled_localized(rows_by_arm, "A_CONTROL", "source_hop_to_nearest_sensor_normalized")
    distance_median = statistics.median(row["source_hop_to_nearest_sensor_normalized"] for row in reference_distance)
    long_distance_ids = {row["scenario_id"] for row in reference_distance if row["source_hop_to_nearest_sensor_normalized"] > distance_median}

    subgroups = {"low_centrality": low_centrality_ids, "long_distance": long_distance_ids}
    control_rows_by_id = {row["scenario_id"]: row for population in POPULATIONS for row in rows_by_arm["A_CONTROL"][population] if row.get("has_source")}

    result: dict[str, Any] = {"tercile_cutoffs": {"low_cut": low_cut, "high_cut": high_cut}, "distance_median": distance_median, "by_arm": {}}
    for arm in ARM_NAMES:
        if arm == "A_CONTROL":
            continue
        arm_rows_by_id = {row["scenario_id"]: row for population in POPULATIONS for row in rows_by_arm[arm][population] if row.get("has_source")}
        arm_result: dict[str, Any] = {}
        for subgroup_name, ids in subgroups.items():
            shared_ids = sorted(ids & set(control_rows_by_id) & set(arm_rows_by_id))
            control_top1 = [float(control_rows_by_id[sid]["top1"]) for sid in shared_ids]
            arm_top1 = [float(arm_rows_by_id[sid]["top1"]) for sid in shared_ids]
            arm_result[subgroup_name] = {
                "n": len(shared_ids),
                "control_top1_mean": statistics.fmean(control_top1) if control_top1 else None,
                "arm_top1_mean": statistics.fmean(arm_top1) if arm_top1 else None,
                **paired_bootstrap(control_top1, arm_top1),
            }
        result["by_arm"][arm] = arm_result
    return result


# ---------------------------------------------------------------------------
# 3. Paired transitions (each arm vs A_CONTROL, identical scenario_ids)
# ---------------------------------------------------------------------------


def paired_transitions(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for population in POPULATIONS:
        control_by_id = {row["scenario_id"]: row for row in rows_by_arm["A_CONTROL"][population] if row.get("has_source")}
        result[population] = {}
        for arm in ARM_NAMES:
            if arm == "A_CONTROL":
                continue
            arm_by_id = {row["scenario_id"]: row for row in rows_by_arm[arm][population] if row.get("has_source")}
            shared_ids = sorted(set(control_by_id) & set(arm_by_id))
            if not shared_ids:
                result[population][arm] = {"n": 0}
                continue
            control_rows = [control_by_id[sid] for sid in shared_ids]
            arm_rows = [arm_by_id[sid] for sid in shared_ids]

            top1_table = {"both_correct": 0, "control_only": 0, "arm_only": 0, "both_wrong": 0}
            for control_row, arm_row in zip(control_rows, arm_rows):
                if control_row["top1"] and arm_row["top1"]:
                    top1_table["both_correct"] += 1
                elif control_row["top1"] and not arm_row["top1"]:
                    top1_table["control_only"] += 1
                elif not control_row["top1"] and arm_row["top1"]:
                    top1_table["arm_only"] += 1
                else:
                    top1_table["both_wrong"] += 1

            rank_deltas = [
                arm_row["true_source_rank"] - control_row["true_source_rank"]
                for control_row, arm_row in zip(control_rows, arm_rows)
                if control_row.get("true_source_rank") is not None and arm_row.get("true_source_rank") is not None
            ]
            margin_deltas = [
                arm_row["margin_top1_top2"] - control_row["margin_top1_top2"]
                for control_row, arm_row in zip(control_rows, arm_rows)
                if control_row.get("margin_top1_top2") is not None and arm_row.get("margin_top1_top2") is not None
            ]

            result[population][arm] = {
                "n": len(shared_ids),
                "top1_transition_table": top1_table,
                "top1_bootstrap_delta": paired_bootstrap([float(r["top1"]) for r in control_rows], [float(r["top1"]) for r in arm_rows]),
                "top3_bootstrap_delta": paired_bootstrap([float(r["top3"]) for r in control_rows], [float(r["top3"]) for r in arm_rows]),
                "mean_rank_delta": statistics.fmean(rank_deltas) if rank_deltas else None,
                "rank_improved_unchanged_worsened": [
                    sum(1 for d in rank_deltas if d < 0), sum(1 for d in rank_deltas if d == 0), sum(1 for d in rank_deltas if d > 0),
                ],
                "mean_margin_delta": statistics.fmean(margin_deltas) if margin_deltas else None,
            }
    return result


# ---------------------------------------------------------------------------
# 4. Oracle-gap-closed (this pilot's A_CONTROL vs the fair oracle numbers
#    from the M11.6 audit -- see plan doc Section 8 for why this is
#    reported qualitatively, not as one blended cross-population ratio)
# ---------------------------------------------------------------------------

FAIR_ORACLE_TOP1_ON_M11_6_FAILURE_SUBSET = 0.964  # docs/evaluation/ORACLE_INFORMATION_AUDIT.md Section 4


def oracle_gap_closed(control_top1: float | None, experimental_top1: float | None, oracle_top1: float) -> float | None:
    if control_top1 is None or experimental_top1 is None:
        return None
    denom = oracle_top1 - control_top1
    if abs(denom) < 1e-6:
        return None
    return (experimental_top1 - control_top1) / denom


def main() -> None:
    evaluations = {arm: load_evaluation(arm) for arm in ARM_NAMES}
    rows_by_arm = {arm: {population: load_rows(arm, population) for population in POPULATIONS} for arm in ARM_NAMES}

    table = build_metric_table(evaluations)
    (RESULTS_DIR / "metric-table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS_DIR / "metric-table.md").write_text(metric_table_markdown(table) + "\n", encoding="utf-8")

    centrality = centrality_subgroups(rows_by_arm)
    (RESULTS_DIR / "centrality-subgroups.json").write_text(json.dumps(centrality, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    distance = distance_subgroups(rows_by_arm)
    (RESULTS_DIR / "distance-subgroups.json").write_text(json.dumps(distance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subgroup_bootstrap = subgroup_paired_bootstrap(rows_by_arm)
    (RESULTS_DIR / "subgroup-paired-bootstrap.json").write_text(json.dumps(subgroup_bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    transitions = paired_transitions(rows_by_arm)
    (RESULTS_DIR / "paired-transitions.json").write_text(json.dumps(transitions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gap_closed: dict[str, Any] = {}
    for population in POPULATIONS:
        control_top1 = table[population]["A_CONTROL"]["top1"]
        gap_closed[population] = {
            arm: oracle_gap_closed(control_top1, table[population][arm]["top1"], FAIR_ORACLE_TOP1_ON_M11_6_FAILURE_SUBSET)
            for arm in ARM_NAMES
            if arm != "A_CONTROL"
        }
    (RESULTS_DIR / "oracle-gap-closed.json").write_text(json.dumps(gap_closed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Wrote metric-table.{json,md}, centrality-subgroups.json, distance-subgroups.json,")
    print("subgroup-paired-bootstrap.json, paired-transitions.json, oracle-gap-closed.json")
    print(f"under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
