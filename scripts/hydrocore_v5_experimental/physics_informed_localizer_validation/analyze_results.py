"""physics-informed-localizer-validation (EXPERIMENTAL, NON-RELEASE): paired
statistical analysis of `run_experiment.py`'s per-seed evaluation outputs.

Generalizes `candidate_conditioned_localizer_v1`'s `analyze_results.py`
(same bootstrap convention: 2000 resamples, seed 20260826, 90% percentile
interval -- "HydroSwarm's established convention", unchanged here) from a
single seed / 3-arm comparison to:

  - per-seed metric tables + paired bootstrap CIs (same shape as the
    pilot's own outputs, one directory per seed);
  - a POOLED cross-seed summary: per-seed Top-1/Top-3/MRR per arm/
    population, mean/median/stdev across seeds, per-seed paired deltas vs
    A_CONTROL, and a pooled paired-bootstrap CI over every (seed, example)
    pair concatenated together (Phase 3/7's explicit requirement: do not
    report only pooled metrics if individual seeds disagree materially --
    both views are written, never one in place of the other);
  - the same centrality/distance subgroup and paired-transition tables as
    the pilot, computed per seed AND pooled across seeds.

Produces (all under reports/evaluation/physics-informed-localizer-
validation/):
  - seed-<seed>/metric-table.{json,md}, centrality-subgroups.json,
    distance-subgroups.json, subgroup-paired-bootstrap.json,
    paired-transitions.json (per-seed, same schema as the pilot branch)
  - pooled/cross-seed-summary.json (per-arm/population: per-seed top1/top3/
    mrr, mean, median, stdev, n_seeds_positive_delta, n_seeds_negative_delta)
  - pooled/pooled-paired-bootstrap.json (concatenate every seed's paired
    per-example deltas vs A_CONTROL, one bootstrap CI per arm/population)
  - pooled/parameter-counts.json (every arm's exact parameter report,
    seed-independent by construction -- same architecture/seed only
    changes initialization)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "reports" / "evaluation" / "physics-informed-localizer-validation"

ARM_NAMES = (
    "A_CONTROL",
    "A_CAPACITY_MATCHED",
    "B_CANDIDATE_CONDITIONED",
    "C_FULL",
    "C1",
    "C2",
    "C3",
    "C1_C2",
    "C1_C3",
    "C2_C3",
)
POPULATIONS = ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_INTERVAL = 0.90


def _results_dir(seed: int) -> Path:
    return RESULTS_ROOT / f"seed-{seed}"


def discover_seeds() -> list[int]:
    seeds = []
    for path in sorted(RESULTS_ROOT.glob("seed-*")):
        if path.is_dir():
            seeds.append(int(path.name.split("-")[1]))
    return sorted(seeds)


def discover_arms(seed: int) -> list[str]:
    results_dir = _results_dir(seed)
    found = []
    for arm in ARM_NAMES:
        if (results_dir / f"{arm.lower()}-evaluation.json").exists():
            found.append(arm)
    return found


def load_evaluation(seed: int, arm: str) -> dict[str, Any] | None:
    path = _results_dir(seed) / f"{arm.lower()}-evaluation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(seed: int, arm: str, population: str) -> list[dict[str, Any]]:
    path = _results_dir(seed) / f"{arm.lower()}-{population}-rows.jsonl"
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
# Per-seed analysis (same shape as candidate-conditioned-localizer-v1's own
# analyze_results.py, one full copy per seed)
# ---------------------------------------------------------------------------


def build_metric_table(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for population in POPULATIONS:
        table[population] = {}
        for arm, evaluation in evaluations.items():
            pop_summary = evaluation["populations"].get(population, {})
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


def metric_table_markdown(table: dict[str, Any], arms: list[str]) -> str:
    lines = ["| population | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for population in POPULATIONS:
        cells = []
        for arm in arms:
            row = table[population].get(arm, {})
            top1, top3, mrr = row.get("top1"), row.get("top3"), row.get("mrr")
            cells.append(f"{top1:.3f}/{top3:.3f}/{mrr:.3f}" if top1 is not None else "n/a")
        lines.append(f"| {population} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
        for row in rows_by_arm.get(arm, {}).get(population, [])
        if row.get("has_source") and key in row
    ]


def centrality_subgroups(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]], arms: list[str]) -> dict[str, Any]:
    reference = _pooled_localized(rows_by_arm, "A_CONTROL", "source_betweenness_centrality")
    if not reference:
        return {"tercile_cutoffs": None, "by_arm": {}}
    low_cut, high_cut = _tercile_bounds([row["source_betweenness_centrality"] for row in reference])
    result: dict[str, Any] = {"tercile_cutoffs": {"low_cut": low_cut, "high_cut": high_cut}, "by_arm": {}}
    for arm in arms:
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


def distance_subgroups(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]], arms: list[str]) -> dict[str, Any]:
    reference = _pooled_localized(rows_by_arm, "A_CONTROL", "source_hop_to_nearest_sensor_normalized")
    if not reference:
        return {"distance_median_split": None, "by_arm": {}}
    median = statistics.median(row["source_hop_to_nearest_sensor_normalized"] for row in reference)
    result: dict[str, Any] = {"distance_median_split": median, "by_arm": {}}
    for arm in arms:
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


def subgroup_paired_bootstrap(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]], arms: list[str]) -> dict[str, Any]:
    reference_centrality = _pooled_localized(rows_by_arm, "A_CONTROL", "source_betweenness_centrality")
    if not reference_centrality:
        return {"by_arm": {}}
    low_cut, high_cut = _tercile_bounds([row["source_betweenness_centrality"] for row in reference_centrality])
    control_by_id = {row["scenario_id"]: row for row in reference_centrality}
    low_centrality_ids = {sid for sid, row in control_by_id.items() if _bucket(row["source_betweenness_centrality"], low_cut, high_cut) == "low"}

    reference_distance = _pooled_localized(rows_by_arm, "A_CONTROL", "source_hop_to_nearest_sensor_normalized")
    distance_median = statistics.median(row["source_hop_to_nearest_sensor_normalized"] for row in reference_distance)
    long_distance_ids = {row["scenario_id"] for row in reference_distance if row["source_hop_to_nearest_sensor_normalized"] > distance_median}

    subgroups = {"low_centrality": low_centrality_ids, "long_distance": long_distance_ids}
    control_rows_by_id = {row["scenario_id"]: row for population in POPULATIONS for row in rows_by_arm.get("A_CONTROL", {}).get(population, []) if row.get("has_source")}

    result: dict[str, Any] = {"tercile_cutoffs": {"low_cut": low_cut, "high_cut": high_cut}, "distance_median": distance_median, "by_arm": {}}
    for arm in arms:
        if arm == "A_CONTROL":
            continue
        arm_rows_by_id = {row["scenario_id"]: row for population in POPULATIONS for row in rows_by_arm.get(arm, {}).get(population, []) if row.get("has_source")}
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


def paired_transitions(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]], arms: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for population in POPULATIONS:
        control_by_id = {row["scenario_id"]: row for row in rows_by_arm.get("A_CONTROL", {}).get(population, []) if row.get("has_source")}
        result[population] = {}
        for arm in arms:
            if arm == "A_CONTROL":
                continue
            arm_by_id = {row["scenario_id"]: row for row in rows_by_arm.get(arm, {}).get(population, []) if row.get("has_source")}
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
                "mrr_bootstrap_delta": paired_bootstrap([float(r["reciprocal_rank"]) for r in control_rows], [float(r["reciprocal_rank"]) for r in arm_rows]),
                "mean_rank_delta": statistics.fmean(rank_deltas) if rank_deltas else None,
                "rank_improved_unchanged_worsened": [
                    sum(1 for d in rank_deltas if d < 0), sum(1 for d in rank_deltas if d == 0), sum(1 for d in rank_deltas if d > 0),
                ],
                "mean_margin_delta": statistics.fmean(margin_deltas) if margin_deltas else None,
            }
    return result


def analyze_seed(seed: int, arms: list[str]) -> None:
    results_dir = _results_dir(seed)
    evaluations = {arm: load_evaluation(seed, arm) for arm in arms}
    evaluations = {arm: evaluation for arm, evaluation in evaluations.items() if evaluation is not None}
    rows_by_arm = {arm: {population: load_rows(seed, arm, population) for population in POPULATIONS} for arm in evaluations}

    table = build_metric_table(evaluations)
    (results_dir / "metric-table.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / "metric-table.md").write_text(metric_table_markdown(table, list(evaluations)) + "\n", encoding="utf-8")

    (results_dir / "centrality-subgroups.json").write_text(
        json.dumps(centrality_subgroups(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "distance-subgroups.json").write_text(
        json.dumps(distance_subgroups(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "subgroup-paired-bootstrap.json").write_text(
        json.dumps(subgroup_paired_bootstrap(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results_dir / "paired-transitions.json").write_text(
        json.dumps(paired_transitions(rows_by_arm, list(evaluations)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[seed {seed}] wrote per-seed metric-table/{{centrality,distance}}-subgroups/subgroup-paired-bootstrap/paired-transitions")


# ---------------------------------------------------------------------------
# Cross-seed (pooled) analysis
# ---------------------------------------------------------------------------


def cross_seed_summary(seeds: list[int], arms: list[str]) -> dict[str, Any]:
    """Per-arm/population: per-seed top1/top3/mrr, mean, median, stdev,
    per-seed delta vs A_CONTROL (same seed), and how many seeds show a
    positive vs negative Top-1 delta -- Phase 3/7's explicit requirement,
    reported ALONGSIDE (never instead of) the pooled paired-bootstrap CI
    below."""

    result: dict[str, Any] = {}
    for population in POPULATIONS:
        result[population] = {}
        control_top1_by_seed: dict[int, float | None] = {}
        for seed in seeds:
            evaluation = load_evaluation(seed, "A_CONTROL")
            control_top1_by_seed[seed] = evaluation["populations"][population]["top1"] if evaluation else None
        for arm in arms:
            per_seed: dict[str, Any] = {}
            top1_values: list[float] = []
            top1_deltas: list[float] = []
            for seed in seeds:
                evaluation = load_evaluation(seed, arm)
                if evaluation is None:
                    continue
                pop_summary = evaluation["populations"].get(population, {})
                top1, top3, mrr = pop_summary.get("top1"), pop_summary.get("top3"), pop_summary.get("mrr")
                control_top1 = control_top1_by_seed.get(seed)
                delta = (top1 - control_top1) if (top1 is not None and control_top1 is not None) else None
                per_seed[str(seed)] = {"top1": top1, "top3": top3, "mrr": mrr, "top1_delta_vs_control_same_seed": delta}
                if top1 is not None:
                    top1_values.append(top1)
                if delta is not None:
                    top1_deltas.append(delta)
            result[population][arm] = {
                "per_seed": per_seed,
                "n_seeds": len(top1_values),
                "top1_mean": statistics.fmean(top1_values) if top1_values else None,
                "top1_median": statistics.median(top1_values) if top1_values else None,
                "top1_stdev": statistics.stdev(top1_values) if len(top1_values) > 1 else 0.0 if top1_values else None,
                "top1_delta_mean": statistics.fmean(top1_deltas) if top1_deltas else None,
                "top1_delta_median": statistics.median(top1_deltas) if top1_deltas else None,
                "top1_delta_stdev": statistics.stdev(top1_deltas) if len(top1_deltas) > 1 else 0.0 if top1_deltas else None,
                "n_seeds_positive_delta": sum(1 for d in top1_deltas if d > 0),
                "n_seeds_negative_delta": sum(1 for d in top1_deltas if d < 0),
                "n_seeds_zero_delta": sum(1 for d in top1_deltas if d == 0),
            }
    return result


def pooled_paired_bootstrap(seeds: list[int], arms: list[str]) -> dict[str, Any]:
    """Concatenates every seed's paired (control, arm) Top-1/Top-3/MRR
    values -- matched by (seed, scenario_id) so cross-seed pooling never
    mixes an example from one seed's split with another's -- into one
    bootstrap resampling per arm/population, alongside (not instead of)
    the per-seed CIs already in each seed-<seed>/paired-transitions.json."""

    result: dict[str, Any] = {}
    for population in POPULATIONS:
        result[population] = {}
        for arm in arms:
            if arm == "A_CONTROL":
                continue
            control_top1: list[float] = []
            arm_top1: list[float] = []
            control_top3: list[float] = []
            arm_top3: list[float] = []
            control_mrr: list[float] = []
            arm_mrr: list[float] = []
            per_seed_n: dict[str, int] = {}
            for seed in seeds:
                control_rows = {row["scenario_id"]: row for row in load_rows(seed, "A_CONTROL", population) if row.get("has_source")}
                arm_rows = {row["scenario_id"]: row for row in load_rows(seed, arm, population) if row.get("has_source")}
                shared_ids = sorted(set(control_rows) & set(arm_rows))
                per_seed_n[str(seed)] = len(shared_ids)
                for sid in shared_ids:
                    control_top1.append(float(control_rows[sid]["top1"]))
                    arm_top1.append(float(arm_rows[sid]["top1"]))
                    control_top3.append(float(control_rows[sid]["top3"]))
                    arm_top3.append(float(arm_rows[sid]["top3"]))
                    control_mrr.append(float(control_rows[sid]["reciprocal_rank"]))
                    arm_mrr.append(float(arm_rows[sid]["reciprocal_rank"]))
            result[population][arm] = {
                "per_seed_n": per_seed_n,
                "total_n": len(control_top1),
                "top1": paired_bootstrap(control_top1, arm_top1),
                "top3": paired_bootstrap(control_top3, arm_top3),
                "mrr": paired_bootstrap(control_mrr, arm_mrr),
            }
    return result


def parameter_counts(seeds: list[int], arms: list[str]) -> dict[str, Any]:
    """Parameter counts are architecture-determined, not seed-determined
    (only initialization differs across seeds) -- report each arm's report
    from its first available seed, and flag any accidental mismatch across
    seeds (would indicate the arm's model_kwargs is not actually seed-
    invariant, a real bug)."""

    result: dict[str, Any] = {}
    for arm in arms:
        reports: dict[int, Any] = {}
        for seed in seeds:
            evaluation = load_evaluation(seed, arm)
            if evaluation is not None:
                reports[seed] = evaluation["parameter_report"]
        totals = {report["total"] for report in reports.values()}
        result[arm] = {
            "by_seed": reports,
            "consistent_across_seeds": len(totals) <= 1,
        }
    control_total = None
    if "A_CONTROL" in result and result["A_CONTROL"]["by_seed"]:
        control_total = next(iter(result["A_CONTROL"]["by_seed"].values()))["total"]
    for arm, entry in result.items():
        if entry["by_seed"]:
            total = next(iter(entry["by_seed"].values()))["total"]
            entry["total"] = total
            entry["delta_vs_control"] = (total - control_total) if control_total else None
            entry["delta_vs_control_pct"] = ((total / control_total - 1) * 100) if control_total else None
    return result


def main() -> None:
    seeds = discover_seeds()
    if not seeds:
        raise SystemExit(f"no seed-* result directories found under {RESULTS_ROOT}")
    print(f"Discovered seeds: {seeds}")

    all_arms: set[str] = set()
    for seed in seeds:
        arms_here = discover_arms(seed)
        all_arms.update(arms_here)
        analyze_seed(seed, arms_here)

    arms = [a for a in ARM_NAMES if a in all_arms]
    pooled_dir = RESULTS_ROOT / "pooled"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    summary = cross_seed_summary(seeds, arms)
    (pooled_dir / "cross-seed-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pooled_bootstrap = pooled_paired_bootstrap(seeds, arms)
    (pooled_dir / "pooled-paired-bootstrap.json").write_text(json.dumps(pooled_bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    params = parameter_counts(seeds, arms)
    (pooled_dir / "parameter-counts.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote pooled/cross-seed-summary.json, pooled/pooled-paired-bootstrap.json, pooled/parameter-counts.json under {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
