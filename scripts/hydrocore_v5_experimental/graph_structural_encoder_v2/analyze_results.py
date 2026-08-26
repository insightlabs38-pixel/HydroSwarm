"""Graph-structural-encoder-v2 (EXPERIMENTAL, NON-RELEASE): paired
statistical analysis of `run_experiment.py`'s per-arm evaluation outputs.

Produces (all under reports/evaluation/graph-structural-encoder-v2/):
  - metric-table.json / .md: baseline vs all-arm metrics per population.
  - condition-matched-known-vs-unseen.json: CLEAN-stage known vs
    ood-UNSEEN_TOPOLOGY comparison per arm.
  - centrality-subgroups.json / distance-subgroups.json: low/med/high
    centrality terciles and short/long sensor-distance groups.
  - paired-transitions/<arm>-vs-control-<population>.json: 2x2 top1/top3
    transition tables, true-source-rank deltas, bootstrap CIs, per arm vs
    A_CONTROL, on identical paired examples (same scenario_id).
  - centrality_vs_observability.json: post-hoc stratified/logistic-
    regression-style analysis of whether centrality remains associated
    with failure after conditioning on sensor distance.

Bootstrap convention (2000 resamples, seed, 90% percentile interval) matches
`exp/failure-mode-diagnostics`'s own established convention
(`bootstrap_followup.py`) -- reimplemented here, not imported (that branch
is not merged; see plan doc Section 0).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "reports" / "evaluation" / "graph-structural-encoder-v2"

ARM_NAMES = ("A_CONTROL", "B_CENTRALITY", "C_OBSERVABILITY", "D_STRUCTURAL_AGG", "D_CAPACITY_CONTROL", "E_COMBINED")
POPULATIONS = ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_INTERVAL = 0.90


def load_evaluation(arm: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{arm.lower()}-evaluation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(arm: str, population: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"{arm.lower()}-{population}-rows.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


# --------------------------------------------------------------------------
# 1. Metric table
# --------------------------------------------------------------------------


def build_metric_table(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for population in POPULATIONS:
        table[population] = {}
        for arm in ARM_NAMES:
            populations = evaluations[arm].get("populations", {})
            stats = populations.get(population, {})
            table[population][arm] = {
                key: stats.get(key)
                for key in (
                    "n",
                    "n_localized",
                    "top1",
                    "top3",
                    "mrr",
                    "proxy_actionable_rate",
                    "proxy_abstention_rate",
                    "proxy_candidate_set_size",
                    "proxy_calibrated_coverage",
                    "ood_caution_or_outside_rate",
                    "known_family_fraction",
                    "event_presence_accuracy",
                )
            }
    table["calibration"] = {arm: evaluations[arm].get("calibration") for arm in ARM_NAMES}
    table["parameter_report"] = {arm: evaluations[arm].get("parameter_report") for arm in ARM_NAMES}
    table["hard_safety_counters"] = {arm: evaluations[arm].get("hard_safety_counters") for arm in ARM_NAMES}
    return table


def metric_table_markdown(table: dict[str, Any]) -> str:
    lines = ["# Graph-structural-encoder-v2: baseline vs all-arm metrics\n"]
    for population in POPULATIONS:
        lines.append(f"## {population}\n")
        lines.append("| arm | n | n_localized | top1 | top3 | mrr | actionable | abstention | candidate_size | coverage |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARM_NAMES:
            row = table[population][arm]

            def fmt(value: Any) -> str:
                if value is None:
                    return "-"
                if isinstance(value, float):
                    return f"{value:.3f}"
                return str(value)

            lines.append(
                f"| {arm} | {fmt(row['n'])} | {fmt(row['n_localized'])} | {fmt(row['top1'])} | {fmt(row['top3'])} | "
                f"{fmt(row['mrr'])} | {fmt(row['proxy_actionable_rate'])} | {fmt(row['proxy_abstention_rate'])} | "
                f"{fmt(row['proxy_candidate_set_size'])} | {fmt(row['proxy_calibrated_coverage'])} |"
            )
        lines.append("")
    lines.append("## Parameter counts (total)\n")
    lines.append("| arm | total | encoders (graph_encoder-inclusive) |")
    lines.append("|---|---|---|")
    for arm in ARM_NAMES:
        report = table["parameter_report"][arm]
        lines.append(f"| {arm} | {report['total']} | {report['encoders']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 2. Condition-matched known vs unseen (CLEAN stage proxy for NOMINAL)
# --------------------------------------------------------------------------


def condition_matched_known_vs_unseen(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARM_NAMES:
        known_rows = [
            row
            for population in ("validation", "development_holdout")
            for row in rows_by_arm[arm][population]
            if row.get("has_source") and row.get("stage") == "CLEAN"
        ]
        unseen_rows = [
            row
            for row in rows_by_arm[arm]["ood-UNSEEN_TOPOLOGY"]
            if row.get("has_source") and row.get("stage") == "CLEAN"
        ]
        known_top1 = statistics.fmean(row["top1"] for row in known_rows) if known_rows else None
        unseen_top1 = statistics.fmean(row["top1"] for row in unseen_rows) if unseen_rows else None
        result[arm] = {
            "known_n": len(known_rows),
            "known_top1": known_top1,
            "unseen_n": len(unseen_rows),
            "unseen_top1": unseen_top1,
            "gap_known_minus_unseen": (known_top1 - unseen_top1) if (known_top1 is not None and unseen_top1 is not None) else None,
            "note": "CLEAN (CurriculumStage) used as this corpus's own least-stressed-condition proxy; NOT the same taxonomy as M11.6's condition_kind (that field does not exist in this corpus -- see report Limitations).",
        }
    return result


# --------------------------------------------------------------------------
# 3. Centrality / distance subgroups
# --------------------------------------------------------------------------


def _tercile_bounds(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    low_cut = ordered[max(0, n // 3 - 1)]
    high_cut = ordered[max(0, 2 * n // 3 - 1)]
    return low_cut, high_cut


def _bucket(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value <= high_cut:
        return "mid"
    return "high"


def centrality_subgroups(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    # Reference distribution for tercile cutoffs: A_CONTROL's pooled
    # localized rows across all three populations. Centrality is a
    # deterministic function of the true source's position in its own
    # topology, identical across arms for the same physical example (every
    # arm evaluates the same scenario_ids) -- so using CONTROL's cutoffs for
    # every arm keeps bucket boundaries identical across arms, which is what
    # makes the per-arm comparison paired/apples-to-apples.
    reference = [
        row
        for population in POPULATIONS
        for row in rows_by_arm["A_CONTROL"][population]
        if row.get("has_source") and "source_betweenness_centrality" in row
    ]
    low_cut, high_cut = _tercile_bounds([row["source_betweenness_centrality"] for row in reference])

    result: dict[str, Any] = {"tercile_cutoffs": {"low_cut": low_cut, "high_cut": high_cut}, "by_arm": {}}
    for arm in ARM_NAMES:
        pooled = [
            row
            for population in POPULATIONS
            for row in rows_by_arm[arm][population]
            if row.get("has_source") and "source_betweenness_centrality" in row
        ]
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
    reference = [
        row
        for population in POPULATIONS
        for row in rows_by_arm["A_CONTROL"][population]
        if row.get("has_source") and "source_hop_to_nearest_sensor_normalized" in row
    ]
    median = statistics.median(row["source_hop_to_nearest_sensor_normalized"] for row in reference) if reference else 0.0
    coverage_median = (
        statistics.median(row["source_local_sensor_coverage_density"] for row in reference) if reference else 0.0
    )

    result: dict[str, Any] = {
        "distance_median_split": median,
        "coverage_median_split": coverage_median,
        "by_arm": {},
    }
    for arm in ARM_NAMES:
        pooled = [
            row
            for population in POPULATIONS
            for row in rows_by_arm[arm][population]
            if row.get("has_source") and "source_hop_to_nearest_sensor_normalized" in row
        ]
        short = [row for row in pooled if row["source_hop_to_nearest_sensor_normalized"] <= median]
        long = [row for row in pooled if row["source_hop_to_nearest_sensor_normalized"] > median]
        sparse = [row for row in pooled if row["source_local_sensor_coverage_density"] <= coverage_median]
        dense = [row for row in pooled if row["source_local_sensor_coverage_density"] > coverage_median]

        def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "n": len(group),
                "top1": statistics.fmean(row["top1"] for row in group) if group else None,
                "top3": statistics.fmean(row["top3"] for row in group) if group else None,
                "mrr": statistics.fmean(row["reciprocal_rank"] for row in group) if group else None,
            }

        result["by_arm"][arm] = {
            "short_distance": summarize(short),
            "long_distance": summarize(long),
            "sparse_coverage": summarize(sparse),
            "dense_coverage": summarize(dense),
        }
    return result


# --------------------------------------------------------------------------
# 4. Paired transitions (each arm vs A_CONTROL, identical scenario_ids)
# --------------------------------------------------------------------------


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
            top3_table = {"both_correct": 0, "control_only": 0, "arm_only": 0, "both_wrong": 0}
            for control_row, arm_row in zip(control_rows, arm_rows):
                if control_row["top1"] and arm_row["top1"]:
                    top1_table["both_correct"] += 1
                elif control_row["top1"] and not arm_row["top1"]:
                    top1_table["control_only"] += 1
                elif not control_row["top1"] and arm_row["top1"]:
                    top1_table["arm_only"] += 1
                else:
                    top1_table["both_wrong"] += 1
                if control_row["top3"] and arm_row["top3"]:
                    top3_table["both_correct"] += 1
                elif control_row["top3"] and not arm_row["top3"]:
                    top3_table["control_only"] += 1
                elif not control_row["top3"] and arm_row["top3"]:
                    top3_table["arm_only"] += 1
                else:
                    top3_table["both_wrong"] += 1

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
            entropy_deltas = [
                arm_row["posterior_entropy_bits"] - control_row["posterior_entropy_bits"]
                for control_row, arm_row in zip(control_rows, arm_rows)
            ]

            top1_bootstrap = paired_bootstrap(
                [float(row["top1"]) for row in control_rows], [float(row["top1"]) for row in arm_rows]
            )
            top3_bootstrap = paired_bootstrap(
                [float(row["top3"]) for row in control_rows], [float(row["top3"]) for row in arm_rows]
            )

            result[population][arm] = {
                "n": len(shared_ids),
                "top1_transition_table": top1_table,
                "top3_transition_table": top3_table,
                "top1_bootstrap_delta": top1_bootstrap,
                "top3_bootstrap_delta": top3_bootstrap,
                "mean_rank_delta": statistics.fmean(rank_deltas) if rank_deltas else None,
                "rank_improved_unchanged_worsened": [
                    sum(1 for delta in rank_deltas if delta < 0),
                    sum(1 for delta in rank_deltas if delta == 0),
                    sum(1 for delta in rank_deltas if delta > 0),
                ],
                "mean_margin_delta": statistics.fmean(margin_deltas) if margin_deltas else None,
                "mean_entropy_delta": statistics.fmean(entropy_deltas) if entropy_deltas else None,
                "fraction_identical_top1_and_top3": statistics.fmean(
                    1.0
                    if (control_row["top1"] == arm_row["top1"] and control_row["top3"] == arm_row["top3"])
                    else 0.0
                    for control_row, arm_row in zip(control_rows, arm_rows)
                ),
            }
    return result


# --------------------------------------------------------------------------
# 5. Centrality vs observability post-hoc diagnostic
# --------------------------------------------------------------------------


def _fit_logistic(features: np.ndarray, labels: np.ndarray, *, steps: int = 2000, lr: float = 0.1) -> np.ndarray:
    """Tiny dependency-free logistic regression (no sklearn in this
    environment) via full-batch gradient descent on standardized features.
    Returns coefficients in STANDARDIZED feature units (comparable
    magnitudes across features of different natural scale) plus an
    intercept as the last element."""

    x = torch.tensor(features, dtype=torch.float64)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    x = torch.cat((x, torch.ones(x.shape[0], 1, dtype=torch.float64)), dim=-1)
    y = torch.tensor(labels, dtype=torch.float64)
    weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = x @ weights
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return weights.detach().numpy()


def centrality_vs_observability(rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Uses A_CONTROL's own rows only (this question is about what predicts
    failure in a model that has neither feature type -- i.e. is centrality/
    distance predictive of the *existing* architecture's failures, the exact
    question the diagnostics report's Section 2 asked, replicated here on
    this pilot's own paired examples)."""

    pooled = [
        row
        for population in POPULATIONS
        for row in rows_by_arm["A_CONTROL"][population]
        if row.get("has_source")
        and "source_betweenness_centrality" in row
        and "source_hop_to_nearest_sensor_normalized" in row
    ]
    if len(pooled) < 20:
        return {"n": len(pooled), "note": "too few localized examples for a meaningful post-hoc fit"}

    centrality = np.array([row["source_betweenness_centrality"] for row in pooled])
    distance = np.array([row["source_hop_to_nearest_sensor_normalized"] for row in pooled])
    top1 = np.array([1.0 if row["top1"] else 0.0 for row in pooled])

    univariate_centrality = _fit_logistic(centrality.reshape(-1, 1), top1)
    univariate_distance = _fit_logistic(distance.reshape(-1, 1), top1)
    joint = _fit_logistic(np.stack((centrality, distance), axis=-1), top1)

    correlation = float(np.corrcoef(centrality, distance)[0, 1])

    # Stratified check: does centrality's association with top1 survive
    # within each distance tercile? (plan doc Section 9 "D": does the
    # apparent centrality effect disappear after conditioning on
    # observability.)
    low_cut, high_cut = _tercile_bounds(list(distance))
    stratified: dict[str, Any] = {}
    for name, mask in (
        ("short_distance", distance <= low_cut),
        ("mid_distance", (distance > low_cut) & (distance <= high_cut)),
        ("long_distance", distance > high_cut),
    ):
        subset_centrality = centrality[mask]
        subset_top1 = top1[mask]
        if len(subset_centrality) >= 10 and len(set(subset_top1.tolist())) > 1:
            coefficient = float(_fit_logistic(subset_centrality.reshape(-1, 1), subset_top1)[0])
        else:
            coefficient = None
        stratified[name] = {
            "n": int(mask.sum()),
            "mean_top1": float(subset_top1.mean()) if len(subset_top1) else None,
            "centrality_coefficient_standardized": coefficient,
        }

    return {
        "n": len(pooled),
        "centrality_distance_correlation": correlation,
        "univariate_centrality_coefficient_standardized": float(univariate_centrality[0]),
        "univariate_distance_coefficient_standardized": float(univariate_distance[0]),
        "joint_centrality_coefficient_standardized": float(joint[0]),
        "joint_distance_coefficient_standardized": float(joint[1]),
        "interpretation_note": (
            "Coefficients are on standardized (z-scored) features, so magnitudes are "
            "directly comparable within this table. A joint centrality coefficient "
            "much smaller than the univariate one indicates centrality's apparent "
            "effect is largely explained by its correlation with sensor distance; a "
            "joint coefficient of comparable magnitude indicates centrality carries "
            "independent information. This is a descriptive/associational fit, not a "
            "causal estimate -- see plan doc Section 9's explicit caveat."
        ),
        "stratified_by_distance_tercile": stratified,
    }


def main() -> None:
    evaluations = {arm: load_evaluation(arm) for arm in ARM_NAMES}
    rows_by_arm = {arm: {population: load_rows(arm, population) for population in POPULATIONS} for arm in ARM_NAMES}

    metric_table = build_metric_table(evaluations)
    (RESULTS_DIR / "metric-table.json").write_text(json.dumps(metric_table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS_DIR / "metric-table.md").write_text(metric_table_markdown(metric_table), encoding="utf-8")

    condition_matched = condition_matched_known_vs_unseen(rows_by_arm)
    (RESULTS_DIR / "condition-matched-known-vs-unseen.json").write_text(
        json.dumps(condition_matched, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    centrality = centrality_subgroups(rows_by_arm)
    (RESULTS_DIR / "centrality-subgroups.json").write_text(json.dumps(centrality, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    distance = distance_subgroups(rows_by_arm)
    (RESULTS_DIR / "distance-subgroups.json").write_text(json.dumps(distance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    transitions = paired_transitions(rows_by_arm)
    (RESULTS_DIR / "paired-transitions.json").write_text(json.dumps(transitions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    centrality_vs_obs = centrality_vs_observability(rows_by_arm)
    (RESULTS_DIR / "centrality-vs-observability.json").write_text(
        json.dumps(centrality_vs_obs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote analysis outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
