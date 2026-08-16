"""Milestone 9.2 pure-computation analysis library.

Every function here consumes the canonical diagnostic table (or a slice of
it) and returns plain dict/list structures ready for `json.dumps`. No I/O,
no model construction, no governance assertions -- those live in
`run_m9_2_build_table.py` and `run_m9_2_analyze.py` respectively. Kept
separate so the statistics themselves are unit-testable without paying for
table construction or model inference in every test.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

import numpy as np
import pandas as pd

import m9_2_common as m92

DEPTHS = list(m92.CAUSAL_PREFIX_DEPTHS)
NOVEL_ARMS = list(m92.NOVEL_ARMS)
SEEDS = list(m92.SCREENING_SEEDS)


# ---------------------------------------------------------------------------
# Section 5: depth-by-depth metrics + paired deltas + bootstrap.
# ---------------------------------------------------------------------------


def _metric_block(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"n": 0}
    return {
        "n": int(len(rows)),
        "top1": float(rows["top1_correct"].mean()),
        "mrr": float(rows["reciprocal_rank"].mean()),
        "median_true_source_rank": float(rows["true_source_rank"].median()),
        "mean_true_source_rank": float(rows["true_source_rank"].mean()),
        "nll": float(rows["nll"].replace([np.inf], np.nan).mean()),
        "brier": float(rows["brier"].mean()),
        "entropy": float(rows["entropy"].mean()),
        "mean_probability_true_source": float(rows["probability_true_source"].mean()),
        "mean_max_probability": float(rows["max_predicted_probability"].mean()),
        "conformal_coverage": float(rows["true_source_covered"].mean()) if rows["true_source_covered"].notna().any() else None,
        "mean_normalized_candidate_set_size": float(rows["conformal_normalized_set_size"].mean()) if rows["conformal_normalized_set_size"].notna().any() else None,
    }


def depth_metrics_by_arm_seed(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in m92.ALL_ARMS:
        out[arm] = {}
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            sub_arm_seed = df[(df["arm"] == arm) & (df["training_seed"] == seed)]
            for depth in DEPTHS:
                out[arm][str(seed)][str(depth)] = _metric_block(sub_arm_seed[sub_arm_seed["prefix_depth"] == depth])
    return out


def paired_deltas_by_depth(df: pd.DataFrame) -> dict[str, Any]:
    """For every novel arm vs CURRENT, per screening seed and per depth:
    paired delta (novel - CURRENT) on top1 and MRR, plus a two-seed mean and
    an incident-level bootstrap 90% CI (M9.2's own bootstrap seed 20260816,
    diagnostic only -- Section 5)."""

    out: dict[str, Any] = {}
    for arm in NOVEL_ARMS:
        out[arm] = {}
        for depth in DEPTHS:
            per_seed: dict[str, Any] = {}
            seed_mean_top1_deltas = []
            seed_mean_mrr_deltas = []
            pooled_novel_top1: list[float] = []
            pooled_current_top1: list[float] = []
            for seed in SEEDS:
                novel = df[(df["arm"] == arm) & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")
                current = df[(df["arm"] == "CURRENT") & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")
                novel, current = novel.align(current, join="inner", axis=0)
                assert len(novel) == len(current) and len(novel) > 0
                top1_delta = float(novel["top1_correct"].astype(float).mean() - current["top1_correct"].astype(float).mean())
                mrr_delta = float(novel["reciprocal_rank"].mean() - current["reciprocal_rank"].mean())
                bootstrap = m92.paired_bootstrap_m9_2(
                    novel["top1_correct"].astype(float).tolist(), current["top1_correct"].astype(float).tolist()
                )
                per_seed[str(seed)] = {
                    "top1_delta": top1_delta,
                    "mrr_delta": mrr_delta,
                    "n_incidents": int(len(novel)),
                    "bootstrap_top1_delta": bootstrap,
                }
                seed_mean_top1_deltas.append(top1_delta)
                seed_mean_mrr_deltas.append(mrr_delta)
                pooled_novel_top1.extend(novel["top1_correct"].astype(float).tolist())
                pooled_current_top1.extend(current["top1_correct"].astype(float).tolist())
            out[arm][str(depth)] = {
                "per_seed": per_seed,
                "two_seed_mean_top1_delta": statistics.fmean(seed_mean_top1_deltas),
                "two_seed_mean_mrr_delta": statistics.fmean(seed_mean_mrr_deltas),
            }
    return out


# ---------------------------------------------------------------------------
# Section 6: paired disagreement / complementarity analysis.
# ---------------------------------------------------------------------------


def disagreement_tables(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"by_arm_depth": {}, "cross_arm_overlap": {}}
    current_by_seed_depth_incident = {
        (seed, depth): df[(df["arm"] == "CURRENT") & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")["top1_correct"]
        for seed in SEEDS
        for depth in DEPTHS
    }
    for arm in NOVEL_ARMS:
        out["by_arm_depth"][arm] = {}
        for depth in DEPTHS:
            per_seed = {}
            for seed in SEEDS:
                current_correct = current_by_seed_depth_incident[(seed, depth)]
                novel_correct = df[(df["arm"] == arm) & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")["top1_correct"]
                current_correct, novel_correct = current_correct.align(novel_correct, join="inner")
                a = int(((current_correct) & (novel_correct)).sum())
                b = int(((current_correct) & (~novel_correct)).sum())
                c = int(((~current_correct) & (novel_correct)).sum())
                d = int(((~current_correct) & (~novel_correct)).sum())
                n = a + b + c + d
                current_only_wins = df[(df["arm"] == "CURRENT") & (df["training_seed"] == seed) & (df["prefix_depth"] == depth) & (df["incident_id"].isin(current_correct[current_correct & ~novel_correct].index))]
                novel_only_wins_ids = novel_correct[(~current_correct) & novel_correct].index.tolist()
                current_only_wins_ids = current_correct[current_correct & (~novel_correct)].index.tolist()
                per_seed[str(seed)] = {
                    "both_correct_A": a,
                    "current_only_B": b,
                    "novel_only_C": c,
                    "both_wrong_D": d,
                    "n": n,
                    "pct_both_correct": a / n if n else None,
                    "pct_current_only": b / n if n else None,
                    "pct_novel_only": c / n if n else None,
                    "pct_both_wrong": d / n if n else None,
                    "net_paired_advantage_C_minus_B": c - b,
                    "complementarity_ratio_C_over_maxB1": c / max(b, 1),
                    "current_only_win_incident_ids": sorted(int(i) for i in current_only_wins_ids),
                    "novel_only_win_incident_ids": sorted(int(i) for i in novel_only_wins_ids),
                }
            out["by_arm_depth"][arm][str(depth)] = per_seed
    # Cross-arm overlap among the three CT arms' failures/wins at each seed+depth.
    for seed in SEEDS:
        out["cross_arm_overlap"][str(seed)] = {}
        for depth in DEPTHS:
            current_correct = current_by_seed_depth_incident[(seed, depth)]
            per_arm_correct = {}
            for arm in NOVEL_ARMS:
                s = df[(df["arm"] == arm) & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")["top1_correct"]
                per_arm_correct[arm] = s
            aligned = pd.DataFrame(per_arm_correct).join(current_correct.rename("CURRENT")).dropna()
            all_ct_lose_current_wins = aligned[(~aligned[NOVEL_ARMS[0]]) & (~aligned[NOVEL_ARMS[1]]) & (~aligned[NOVEL_ARMS[2]]) & aligned["CURRENT"]]
            all_ct_win_current_loses = aligned[aligned[NOVEL_ARMS[0]] & aligned[NOVEL_ARMS[1]] & aligned[NOVEL_ARMS[2]] & (~aligned["CURRENT"])]
            unique_contributors = {}
            for arm in NOVEL_ARMS:
                others = [a for a in NOVEL_ARMS if a != arm]
                unique = aligned[aligned[arm] & (~aligned[others[0]]) & (~aligned[others[1]]) & (~aligned["CURRENT"])]
                unique_contributors[arm] = sorted(int(i) for i in unique.index)
            out["cross_arm_overlap"][str(seed)][str(depth)] = {
                "n_incidents": int(len(aligned)),
                "all_ct_lose_current_wins_ids": sorted(int(i) for i in all_ct_lose_current_wins.index),
                "all_ct_win_current_loses_ids": sorted(int(i) for i in all_ct_win_current_loses.index),
                "unique_contributor_wins_by_arm": unique_contributors,
            }
    return out


# ---------------------------------------------------------------------------
# Section 7: true-source rank movement.
# ---------------------------------------------------------------------------


def rank_movement(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in NOVEL_ARMS:
        out[arm] = {}
        for depth in DEPTHS:
            per_seed = {}
            for seed in SEEDS:
                novel = df[(df["arm"] == arm) & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")["true_source_rank"]
                current = df[(df["arm"] == "CURRENT") & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)].set_index("incident_id")["true_source_rank"]
                novel, current = novel.align(current, join="inner")
                delta = novel - current
                n = len(delta)
                per_seed[str(seed)] = {
                    "n": int(n),
                    "mean": float(delta.mean()),
                    "median": float(delta.median()),
                    "q1": float(delta.quantile(0.25)),
                    "q3": float(delta.quantile(0.75)),
                    "fraction_improved": float((delta < 0).mean()),
                    "fraction_unchanged": float((delta == 0).mean()),
                    "fraction_worsened": float((delta > 0).mean()),
                    "large_regressions": {
                        f">={t}": int((delta >= t).sum()) for t in m92.RANK_DELTA_LARGE_THRESHOLDS
                    },
                    "large_improvements": {
                        f"<=-{t}": int((delta <= -t).sum()) for t in m92.RANK_DELTA_LARGE_THRESHOLDS
                    },
                }
            out[arm][str(depth)] = per_seed
    return out


# ---------------------------------------------------------------------------
# Section 8: topology/spatial error analysis.
# ---------------------------------------------------------------------------


def topology_error_analysis(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in m92.ALL_ARMS:
        out[arm] = {}
        for depth in DEPTHS:
            per_seed = {}
            for seed in SEEDS:
                sub = df[(df["arm"] == arm) & (df["training_seed"] == seed) & (df["prefix_depth"] == depth)]
                wrong = sub[~sub["top1_correct"]]
                distances = wrong["graph_distance_pred_to_true"].dropna()
                per_seed[str(seed)] = {
                    "n": int(len(sub)),
                    "exact_source_accuracy": float(sub["top1_correct"].mean()),
                    "n_wrong": int(len(wrong)),
                    "within_1_hop_accuracy": float((distances <= 1).mean()) if len(distances) else None,
                    "within_2_hop_accuracy": float((distances <= 2).mean()) if len(distances) else None,
                    "mean_graph_distance_when_wrong": float(distances.mean()) if len(distances) else None,
                    "median_graph_distance_when_wrong": float(distances.median()) if len(distances) else None,
                }
            out[arm][str(depth)] = per_seed
    # Stratify errors by true-source node degree (structural property, predeclared not outcome-chosen).
    by_degree: dict[str, Any] = {}
    for arm in m92.ALL_ARMS:
        by_degree[arm] = {}
        for degree_value in sorted(df["true_source_degree"].dropna().unique().tolist()):
            sub = df[(df["arm"] == arm) & (df["true_source_degree"] == degree_value)]
            by_degree[arm][str(int(degree_value))] = {"n": int(len(sub)), "top1": float(sub["top1_correct"].mean())}
    out["by_true_source_degree"] = by_degree
    return out


# ---------------------------------------------------------------------------
# Sections 9-10: missingness / difficulty stratification (predeclared quartile bins).
# ---------------------------------------------------------------------------


PREDECLARED_COVARIATES = (
    "fraction_missing",
    "gap_coefficient_of_variation",
    "mean_gap_seconds",
    "n_valid_observations",
)


def _quartile_bins(series: pd.Series) -> pd.Series:
    """Fixed quartile edges computed from the covariate's OWN complete
    development_holdout distribution (Section 9), independent of model
    correctness. Falls back to fewer bins if the covariate has too few
    distinct values for 4 quantile groups (duplicate edges)."""

    clean = series.dropna()
    if clean.nunique() < 4:
        return pd.cut(series, bins=max(1, clean.nunique()), labels=False, duplicates="drop")
    try:
        return pd.qcut(series, q=4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"], duplicates="drop")
    except ValueError:
        return pd.cut(series, bins=4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"], duplicates="drop")


def covariate_stratified_deltas(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Bins computed once over the FULL covariate distribution across all
    # arms/seeds/depths pooled (the covariate itself is arm-independent --
    # identical for a given incident+depth across arms -- so pooling here
    # only affects the quartile-edge estimate, not which arm a row belongs to).
    base = df[df["arm"] == "CURRENT"]  # one row per (incident, depth) pair per seed, arm-independent covariates
    for covariate in PREDECLARED_COVARIATES:
        edges_source = base[covariate]
        bins = _quartile_bins(edges_source)
        bin_map = dict(zip(base.set_index(["training_seed", "incident_id", "prefix_depth"]).index, bins))
        df_key = list(zip(df["training_seed"], df["incident_id"], df["prefix_depth"]))
        df = df.assign(**{f"_bin_{covariate}": [bin_map.get(k) for k in df_key]})

    out["covariates"] = {}
    for covariate in PREDECLARED_COVARIATES:
        bin_col = f"_bin_{covariate}"
        out["covariates"][covariate] = {"by_bin": {}}
        for bin_label in [b for b in df[bin_col].dropna().unique()]:
            bin_label_key = str(bin_label)
            out["covariates"][covariate]["by_bin"][bin_label_key] = {}
            for arm in NOVEL_ARMS:
                per_seed = {}
                for seed in SEEDS:
                    novel = df[(df["arm"] == arm) & (df["training_seed"] == seed) & (df[bin_col] == bin_label)].set_index(["incident_id", "prefix_depth"])["top1_correct"]
                    current = df[(df["arm"] == "CURRENT") & (df["training_seed"] == seed) & (df[bin_col] == bin_label)].set_index(["incident_id", "prefix_depth"])["top1_correct"]
                    novel, current = novel.align(current, join="inner")
                    n = len(novel)
                    per_seed[str(seed)] = {
                        "n": int(n),
                        "novel_top1": float(novel.astype(float).mean()) if n else None,
                        "current_top1": float(current.astype(float).mean()) if n else None,
                        "paired_delta": float(novel.astype(float).mean() - current.astype(float).mean()) if n else None,
                    }
                out["covariates"][covariate]["by_bin"][bin_label_key][arm] = per_seed
    return out


def current_conditioned_difficulty(df: pd.DataFrame) -> dict[str, Any]:
    """Section 10: difficulty strata derived from CURRENT's own predictions
    (probability of true source, entropy, rank), CLEARLY LABELED
    CURRENT-CONDITIONED per the milestone brief -- descriptive only, never
    used to select a novel arm."""

    current = df[df["arm"] == "CURRENT"].copy()
    out: dict[str, Any] = {"label": "CURRENT_CONDITIONED_DESCRIPTIVE", "by_covariate": {}}
    for covariate in ("probability_true_source", "true_source_rank", "entropy"):
        bins = _quartile_bins(current[covariate])
        bin_map = dict(zip(current.set_index(["training_seed", "incident_id", "prefix_depth"]).index, bins))
        df_key = list(zip(df["training_seed"], df["incident_id"], df["prefix_depth"]))
        tagged = df.assign(_bin=[bin_map.get(k) for k in df_key])
        out["by_covariate"][covariate] = {}
        for bin_label in [b for b in tagged["_bin"].dropna().unique()]:
            out["by_covariate"][covariate][str(bin_label)] = {}
            for arm in NOVEL_ARMS:
                per_seed = {}
                for seed in SEEDS:
                    novel = tagged[(tagged["arm"] == arm) & (tagged["training_seed"] == seed) & (tagged["_bin"] == bin_label)].set_index(["incident_id", "prefix_depth"])["top1_correct"]
                    curr = tagged[(tagged["arm"] == "CURRENT") & (tagged["training_seed"] == seed) & (tagged["_bin"] == bin_label)].set_index(["incident_id", "prefix_depth"])["top1_correct"]
                    novel, curr = novel.align(curr, join="inner")
                    n = len(novel)
                    per_seed[str(seed)] = {
                        "n": int(n),
                        "paired_delta": float(novel.astype(float).mean() - curr.astype(float).mean()) if n else None,
                    }
                out["by_covariate"][covariate][str(bin_label)][arm] = per_seed
    return out


# ---------------------------------------------------------------------------
# Section 11: confidence / calibration diagnostics.
# ---------------------------------------------------------------------------


def calibration_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"by_arm_depth_correctness": {}, "paired_case_probability_comparison": {}}
    for arm in m92.ALL_ARMS:
        out["by_arm_depth_correctness"][arm] = {}
        for depth in DEPTHS:
            sub = df[(df["arm"] == arm) & (df["prefix_depth"] == depth)]
            correct = sub[sub["top1_correct"]]
            incorrect = sub[~sub["top1_correct"]]
            out["by_arm_depth_correctness"][arm][str(depth)] = {
                "correct": {
                    "n": int(len(correct)),
                    "mean_max_probability": float(correct["max_predicted_probability"].mean()) if len(correct) else None,
                    "mean_entropy": float(correct["entropy"].mean()) if len(correct) else None,
                    "mean_probability_true_source": float(correct["probability_true_source"].mean()) if len(correct) else None,
                    "coverage": float(correct["true_source_covered"].mean()) if correct["true_source_covered"].notna().any() else None,
                    "mean_normalized_set_size": float(correct["conformal_normalized_set_size"].mean()) if correct["conformal_normalized_set_size"].notna().any() else None,
                },
                "incorrect": {
                    "n": int(len(incorrect)),
                    "mean_max_probability": float(incorrect["max_predicted_probability"].mean()) if len(incorrect) else None,
                    "mean_entropy": float(incorrect["entropy"].mean()) if len(incorrect) else None,
                    "mean_probability_true_source": float(incorrect["probability_true_source"].mean()) if len(incorrect) else None,
                    "coverage": float(incorrect["true_source_covered"].mean()) if incorrect["true_source_covered"].notna().any() else None,
                    "mean_normalized_set_size": float(incorrect["conformal_normalized_set_size"].mean()) if incorrect["conformal_normalized_set_size"].notna().any() else None,
                },
            }
    for arm in NOVEL_ARMS:
        out["paired_case_probability_comparison"][arm] = {}
        for seed in SEEDS:
            current = df[(df["arm"] == "CURRENT") & (df["training_seed"] == seed)].set_index(["incident_id", "prefix_depth"])
            novel = df[(df["arm"] == arm) & (df["training_seed"] == seed)].set_index(["incident_id", "prefix_depth"])
            joined = current[["top1_correct", "probability_true_source", "entropy", "top1_top2_margin"]].join(
                novel[["top1_correct", "probability_true_source", "entropy", "top1_top2_margin"]], lsuffix="_current", rsuffix="_novel", how="inner"
            )
            current_correct_novel_wrong = joined[joined["top1_correct_current"] & (~joined["top1_correct_novel"])]
            novel_correct_current_wrong = joined[(~joined["top1_correct_current"]) & joined["top1_correct_novel"]]
            out["paired_case_probability_comparison"][arm][str(seed)] = {
                "current_correct_novel_wrong": {
                    "n": int(len(current_correct_novel_wrong)),
                    "mean_probability_true_source_current": float(current_correct_novel_wrong["probability_true_source_current"].mean()) if len(current_correct_novel_wrong) else None,
                    "mean_probability_true_source_novel": float(current_correct_novel_wrong["probability_true_source_novel"].mean()) if len(current_correct_novel_wrong) else None,
                    "mean_entropy_current": float(current_correct_novel_wrong["entropy_current"].mean()) if len(current_correct_novel_wrong) else None,
                    "mean_entropy_novel": float(current_correct_novel_wrong["entropy_novel"].mean()) if len(current_correct_novel_wrong) else None,
                },
                "novel_correct_current_wrong": {
                    "n": int(len(novel_correct_current_wrong)),
                    "mean_probability_true_source_current": float(novel_correct_current_wrong["probability_true_source_current"].mean()) if len(novel_correct_current_wrong) else None,
                    "mean_probability_true_source_novel": float(novel_correct_current_wrong["probability_true_source_novel"].mean()) if len(novel_correct_current_wrong) else None,
                    "mean_entropy_current": float(novel_correct_current_wrong["entropy_current"].mean()) if len(novel_correct_current_wrong) else None,
                    "mean_entropy_novel": float(novel_correct_current_wrong["entropy_novel"].mean()) if len(novel_correct_current_wrong) else None,
                },
                "rank_probability_correlation_spearman": float(
                    joined["probability_true_source_current"].corr(joined["probability_true_source_novel"], method="spearman")
                ) if len(joined) > 1 else None,
            }
    return out


# ---------------------------------------------------------------------------
# Section 12: deterministic, outcome-independent case-study selection.
# ---------------------------------------------------------------------------


def case_studies(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mature = df[df["depth_bucket"] == "MATURE"]

    def _row_dict(row: pd.Series) -> dict[str, Any]:
        return {
            "incident_id": int(row["incident_id"]),
            "seed": int(row["training_seed"]),
            "depth": int(row["prefix_depth"]),
            "arm": row["arm"],
            "true_source_node": row["true_source_node"],
            "predicted_top1_node": row["predicted_top1_node"],
            "true_source_rank": int(row["true_source_rank"]),
            "probability_true_source": float(row["probability_true_source"]),
            "entropy": float(row["entropy"]),
            "conformal_candidate_set_size": row["conformal_candidate_set_size"],
            "fraction_missing": row["fraction_missing"],
            "graph_distance_pred_to_true": row["graph_distance_pred_to_true"],
            "curriculum_stage": row["curriculum_stage"],
        }

    for arm in NOVEL_ARMS:
        rows = []
        for seed in SEEDS:
            novel = mature[(mature["arm"] == arm) & (mature["training_seed"] == seed)].set_index(["incident_id", "prefix_depth"])
            current = mature[(mature["arm"] == "CURRENT") & (mature["training_seed"] == seed)].set_index(["incident_id", "prefix_depth"])
            joined_idx = novel.index.intersection(current.index)
            for idx in joined_idx:
                delta = int(novel.loc[idx, "true_source_rank"]) - int(current.loc[idx, "true_source_rank"])
                rows.append((delta, seed, idx))
        rows.sort(key=lambda t: t[0])
        largest_improvements = rows[: m92.CASE_STUDY_TOP_N]
        largest_regressions = rows[-m92.CASE_STUDY_TOP_N :][::-1]

        def _export(selected):
            out_rows = []
            for delta, seed, (incident_id, depth) in selected:
                novel_row = mature[(mature["arm"] == arm) & (mature["training_seed"] == seed) & (mature["incident_id"] == incident_id) & (mature["prefix_depth"] == depth)].iloc[0]
                current_row = mature[(mature["arm"] == "CURRENT") & (mature["training_seed"] == seed) & (mature["incident_id"] == incident_id) & (mature["prefix_depth"] == depth)].iloc[0]
                out_rows.append({"rank_delta": delta, "novel": _row_dict(novel_row), "current": _row_dict(current_row)})
            return out_rows

        out[arm] = {
            "largest_mature_rank_regressions": _export(largest_regressions),
            "largest_mature_rank_improvements": _export(largest_improvements),
        }

    # All-CT-fail-CURRENT-succeeds / all-CT-succeed-CURRENT-fails at MATURE depths.
    all_fail_win = []
    all_win_fail = []
    for seed in SEEDS:
        for depth in m92.MATURE_DEPTHS:
            sub = mature[(mature["training_seed"] == seed) & (mature["prefix_depth"] == depth)]
            current_correct = sub[sub["arm"] == "CURRENT"].set_index("incident_id")["top1_correct"]
            per_arm = {arm: sub[sub["arm"] == arm].set_index("incident_id")["top1_correct"] for arm in NOVEL_ARMS}
            aligned = pd.DataFrame(per_arm).join(current_correct.rename("CURRENT")).dropna()
            fail_win = aligned[(~aligned[NOVEL_ARMS[0]]) & (~aligned[NOVEL_ARMS[1]]) & (~aligned[NOVEL_ARMS[2]]) & aligned["CURRENT"]]
            win_fail = aligned[aligned[NOVEL_ARMS[0]] & aligned[NOVEL_ARMS[1]] & aligned[NOVEL_ARMS[2]] & (~aligned["CURRENT"])]
            for incident_id in fail_win.index[: m92.CASE_STUDY_TOP_N]:
                all_fail_win.append({"seed": int(seed), "depth": int(depth), "incident_id": int(incident_id)})
            for incident_id in win_fail.index[: m92.CASE_STUDY_TOP_N]:
                all_win_fail.append({"seed": int(seed), "depth": int(depth), "incident_id": int(incident_id)})
    out["all_ct_fail_current_succeeds_mature"] = all_fail_win[: m92.CASE_STUDY_TOP_N]
    out["all_ct_succeed_current_fails_mature"] = all_win_fail[: m92.CASE_STUDY_TOP_N]

    # High-missingness / high-irregularity representative examples (predeclared: top decile fraction_missing at MATURE).
    high_missing = mature[mature["arm"] == "CURRENT"].nlargest(m92.CASE_STUDY_TOP_N, "fraction_missing")
    out["high_missingness_representative_examples"] = [
        {
            "incident_id": int(r["incident_id"]),
            "seed": int(r["training_seed"]),
            "depth": int(r["prefix_depth"]),
            "fraction_missing": r["fraction_missing"],
            "current_top1_correct": bool(r["top1_correct"]),
        }
        for _, r in high_missing.iterrows()
    ]
    return out


# ---------------------------------------------------------------------------
# Section 13: cross-seed consistency classification.
# ---------------------------------------------------------------------------


def classify_cross_seed(seed_values: Sequence[float | None], *, sign_tolerance: float = 0.0) -> str:
    values = [v for v in seed_values if v is not None]
    if len(values) < 2:
        return "SINGLE_SEED_ONLY"
    signs = {(1 if v > sign_tolerance else (-1 if v < -sign_tolerance else 0)) for v in values}
    if len(signs) == 1 and 0 not in signs:
        return "ROBUST"
    if len(signs) == 1 and 0 in signs:
        return "ROBUST"
    return "MIXED"
