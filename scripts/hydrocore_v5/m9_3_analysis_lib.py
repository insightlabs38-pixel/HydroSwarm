"""Milestone 9.3 pure-computation analysis library. Consumes the canonical
calibration diagnostic table (or a slice of it) and returns plain dict/list
structures for `json.dumps`. No I/O, no model construction, no governance
assertions, no new inference -- everything here operates on already-computed
per-row `nonconformity_score`/`schemes[...]` fields in the canonical table.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import m9_3_common as m93
from hydroswarm.calibration.conformal import _quantile

DEPTHS = list(m93.DEPTHS)
SEEDS = list(m93.SEEDS)
KNOWN_FAMILIES = list(m93.KNOWN_FAMILIES)
PRIMARY_SCHEME = "CURRENT_FAMILY_DEPTH"


def _scheme_field(df: pd.DataFrame, scheme: str, field: str) -> pd.Series:
    return df["schemes"].apply(lambda d: d.get(scheme, {}).get(field))


# ---------------------------------------------------------------------------
# Section 7: calibration support analysis.
# ---------------------------------------------------------------------------


def support_analysis(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cal = df[df["split"] == "calibration"]
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for bucket in ("EARLY", "MID", "MATURE"):
                    subset = cal[(cal.predictor_arm == arm) & (cal.training_seed == seed) & (cal.topology_family == family) & (cal.depth_bucket == bucket)]
                    n = len(subset)
                    n_incidents = subset["incident_id"].nunique()
                    scores = subset["nonconformity_score"].tolist()
                    quantile = _quantile(scores, m93.ALPHA) if scores else None
                    rank = min(n, max(1, int(np.ceil((n + 1) * (1 - m93.ALPHA))))) if n else None
                    allowable_exceedances = int(np.floor(m93.ALPHA * (n + 1))) if n else None
                    out[arm][str(seed)][family][bucket] = {
                        "n_calibration_rows": n, "n_distinct_incidents": int(n_incidents),
                        "quantile_rank": rank, "resulting_quantile": quantile,
                        "allowable_exceedances_at_alpha": allowable_exceedances,
                        "finite_sample_resolution": m93.finite_sample_resolution(n),
                    }
    return out


def fallback_frequency(df: pd.DataFrame) -> dict[str, Any]:
    dev = df[df["split"] == "development"]
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in SEEDS:
            subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed)]
            sources = _scheme_field(subset, PRIMARY_SCHEME, "selection_source")
            counts = sources.value_counts().to_dict()
            out[arm][str(seed)] = {"n": int(len(subset)), "selection_source_counts": {str(k): int(v) for k, v in counts.items()}}
    return out


# ---------------------------------------------------------------------------
# Section 8: empirical coverage uncertainty (Wilson 90%, predeclared).
# ---------------------------------------------------------------------------


def coverage_uncertainty(df: pd.DataFrame, *, scheme: str = PRIMARY_SCHEME) -> dict[str, Any]:
    dev = df[df["split"] == "development"]
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for depth in DEPTHS:
                    subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed) & (dev.topology_family == family) & (dev.prefix_depth == depth)]
                    n = len(subset)
                    if n == 0:
                        continue
                    covered = _scheme_field(subset, scheme, "covered")
                    n_covered = int(covered.sum())
                    empirical = n_covered / n
                    lower, upper = m93.wilson_interval_90(n_covered, n)
                    out[arm][str(seed)][family][str(depth)] = {
                        "n": n, "empirical_coverage": empirical, "wilson_90_ci": [lower, upper],
                        "distance_to_nominal_0_90": empirical - m93.NOMINAL_COVERAGE_TARGET,
                        "distance_to_operational_floor_0_85": empirical - m93.OPERATIONAL_COVERAGE_FLOOR,
                        "ci_excludes_operational_floor_below": upper < m93.OPERATIONAL_COVERAGE_FLOOR,
                    }
    return out


# ---------------------------------------------------------------------------
# Section 9: calibration vs development nonconformity score-shift diagnostics.
# ---------------------------------------------------------------------------


def _score_stats(scores: Sequence[float]) -> dict[str, Any]:
    if not scores:
        return {"n": 0}
    arr = np.asarray(scores, dtype=float)
    return {
        "n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)), "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def score_shift(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for bucket in ("EARLY", "MID", "MATURE"):
                    cal_scores = df[(df.predictor_arm == arm) & (df.training_seed == seed) & (df.topology_family == family) & (df.depth_bucket == bucket) & (df.split == "calibration")]["nonconformity_score"].tolist()
                    dev_scores = df[(df.predictor_arm == arm) & (df.training_seed == seed) & (df.topology_family == family) & (df.depth_bucket == bucket) & (df.split == "development")]["nonconformity_score"].tolist()
                    cal_stats = _score_stats(cal_scores)
                    dev_stats = _score_stats(dev_scores)
                    result: dict[str, Any] = {"calibration": cal_stats, "development": dev_stats}
                    if cal_scores and dev_scores:
                        ks = scipy_stats.ks_2samp(cal_scores, dev_scores)
                        wasserstein = scipy_stats.wasserstein_distance(cal_scores, dev_scores)
                        result["ks_statistic"] = float(ks.statistic)
                        result["ks_pvalue"] = float(ks.pvalue)
                        result["wasserstein_distance"] = float(wasserstein)
                        result["score_shift_median"] = dev_stats["median"] - cal_stats["median"]
                        result["score_shift_p90"] = dev_stats["p90"] - cal_stats["p90"]
                        result["score_shift_mean"] = dev_stats["mean"] - cal_stats["mean"]
                    out[arm][str(seed)][family][bucket] = result
    return out


# ---------------------------------------------------------------------------
# Section 10: quantile stability under incident-level bootstrap resampling.
# ---------------------------------------------------------------------------


def quantile_stability(df: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(m93.M9_3_BOOTSTRAP_SEED)
    out: dict[str, Any] = {}
    cal = df[df["split"] == "calibration"]
    dev = df[df["split"] == "development"]
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for bucket in ("EARLY", "MID", "MATURE"):
                    cal_subset = cal[(cal.predictor_arm == arm) & (cal.training_seed == seed) & (cal.topology_family == family) & (cal.depth_bucket == bucket)]
                    if cal_subset.empty:
                        continue
                    by_incident = cal_subset.groupby("incident_id")["nonconformity_score"].apply(list)
                    incident_ids = by_incident.index.to_numpy()
                    n_incidents = len(incident_ids)
                    quantiles = np.empty(m93.M9_3_BOOTSTRAP_RESAMPLES)
                    for i in range(m93.M9_3_BOOTSTRAP_RESAMPLES):
                        sampled = rng.choice(incident_ids, size=n_incidents, replace=True)
                        pooled_scores = [s for incident in sampled for s in by_incident.loc[incident]]
                        quantiles[i] = _quantile(pooled_scores, m93.ALPHA)
                    median_q = float(np.median(quantiles))
                    p5, p95 = float(np.percentile(quantiles, 5)), float(np.percentile(quantiles, 95))
                    dev_subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed) & (dev.topology_family == family) & (dev.depth_bucket == bucket)]
                    coverage_at_p5 = float((dev_subset["nonconformity_score"] <= p5).mean()) if not dev_subset.empty else None
                    coverage_at_p95 = float((dev_subset["nonconformity_score"] <= p95).mean()) if not dev_subset.empty else None
                    out[arm][str(seed)][family][bucket] = {
                        "n_incidents": int(n_incidents), "resamples": m93.M9_3_BOOTSTRAP_RESAMPLES,
                        "bootstrap_seed": m93.M9_3_BOOTSTRAP_SEED,
                        "median_quantile": median_q, "p5_quantile": p5, "p95_quantile": p95,
                        "absolute_span": p95 - p5, "relative_span": (p95 - p5) / median_q if median_q else None,
                        "development_coverage_at_p5_quantile": coverage_at_p5,
                        "development_coverage_at_p95_quantile": coverage_at_p95,
                    }
    return out


# ---------------------------------------------------------------------------
# Section 11: calibration-support learning curves (nested, predeclared fractions).
# ---------------------------------------------------------------------------


def support_learning_curves(df: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(m93.LEARNING_CURVE_SEED)
    out: dict[str, Any] = {}
    cal = df[df["split"] == "calibration"]
    dev = df[df["split"] == "development"]
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for bucket in ("EARLY", "MID", "MATURE"):
                    cal_subset = cal[(cal.predictor_arm == arm) & (cal.training_seed == seed) & (cal.topology_family == family) & (cal.depth_bucket == bucket)]
                    dev_subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed) & (dev.topology_family == family) & (dev.depth_bucket == bucket)]
                    if cal_subset.empty or dev_subset.empty:
                        continue
                    by_incident = cal_subset.groupby("incident_id")["nonconformity_score"].apply(list)
                    incident_ids = by_incident.index.to_numpy()
                    n_incidents_full = len(incident_ids)
                    dev_scores = dev_subset["nonconformity_score"].to_numpy()
                    fractions_out = {}
                    for fraction in m93.LEARNING_CURVE_FRACTIONS:
                        n_sub = max(1, int(round(fraction * n_incidents_full)))
                        quantiles = np.empty(m93.LEARNING_CURVE_RESAMPLES_PER_FRACTION)
                        coverages = np.empty(m93.LEARNING_CURVE_RESAMPLES_PER_FRACTION)
                        for i in range(m93.LEARNING_CURVE_RESAMPLES_PER_FRACTION):
                            chosen = rng.choice(incident_ids, size=n_sub, replace=False) if n_sub <= n_incidents_full else rng.choice(incident_ids, size=n_sub, replace=True)
                            pooled = [s for incident in chosen for s in by_incident.loc[incident]]
                            q = _quantile(pooled, m93.ALPHA)
                            quantiles[i] = q
                            coverages[i] = float((dev_scores <= q).mean())
                        fractions_out[str(fraction)] = {
                            "n_incidents_sampled": n_sub,
                            "quantile_mean": float(quantiles.mean()), "quantile_std": float(quantiles.std()),
                            "development_coverage_mean": float(coverages.mean()), "development_coverage_std": float(coverages.std()),
                        }
                    out[arm][str(seed)][family][bucket] = fractions_out
    return out


# ---------------------------------------------------------------------------
# Section 12: family heterogeneity (ARM_B2 only -- 3 known families).
# ---------------------------------------------------------------------------


def family_heterogeneity(df: pd.DataFrame) -> dict[str, Any]:
    cal = df[(df["split"] == "calibration") & (df["predictor_arm"] == "ARM_B2")]
    out: dict[str, Any] = {}
    for seed in SEEDS:
        out[str(seed)] = {}
        for bucket in ("EARLY", "MID", "MATURE"):
            matrix: dict[str, Any] = {}
            for i, family_a in enumerate(KNOWN_FAMILIES):
                for family_b in KNOWN_FAMILIES[i + 1 :]:
                    scores_a = cal[(cal.training_seed == seed) & (cal.topology_family == family_a) & (cal.depth_bucket == bucket)]["nonconformity_score"].tolist()
                    scores_b = cal[(cal.training_seed == seed) & (cal.topology_family == family_b) & (cal.depth_bucket == bucket)]["nonconformity_score"].tolist()
                    if not scores_a or not scores_b:
                        continue
                    ks = scipy_stats.ks_2samp(scores_a, scores_b)
                    wasserstein = scipy_stats.wasserstein_distance(scores_a, scores_b)
                    matrix[f"{family_a}__vs__{family_b}"] = {
                        "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                        "wasserstein_distance": float(wasserstein),
                        "mean_delta": statistics.fmean(scores_b) - statistics.fmean(scores_a),
                        "median_delta": statistics.median(scores_b) - statistics.median(scores_a),
                        "p90_delta": float(np.percentile(scores_b, 90) - np.percentile(scores_a, 90)),
                    }
            out[str(seed)][bucket] = matrix
    return out


# ---------------------------------------------------------------------------
# Section 13: exact depth-level calibration behavior (using the BUCKET's
# fitted quantile, applied at each individual depth within the bucket).
# ---------------------------------------------------------------------------


def depth_root_cause(df: pd.DataFrame, *, scheme: str = PRIMARY_SCHEME) -> dict[str, Any]:
    dev = df[df["split"] == "development"]
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for depth in DEPTHS:
                    subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed) & (dev.topology_family == family) & (dev.prefix_depth == depth)]
                    if subset.empty:
                        continue
                    covered = _scheme_field(subset, scheme, "covered")
                    set_size = _scheme_field(subset, scheme, "candidate_set_size")
                    quantile_used = _scheme_field(subset, scheme, "quantile_used")
                    out[arm][str(seed)][family][str(depth)] = {
                        "n": int(len(subset)), "top1": float(subset["top1_correct"].mean()),
                        "mean_probability_true_source": float(subset["probability_true_source"].mean()),
                        "mean_entropy": float(subset["entropy"].mean()),
                        "mean_nonconformity_score": float(subset["nonconformity_score"].mean()),
                        "quantile_used": float(quantile_used.iloc[0]) if quantile_used.notna().any() else None,
                        "coverage": float(covered.mean()),
                        "mean_candidate_set_size": float(set_size.mean()),
                    }
    return out


# ---------------------------------------------------------------------------
# Section 14: CURRENT (ARM_A) vs INTERLEAVED (ARM_B2) confidence/overconfidence,
# same family (golden-reference, the only family both arms share).
# ---------------------------------------------------------------------------


def confidence_overconfidence(df: pd.DataFrame) -> dict[str, Any]:
    dev = df[(df["split"] == "development") & (df["topology_family"] == "golden-reference")]
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for depth in DEPTHS:
                subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed) & (dev.prefix_depth == depth)]
                if subset.empty:
                    continue
                correct = subset[subset["top1_correct"]]
                incorrect = subset[~subset["top1_correct"]]
                def _block(s: pd.DataFrame) -> dict[str, Any]:
                    if s.empty:
                        return {"n": 0}
                    return {
                        "n": int(len(s)), "mean_max_probability": float(s["max_predicted_probability"].mean()),
                        "mean_probability_true_source": float(s["probability_true_source"].mean()),
                        "mean_entropy": float(s["entropy"].mean()), "mean_nll": float(s["nll"].replace([np.inf], np.nan).mean()),
                        "mean_brier": float(s["brier"].mean()), "mean_top1_top2_margin": float(s["top1_top2_margin"].mean()),
                    }
                out[arm][str(seed)][str(depth)] = {"correct": _block(correct), "incorrect": _block(incorrect)}
    return out


def reliability_bins(df: pd.DataFrame, *, n_bins: int = 10) -> dict[str, Any]:
    dev = df[(df["split"] == "development") & (df["topology_family"] == "golden-reference")]
    out: dict[str, Any] = {}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in SEEDS:
            subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed)]
            if subset.empty:
                continue
            conf = subset["max_predicted_probability"].to_numpy()
            correct = subset["top1_correct"].to_numpy(dtype=float)
            bins = []
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
                if mask.any():
                    bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()), "mean_confidence": float(conf[mask].mean()), "empirical_accuracy": float(correct[mask].mean())})
            out[arm][str(seed)] = bins
    return out


# ---------------------------------------------------------------------------
# Section 15: source-conditional failure analysis.
# ---------------------------------------------------------------------------


def source_conditional(df: pd.DataFrame, *, scheme: str = PRIMARY_SCHEME) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                dev_subset = df[(df.predictor_arm == arm) & (df.training_seed == seed) & (df.topology_family == family) & (df.split == "development")]
                if dev_subset.empty:
                    continue
                out[arm][str(seed)][family] = {}
                for source_node in sorted(dev_subset["true_source_node"].dropna().unique().tolist()):
                    node_rows = dev_subset[dev_subset["true_source_node"] == source_node]
                    covered = _scheme_field(node_rows, scheme, "covered")
                    set_size = _scheme_field(node_rows, scheme, "candidate_set_size")
                    out[arm][str(seed)][family][source_node] = {
                        "n_development": int(len(node_rows)), "empirical_coverage": float(covered.mean()),
                        "mean_nonconformity_score": float(node_rows["nonconformity_score"].mean()),
                        "top1_accuracy": float(node_rows["top1_correct"].mean()),
                        "mean_candidate_set_size": float(set_size.mean()),
                        "true_source_degree": (node_rows["true_source_degree"].iloc[0] if node_rows["true_source_degree"].notna().any() else None),
                    }
    return out


# ---------------------------------------------------------------------------
# Section 17: miscoverage severity.
# ---------------------------------------------------------------------------


def miscoverage_severity(df: pd.DataFrame, *, scheme: str = PRIMARY_SCHEME) -> dict[str, Any]:
    dev = df[df["split"] == "development"].copy()
    dev["_covered"] = _scheme_field(dev, scheme, "covered")
    dev["_quantile"] = _scheme_field(dev, scheme, "quantile_used")
    uncovered = dev[dev["_covered"] == False]  # noqa: E712
    uncovered = uncovered[uncovered["_quantile"].notna()]
    excess = (uncovered["nonconformity_score"] - uncovered["_quantile"]).to_numpy()
    out: dict[str, Any] = {"overall": {}}
    if excess.size:
        out["overall"] = {
            "n_uncovered": int(excess.size), "median_excess": float(np.median(excess)),
            "p75_excess": float(np.percentile(excess, 75)), "p90_excess": float(np.percentile(excess, 90)),
            "max_excess": float(excess.max()), "min_excess": float(excess.min()),
        }
    out["by_arm_family"] = {}
    for arm in ("ARM_A", "ARM_B2"):
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        out["by_arm_family"][arm] = {}
        for family in families:
            subset = uncovered[(uncovered.predictor_arm == arm) & (uncovered.topology_family == family)]
            excess_f = (subset["nonconformity_score"] - subset["_quantile"]).to_numpy()
            if excess_f.size:
                out["by_arm_family"][arm][family] = {
                    "n_uncovered": int(excess_f.size), "median_excess": float(np.median(excess_f)),
                    "p90_excess": float(np.percentile(excess_f, 90)), "max_excess": float(excess_f.max()),
                }
            else:
                out["by_arm_family"][arm][family] = {"n_uncovered": 0}
    return out


# ---------------------------------------------------------------------------
# Section 16: deterministic, outcome-independent case-study export.
# ---------------------------------------------------------------------------


def _case_row(row: pd.Series, scheme: str) -> dict[str, Any]:
    block = row["schemes"].get(scheme, {})
    quantile_used = block.get("quantile_used")
    excess = (row["nonconformity_score"] - quantile_used) if quantile_used is not None else None
    return {
        "predictor_arm": row["predictor_arm"], "training_seed": int(row["training_seed"]),
        "topology_family": row["topology_family"], "incident_id": int(row["incident_id"]), "prefix_depth": int(row["prefix_depth"]),
        "true_source_node": row.get("true_source_node"), "predicted_top1_node": row.get("predicted_top1_node"),
        "top1_correct": bool(row["top1_correct"]), "true_source_rank": int(row["true_source_rank"]),
        "probability_true_source": row["probability_true_source"], "max_predicted_probability": row["max_predicted_probability"],
        "entropy": row["entropy"], "nonconformity_score": row["nonconformity_score"],
        "quantile_used": quantile_used, "excess": excess,
        "group_key": block.get("group_key"), "selection_source": block.get("selection_source"),
        "candidate_set_size": block.get("candidate_set_size"), "covered": block.get("covered"),
    }


def case_studies(df: pd.DataFrame, *, scheme: str = PRIMARY_SCHEME, top_n: int = m93.CASE_STUDY_TOP_N) -> dict[str, Any]:
    dev = df[df["split"] == "development"].copy()
    dev["_covered"] = _scheme_field(dev, scheme, "covered")
    dev["_quantile"] = _scheme_field(dev, scheme, "quantile_used")
    out: dict[str, Any] = {}

    uncovered_known = dev[(dev["_covered"] == False) & (dev["known_family"])]  # noqa: E712
    out["A_trained_family_uncovered_examples"] = [_case_row(r, scheme) for _, r in uncovered_known.iterrows()]

    golden = dev[dev["topology_family"] == "golden-reference"]
    b_rows, c_rows = [], []
    for seed in SEEDS:
        arm_a = golden[(golden.predictor_arm == "ARM_A") & (golden.training_seed == seed)].set_index(["incident_id", "prefix_depth"], drop=False)
        arm_b2 = golden[(golden.predictor_arm == "ARM_B2") & (golden.training_seed == seed)].set_index(["incident_id", "prefix_depth"], drop=False)
        a_cov = _scheme_field(arm_a, scheme, "covered")
        b_cov = _scheme_field(arm_b2, scheme, "covered")
        common_idx = arm_a.index.intersection(arm_b2.index)
        for idx in common_idx:
            if bool(a_cov.loc[idx]) and not bool(b_cov.loc[idx]):
                b_rows.append(_case_row(arm_b2.loc[idx], scheme))
            elif bool(b_cov.loc[idx]) and not bool(a_cov.loc[idx]):
                c_rows.append(_case_row(arm_b2.loc[idx], scheme))
    out["B_baseline_covers_interleaved_fails"] = b_rows[:top_n]
    out["C_interleaved_covers_baseline_fails"] = c_rows[:top_n]

    uncovered_known_sorted = uncovered_known.copy()
    uncovered_known_sorted["_excess"] = uncovered_known_sorted["nonconformity_score"] - uncovered_known_sorted["_quantile"]
    uncovered_known_sorted = uncovered_known_sorted[uncovered_known_sorted["_excess"].notna()].sort_values("_excess", ascending=False)
    out["D_largest_nonconformity_excess"] = [_case_row(r, scheme) for _, r in uncovered_known_sorted.head(top_n).iterrows()]

    loop_grid_uncovered = dev[(dev["topology_family"] == "loop-grid") & (dev["_covered"] == False)]  # noqa: E712
    out["E_loop_grid_miscoverage_cases"] = [_case_row(r, scheme) for _, r in loop_grid_uncovered.head(top_n * 3).iterrows()]

    return out


# ---------------------------------------------------------------------------
# Section 18: counterfactual diagnostic decomposition (read-only, no scheme
# selection). B/E computed directly from canonical-table scores; A/C/D reuse
# the already-stored scheme quantiles -- no new model inference.
# ---------------------------------------------------------------------------


def counterfactual_decomposition(df: pd.DataFrame) -> dict[str, Any]:
    cal = df[df["split"] == "calibration"]
    dev = df[df["split"] == "development"]
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                out[arm][str(seed)][family] = {}
                for bucket in ("EARLY", "MID", "MATURE"):
                    dev_subset = dev[(dev.predictor_arm == arm) & (dev.training_seed == seed) & (dev.topology_family == family) & (dev.depth_bucket == bucket)]
                    if dev_subset.empty:
                        continue
                    n_dev = len(dev_subset)

                    q_a = _scheme_field(dev_subset, "CURRENT_FAMILY_DEPTH", "quantile_used").dropna()
                    q_a = float(q_a.iloc[0]) if len(q_a) else None
                    cov_a = float(_scheme_field(dev_subset, "CURRENT_FAMILY_DEPTH", "covered").mean())

                    family_cal_scores = cal[(cal.predictor_arm == arm) & (cal.training_seed == seed) & (cal.topology_family == family)]["nonconformity_score"].tolist()
                    q_b = _quantile(family_cal_scores, m93.ALPHA) if family_cal_scores else None
                    cov_b = float((dev_subset["nonconformity_score"] <= q_b).mean()) if q_b is not None else None

                    q_c = _scheme_field(dev_subset, "POOLED_DEPTH_AWARE", "quantile_used").dropna()
                    q_c = float(q_c.iloc[0]) if len(q_c) else None
                    cov_c = float(_scheme_field(dev_subset, "POOLED_DEPTH_AWARE", "covered").mean())

                    q_d = _scheme_field(dev_subset, "BROAD_FALLBACK_CONTROL", "quantile_used").dropna()
                    q_d = float(q_d.iloc[0]) if len(q_d) else None
                    cov_d = float(_scheme_field(dev_subset, "BROAD_FALLBACK_CONTROL", "covered").mean())

                    dev_scores = dev_subset["nonconformity_score"].tolist()
                    q_e = _quantile(dev_scores, m93.ALPHA) if dev_scores else None
                    cov_e = float((dev_subset["nonconformity_score"] <= q_e).mean()) if q_e is not None else None

                    out[arm][str(seed)][family][bucket] = {
                        "n_development": n_dev,
                        "A_actual_current_family_depth": {"quantile": q_a, "diagnostic_coverage": cov_a},
                        "B_family_only_pooled_depth_within_family": {"quantile": q_b, "diagnostic_coverage": cov_b},
                        "C_pooled_depth_aware": {"quantile": q_c, "diagnostic_coverage": cov_c},
                        "D_broad_fallback_control": {"quantile": q_d, "diagnostic_coverage": cov_d},
                        "E_DEVELOPMENT_ORACLE_NOT_VALID_FOR_DEPLOYMENT": {"quantile": q_e, "diagnostic_coverage": cov_e},
                        "required_quantile_shift_oracle_minus_actual": (q_e - q_a) if (q_e is not None and q_a is not None) else None,
                    }
    return out


# ---------------------------------------------------------------------------
# Section 19: sample-size / support-tier estimation (diagnostic only, no new
# scenario generation -- extrapolates from the ALREADY-OBSERVED calibration
# score distribution's quantile-bootstrap behavior).
# ---------------------------------------------------------------------------


def sample_size_estimation(df: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(m93.LEARNING_CURVE_SEED)
    cal = df[(df["split"] == "calibration") & (df["predictor_arm"] == "ARM_B2")]
    out: dict[str, Any] = {}
    for seed in SEEDS:
        out[str(seed)] = {}
        for family in KNOWN_FAMILIES:
            out[str(seed)][family] = {}
            for bucket in ("EARLY", "MID", "MATURE"):
                subset = cal[(cal.training_seed == seed) & (cal.topology_family == family) & (cal.depth_bucket == bucket)]
                scores = subset["nonconformity_score"].to_numpy()
                if scores.size == 0:
                    continue
                tiers = {}
                for n_tier in m93.SUPPORT_TIERS:
                    quantiles = np.empty(200)
                    for i in range(200):
                        sampled = rng.choice(scores, size=n_tier, replace=True)
                        quantiles[i] = _quantile(sampled.tolist(), m93.ALPHA)
                    tiers[str(n_tier)] = {
                        "finite_sample_resolution": m93.finite_sample_resolution(n_tier),
                        "quantile_std_at_this_tier": float(quantiles.std()),
                        "quantile_relative_span_5_95": float((np.percentile(quantiles, 95) - np.percentile(quantiles, 5)) / max(np.median(quantiles), 1e-9)),
                    }
                out[str(seed)][family][bucket] = {"n_observed": int(scores.size), "tiers": tiers}
    return out


# ---------------------------------------------------------------------------
# Section 20: exchangeability / corpus audit -- descriptive covariate
# comparison between calibration and development splits, using only
# already-defined per-row covariates (no new scenario metadata invented).
# ---------------------------------------------------------------------------


def exchangeability_audit(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        families = ("golden-reference",) if arm == "ARM_A" else KNOWN_FAMILIES
        for seed in SEEDS:
            out[arm][str(seed)] = {}
            for family in families:
                cal_subset = df[(df.predictor_arm == arm) & (df.training_seed == seed) & (df.topology_family == family) & (df.split == "calibration")]
                dev_subset = df[(df.predictor_arm == arm) & (df.training_seed == seed) & (df.topology_family == family) & (df.split == "development")]
                result: dict[str, Any] = {
                    "n_calibration": int(len(cal_subset)), "n_development": int(len(dev_subset)),
                    "calibration_condition_distribution": cal_subset["condition"].value_counts(normalize=True).to_dict(),
                    "development_condition_distribution": dev_subset["condition"].value_counts(normalize=True).to_dict(),
                }
                if not dev_subset.empty and dev_subset["missing_rate"].notna().any():
                    result["development_mean_missing_rate"] = float(dev_subset["missing_rate"].mean())
                    result["development_mean_healthy_sensor_fraction"] = float(dev_subset["healthy_sensor_fraction"].mean())
                if not dev_subset.empty and "true_source_node" in dev_subset:
                    result["development_source_node_distribution"] = dev_subset["true_source_node"].value_counts(normalize=True).to_dict()
                out[arm][str(seed)][family] = result
    return out
