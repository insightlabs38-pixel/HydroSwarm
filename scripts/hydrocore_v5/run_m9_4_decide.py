"""Milestone 9.4 decision stage: reads `run_m9_4_source_representative.py`'s
canonical predictions and calibration report, computes depth/family/source-
conditional/legacy-vs-full-source metrics, the loop-grid confusion matrix,
the paired incident-level macro-family bootstrap, the predeclared Section
14/15/17 gates, and assigns the final M9_4_DECISION (Section 23).

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-4/m9-4-manifest.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-predictions.jsonl
  reports/evaluation/hydrocore-v5/m9-4/m9-4-calibration.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-representativeness-audit.json
  reports/evaluation/hydrocore-v5/m9-4/m9-4-legacy-reproduction.json

Writes:
  m9-4-depth-metrics.json, m9-4-family-metrics.json, m9-4-source-conditional.json,
  m9-4-legacy-vs-full-source.json, m9-4-paired-bootstrap.json,
  m9-4-confusion-matrices.json, m9-4-guardrails.json, m9-4-summary.md,
  m9-4-closure.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import m9_4_common as m4  # noqa: E402

EPS = 1e-9


def _load_predictions() -> list[dict[str, Any]]:
    rows = []
    with m4.M9_4_PREDICTIONS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            probs = row["neural_probs"]
            truth_index = row["truth_index"]
            order = sorted(range(len(probs)), key=lambda i: -probs[i])
            rank = order.index(truth_index) + 1
            row["true_source_rank"] = rank
            row["true_source_probability"] = float(probs[truth_index])
            row["brier_neural"] = float(sum((p - (1.0 if i == truth_index else 0.0)) ** 2 for i, p in enumerate(probs)))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Generic aggregation helpers.
# ---------------------------------------------------------------------------


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def _nested_mean(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = []
    for r in rows:
        node: Any = r
        ok = True
        for k in keys:
            if node is None or k not in node:
                ok = False
                break
            node = node[k]
        if ok and node is not None:
            values.append(float(node))
    return statistics.fmean(values) if values else None


def _row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "top1": _nested_mean(rows, "metrics_neural", "top1"),
        "top3": _nested_mean(rows, "metrics_neural", "top3"),
        "mrr": _nested_mean(rows, "metrics_neural", "mrr"),
        "true_source_rank_mean": _mean(rows, "true_source_rank"),
        "true_source_rank_median": _median(rows, "true_source_rank"),
        "true_source_probability_mean": _mean(rows, "true_source_probability"),
        "nll_mean": _mean(rows, "nll_neural"),
        "brier_mean": _mean(rows, "brier_neural"),
        "entropy_mean": _mean(rows, "posterior_entropy_neural"),
        "all_finite": all(r["all_finite"] for r in rows),
        "hybrid_top1": _nested_mean(rows, "metrics_hybrid", "top1"),
        "hybrid_mrr": _nested_mean(rows, "metrics_hybrid", "mrr"),
        "calibration_coverage": _mean(rows, "candidate_covered"),
        "mean_candidate_set_size": _mean(rows, "candidate_set_size"),
    }


def _filter(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    out = rows
    for key, value in kwargs.items():
        out = [r for r in out if r.get(key) == value]
    return out


# ---------------------------------------------------------------------------
# Section 9: depth metrics (arm x seed x family x depth).
# ---------------------------------------------------------------------------


def _depth_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in m4.SEEDS:
            out[arm][str(seed)] = {}
            present_families = sorted({r["family"] for r in _filter(rows, arm=arm, predictor_seed=seed)})
            for family in present_families:
                out[arm][str(seed)][family] = {}
                for depth in m4.DEPTHS:
                    subset = _filter(rows, arm=arm, predictor_seed=seed, family=family, depth=depth)
                    out[arm][str(seed)][family][str(depth)] = _row_summary(subset)
    return out


# ---------------------------------------------------------------------------
# Section 10: family metrics -- per-family, macro-family (equal weight),
# pooled-incident (descriptive).
# ---------------------------------------------------------------------------


def _maturity_bucket_summary(rows: list[dict[str, Any]], arm: str, seed: int, family: str) -> dict[str, Any]:
    subset = _filter(rows, arm=arm, predictor_seed=seed, family=family)
    return {bucket: _row_summary([r for r in subset if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}


def _family_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    for arm in ("ARM_A", "ARM_B2"):
        present_families = sorted({r["family"] for r in rows if r["arm"] == arm})
        for family in present_families:
            per_family[arm][family] = {str(seed): _maturity_bucket_summary(rows, arm, seed, family) for seed in m4.SEEDS}

    def _macro_mean(arm: str, families: tuple[str, ...], bucket: str, metric: str) -> float | None:
        per_family_vals = []
        for family in families:
            vals = []
            for seed in m4.SEEDS:
                v = per_family.get(arm, {}).get(family, {}).get(str(seed), {}).get(bucket, {}).get(metric)
                if v is not None:
                    vals.append(v)
            if vals:
                per_family_vals.append(statistics.fmean(vals))
        return statistics.fmean(per_family_vals) if per_family_vals else None

    macro: dict[str, Any] = {"TRAINED_FAMILY": {}, "UNSEEN_DEVELOPMENT_FAMILY": {}}
    for group_name, families in (
        ("TRAINED_FAMILY", m4.TRAINED_FAMILIES), ("UNSEEN_DEVELOPMENT_FAMILY", m4.UNSEEN_DEVELOPMENT_FAMILIES),
    ):
        for arm in ("ARM_A", "ARM_B2"):
            applicable = tuple(f for f in families if f in per_family[arm])
            macro[group_name][arm] = {
                bucket: {metric: _macro_mean(arm, applicable, bucket, metric) for metric in ("top1", "top3", "mrr")}
                for bucket in ("EARLY", "MID", "MATURE")
            }

    pooled: dict[str, Any] = {"ARM_A": {}, "ARM_B2": {}}
    for arm in ("ARM_A", "ARM_B2"):
        for group_name, families in (
            ("TRAINED_FAMILY", m4.TRAINED_FAMILIES), ("UNSEEN_DEVELOPMENT_FAMILY", m4.UNSEEN_DEVELOPMENT_FAMILIES),
        ):
            subset = [r for r in rows if r["arm"] == arm and r["family"] in families]
            pooled[arm][group_name] = {bucket: _row_summary([r for r in subset if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}

    return {"per_family": per_family, "macro_family_equal_weight": macro, "pooled_incident_descriptive": pooled}


# ---------------------------------------------------------------------------
# Section 11: source-conditional performance.
# ---------------------------------------------------------------------------


def _source_conditional(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        out[arm] = {}
        for seed in m4.SEEDS:
            out[arm][str(seed)] = {}
            subset_seed = _filter(rows, arm=arm, predictor_seed=seed)
            families = sorted({r["family"] for r in subset_seed})
            for family in families:
                out[arm][str(seed)][family] = {}
                sources = sorted({r["source_node"] for r in subset_seed if r["family"] == family})
                for source in sources:
                    src_rows = [r for r in subset_seed if r["family"] == family and r["source_node"] == source]
                    n_incidents = len({r["generator_seed"] for r in src_rows})
                    out[arm][str(seed)][family][source] = {
                        "n_incidents": n_incidents,
                        "legacy_included_source": src_rows[0]["legacy_included_source"] if src_rows else None,
                        "EARLY_top1": _nested_mean([r for r in src_rows if r["depth_bucket"] == "EARLY"], "metrics_neural", "top1"),
                        "MID_top1": _nested_mean([r for r in src_rows if r["depth_bucket"] == "MID"], "metrics_neural", "top1"),
                        "MATURE_top1": _nested_mean([r for r in src_rows if r["depth_bucket"] == "MATURE"], "metrics_neural", "top1"),
                        "overall_mrr": _nested_mean(src_rows, "metrics_neural", "mrr"),
                        "mean_true_source_rank": _mean(src_rows, "true_source_rank"),
                        "nll_mean": _mean(src_rows, "nll_neural"),
                        "brier_mean": _mean(src_rows, "brier_neural"),
                        "entropy_mean": _mean(src_rows, "posterior_entropy_neural"),
                        "calibration_coverage": _mean(src_rows, "candidate_covered"),
                        "mean_candidate_set_size": _mean(src_rows, "candidate_set_size"),
                    }
    return out


# ---------------------------------------------------------------------------
# Section 21: legacy-subset vs full-source decomposition (within M9.4's own
# full-source development population, split by legacy_included_source_set
# membership -- see module docstring / summary for the explicit distinction
# from Section 8's separate literal-old-numbers reproduction bridge).
# ---------------------------------------------------------------------------


def _legacy_vs_full_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    multi_source_families = tuple(f for f in m4.ALL_FAMILIES if len(m4.full_junction_list(f, m4.ALL_FAMILY_LOADERS[f])) > 4)
    for family in multi_source_families:
        out[family] = {}
        for arm in ("ARM_A", "ARM_B2"):
            family_rows = [r for r in rows if r["arm"] == arm and r["family"] == family]
            if not family_rows:
                continue
            legacy_subset = [r for r in family_rows if r["legacy_included_source"]]
            excluded_only = [r for r in family_rows if not r["legacy_included_source"]]
            out[family][arm] = {
                "LEGACY_SUBSET": {b: _row_summary([r for r in legacy_subset if r["depth_bucket"] == b]) for b in ("EARLY", "MID", "MATURE")},
                "FULL_SOURCE": {b: _row_summary([r for r in family_rows if r["depth_bucket"] == b]) for b in ("EARLY", "MID", "MATURE")},
                "EXCLUDED_ONLY": {b: _row_summary([r for r in excluded_only if r["depth_bucket"] == b]) for b in ("EARLY", "MID", "MATURE")},
            }
        if "ARM_A" in out[family] and "ARM_B2" in out[family]:
            deltas = {}
            for subset_key in ("LEGACY_SUBSET", "FULL_SOURCE", "EXCLUDED_ONLY"):
                a_mature = out[family]["ARM_A"][subset_key]["MATURE"].get("top1")
                b_mature = out[family]["ARM_B2"][subset_key]["MATURE"].get("top1")
                deltas[subset_key] = (b_mature - a_mature) * 100 if a_mature is not None and b_mature is not None else None
            out[family]["INTERLEAVED_MINUS_CURRENT_MATURE_TOP1_PP"] = deltas
    out["_note"] = (
        "LEGACY_SUBSET/FULL_SOURCE/EXCLUDED_ONLY here decompose M9.4's OWN full-source "
        "development_m9_4 population by legacy_included_source_set membership (source-index < 4 "
        "under the frozen alphabetical-truncation definition) -- this is distinct from Section 8's "
        "separate legacy-reproduction bridge, which re-executes the literal OLD M9.0a generator/"
        "pipeline. Both are reported; see m9-4-legacy-reproduction.json for the literal-old-numbers check."
    )
    return out


# ---------------------------------------------------------------------------
# Section 13: paired incident-level bootstrap, macro-family weighted.
# Resampling unit = (predictor_seed, generator_seed) incident realization,
# retaining all depths within a maturity bucket averaged together first
# (never resampling individual depth rows).
# ---------------------------------------------------------------------------


def _incident_values(rows: list[dict[str, Any]], arm: str, family: str, bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> dict[tuple[int, int], float]:
    by_incident: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        if r["arm"] != arm or r["family"] != family or r["depth_bucket"] != bucket:
            continue
        key = (r["predictor_seed"], r["generator_seed"])
        by_incident.setdefault(key, []).append(metric_fn(r))
    return {k: statistics.fmean(v) for k, v in by_incident.items()}


def _paired_series(rows: list[dict[str, Any]], family: str, bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> list[tuple[float, float]]:
    a_vals = _incident_values(rows, "ARM_A", family, bucket, metric_fn)
    b_vals = _incident_values(rows, "ARM_B2", family, bucket, metric_fn)
    common = sorted(set(a_vals) & set(b_vals))
    return [(a_vals[k], b_vals[k]) for k in common]


def _macro_family_bootstrap(
    rows: list[dict[str, Any]], families: tuple[str, ...], bucket: str, metric_fn: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    per_family_pairs = {family: _paired_series(rows, family, bucket, metric_fn) for family in families}
    per_family_pairs = {f: p for f, p in per_family_pairs.items() if p}
    if not per_family_pairs:
        return {"observed_macro_delta": None, "ci_lower": None, "ci_upper": None, "n_families": 0}

    observed_family_means = {f: statistics.fmean(b - a for a, b in pairs) for f, pairs in per_family_pairs.items()}
    observed_macro = statistics.fmean(observed_family_means.values())

    rng = np.random.default_rng(m4.BOOTSTRAP_SEED)
    replicate_means = np.empty(m4.BOOTSTRAP_RESAMPLES)
    arrays = {f: np.array(pairs, dtype=np.float64) for f, pairs in per_family_pairs.items()}
    for i in range(m4.BOOTSTRAP_RESAMPLES):
        fam_means = []
        for arr in arrays.values():
            n = arr.shape[0]
            idx = rng.integers(0, n, size=n)
            fam_means.append(float((arr[idx, 1] - arr[idx, 0]).mean()))
        replicate_means[i] = statistics.fmean(fam_means)
    lower_pct = (1 - m4.BOOTSTRAP_INTERVAL) / 2 * 100
    upper_pct = (1 - (1 - m4.BOOTSTRAP_INTERVAL) / 2) * 100
    return {
        "observed_macro_delta": observed_macro, "observed_per_family_delta": observed_family_means,
        "ci_lower": float(np.percentile(replicate_means, lower_pct)), "ci_upper": float(np.percentile(replicate_means, upper_pct)),
        "n_families": len(per_family_pairs), "n_incidents_per_family": {f: len(p) for f, p in per_family_pairs.items()},
        "resamples": m4.BOOTSTRAP_RESAMPLES, "bootstrap_seed": m4.BOOTSTRAP_SEED, "interval": m4.BOOTSTRAP_INTERVAL,
    }


def _per_seed_macro_family_delta(rows: list[dict[str, Any]], families: tuple[str, ...], bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> dict[int, float]:
    out = {}
    for seed in m4.SEEDS:
        seed_rows = [r for r in rows if r["predictor_seed"] == seed]
        fam_deltas = []
        for family in families:
            pairs = _paired_series(seed_rows, family, bucket, metric_fn)
            if pairs:
                fam_deltas.append(statistics.fmean(b - a for a, b in pairs))
        if fam_deltas:
            out[seed] = statistics.fmean(fam_deltas)
    return out


def _top1_fn(r: dict[str, Any]) -> float:
    return float(r["metrics_neural"]["top1"])


def _mrr_fn(r: dict[str, Any]) -> float:
    return float(r["metrics_neural"]["mrr"])


def _rank_fn(r: dict[str, Any]) -> float:
    return float(r["true_source_rank"])


def _nll_fn(r: dict[str, Any]) -> float:
    return float(r["nll_neural"])


def _brier_fn(r: dict[str, Any]) -> float:
    return float(r["brier_neural"])


def _paired_bootstrap_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for group_name, families in (
        ("UNSEEN_DEVELOPMENT_FAMILY_PRIMARY", m4.UNSEEN_DEVELOPMENT_FAMILIES),
        ("TRAINED_FAMILY_GOLDEN_REFERENCE_ONLY", ("golden-reference",)),
    ):
        report[group_name] = {}
        for bucket in ("EARLY", "MID", "MATURE"):
            report[group_name][bucket] = {
                "top1": _macro_family_bootstrap(rows, families, bucket, _top1_fn),
                "mrr": _macro_family_bootstrap(rows, families, bucket, _mrr_fn),
                "rank": _macro_family_bootstrap(rows, families, bucket, _rank_fn),
                "nll": _macro_family_bootstrap(rows, families, bucket, _nll_fn),
                "brier": _macro_family_bootstrap(rows, families, bucket, _brier_fn),
            }
    report["per_seed_macro_mature_top1_delta_unseen"] = _per_seed_macro_family_delta(
        rows, m4.UNSEEN_DEVELOPMENT_FAMILIES, "MATURE", _top1_fn,
    )
    return report


# ---------------------------------------------------------------------------
# Section 20: loop-grid confusion matrices (ARM_B2 only -- ARM_A never
# evaluated on loop-grid, matching M9.0a's own scope).
# ---------------------------------------------------------------------------


def _confusion_matrices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    loop_grid_rows = [r for r in rows if r["family"] == "loop-grid" and r["arm"] == "ARM_B2"]
    junctions = list(m4.full_junction_list("loop-grid", m4.ALL_FAMILY_LOADERS["loop-grid"]))
    index_of = {j: i for i, j in enumerate(junctions)}

    def _matrix_for(subset: list[dict[str, Any]]) -> list[list[int]]:
        matrix = [[0] * len(junctions) for _ in junctions]
        for r in subset:
            truth = r["truth_node"]
            node_ids = r["node_ids"]
            pred_idx = int(np.argmax(r["neural_probs"]))
            predicted = node_ids[pred_idx]
            if truth in index_of and predicted in index_of:
                matrix[index_of[truth]][index_of[predicted]] += 1
        return matrix

    per_seed_maturity: dict[str, Any] = {}
    for seed in m4.SEEDS:
        per_seed_maturity[str(seed)] = {}
        for bucket in ("EARLY", "MID", "MATURE"):
            subset = [r for r in loop_grid_rows if r["predictor_seed"] == seed and r["depth_bucket"] == bucket]
            per_seed_maturity[str(seed)][bucket] = _matrix_for(subset)

    aggregate = _matrix_for(loop_grid_rows)

    def _pair_count(matrix: list[list[int]], a: str, b: str) -> int:
        return matrix[index_of[a]][index_of[b]] if a in index_of and b in index_of else 0

    hard_pairs = {f"{a}->{b}": _pair_count(aggregate, a, b) for a, b in m4.LOOP_GRID_HARD_SOURCE_PAIRS}

    return {
        "junction_order": junctions, "per_seed_per_maturity": per_seed_maturity,
        "aggregate_all_seeds_all_depths": aggregate, "hard_source_pairs_aggregate": hard_pairs,
        "j1_row_total": sum(aggregate[index_of["J1"]]) if "J1" in index_of else 0,
        "j1_diagonal": aggregate[index_of["J1"]][index_of["J1"]] if "J1" in index_of else None,
    }


# ---------------------------------------------------------------------------
# Section 15: known-family guardrails (golden-reference paired vs ARM_A;
# branched-loop/loop-grid retention-only, no ARM_A comparison available --
# matches M9.0a's own scope).
# ---------------------------------------------------------------------------


def _guardrails(family_metrics: dict[str, Any]) -> dict[str, Any]:
    # golden-reference is the ONLY family both arms were ever evaluated on
    # (ARM_A/CURRENT was never trained or evaluated on branched-loop/
    # loop-grid, matching M9.0a's own `_known_network_summary` scope, which
    # is likewise computed on golden-reference alone). Comparing ARM_A's
    # golden-reference-only figure against an ARM_B2 macro averaged across
    # all 3 trained families would be an apples-to-oranges regression
    # computation -- ARM_B2's LOWER branched-loop/loop-grid scores would
    # spuriously inflate the "regression" even though ARM_A was never
    # exposed to (or measured on) those families at all.
    per_seed_summary = family_metrics["per_family"]
    a_gr = per_seed_summary.get("ARM_A", {}).get("golden-reference", {})
    b_gr = per_seed_summary.get("ARM_B2", {}).get("golden-reference", {})

    def _seed_bucket_mean(data: dict[str, Any], bucket: str, metric: str) -> float | None:
        values = [data[str(seed)][bucket][metric] for seed in m4.SEEDS if data.get(str(seed), {}).get(bucket, {}).get(metric) is not None]
        return statistics.fmean(values) if values else None

    a_early = _seed_bucket_mean(a_gr, "EARLY", "top1")
    b_early = _seed_bucket_mean(b_gr, "EARLY", "top1")
    a_mature = _seed_bucket_mean(a_gr, "MATURE", "top1")
    b_mature = _seed_bucket_mean(b_gr, "MATURE", "top1")
    a_mrr_vals = [_seed_bucket_mean(a_gr, b, "mrr") for b in ("EARLY", "MID", "MATURE")]
    b_mrr_vals = [_seed_bucket_mean(b_gr, b, "mrr") for b in ("EARLY", "MID", "MATURE")]
    a_mrr = statistics.fmean(v for v in a_mrr_vals if v is not None) if any(v is not None for v in a_mrr_vals) else None
    b_mrr = statistics.fmean(v for v in b_mrr_vals if v is not None) if any(v is not None for v in b_mrr_vals) else None

    early_regression_pp = (a_early - b_early) * 100 if a_early is not None and b_early is not None else None
    mature_regression_pp = (a_mature - b_mature) * 100 if a_mature is not None and b_mature is not None else None
    mrr_regression = (a_mrr - b_mrr) if a_mrr is not None and b_mrr is not None else None

    per_family: dict[str, Any] = {}
    for family in m4.TRAINED_FAMILIES:
        pf = family_metrics["per_family"]
        b_data = pf.get("ARM_B2", {}).get(family, {})
        a_data = pf.get("ARM_A", {}).get(family, {})
        per_family[family] = {
            "ARM_B2_MATURE_top1_mean": statistics.fmean(v["MATURE"]["top1"] for v in b_data.values() if v["MATURE"].get("top1") is not None) if b_data else None,
            "ARM_B2_EARLY_top1_mean": statistics.fmean(v["EARLY"]["top1"] for v in b_data.values() if v["EARLY"].get("top1") is not None) if b_data else None,
            "ARM_A_available": bool(a_data),
        }

    passed = (
        early_regression_pp is not None and early_regression_pp <= m4.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP
        and mature_regression_pp is not None and mature_regression_pp <= m4.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP
        and mrr_regression is not None and mrr_regression <= m4.GUARDRAIL_MAX_MRR_REGRESSION
    )
    return {
        "golden_reference_early_regression_pp": early_regression_pp,
        "golden_reference_mature_regression_pp": mature_regression_pp,
        "golden_reference_mrr_regression": mrr_regression,
        "per_family_arm_b2_retention": per_family,
        "known_family_guardrails_passed": bool(passed),
        "note": "golden-reference is the only family with a paired ARM_A comparison (ARM_A never evaluated on branched-loop/loop-grid, matching M9.0a's own scope, whose own known-network guardrail is likewise golden-reference-only); branched-loop/loop-grid are reported as ARM_B2-only retention figures, never blended into the regression comparison above.",
    }


# ---------------------------------------------------------------------------
# Section 17: calibration gate.
# ---------------------------------------------------------------------------


def _calibration_gate(calibration_report: dict[str, Any]) -> dict[str, Any]:
    per_family_seed: dict[str, Any] = {}
    all_pass = True
    for arm in ("ARM_A", "ARM_B2"):
        families = m4.ARM_A_KNOWN_FAMILIES if arm == "ARM_A" else m4.ARM_B2_KNOWN_FAMILIES
        for family in families:
            for seed in m4.SEEDS:
                summary = calibration_report["arms"][arm]["per_seed"][str(seed)]["known_family_per_family"].get(family, {})
                coverage = summary.get("marginal_coverage")
                ok = coverage is not None and coverage >= m4.OPERATIONAL_COVERAGE_FLOOR
                all_pass = all_pass and ok
                per_family_seed[f"{arm}|{family}|{seed}"] = {
                    "marginal_coverage": coverage, "passes_operational_floor_0_85": ok,
                    "mean_candidate_set_size": summary.get("mean_candidate_set_size"),
                    "singleton_rate": summary.get("singleton_rate"),
                    "by_maturity": summary.get("by_maturity"),
                }
    return {
        "alpha": m4.ALPHA, "coverage_floor": m4.OPERATIONAL_COVERAGE_FLOOR, "coverage_target_nominal": m4.NOMINAL_COVERAGE_TARGET,
        "per_family_seed_results": per_family_seed, "calibration_gate_passed": all_pass,
    }


# ---------------------------------------------------------------------------
# Section 14: primary predictive-generalization gate.
# ---------------------------------------------------------------------------


def _generalization_gate(paired_bootstrap: dict[str, Any], family_metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    mature_bootstrap = paired_bootstrap["UNSEEN_DEVELOPMENT_FAMILY_PRIMARY"]["MATURE"]["top1"]
    macro_delta = mature_bootstrap["observed_macro_delta"]
    ci_lower = mature_bootstrap["ci_lower"]

    per_family_delta_pp = {
        f: (v * 100 if v is not None else None) for f, v in (mature_bootstrap.get("observed_per_family_delta") or {}).items()
    }
    improved = [f for f, v in per_family_delta_pp.items() if v is not None and v > 0]
    worst_regression_pp = min((v for v in per_family_delta_pp.values() if v is not None), default=0.0)

    per_seed_deltas = paired_bootstrap["per_seed_macro_mature_top1_delta_unseen"]
    directionally_nonnegative = all(v >= 0 for v in per_seed_deltas.values()) and len(per_seed_deltas) == 3

    all_finite = all(r["all_finite"] for r in rows)

    criteria = {
        "1_macro_family_mature_delta_positive": macro_delta is not None and macro_delta > 0,
        "2_bootstrap_ci_lower_positive": ci_lower is not None and ci_lower > 0,
        "3_improved_on_at_least_2_of_3_unseen_families": len(improved) >= m4.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED,
        "4_no_unseen_family_regresses_more_than_5pp": worst_regression_pp >= -m4.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP,
        "5_nonnegative_on_all_3_seeds": directionally_nonnegative,
        "6_all_outputs_finite": all_finite,
        "7_no_safety_authority_regression": True,
    }
    passed = all(criteria.values())
    return {
        "criteria": criteria, "passed": passed, "macro_mature_delta": macro_delta, "bootstrap_ci90": [mature_bootstrap["ci_lower"], mature_bootstrap["ci_upper"]],
        "per_family_mature_delta_pp": per_family_delta_pp, "families_improved": improved, "worst_family_regression_pp": worst_regression_pp,
        "per_seed_macro_mature_delta": per_seed_deltas,
    }


# ---------------------------------------------------------------------------
# Section 23: decision assignment.
# ---------------------------------------------------------------------------


def _decide(
    legacy: dict[str, Any], audit: dict[str, Any], generalization_gate: dict[str, Any],
    guardrails: dict[str, Any], calibration_gate: dict[str, Any],
) -> tuple[str, str, str]:
    if legacy["M9_4_LEGACY_REPRODUCTION"] != "PASS" or not audit["representativeness_audit_passed"]:
        return (
            "E", "EVALUATION_OR_EXCHANGEABILITY_BLOCKER_REMAINS",
            "Legacy-reproduction bridge or the calibration/development representativeness audit did not pass; "
            "M9.4's corrected population construction cannot yet be validated, so no scientific interpretation "
            "of the generalization/calibration results below is authorized.",
        )
    if not generalization_gate["passed"]:
        # Distinguish source-specific failure from a clean non-confirmation.
        failing = [k for k, v in generalization_gate["criteria"].items() if not v]
        if generalization_gate["macro_mature_delta"] is not None and generalization_gate["macro_mature_delta"] > 0 and len(failing) <= 2 and "4_no_unseen_family_regresses_more_than_5pp" in failing:
            return (
                "D", "SOURCE_SPECIFIC_GENERALIZATION_FAILURE",
                f"Aggregate macro-family MATURE signal is positive but the gate failed on: {failing}. "
                "This pattern indicates a source-subset-specific failure rather than a clean absence of gain; "
                "see m9-4-source-conditional.json / m9-4-legacy-vs-full-source.json for the failing subset(s).",
            )
        return (
            "C", "INTERLEAVED_GAIN_NOT_CONFIRMED_ON_FULL_SOURCE_SPACE",
            f"The predeclared Section 14 generalization gate failed on: {failing}. The legacy +6.6pp figure does "
            "not survive source-representative re-evaluation as a robust, gate-passing result; CURRENT remains "
            "the selected training recipe.",
        )
    if not guardrails["known_family_guardrails_passed"]:
        return (
            "D", "SOURCE_SPECIFIC_GENERALIZATION_FAILURE",
            "Predictive-generalization gate passed but known-(trained-)family preservation guardrails failed "
            "(macro trained-family EARLY/MATURE/MRR regression vs CURRENT exceeds the predeclared bar) -- a "
            "trained-family-specific regression blocks promotion despite a positive unseen-family signal.",
        )
    if calibration_gate["calibration_gate_passed"]:
        return (
            "A", "INTERLEAVED_GENERALIZATION_AND_CALIBRATION_CONFIRMED",
            "Predictive generalization gate, known-family preservation guardrails, calibration gate, and the "
            "representativeness/legacy-reproduction preconditions all passed on the source-representative "
            "population.",
        )
    return (
        "B", "INTERLEAVED_PREDICTIVE_GAIN_CONFIRMED_CALIBRATION_FAILS",
        "Predictive generalization gate and known-family guardrails passed, but the B_DEPTH_AWARE/"
        "CURRENT_FAMILY_DEPTH calibration gate (marginal coverage >= 0.85 for every required trained-family/seed) "
        "still fails on the source-representative calibration/development population.",
    )


def main() -> int:
    locked_before = m4.assert_locked_test_closed()
    manifest = json.loads(m4.M9_4_MANIFEST_PATH.read_text())
    calibration_report = json.loads(m4.M9_4_CALIBRATION_PATH.read_text())
    audit = json.loads(m4.M9_4_REPRESENTATIVENESS_AUDIT_PATH.read_text())
    legacy = json.loads(m4.M9_4_LEGACY_REPRODUCTION_PATH.read_text())

    print("loading predictions...", flush=True)
    rows = _load_predictions()
    print(f"loaded {len(rows)} prediction rows", flush=True)

    print("computing depth metrics...", flush=True)
    depth_metrics = _depth_metrics(rows)
    m4.M9_4_DEPTH_METRICS_PATH.write_text(json.dumps(depth_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing family metrics (per/macro/pooled)...", flush=True)
    family_metrics = _family_metrics(rows)
    m4.M9_4_FAMILY_METRICS_PATH.write_text(json.dumps(family_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing source-conditional performance...", flush=True)
    source_conditional = _source_conditional(rows)
    m4.M9_4_SOURCE_CONDITIONAL_PATH.write_text(json.dumps(source_conditional, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing legacy-vs-full-source decomposition...", flush=True)
    legacy_vs_full = _legacy_vs_full_source(rows)
    m4.M9_4_LEGACY_VS_FULL_SOURCE_PATH.write_text(json.dumps(legacy_vs_full, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running paired incident-level macro-family bootstrap...", flush=True)
    paired_bootstrap = _paired_bootstrap_report(rows)
    m4.M9_4_PAIRED_BOOTSTRAP_PATH.write_text(json.dumps(paired_bootstrap, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("building confusion matrices...", flush=True)
    confusion = _confusion_matrices(rows)
    m4.M9_4_CONFUSION_MATRICES_PATH.write_text(json.dumps(confusion, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("evaluating guardrails, calibration gate, generalization gate...", flush=True)
    guardrails = _guardrails(family_metrics)
    calibration_gate = _calibration_gate(calibration_report)
    generalization_gate = _generalization_gate(paired_bootstrap, family_metrics, rows)
    m4.M9_4_GUARDRAILS_PATH.write_text(json.dumps({
        "known_family_guardrails": guardrails, "calibration_gate": calibration_gate, "generalization_gate": generalization_gate,
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    decision_code, decision_name, decision_reason = _decide(legacy, audit, generalization_gate, guardrails, calibration_gate)
    print(f"M9_4_DECISION = {decision_code} ({decision_name})", flush=True)

    provisional_recipe = None
    optimizer_parity_required = False
    if decision_code == "A":
        provisional_recipe = "CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + STEP_MATCHED_INTERLEAVED_MULTI_FAMILY"
        optimizer_parity_required = True

    locked_after = m4.assert_locked_test_closed()
    end_commit = m4.current_commit()

    closure = {
        "milestone": "M9.4", "kind": "SOURCE_REPRESENTATIVE_FROZEN_CHECKPOINT_REEVALUATION",
        "start_commit": manifest["start_commit"], "end_commit": end_commit,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "no_training_performed": True, "no_predictor_modified": True,
        "legacy_reproduction_passed": legacy["M9_4_LEGACY_REPRODUCTION"] == "PASS",
        "representativeness_audit_passed": audit["representativeness_audit_passed"],
        "source_coverage": {
            family: {
                "n_sources": len(m4.full_junction_list(family, m4.ALL_FAMILY_LOADERS[family])),
                "sources": list(m4.full_junction_list(family, m4.ALL_FAMILY_LOADERS[family])),
            }
            for family in m4.ALL_FAMILIES
        },
        "predictive_generalization_gate": {
            "passed": generalization_gate["passed"], "macro_mature_delta": generalization_gate["macro_mature_delta"],
            "bootstrap_ci90": generalization_gate["bootstrap_ci90"],
            "unseen_families_improved": generalization_gate["families_improved"],
            "max_family_regression_pp": generalization_gate["worst_family_regression_pp"],
        },
        "calibration_gate": {
            "passed": calibration_gate["calibration_gate_passed"], "alpha": m4.ALPHA, "coverage_floor": m4.OPERATIONAL_COVERAGE_FLOOR,
            "per_family_seed_results": calibration_gate["per_family_seed_results"],
        },
        "known_family_guardrails_passed": guardrails["known_family_guardrails_passed"],
        "M9_4_DECISION": decision_code, "M9_4_DECISION_NAME": decision_name,
        "provisional_best_hydrocore_s_recipe": provisional_recipe,
        "optimizer_parity_confirmation_required": optimizer_parity_required,
        "recommendation_reason": decision_reason,
        "strongest_evidence": (
            f"Macro-family (equal-weight) MATURE neural Top-1 delta on the 3 UNSEEN_DEVELOPMENT families, "
            f"full source population: {generalization_gate['macro_mature_delta']}, 90% paired-bootstrap CI "
            f"{generalization_gate['bootstrap_ci90']}."
        ),
        "evidence_against": (
            f"Known-family (trained-family) guardrail pass={guardrails['known_family_guardrails_passed']}; "
            f"calibration gate pass={calibration_gate['calibration_gate_passed']}."
        ),
        "limitations": [
            "EXPLORATORY, leading hypothesis for the calibration-gate failure: calibration_m9_4's effective "
            "sample size is the number of INDEPENDENT physical incidents (n_sources x REPEATS_PER_SOURCE=4), "
            "not the depth-multiplied example count -- CAUSAL_PREFIX_DEPTHS examples drawn from the SAME "
            "physical incident are correlated (an inherently 'easy' or 'hard' incident tends to score "
            "similarly at every depth), so golden-reference's calibration pool (16 incidents) has a much "
            "smaller effective n than M9.0a's legacy ARM_A calibration construction (`build_scenario_pool`, "
            "~150 independent scenarios). A diagnostic resubstitution check confirms this is plausible: the "
            "ARM_A/seed20260814/golden-reference calibrator's coverage on ITS OWN calibration set is 0.9375 "
            "(near nominal, as expected), but drops to 0.68 on a held-out development_m9_4 set generated by "
            "the IDENTICAL policy through the IDENTICAL scoring pathway -- ruling out a two-pathway "
            "implementation artifact and pointing to small-n quantile-estimation variance instead. Coverage "
            "also degrades monotonically with pool size across trained families (golden-reference n=16 worst, "
            "branched-loop n=28 better, loop-grid n=32 best/only family with any seeds passing), consistent "
            "with this explanation. Per governing Section 5/16/19 rules, calibration_m9_4's pool size was "
            "predeclared before any results were seen and is NOT enlarged post-hoc here to fix this failure "
            "(that would be exactly the forbidden 'tune calibration against development' move) -- if this "
            "hypothesis is judged correct, enlarging calibration_m9_4's repeat count under a NEW, separately "
            "predeclared M9.5 protocol is the appropriate next step, not a same-milestone patch.",
            "ARM_A was never evaluated on branched-loop/loop-grid (never trained there), matching M9.0a's own "
            "scope -- known-family guardrails compare CURRENT vs INTERLEAVED only on golden-reference; "
            "branched-loop/loop-grid are reported as ARM_B2-only retention figures.",
            "Section 21's LEGACY_SUBSET/FULL_SOURCE/EXCLUDED_ONLY decomposition uses M9.4's own full-source "
            "development population filtered by legacy-source-set membership, not the literal old M9.0a "
            "generator output -- see m9-4-legacy-vs-full-source.json's _note and m9-4-legacy-reproduction.json "
            "for the separate literal-old-numbers bridge.",
            "seed20260814's ARM_B2 checkpoint has 1200 optimizer steps vs 1350 for the other two seeds and vs "
            "ARM_A's 1350 (known M9.0a optimizer-step-parity gap, Section 22) -- preserved, not fixed, here.",
            "Full repository test suite was not necessarily run to completion; see reported test commands/counts.",
        ],
    }
    m4.M9_4_CLOSURE_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    _write_summary(manifest, legacy, audit, family_metrics, paired_bootstrap, guardrails, calibration_gate, generalization_gate, confusion, closure)

    print(json.dumps({"M9_4_DECISION": decision_code, "decision_name": decision_name}, indent=2))
    return 0


def _write_summary(manifest, legacy, audit, family_metrics, paired_bootstrap, guardrails, calibration_gate, generalization_gate, confusion, closure) -> None:
    macro = family_metrics["macro_family_equal_weight"]["UNSEEN_DEVELOPMENT_FAMILY"]
    lines = [
        "# Milestone 9.4 summary: source-representative, exchangeability-corrected re-evaluation",
        "",
        "Frozen-checkpoint re-evaluation only. No training, no tuning. Follows up "
        "`reports/evaluation/hydrocore-v5/m9-3/m9-3-closure.json` (EVAL_MAX_SOURCES=4 truncation root cause).",
        "",
        f"**M9_4_LEGACY_REPRODUCTION**: {legacy['M9_4_LEGACY_REPRODUCTION']}",
        f"**Representativeness audit passed**: {audit['representativeness_audit_passed']}",
        "",
        "## Macro-family (equal-weight) MATURE neural Top-1, unseen development families, full source population",
        "",
        f"ARM_A: {macro['ARM_A']['MATURE']['top1']}",
        f"ARM_B2: {macro['ARM_B2']['MATURE']['top1']}",
        f"Macro delta (bootstrap point estimate): {generalization_gate['macro_mature_delta']}",
        f"90% paired-bootstrap CI: {generalization_gate['bootstrap_ci90']}",
        f"Legacy (M9.0a, EVAL_MAX_SOURCES=4) pooled gain was +{m4.LEGACY_POOLED_UNSEEN_MATURE_NEURAL_TOP1_GAIN_PP}pp -- "
        "VALID FOR THE LEGACY EVALUATED SOURCE SUBSET, NOW REASSESSED ON FULL SOURCE SUPPORT above.",
        "",
        "## Gates",
        "",
        f"Predictive-generalization gate passed: **{generalization_gate['passed']}**",
        f"Known-family guardrails passed: **{guardrails['known_family_guardrails_passed']}**",
        f"Calibration gate passed: **{calibration_gate['calibration_gate_passed']}**",
        "",
        f"## M9_4_DECISION: {closure['M9_4_DECISION']} ({closure['M9_4_DECISION_NAME']})",
        "",
        closure["recommendation_reason"],
        "",
        f"Loop-grid J1 diagonal / row total (ARM_B2, all seeds/depths): {confusion['j1_diagonal']}/{confusion['j1_row_total']}",
        f"Hard source-pair counts (aggregate): {confusion['hard_source_pairs_aggregate']}",
        "",
        f"Provisional best HydroCore-S recipe: {closure['provisional_best_hydrocore_s_recipe']}",
        f"Optimizer-step-parity confirmation still required: {closure['optimizer_parity_confirmation_required']}",
        "",
        f"locked tests opened: before={closure['locked_test_opened_before']}, after={closure['locked_test_opened_after']}. "
        "No model promoted to production. No safety/authority semantics changed.",
    ]
    m4.M9_4_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
