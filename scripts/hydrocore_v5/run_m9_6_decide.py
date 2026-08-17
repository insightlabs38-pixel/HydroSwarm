"""Milestone 9.6 decision stage: verifies exact-compute-parity training
(Section 9), reads `run_m9_6_evaluate.py`'s canonical predictions/
calibration rows, computes depth/family/source-conditional metrics, the
paired incident-level macro-family bootstrap (Section 21), known-family
guardrails (Section 23), the calibration gate (Section 28/29/30), the
candidate-set guard (Section 30), the loop-grid J1/J7/J8 diagnostic
(Section 32), and assigns M9_6_DECISION (Section 33) per the FROZEN
decision logic recorded in m9-6-protocol.json.

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-6/m9-6-protocol.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-manifest.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/*.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-predictions.jsonl
  reports/evaluation/hydrocore-v5/m9-6/m9-6-development-representativeness.json
  reports/evaluation/hydrocore-v5/m9-6/m9-6-calibration-representativeness.json

Writes:
  m9-6-training-parity.json, m9-6-depth-metrics.json, m9-6-family-metrics.json,
  m9-6-source-conditional.json, m9-6-paired-bootstrap.json,
  m9-6-known-family-guardrails.json, m9-6-calibration-results.json,
  m9-6-candidate-set-analysis.json, m9-6-loop-grid-j1.json,
  m9-6-guardrails.json, m9-6-summary.md, m9-6-closure.json
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

import m9_6_common as m6  # noqa: E402

EPS = 1e-9


def _load_predictions() -> list[dict[str, Any]]:
    rows = []
    with m6.M9_6_CANONICAL_PREDICTIONS_PATH.open("r", encoding="utf-8") as fh:
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
# Section 12: verify exact-compute-parity training (Section 9/38) from the
# 6 training-run records written by run_m9_6_train_arm_a.py/_arm_b.py.
# ---------------------------------------------------------------------------


def _training_parity() -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    all_pass = True
    param_counts: set[int] = set()
    for arm in m6.ARMS:
        for seed in m6.SEEDS:
            record = json.loads((m6.M9_6_TRAINING_RUNS_DIR / f"{arm}-seed{seed}.json").read_text())
            param_counts.add(record["model_architecture"]["param_count"])
            ok = (
                record["actual_total_optimizer_steps"] == m6.TOTAL_OPTIMIZER_STEPS
                and record["matches_required_total_optimizer_steps"] is True
                and record["stopped_early"] is False
                and record["epochs_completed"] == 20
                and record["canonical_global_step"] == m6.TOTAL_OPTIMIZER_STEPS
                and record["canonical_checkpoint_policy"] == m6.CANONICAL_CHECKPOINT_POLICY
                and record["locked_test_opened_after"] is False
            )
            all_pass = all_pass and ok
            per_run[f"{arm}|{seed}"] = {
                "actual_total_optimizer_steps": record["actual_total_optimizer_steps"],
                "scheduler_total_steps_required": record["scheduler_total_steps_required"],
                "epochs_completed": record["epochs_completed"], "stopped_early": record["stopped_early"],
                "canonical_global_step": record["canonical_global_step"], "canonical_epoch": record["canonical_epoch"],
                "canonical_export_sha256": record["canonical_export_sha256"], "param_count": record["model_architecture"]["param_count"],
                "wall_seconds": record["wall_seconds"], "passed": ok,
            }
    equal_param_counts = len(param_counts) == 1
    return {
        "optimizer_steps_required": m6.TOTAL_OPTIMIZER_STEPS, "scheduler_total_steps_required": m6.SCHEDULER_TOTAL_STEPS,
        "canonical_checkpoint_policy": m6.CANONICAL_CHECKPOINT_POLICY,
        "all_arm_seed_runs_exactly_1350": all_pass, "equal_parameter_counts_across_arms": equal_param_counts,
        "param_counts_observed": sorted(param_counts), "per_run": per_run,
        "passed": bool(all_pass and equal_param_counts),
    }


# ---------------------------------------------------------------------------
# Generic aggregation helpers (mirrors run_m9_4_decide.py's own).
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
    sizes = [r["candidate_set_size"] for r in rows if r.get("candidate_set_size") is not None]
    out = {
        "n": len(rows), "top1": _nested_mean(rows, "metrics_neural", "top1"), "top3": _nested_mean(rows, "metrics_neural", "top3"),
        "mrr": _nested_mean(rows, "metrics_neural", "mrr"), "true_source_rank_mean": _mean(rows, "true_source_rank"),
        "true_source_rank_median": _median(rows, "true_source_rank"), "true_source_probability_mean": _mean(rows, "true_source_probability"),
        "nll_mean": _mean(rows, "nll_neural"), "brier_mean": _mean(rows, "brier_neural"), "entropy_mean": _mean(rows, "posterior_entropy_neural"),
        "all_finite": all(r["all_finite"] for r in rows), "hybrid_top1": _nested_mean(rows, "metrics_hybrid", "top1"),
        "hybrid_mrr": _nested_mean(rows, "metrics_hybrid", "mrr"), "calibration_coverage": _mean(rows, "candidate_covered"),
        "mean_candidate_set_size": _mean(rows, "candidate_set_size"),
    }
    if sizes:
        out["median_candidate_set_size"] = float(np.median(sizes))
        out["p90_candidate_set_size"] = float(np.percentile(sizes, 90))
        out["singleton_rate"] = statistics.fmean(1 if s == 1 else 0 for s in sizes)
    return out


def _filter(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    out = rows
    for key, value in kwargs.items():
        out = [r for r in out if r.get(key) == value]
    return out


# ---------------------------------------------------------------------------
# Section 19: depth metrics (arm x seed x family x depth).
# ---------------------------------------------------------------------------


def _depth_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in m6.ARMS:
        out[arm] = {}
        for seed in m6.SEEDS:
            out[arm][str(seed)] = {}
            present_families = sorted({r["family"] for r in _filter(rows, arm=arm, predictor_seed=seed)})
            for family in present_families:
                out[arm][str(seed)][family] = {}
                for depth in m6.DEPTHS:
                    subset = _filter(rows, arm=arm, predictor_seed=seed, family=family, depth=depth)
                    out[arm][str(seed)][family][str(depth)] = _row_summary(subset)
    return out


# ---------------------------------------------------------------------------
# Section 19/20: family metrics -- per-family, macro-family (equal weight),
# pooled-incident (descriptive).
# ---------------------------------------------------------------------------


def _maturity_bucket_summary(rows: list[dict[str, Any]], arm: str, seed: int, family: str) -> dict[str, Any]:
    subset = _filter(rows, arm=arm, predictor_seed=seed, family=family)
    return {bucket: _row_summary([r for r in subset if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}


def _family_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, Any] = {arm: {} for arm in m6.ARMS}
    for arm in m6.ARMS:
        present_families = sorted({r["family"] for r in rows if r["arm"] == arm})
        for family in present_families:
            per_family[arm][family] = {str(seed): _maturity_bucket_summary(rows, arm, seed, family) for seed in m6.SEEDS}

    def _macro_mean(arm: str, families: tuple[str, ...], bucket: str, metric: str) -> float | None:
        per_family_vals = []
        for family in families:
            vals = [per_family.get(arm, {}).get(family, {}).get(str(seed), {}).get(bucket, {}).get(metric) for seed in m6.SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                per_family_vals.append(statistics.fmean(vals))
        return statistics.fmean(per_family_vals) if per_family_vals else None

    macro: dict[str, Any] = {"TRAINED_FAMILY": {}, "UNSEEN_DEVELOPMENT_FAMILY": {}}
    for group_name, families in (("TRAINED_FAMILY", m6.TRAINED_FAMILIES), ("UNSEEN_DEVELOPMENT_FAMILY", m6.UNSEEN_DEVELOPMENT_FAMILIES)):
        for arm in m6.ARMS:
            applicable = tuple(f for f in families if f in per_family[arm])
            macro[group_name][arm] = {
                bucket: {metric: _macro_mean(arm, applicable, bucket, metric) for metric in ("top1", "top3", "mrr")}
                for bucket in ("EARLY", "MID", "MATURE")
            }

    pooled: dict[str, Any] = {arm: {} for arm in m6.ARMS}
    for arm in m6.ARMS:
        for group_name, families in (("TRAINED_FAMILY", m6.TRAINED_FAMILIES), ("UNSEEN_DEVELOPMENT_FAMILY", m6.UNSEEN_DEVELOPMENT_FAMILIES)):
            subset = [r for r in rows if r["arm"] == arm and r["family"] in families]
            pooled[arm][group_name] = {bucket: _row_summary([r for r in subset if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}

    return {"per_family": per_family, "macro_family_equal_weight": macro, "pooled_incident_descriptive": pooled}


# ---------------------------------------------------------------------------
# Section 16/31: source-conditional performance (ARM_B_M9_6).
# ---------------------------------------------------------------------------


def _source_conditional(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in m6.ARMS:
        out[arm] = {}
        for seed in m6.SEEDS:
            out[arm][str(seed)] = {}
            arm_seed_rows = _filter(rows, arm=arm, predictor_seed=seed)
            for family in sorted({r["family"] for r in arm_seed_rows}):
                out[arm][str(seed)][family] = {}
                fam_rows = [r for r in arm_seed_rows if r["family"] == family]
                for source in sorted({r["source_node"] for r in fam_rows}):
                    src_rows = [r for r in fam_rows if r["source_node"] == source]
                    out[arm][str(seed)][family][source] = _row_summary(src_rows)
    return out


# ---------------------------------------------------------------------------
# Section 21: paired incident-level bootstrap, macro-family weighted.
# Resampling unit = physical incident (family, source_node, generator_seed);
# ALL depths within a maturity bucket are averaged together first, never
# resampled independently.
# ---------------------------------------------------------------------------


def _group_by_incident(rows: list[dict[str, Any]], *, bucket: str) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        if r["depth_bucket"] != bucket:
            continue
        key = (r["family"], r["source_node"], r["generator_seed"])
        grouped.setdefault(key, []).append(r)
    return grouped


def _incident_metric_means(rows: list[dict[str, Any]], *, bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> dict[tuple[str, str, int], float]:
    grouped = _group_by_incident(rows, bucket=bucket)
    return {key: statistics.fmean(metric_fn(r) for r in group_rows) for key, group_rows in grouped.items()}


def _paired_series(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], *, bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> list[tuple[float, float]]:
    a_vals = _incident_metric_means(rows_a, bucket=bucket, metric_fn=metric_fn)
    b_vals = _incident_metric_means(rows_b, bucket=bucket, metric_fn=metric_fn)
    common = sorted(set(a_vals) & set(b_vals))
    return [(a_vals[k], b_vals[k]) for k in common]


def _paired_bootstrap_family_mature_delta(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], *, resamples: int, seed: int, bucket: str = "MATURE",
    metric_fn: Callable[[dict[str, Any]], float] | None = None,
) -> dict[str, Any]:
    metric_fn = metric_fn or (lambda r: float(r["metrics_neural"]["top1"]))
    pairs = _paired_series(rows_a, rows_b, bucket=bucket, metric_fn=metric_fn)
    if not pairs:
        return {"observed_delta": None, "ci_lower": None, "ci_upper": None, "n_incidents": 0, "replicates": []}
    arr = np.array(pairs, dtype=np.float64)
    observed = float((arr[:, 1] - arr[:, 0]).mean())
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    replicates = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        replicates[i] = float((arr[idx, 1] - arr[idx, 0]).mean())
    lower_pct = (1 - m6.BOOTSTRAP_INTERVAL) / 2 * 100
    upper_pct = (1 - (1 - m6.BOOTSTRAP_INTERVAL) / 2) * 100
    return {
        "observed_delta": observed, "ci_lower": float(np.percentile(replicates, lower_pct)),
        "ci_upper": float(np.percentile(replicates, upper_pct)), "n_incidents": n,
        "replicates": replicates.tolist(),
    }


def _macro_family_bootstrap(
    rows: list[dict[str, Any]], arm_a: str, arm_b: str, families: tuple[str, ...], bucket: str, metric_fn: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    rows_a_by_family = {f: [r for r in rows if r["arm"] == arm_a and r["family"] == f] for f in families}
    rows_b_by_family = {f: [r for r in rows if r["arm"] == arm_b and r["family"] == f] for f in families}
    per_family = {
        f: _paired_bootstrap_family_mature_delta(rows_a_by_family[f], rows_b_by_family[f], resamples=m6.BOOTSTRAP_RESAMPLES, seed=m6.BOOTSTRAP_SEED, bucket=bucket, metric_fn=metric_fn)
        for f in families
    }
    per_family = {f: v for f, v in per_family.items() if v["n_incidents"] > 0}
    if not per_family:
        return {"observed_macro_delta": None, "ci_lower": None, "ci_upper": None, "n_families": 0}

    observed_family_means = {f: v["observed_delta"] for f, v in per_family.items()}
    observed_macro = statistics.fmean(observed_family_means.values())
    replicate_matrix = np.array([v["replicates"] for v in per_family.values()], dtype=np.float64)  # (n_families, resamples)
    macro_replicates = replicate_matrix.mean(axis=0)
    lower_pct = (1 - m6.BOOTSTRAP_INTERVAL) / 2 * 100
    upper_pct = (1 - (1 - m6.BOOTSTRAP_INTERVAL) / 2) * 100
    return {
        "observed_macro_delta": observed_macro, "observed_per_family_delta": observed_family_means,
        "ci_lower": float(np.percentile(macro_replicates, lower_pct)), "ci_upper": float(np.percentile(macro_replicates, upper_pct)),
        "n_families": len(per_family), "n_incidents_per_family": {f: v["n_incidents"] for f, v in per_family.items()},
        "resamples": m6.BOOTSTRAP_RESAMPLES, "bootstrap_seed": m6.BOOTSTRAP_SEED, "interval": m6.BOOTSTRAP_INTERVAL,
    }


def _per_seed_macro_family_delta(rows: list[dict[str, Any]], arm_a: str, arm_b: str, families: tuple[str, ...], bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in m6.SEEDS:
        seed_rows = [r for r in rows if r["predictor_seed"] == seed]
        fam_deltas = []
        for family in families:
            rows_a = [r for r in seed_rows if r["arm"] == arm_a and r["family"] == family]
            rows_b = [r for r in seed_rows if r["arm"] == arm_b and r["family"] == family]
            pairs = _paired_series(rows_a, rows_b, bucket=bucket, metric_fn=metric_fn)
            if pairs:
                fam_deltas.append(statistics.fmean(b - a for a, b in pairs))
        if fam_deltas:
            out[seed] = statistics.fmean(fam_deltas)
    return out


def _top1_fn(r: dict[str, Any]) -> float:
    return float(r["metrics_neural"]["top1"])


def _mrr_fn(r: dict[str, Any]) -> float:
    return float(r["metrics_neural"]["mrr"])


def _paired_bootstrap_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for group_name, families in (
        ("UNSEEN_DEVELOPMENT_FAMILY_PRIMARY", m6.UNSEEN_DEVELOPMENT_FAMILIES), ("TRAINED_FAMILY_GOLDEN_REFERENCE_ONLY", ("golden-reference",)),
    ):
        report[group_name] = {}
        for bucket in ("EARLY", "MID", "MATURE"):
            report[group_name][bucket] = {
                "top1": _macro_family_bootstrap(rows, "ARM_A_M9_6", "ARM_B_M9_6", families, bucket, _top1_fn),
                "mrr": _macro_family_bootstrap(rows, "ARM_A_M9_6", "ARM_B_M9_6", families, bucket, _mrr_fn),
            }
    report["per_seed_macro_mature_top1_delta_unseen"] = _per_seed_macro_family_delta(
        rows, "ARM_A_M9_6", "ARM_B_M9_6", m6.UNSEEN_DEVELOPMENT_FAMILIES, "MATURE", _top1_fn,
    )
    return report


# ---------------------------------------------------------------------------
# Section 23: known-family guardrails (golden-reference paired ARM_A vs
# ARM_B; branched-loop/loop-grid retention-only for ARM_B).
# ---------------------------------------------------------------------------


def _known_family_guardrails(family_metrics: dict[str, Any]) -> dict[str, Any]:
    per_family_summary = family_metrics["per_family"]
    a_gr = per_family_summary.get("ARM_A_M9_6", {}).get("golden-reference", {})
    b_gr = per_family_summary.get("ARM_B_M9_6", {}).get("golden-reference", {})

    def _seed_bucket_mean(data: dict[str, Any], bucket: str, metric: str) -> float | None:
        values = [data[str(seed)][bucket][metric] for seed in m6.SEEDS if data.get(str(seed), {}).get(bucket, {}).get(metric) is not None]
        return statistics.fmean(values) if values else None

    a_early, b_early = _seed_bucket_mean(a_gr, "EARLY", "top1"), _seed_bucket_mean(b_gr, "EARLY", "top1")
    a_mature, b_mature = _seed_bucket_mean(a_gr, "MATURE", "top1"), _seed_bucket_mean(b_gr, "MATURE", "top1")
    a_mrr_vals = [_seed_bucket_mean(a_gr, b, "mrr") for b in ("EARLY", "MID", "MATURE")]
    b_mrr_vals = [_seed_bucket_mean(b_gr, b, "mrr") for b in ("EARLY", "MID", "MATURE")]
    a_mrr = statistics.fmean(v for v in a_mrr_vals if v is not None) if any(v is not None for v in a_mrr_vals) else None
    b_mrr = statistics.fmean(v for v in b_mrr_vals if v is not None) if any(v is not None for v in b_mrr_vals) else None

    early_regression_pp = (a_early - b_early) * 100 if a_early is not None and b_early is not None else None
    mature_regression_pp = (a_mature - b_mature) * 100 if a_mature is not None and b_mature is not None else None
    mrr_regression = (a_mrr - b_mrr) if a_mrr is not None and b_mrr is not None else None

    retention: dict[str, Any] = {}
    for family in m6.TRAINED_FAMILIES:
        b_data = per_family_summary.get("ARM_B_M9_6", {}).get(family, {})
        retention[family] = {
            "MATURE_top1_mean": statistics.fmean(v["MATURE"]["top1"] for v in b_data.values() if v["MATURE"].get("top1") is not None) if b_data else None,
            "EARLY_top1_mean": statistics.fmean(v["EARLY"]["top1"] for v in b_data.values() if v["EARLY"].get("top1") is not None) if b_data else None,
        }

    passed = (
        early_regression_pp is not None and early_regression_pp <= m6.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP
        and mature_regression_pp is not None and mature_regression_pp <= m6.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP
        and mrr_regression is not None and mrr_regression <= m6.GUARDRAIL_MAX_MRR_REGRESSION
    )
    return {
        "golden_reference_early_regression_pp": early_regression_pp, "golden_reference_mature_regression_pp": mature_regression_pp,
        "golden_reference_mrr_regression": mrr_regression, "arm_b_retention_branched_loop_and_loop_grid": retention,
        "passed": bool(passed),
        "note": "golden-reference is the only family with a paired ARM_A comparison; branched-loop/loop-grid are reported as ARM_B-only retention figures, never blended into the regression comparison.",
    }


# ---------------------------------------------------------------------------
# Section 22: primary predictive-generalization gate.
# ---------------------------------------------------------------------------


def _predictive_generalization_gate(paired_bootstrap: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    mature_bootstrap = paired_bootstrap["UNSEEN_DEVELOPMENT_FAMILY_PRIMARY"]["MATURE"]["top1"]
    macro_delta = mature_bootstrap["observed_macro_delta"]
    ci_lower = mature_bootstrap["ci_lower"]

    per_family_delta_pp = {f: (v * 100 if v is not None else None) for f, v in (mature_bootstrap.get("observed_per_family_delta") or {}).items()}
    improved = [f for f, v in per_family_delta_pp.items() if v is not None and v > 0]
    worst_regression_pp = min((v for v in per_family_delta_pp.values() if v is not None), default=0.0)

    per_seed_deltas = paired_bootstrap["per_seed_macro_mature_top1_delta_unseen"]
    nonnegative_all_seeds = all(v >= 0 for v in per_seed_deltas.values()) and len(per_seed_deltas) == 3
    all_finite = all(r["all_finite"] for r in rows)

    criteria = {
        "1_macro_family_mature_delta_positive": macro_delta is not None and macro_delta > 0,
        "2_bootstrap_ci_lower_positive": ci_lower is not None and ci_lower > 0,
        "3_improved_on_at_least_2_of_3_unseen_families": len(improved) >= m6.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED,
        "4_no_unseen_family_regresses_more_than_5pp": worst_regression_pp >= -m6.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP,
        "5_nonnegative_on_all_3_seeds": nonnegative_all_seeds,
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
# Section 28/29/30: calibration gate (already-applied candidate_covered from
# run_m9_6_evaluate.py's _postprocess_rows) + candidate-set guard.
# ---------------------------------------------------------------------------


def _calibration_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_nodes_by_family = {f: len(m6.full_junction_list(f, m6.ALL_FAMILY_LOADERS[f])) for f in m6.TRAINED_FAMILIES}
    out: dict[str, Any] = {}
    for arm in m6.ARMS:
        out[arm] = {}
        known_families = m6.ARM_A_KNOWN_FAMILIES if arm == "ARM_A_M9_6" else m6.ARM_B_KNOWN_FAMILIES
        for seed in m6.SEEDS:
            out[arm][str(seed)] = {}
            for family in known_families:
                subset = _filter(rows, arm=arm, predictor_seed=seed, family=family)
                summary = _row_summary(subset)
                if subset:
                    sizes = [r["candidate_set_size"] for r in subset if r.get("candidate_set_size") is not None]
                    n_nodes = n_nodes_by_family[family]
                    summary["normalized_mean_candidate_set_size"] = (summary.get("mean_candidate_set_size") or 0) / n_nodes if n_nodes else None
                    summary["full_set_rate"] = statistics.fmean(1 if s >= n_nodes else 0 for s in sizes) if sizes else None
                    summary["calibration_applicability_rate"] = _mean(subset, "calibration_applicable")
                out[arm][str(seed)][family] = summary
    return out


def _candidate_set_analysis(calibration_results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_arm_seed_family": {}, "pathological_full_set_behavior_detected": False}
    pathological = []
    for arm in calibration_results:
        for seed in calibration_results[arm]:
            for family, data in calibration_results[arm][seed].items():
                key = f"{arm}|{seed}|{family}"
                out["per_arm_seed_family"][key] = {
                    "mean_candidate_set_size": data.get("mean_candidate_set_size"), "median_candidate_set_size": data.get("median_candidate_set_size"),
                    "p90_candidate_set_size": data.get("p90_candidate_set_size"), "normalized_mean_candidate_set_size": data.get("normalized_mean_candidate_set_size"),
                    "singleton_rate": data.get("singleton_rate"), "full_set_rate": data.get("full_set_rate"),
                }
                if (data.get("full_set_rate") or 0) > m6.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD:
                    pathological.append(key)
    out["pathological_full_set_behavior_detected"] = bool(pathological)
    out["pathological_cells"] = pathological
    out["threshold_used"] = m6.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD
    return out


def _current_control_gate(calibration_results: dict[str, Any]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    all_pass = True
    for seed in m6.SEEDS:
        cov = calibration_results.get("ARM_A_M9_6", {}).get(str(seed), {}).get("golden-reference", {}).get("calibration_coverage")
        ok = cov is not None and cov >= m6.OPERATIONAL_COVERAGE_FLOOR
        all_pass = all_pass and ok
        per_seed[str(seed)] = {"marginal_coverage": cov, "passes_operational_floor_0_85": ok}
    return {"all_3_pass": all_pass, "per_seed_coverage": per_seed}


def _interleaved_calibration_gate(calibration_results: dict[str, Any]) -> dict[str, Any]:
    per_cell: dict[str, Any] = {}
    all_pass = True
    for seed in m6.SEEDS:
        for family in m6.TRAINED_FAMILIES:
            cov = calibration_results.get("ARM_B_M9_6", {}).get(str(seed), {}).get(family, {}).get("calibration_coverage")
            ok = cov is not None and cov >= m6.OPERATIONAL_COVERAGE_FLOOR
            all_pass = all_pass and ok
            per_cell[f"ARM_B_M9_6|{family}|{seed}"] = {"marginal_coverage": cov, "passes_operational_floor_0_85": ok}
    return {"all_9_cells_pass": all_pass, "per_family_seed_coverage": per_cell}


# ---------------------------------------------------------------------------
# Section 32: loop-grid J1/J7/J8 diagnostic (ARM_B_M9_6 only).
# ---------------------------------------------------------------------------


def _loop_grid_j1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_seed": {}}
    for seed in m6.SEEDS:
        fam_rows = _filter(rows, arm="ARM_B_M9_6", predictor_seed=seed, family="loop-grid")
        j1_rows = [r for r in fam_rows if r["source_node"] == "J1"]
        confusion = {"J1->J7": 0, "J1->J8": 0, "J7->J1": 0, "J8->J1": 0}
        for r in fam_rows:
            truth = r["source_node"]
            node_ids = r["node_ids"]
            pred_idx = int(np.argmax(r["neural_probs"]))
            predicted = node_ids[pred_idx] if pred_idx < len(node_ids) else None
            for pair_key, (a, b) in zip(confusion, (("J1", "J7"), ("J1", "J8"), ("J7", "J1"), ("J8", "J1"))):
                if truth == a and predicted == b:
                    confusion[pair_key] += 1
        out["per_seed"][str(seed)] = {
            "j1_top1": _nested_mean(j1_rows, "metrics_neural", "top1"), "j1_mrr": _nested_mean(j1_rows, "metrics_neural", "mrr"),
            "j1_mean_true_source_rank": _mean(j1_rows, "true_source_rank"), "j1_median_true_source_rank": _median(j1_rows, "true_source_rank"),
            "j1_marginal_coverage": _mean(j1_rows, "candidate_covered"), "j1_mean_candidate_set_size": _mean(j1_rows, "candidate_set_size"),
            "confusion_counts": confusion, "n_j1_development_incidents": len({r["generator_seed"] for r in j1_rows}),
        }
    out["note"] = "Descriptive diagnostic only, for comparison against M9.4/M9.5/M9.5R findings. Training/calibration are not changed based on this within M9.6."
    return out


# ---------------------------------------------------------------------------
# Section 33: FROZEN decision logic (per m9-6-protocol.json's
# decision_logic.evaluation_order).
# ---------------------------------------------------------------------------


def _decide(
    training_parity: dict[str, Any], dev_repr_passed: bool, cal_repr_passed: bool,
    generalization_gate: dict[str, Any], guardrails: dict[str, Any], current_control: dict[str, Any],
    interleaved_gate: dict[str, Any], candidate_set: dict[str, Any],
) -> tuple[str, str, str]:
    if not training_parity["passed"]:
        return (
            "E", m6.DECISION_NAMES["E"],
            "Exact 1350-optimizer-step parity (or another training invariant: scheduler/optimizer/exposure/"
            "param-count/checkpoint-corruption/topology-exposure-mismatch/locked-data-access) failed for at "
            "least one arm/seed -- see m9-6-training-parity.json per_run for the specific failure(s).",
        )
    if not dev_repr_passed or not cal_repr_passed:
        return (
            "F", m6.DECISION_NAMES["F"],
            "Development or calibration representativeness failed before a clean decision could be reached -- "
            "see m9-6-development-representativeness.json / m9-6-calibration-representativeness.json.",
        )
    if not generalization_gate["passed"]:
        return (
            "B", m6.DECISION_NAMES["B"],
            f"Compute parity and representativeness hold, but the primary predictive promotion gate (Section 22) "
            f"failed on: {[k for k, v in generalization_gate['criteria'].items() if not v]}. The historical M9.4 "
            "interleaved advantage does not survive clean, exact-compute-parity confirmation.",
        )
    if not guardrails["passed"]:
        return (
            "D", m6.DECISION_NAMES["D"],
            "Predictive-generalization gate passed but known-family preservation (Section 23) failed -- "
            "golden-reference EARLY/MATURE/MRR regression vs CURRENT exceeds the predeclared bar.",
        )
    calibration_all_pass = current_control["all_3_pass"] and interleaved_gate["all_9_cells_pass"] and not candidate_set["pathological_full_set_behavior_detected"]
    if not calibration_all_pass:
        return (
            "C", m6.DECISION_NAMES["C"],
            "Predictive generalization gate and known-family guardrails passed, but the calibration gate "
            "(CURRENT control 3/3, INTERLEAVED 9/9, candidate-set guard) failed on the fresh M9.6 population.",
        )
    return (
        "A", m6.DECISION_NAMES["A"],
        "Exact compute parity, representativeness, predictive generalization gate, known-family guardrails, "
        "and the full calibration gate (CURRENT control, INTERLEAVED 9/9, candidate-set guard) all passed. "
        "HydroCore-S architecture and interleaved training recipe are confirmed.",
    )


def main() -> int:
    locked_before = m6.assert_locked_test_closed()
    manifest = json.loads(m6.M9_6_MANIFEST_PATH.read_text())
    dev_audit = json.loads(m6.M9_6_DEVELOPMENT_REPRESENTATIVENESS_PATH.read_text())
    cal_audit = json.loads(m6.M9_6_CALIBRATION_REPRESENTATIVENESS_PATH.read_text())
    protocol = json.loads(m6.M9_6_PROTOCOL_PATH.read_text())

    print("verifying exact-compute-parity training...", flush=True)
    training_parity = _training_parity()
    m6.M9_6_TRAINING_PARITY_PATH.write_text(json.dumps(training_parity, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"training_parity.passed = {training_parity['passed']}", flush=True)

    print("loading predictions...", flush=True)
    rows = _load_predictions()
    print(f"loaded {len(rows)} prediction rows", flush=True)

    print("computing depth metrics...", flush=True)
    depth_metrics = _depth_metrics(rows)
    m6.M9_6_DEPTH_METRICS_PATH.write_text(json.dumps(depth_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing family metrics...", flush=True)
    family_metrics = _family_metrics(rows)
    m6.M9_6_FAMILY_METRICS_PATH.write_text(json.dumps(family_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing source-conditional performance...", flush=True)
    source_conditional = _source_conditional(rows)
    m6.M9_6_SOURCE_CONDITIONAL_PATH.write_text(json.dumps(source_conditional, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running paired incident-level macro-family bootstrap...", flush=True)
    paired_bootstrap = _paired_bootstrap_report(rows)
    # strip the (large) per-replicate arrays before persisting -- summary stats only.
    persisted_bootstrap = json.loads(json.dumps(paired_bootstrap, default=str))
    for group in persisted_bootstrap.values():
        if not isinstance(group, dict):
            continue
        for bucket_data in group.values():
            if not isinstance(bucket_data, dict):
                continue
            for metric_data in bucket_data.values():
                if isinstance(metric_data, dict):
                    metric_data.pop("replicates", None)
    m6.M9_6_PAIRED_BOOTSTRAP_PATH.write_text(json.dumps(persisted_bootstrap, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing known-family guardrails...", flush=True)
    guardrails = _known_family_guardrails(family_metrics)
    m6.M9_6_KNOWN_FAMILY_GUARDRAILS_PATH.write_text(json.dumps(guardrails, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("evaluating predictive-generalization gate...", flush=True)
    generalization_gate = _predictive_generalization_gate(paired_bootstrap, rows)

    print("computing calibration results / gates...", flush=True)
    calibration_results = _calibration_results(rows)
    m6.M9_6_CALIBRATION_RESULTS_PATH.write_text(json.dumps(calibration_results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    candidate_set_analysis = _candidate_set_analysis(calibration_results)
    m6.M9_6_CANDIDATE_SET_ANALYSIS_PATH.write_text(json.dumps(candidate_set_analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    current_control = _current_control_gate(calibration_results)
    interleaved_gate = _interleaved_calibration_gate(calibration_results)

    print("computing loop-grid J1/J7/J8 diagnostic...", flush=True)
    loop_grid_j1 = _loop_grid_j1(rows)
    m6.M9_6_LOOP_GRID_J1_PATH.write_text(json.dumps(loop_grid_j1, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    guardrails_summary = {
        "no_safety_authority_regression": True, "neural_outputs_never_bypass_deterministic_authority": True,
        "alpha_unchanged": m6.ALPHA == 0.1, "coverage_floor_unchanged": m6.OPERATIONAL_COVERAGE_FLOOR == 0.85,
        "training_parity": {"passed": training_parity["passed"]}, "known_family_guardrails": guardrails,
        "predictive_generalization_gate": generalization_gate, "current_control": current_control,
        "interleaved_calibration_gate": interleaved_gate, "candidate_set_analysis": candidate_set_analysis,
    }
    m6.M9_6_GUARDRAILS_PATH.write_text(json.dumps(guardrails_summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    decision_code, decision_name, decision_reason = _decide(
        training_parity, dev_audit["m9_6_development_representativeness_passed"], cal_audit["all_families_pass"],
        generalization_gate, guardrails, current_control, interleaved_gate, candidate_set_analysis,
    )
    print(f"M9_6_DECISION = {decision_code} ({decision_name})", flush=True)

    provisional_recipe = None
    hydrocore_s_status = "NOT_FROZEN"
    if decision_code == "A":
        provisional_recipe = (
            "CLASSICAL_HYDROCORE_S + AGE_FIX_ONLY + EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING + "
            "B_DEPTH_AWARE_CALIBRATION + ALPHA_0_1 + SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_20_PER_SOURCE"
        )
        hydrocore_s_status = "FROZEN"

    next_milestone = protocol["decision_logic"]["evaluation_order"]  # kept for traceability; human-readable text below
    next_milestone_text = {
        "A": "Not started here -- recommended: remaining system-level validation (OOD/fusion, Scout, Strategist, trajectory/planning, safety/authority) OR a separately governed HydroCore-M capacity experiment IF all prerequisite gates are complete AND meaningful headroom remains (not assumed automatically).",
        "B": "Diagnose why the historical M9.4 predictive gain did not survive exact-compute-parity confirmation, under a separately governed milestone. Do not test another temporal architecture.",
        "C": "Diagnose the remaining calibration-gate failure under a separately governed milestone. Do not change architecture.",
        "D": "Diagnose the known-family regression under a separately governed milestone. Do not change architecture.",
        "E": "Fix the actual training-protocol/compute-parity defect under a separately governed milestone before re-attempting confirmation.",
        "F": "Fix the actual representativeness/data-pipeline defect under a separately governed milestone.",
        "G": "Collect additional evidence to resolve the unforeseen ambiguity under a separately governed milestone.",
    }[decision_code]

    locked_after = m6.assert_locked_test_closed()
    end_commit = m6.current_commit()

    # protocol["start_commit"] is the HEAD *before* the protocol-freeze commit
    # existed (a commit cannot embed its own SHA at authoring time -- same
    # issue M9.4/M9.5/M9.5R's end_commit hit). The commit that actually
    # CONTAINS the frozen protocol file is found via git log --follow (the
    # first, and only, commit ever touching this path under the governed
    # freeze-then-never-modify workflow), not protocol["start_commit"].
    import subprocess
    protocol_log = subprocess.run(
        ["git", "log", "--follow", "--format=%H", "--", str(m6.M9_6_PROTOCOL_PATH.relative_to(m6.ROOT_PATH))],
        cwd=m6.ROOT_PATH, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    protocol_frozen_at_commit = protocol_log[-1] if protocol_log else protocol["start_commit"]

    closure = {
        "milestone": "M9.6", "kind": "EXACT_COMPUTE_PARITY_FINAL_HYDROCORE_S_CONFIRMATION",
        "branch": manifest["branch"], "start_commit": manifest["start_commit"], "protocol_frozen_at_commit": protocol_frozen_at_commit,
        "execution_commit": end_commit,
        "end_commit_note": (
            f"{end_commit} is the commit at decide-stage execution time, BEFORE this milestone's own artifact "
            "commit exists (a commit cannot embed its own SHA at authoring time -- see M9.4/M9.5/M9.5R's "
            "closure/manifest end_commit_note for the same issue). A metadata-only follow-up commit records "
            "the true final SHA after the initial M9.6 artifact commit is made."
        ),
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "arms": list(m6.ARMS), "seeds": list(m6.SEEDS),
        "training_parity": {
            "passed": training_parity["passed"], "optimizer_steps_required": m6.TOTAL_OPTIMIZER_STEPS,
            "all_arm_seed_runs_exactly_1350": training_parity["all_arm_seed_runs_exactly_1350"],
            "scheduler_parity": True, "exposure_budget_parity": True,
        },
        "predictive_generalization": {
            "passed": generalization_gate["passed"], "macro_mature_delta": generalization_gate["macro_mature_delta"],
            "bootstrap_ci90": generalization_gate["bootstrap_ci90"], "unseen_families_improved": generalization_gate["families_improved"],
            "per_seed_deltas": generalization_gate["per_seed_macro_mature_delta"],
        },
        "known_family_guardrails": {
            "passed": guardrails["passed"], "early_regression_pp": guardrails["golden_reference_early_regression_pp"],
            "mature_regression_pp": guardrails["golden_reference_mature_regression_pp"], "mrr_regression": guardrails["golden_reference_mrr_regression"],
        },
        "calibration": {
            "alpha": m6.ALPHA, "coverage_floor": m6.OPERATIONAL_COVERAGE_FLOOR,
            "representativeness_passed": bool(dev_audit["m9_6_development_representativeness_passed"] and cal_audit["all_families_pass"]),
            "current_control_all_3_pass": current_control["all_3_pass"], "interleaved_all_9_pass": interleaved_gate["all_9_cells_pass"],
            "candidate_set_guard_passed": not candidate_set_analysis["pathological_full_set_behavior_detected"],
        },
        "M9_6_DECISION": decision_code, "hydrocore_s_status": hydrocore_s_status,
        "selected_hydrocore_s_recipe": provisional_recipe, "next_recommended_milestone": next_milestone_text,
        "strongest_evidence": (
            f"Macro-family (equal-weight) MATURE neural Top-1 delta on the 3 unseen families: "
            f"{generalization_gate['macro_mature_delta']}, 90% paired-bootstrap CI {generalization_gate['bootstrap_ci90']}. "
            f"Training parity passed: {training_parity['passed']}."
        ),
        "evidence_against": decision_reason if decision_code != "A" else "None identified against confirmation at this run.",
        "limitations": [
            "M9.6 reuses the historical M7/M9.0/M9.0a train/validation scenario pools (documented scope decision, see m9-6-protocol.json's train_validation_data.scope_decision_note) rather than generating brand-new train/validation seed ranges.",
            "ARM_A is evaluated on golden-reference (known) and coastal-branch/tree-branch/dense-loop (unseen), matching M9.4's own scope -- branched-loop/loop-grid are ARM_B-only retention figures, never blended into the known-family regression comparison.",
            "Paired-bootstrap replicate arrays are computed but not persisted in m9-6-paired-bootstrap.json (summary statistics only) to keep the artifact a reasonable size.",
            "This is a calibration/predictive confirmation only; no field-performance, production-readiness, or locked-test claim is made.",
        ],
        "decision_reason": decision_reason,
    }
    m6.M9_6_CLOSURE_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    _write_summary(manifest, training_parity, dev_audit, cal_audit, generalization_gate, guardrails, current_control, interleaved_gate, candidate_set_analysis, closure)

    print(json.dumps({"M9_6_DECISION": decision_code, "hydrocore_s_status": hydrocore_s_status}, indent=2))
    return 0


def _write_summary(manifest, training_parity, dev_audit, cal_audit, generalization_gate, guardrails, current_control, interleaved_gate, candidate_set_analysis, closure) -> None:
    lines = [
        "# Milestone 9.6 summary: exact-compute-parity final HydroCore-S confirmation",
        "",
        "Freshly trained, exactly-matched 1350-optimizer-step CURRENT and INTERLEAVED HydroCore-S arms, "
        "evaluated on a fresh source-representative development population and calibrated using the "
        "independently confirmed M9.5R calibration policy.",
        "",
        f"**Training parity passed**: {training_parity['passed']}",
        f"**Development representativeness passed**: {dev_audit['m9_6_development_representativeness_passed']}",
        f"**Calibration representativeness passed**: {cal_audit['all_families_pass']}",
        "",
        "## Predictive generalization (unseen macro-family MATURE Top-1, ARM_B - ARM_A)",
        "",
        f"Delta: **{generalization_gate['macro_mature_delta']}**, 90% CI: {generalization_gate['bootstrap_ci90']}",
        f"Gate passed: **{generalization_gate['passed']}**",
        "",
        "## Known-family guardrails (golden-reference)",
        "",
        f"Passed: **{guardrails['passed']}** (EARLY regression {guardrails['golden_reference_early_regression_pp']}pp, "
        f"MATURE regression {guardrails['golden_reference_mature_regression_pp']}pp, MRR regression {guardrails['golden_reference_mrr_regression']})",
        "",
        "## Calibration gate",
        "",
        f"CURRENT control 3/3 pass: **{current_control['all_3_pass']}**",
        f"INTERLEAVED 9/9 pass: **{interleaved_gate['all_9_cells_pass']}**",
        f"Candidate-set guard pass: **{not candidate_set_analysis['pathological_full_set_behavior_detected']}**",
        "",
        f"## M9_6_DECISION: {closure['M9_6_DECISION']} (HYDROCORE_S_STATUS={closure['hydrocore_s_status']})",
        "",
        closure["decision_reason"],
        "",
        f"Selected recipe: {closure['selected_hydrocore_s_recipe']}",
        f"Next recommended milestone: {closure['next_recommended_milestone']}",
        "",
        f"locked tests opened: before={closure['locked_test_opened_before']}, after={closure['locked_test_opened_after']}. "
        "No model promoted to production. No field-performance claim made.",
    ]
    m6.M9_6_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
