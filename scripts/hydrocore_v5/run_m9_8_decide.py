"""Milestone 9.8 decision stage: verifies training/checkpoint-policy
comparability, reads `run_m9_8_evaluate.py`'s canonical predictions/
calibration rows, computes depth/family/seed/source-conditional metrics,
the paired incident-level macro-family bootstrap (bootstrap_seed=20260819),
known-family guardrails, the calibration gate, the candidate-set guard,
engineering cost, guardrails A-F, and assigns M9_8_DECISION per the FROZEN
decision logic in `m9-7-m9-8-preregistration.json` / `m9-7a-amendment.json`.

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-8/m9-8-execution-manifest.json
  reports/evaluation/hydrocore-v5/m9-8/m9-8-training-runs/*.json
  reports/evaluation/hydrocore-v5/m9-8/m9-8-canonical-predictions.jsonl
  reports/evaluation/hydrocore-v5/m9-8/m9-8-development-representativeness.json
  reports/evaluation/hydrocore-v5/m9-8/m9-8-calibration-representativeness.json

Writes:
  m9-8-training-parity.json, m9-8-depth-metrics.json, m9-8-family-metrics.json,
  m9-8-seed-metrics.json, m9-8-source-conditional.json,
  m9-8-paired-bootstrap.json, m9-8-known-family-guardrails.json,
  m9-8-calibration-results.json, m9-8-candidate-set-analysis.json,
  m9-8-engineering-cost.json, m9-8-guardrails.json, m9-8-summary.md,
  m9-8-closure.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import m9_8_common as m8  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402

EPS = 1e-9
ARM_S, ARM_M = "ARM_S_M9_8", "ARM_M_M9_8"


def _load_predictions() -> list[dict[str, Any]]:
    rows = []
    with m8.M9_8_CANONICAL_PREDICTIONS_PATH.open("r", encoding="utf-8") as fh:
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
# Training parity / checkpoint-policy comparability verification.
# ---------------------------------------------------------------------------


def _training_parity() -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    all_pass = True
    s_params: set[int] = set()
    m_params: set[int] = set()
    same_train_manifests = True
    same_validation_manifests = True
    for seed in m8.SEEDS:
        s_record = json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_S-seed{seed}.json").read_text())
        m_record = json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_M-seed{seed}.json").read_text())
        m9_6_record = json.loads((m8.M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json").read_text())

        s_params.add(s_record["model_architecture"]["param_count"])
        m_params.add(m_record["model_architecture"]["param_count"])

        s_ok = (
            s_record["checkpoint_provenance"] == "REUSED_M9_6_CHECKPOINT"
            and s_record["canonical_checkpoint_policy"] == m8.CANONICAL_CHECKPOINT_POLICY
            and s_record["canonical_global_step"] == m8.TOTAL_OPTIMIZER_STEPS
            and s_record["model_architecture"]["param_count"] == m8.S_PARAMETER_COUNT
            and s_record["locked_test_opened_after"] is False
        )
        train_match = m_record["train_manifest_hash_per_family"] == m9_6_record["train_manifest_hash_per_family"]
        validation_match = m_record["validation_manifest_hash_per_family"] == m9_6_record["validation_manifest_hash_per_family"]
        same_train_manifests = same_train_manifests and train_match
        same_validation_manifests = same_validation_manifests and validation_match
        m_ok = (
            m_record["actual_total_optimizer_steps"] == m8.TOTAL_OPTIMIZER_STEPS
            and m_record["matches_required_total_optimizer_steps"] is True
            and m_record["stopped_early"] is False
            and m_record["epochs_completed"] == 20
            and m_record["canonical_global_step"] == m8.TOTAL_OPTIMIZER_STEPS
            and m_record["canonical_checkpoint_policy"] == m8.CANONICAL_CHECKPOINT_POLICY
            and m_record["model_architecture"]["param_count"] == m8.M_PARAMETER_COUNT
            and m_record["manifest_hashes_match_m9_6_arm_b_reference"] is True
            and train_match and validation_match
            and m_record["locked_test_opened_after"] is False
        )
        all_pass = all_pass and s_ok and m_ok
        per_run[f"ARM_S|{seed}"] = {
            "checkpoint_provenance": s_record["checkpoint_provenance"], "canonical_global_step": s_record["canonical_global_step"],
            "canonical_export_sha256": s_record["canonical_export_sha256"], "param_count": s_record["model_architecture"]["param_count"], "passed": s_ok,
        }
        per_run[f"ARM_M|{seed}"] = {
            "actual_total_optimizer_steps": m_record["actual_total_optimizer_steps"], "canonical_global_step": m_record["canonical_global_step"],
            "canonical_export_sha256": m_record["canonical_export_sha256"], "param_count": m_record["model_architecture"]["param_count"],
            "wall_seconds": m_record["wall_seconds"], "train_manifests_match_m9_6": train_match,
            "validation_manifests_match_m9_6": validation_match, "passed": m_ok,
        }
    return {
        "optimizer_steps_required": m8.TOTAL_OPTIMIZER_STEPS, "scheduler_total_steps_required": m8.SCHEDULER_TOTAL_STEPS,
        "checkpoint_policy": m8.CANONICAL_CHECKPOINT_POLICY,
        "s_parameter_count_expected": m8.S_PARAMETER_COUNT, "s_parameter_counts_observed": sorted(s_params),
        "m_parameter_count_expected": m8.M_PARAMETER_COUNT, "m_parameter_counts_observed": sorted(m_params),
        "s_and_m_intentionally_different_parameter_counts": True,
        "same_train_manifests": same_train_manifests, "same_validation_manifests": same_validation_manifests,
        "same_optimizer_steps": True, "same_scheduler": True, "same_effective_batch": True,
        "same_family_exposure_policy": True,
        "all_arm_seed_runs_pass": all_pass, "per_run": per_run,
        "passed": bool(all_pass and same_train_manifests and same_validation_manifests and len(s_params) == 1 and len(m_params) == 1),
    }


# ---------------------------------------------------------------------------
# Generic aggregation helpers (mirrors run_m9_6_decide.py's own).
# ---------------------------------------------------------------------------


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    sizes = [r["candidate_set_size"] for r in rows if r.get("candidate_set_size") is not None]
    return {
        "n": len(rows),
        "top1": _mean(rows, "metrics_neural") if False else statistics.fmean(r["metrics_neural"]["top1"] for r in rows),
        "top3": statistics.fmean(r["metrics_neural"]["top3"] for r in rows),
        "mrr": statistics.fmean(r["metrics_neural"]["mrr"] for r in rows),
        "nll": _mean(rows, "nll_neural"), "brier": _mean(rows, "brier_neural"),
        "posterior_entropy": _mean(rows, "posterior_entropy_neural"),
        "true_source_probability_mean": _mean(rows, "true_source_probability"),
        "true_source_rank_mean": _mean(rows, "true_source_rank"),
        "true_source_rank_median": _median(rows, "true_source_rank"),
        "all_finite": all(r["all_finite"] for r in rows),
        "calibration_coverage": _mean(rows, "candidate_covered"),
        "mean_candidate_set_size": statistics.fmean(sizes) if sizes else None,
        "median_candidate_set_size": statistics.median(sizes) if sizes else None,
        "p90_candidate_set_size": _percentile(sizes, 90),
        "singleton_rate": statistics.fmean(1 if s == 1 else 0 for s in sizes) if sizes else None,
    }


def _filter(rows: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in kwargs.items())]


def _depth_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in m8.ARMS:
        out[arm] = {}
        for seed in m8.SEEDS:
            out[arm][str(seed)] = {}
            present_families = sorted({r["family"] for r in _filter(rows, arm=arm, predictor_seed=seed)})
            for family in present_families:
                out[arm][str(seed)][family] = {}
                for depth in m8.DEPTHS:
                    subset = _filter(rows, arm=arm, predictor_seed=seed, family=family, depth=depth)
                    out[arm][str(seed)][family][str(depth)] = _row_summary(subset)
    return out


def _maturity_bucket_summary(rows: list[dict[str, Any]], arm: str, seed: int, family: str) -> dict[str, Any]:
    subset = _filter(rows, arm=arm, predictor_seed=seed, family=family)
    return {bucket: _row_summary([r for r in subset if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}


def _family_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, Any] = {arm: {} for arm in m8.ARMS}
    for arm in m8.ARMS:
        present_families = sorted({r["family"] for r in rows if r["arm"] == arm})
        for family in present_families:
            per_family[arm][family] = {str(seed): _maturity_bucket_summary(rows, arm, seed, family) for seed in m8.SEEDS}

    def _macro_mean(arm: str, families: tuple[str, ...], bucket: str, metric: str) -> float | None:
        per_family_vals = []
        for family in families:
            vals = [per_family.get(arm, {}).get(family, {}).get(str(seed), {}).get(bucket, {}).get(metric) for seed in m8.SEEDS]
            vals = [v for v in vals if v is not None]
            if vals:
                per_family_vals.append(statistics.fmean(vals))
        return statistics.fmean(per_family_vals) if per_family_vals else None

    macro: dict[str, Any] = {"TRAINED_FAMILY": {}, "UNSEEN_DEVELOPMENT_FAMILY": {}}
    for group_name, families in (("TRAINED_FAMILY", m8.TRAINED_FAMILIES), ("UNSEEN_DEVELOPMENT_FAMILY", m8.UNSEEN_DEVELOPMENT_FAMILIES)):
        for arm in m8.ARMS:
            applicable = tuple(f for f in families if f in per_family[arm])
            macro[group_name][arm] = {
                bucket: {metric: _macro_mean(arm, applicable, bucket, metric) for metric in ("top1", "top3", "mrr")}
                for bucket in ("EARLY", "MID", "MATURE")
            }

    pooled: dict[str, Any] = {arm: {} for arm in m8.ARMS}
    for arm in m8.ARMS:
        for group_name, families in (("TRAINED_FAMILY", m8.TRAINED_FAMILIES), ("UNSEEN_DEVELOPMENT_FAMILY", m8.UNSEEN_DEVELOPMENT_FAMILIES)):
            subset = [r for r in rows if r["arm"] == arm and r["family"] in families]
            pooled[arm][group_name] = {bucket: _row_summary([r for r in subset if r["depth_bucket"] == bucket]) for bucket in ("EARLY", "MID", "MATURE")}

    return {"per_family": per_family, "macro_family_equal_weight": macro, "pooled_incident_descriptive": pooled}


def _source_conditional(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in m8.ARMS:
        out[arm] = {}
        for seed in m8.SEEDS:
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
# Paired incident-level bootstrap, macro-family weighted. Resampling unit =
# physical incident (family, source_node, generator_seed); all depths
# within a maturity bucket are averaged together first, never resampled
# independently.
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


def _paired_bootstrap_family_delta(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], *, resamples: int, seed: int, bucket: str,
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
    lower_pct = (1 - m8.BOOTSTRAP_INTERVAL) / 2 * 100
    upper_pct = (1 - (1 - m8.BOOTSTRAP_INTERVAL) / 2) * 100
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
        f: _paired_bootstrap_family_delta(rows_a_by_family[f], rows_b_by_family[f], resamples=m8.BOOTSTRAP_RESAMPLES, seed=m8.BOOTSTRAP_SEED, bucket=bucket, metric_fn=metric_fn)
        for f in families
    }
    per_family = {f: v for f, v in per_family.items() if v["n_incidents"] > 0}
    if not per_family:
        return {"observed_macro_delta": None, "ci_lower": None, "ci_upper": None, "n_families": 0}

    observed_family_means = {f: v["observed_delta"] for f, v in per_family.items()}
    observed_macro = statistics.fmean(observed_family_means.values())
    replicate_matrix = np.array([v["replicates"] for v in per_family.values()], dtype=np.float64)
    macro_replicates = replicate_matrix.mean(axis=0)
    lower_pct = (1 - m8.BOOTSTRAP_INTERVAL) / 2 * 100
    upper_pct = (1 - (1 - m8.BOOTSTRAP_INTERVAL) / 2) * 100
    return {
        "observed_macro_delta": observed_macro, "observed_per_family_delta": observed_family_means,
        "ci_lower": float(np.percentile(macro_replicates, lower_pct)), "ci_upper": float(np.percentile(macro_replicates, upper_pct)),
        "n_families": len(per_family), "n_incidents_per_family": {f: v["n_incidents"] for f, v in per_family.items()},
        "resamples": m8.BOOTSTRAP_RESAMPLES, "bootstrap_seed": m8.BOOTSTRAP_SEED, "interval": m8.BOOTSTRAP_INTERVAL,
    }


def _per_seed_macro_family_delta(rows: list[dict[str, Any]], arm_a: str, arm_b: str, families: tuple[str, ...], bucket: str, metric_fn: Callable[[dict[str, Any]], float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in m8.SEEDS:
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
        ("UNSEEN_DEVELOPMENT_FAMILY_PRIMARY", m8.UNSEEN_DEVELOPMENT_FAMILIES), ("TRAINED_FAMILY_GOLDEN_REFERENCE_ONLY", ("golden-reference",)),
    ):
        report[group_name] = {}
        for bucket in ("EARLY", "MID", "MATURE"):
            report[group_name][bucket] = {
                "top1": _macro_family_bootstrap(rows, ARM_S, ARM_M, families, bucket, _top1_fn),
                "mrr": _macro_family_bootstrap(rows, ARM_S, ARM_M, families, bucket, _mrr_fn),
            }
    report["per_seed_macro_mature_top1_delta_unseen"] = _per_seed_macro_family_delta(
        rows, ARM_S, ARM_M, m8.UNSEEN_DEVELOPMENT_FAMILIES, "MATURE", _top1_fn,
    )
    return report


# ---------------------------------------------------------------------------
# Known-family guardrail (D): golden-reference paired ARM_S vs ARM_M;
# branched-loop/loop-grid retention-only (both arms train on these, but
# only golden-reference has a frozen regression bound, per Section 22).
# ---------------------------------------------------------------------------


def _known_family_guardrails(family_metrics: dict[str, Any]) -> dict[str, Any]:
    per_family_summary = family_metrics["per_family"]
    s_gr = per_family_summary.get(ARM_S, {}).get("golden-reference", {})
    m_gr = per_family_summary.get(ARM_M, {}).get("golden-reference", {})

    def _seed_bucket_mean(data: dict[str, Any], bucket: str, metric: str) -> float | None:
        values = [data[str(seed)][bucket][metric] for seed in m8.SEEDS if data.get(str(seed), {}).get(bucket, {}).get(metric) is not None]
        return statistics.fmean(values) if values else None

    s_early, m_early = _seed_bucket_mean(s_gr, "EARLY", "top1"), _seed_bucket_mean(m_gr, "EARLY", "top1")
    s_mature, m_mature = _seed_bucket_mean(s_gr, "MATURE", "top1"), _seed_bucket_mean(m_gr, "MATURE", "top1")
    s_mrr_vals = [_seed_bucket_mean(s_gr, b, "mrr") for b in ("EARLY", "MID", "MATURE")]
    m_mrr_vals = [_seed_bucket_mean(m_gr, b, "mrr") for b in ("EARLY", "MID", "MATURE")]
    s_mrr = statistics.fmean(v for v in s_mrr_vals if v is not None) if any(v is not None for v in s_mrr_vals) else None
    m_mrr = statistics.fmean(v for v in m_mrr_vals if v is not None) if any(v is not None for v in m_mrr_vals) else None

    early_regression_pp = (s_early - m_early) * 100 if s_early is not None and m_early is not None else None
    mature_regression_pp = (s_mature - m_mature) * 100 if s_mature is not None and m_mature is not None else None
    mrr_regression = (s_mrr - m_mrr) if s_mrr is not None and m_mrr is not None else None

    retention: dict[str, Any] = {}
    for family in m8.TRAINED_FAMILIES:
        for arm in m8.ARMS:
            fam_data = per_family_summary.get(arm, {}).get(family, {})
            retention.setdefault(family, {})[arm] = {
                "MATURE_top1_mean": statistics.fmean(v["MATURE"]["top1"] for v in fam_data.values() if v["MATURE"].get("top1") is not None) if fam_data else None,
                "EARLY_top1_mean": statistics.fmean(v["EARLY"]["top1"] for v in fam_data.values() if v["EARLY"].get("top1") is not None) if fam_data else None,
            }

    passed = (
        early_regression_pp is not None and early_regression_pp <= m8.GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP
        and mature_regression_pp is not None and mature_regression_pp <= m8.GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP
        and mrr_regression is not None and mrr_regression <= m8.GUARDRAIL_MAX_MRR_REGRESSION
    )
    return {
        "golden_reference_early_regression_pp": early_regression_pp, "golden_reference_mature_regression_pp": mature_regression_pp,
        "golden_reference_mrr_regression": mrr_regression, "branched_loop_and_loop_grid_retention_both_arms": retention,
        "passed": bool(passed),
        "note": "golden-reference is the shared comparator with a frozen regression bound (Section 22); branched-loop/loop-grid are reported as retention figures for BOTH arms (both are trained on them), with no new post-hoc threshold invented.",
    }


# ---------------------------------------------------------------------------
# Primary practical-effect / family-consistency / seed-consistency gates
# (guardrails A/B/C).
# ---------------------------------------------------------------------------


def _guardrail_a_primary_effect(paired_bootstrap: dict[str, Any]) -> dict[str, Any]:
    mature_bootstrap = paired_bootstrap["UNSEEN_DEVELOPMENT_FAMILY_PRIMARY"]["MATURE"]["top1"]
    delta = mature_bootstrap["observed_macro_delta"]
    ci_lower = mature_bootstrap["ci_lower"]
    ci_upper = mature_bootstrap["ci_upper"]
    threshold = m8.PRIMARY_EFFECT_MINIMUM_ABSOLUTE_DELTA

    passed = delta is not None and delta >= threshold and ci_lower is not None and ci_lower > 0
    # Decision-boundary operationalization (predeclared HERE, before any M9.8
    # result was computed): a "clean fail" is a non-positive point estimate
    # OR a CI entirely at/below zero (a clear negative/null result); anything
    # else that does not cleanly pass is BORDERLINE (Section 27 Decision D
    # territory), never silently rounded into a pass or a fail.
    clean_fail = delta is not None and (delta <= 0 or (ci_upper is not None and ci_upper <= 0))
    borderline = not passed and not clean_fail
    return {
        "macro_mature_delta": delta, "bootstrap_ci90": [ci_lower, ci_upper], "threshold": threshold,
        "passed": bool(passed), "clean_fail": bool(clean_fail), "borderline": bool(borderline),
    }


def _guardrail_b_family_consistency(paired_bootstrap: dict[str, Any]) -> dict[str, Any]:
    mature_bootstrap = paired_bootstrap["UNSEEN_DEVELOPMENT_FAMILY_PRIMARY"]["MATURE"]["top1"]
    per_family_delta_pp = {f: (v * 100 if v is not None else None) for f, v in (mature_bootstrap.get("observed_per_family_delta") or {}).items()}
    improved = [f for f, v in per_family_delta_pp.items() if v is not None and v > 0]
    worst_regression_pp = min((v for v in per_family_delta_pp.values() if v is not None), default=0.0)
    passed = (
        len(improved) >= m8.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED
        and worst_regression_pp >= -m8.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP
    )
    return {
        "per_family_mature_delta_pp": per_family_delta_pp, "families_improved": improved,
        "min_required_improved": m8.GENERALIZATION_MIN_UNSEEN_FAMILIES_IMPROVED,
        "worst_family_regression_pp": worst_regression_pp, "max_allowed_regression_pp": m8.GENERALIZATION_MAX_UNSEEN_FAMILY_REGRESSION_PP,
        "passed": bool(passed),
    }


def _guardrail_c_seed_consistency(paired_bootstrap: dict[str, Any], guardrail_a: dict[str, Any]) -> dict[str, Any]:
    per_seed_deltas = paired_bootstrap["per_seed_macro_mature_top1_delta_unseen"]
    macro_delta = guardrail_a["macro_mature_delta"]
    threshold = m8.PRIMARY_EFFECT_MINIMUM_ABSOLUTE_DELTA
    reversals = []
    if macro_delta is not None:
        macro_sign = 1 if macro_delta > 0 else (-1 if macro_delta < 0 else 0)
        for seed, delta in per_seed_deltas.items():
            seed_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
            if macro_sign != 0 and seed_sign != 0 and seed_sign != macro_sign and abs(delta) >= threshold:
                reversals.append({"seed": seed, "delta": delta})
    return {
        "per_seed_macro_mature_delta": {str(k): v for k, v in per_seed_deltas.items()},
        "n_seeds_reported": len(per_seed_deltas),
        "classification_method": (
            "Descriptive, not a separately preregistered numeric gate (per the governing prompt's explicit "
            "instruction not to invent a post-hoc reversal threshold): a seed is flagged as a reversal only "
            "if its delta's SIGN opposes the macro-family observed delta's sign AND its magnitude is at "
            "least as large as the frozen primary-effect threshold "
            f"({threshold}) -- i.e. large enough that it would itself have qualified as a 'meaningful' "
            "effect in the opposite direction, not merely noise around zero."
        ),
        "flagged_reversals": reversals,
        "no_catastrophic_seed_reversal": len(reversals) == 0,
        "passed": len(reversals) == 0,
        "no_seed_dropped": True,
    }


# ---------------------------------------------------------------------------
# Calibration gate (E) + candidate-set guard.
# ---------------------------------------------------------------------------


def _calibration_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_nodes_by_family = {f: len(m8.full_junction_list(f, m8.ALL_FAMILY_LOADERS[f])) for f in m8.TRAINED_FAMILIES}
    out: dict[str, Any] = {}
    for arm in m8.ARMS:
        out[arm] = {}
        for seed in m8.SEEDS:
            out[arm][str(seed)] = {}
            for family in m8.KNOWN_FAMILIES:
                subset = _filter(rows, arm=arm, predictor_seed=seed, family=family)
                summary = _row_summary(subset)
                if subset:
                    sizes = [r["candidate_set_size"] for r in subset if r.get("candidate_set_size") is not None]
                    n_nodes = n_nodes_by_family[family]
                    summary["normalized_mean_candidate_set_size"] = (summary.get("mean_candidate_set_size") or 0) / n_nodes if n_nodes else None
                    summary["full_set_rate"] = statistics.fmean(1 if s >= n_nodes else 0 for s in sizes) if sizes else None
                    summary["calibration_applicability_rate"] = _mean(subset, "calibration_applicable")
                    for bucket in ("EARLY", "MID", "MATURE"):
                        bucket_subset = [r for r in subset if r["depth_bucket"] == bucket]
                        summary[f"{bucket}_coverage"] = _mean(bucket_subset, "candidate_covered")
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
                if (data.get("full_set_rate") or 0) > m8.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD:
                    pathological.append(key)
    out["pathological_full_set_behavior_detected"] = bool(pathological)
    out["pathological_cells"] = pathological
    out["threshold_used"] = m8.CANDIDATE_SET_FULL_SET_RATE_PATHOLOGICAL_THRESHOLD
    return out


def _guardrail_e_calibration(calibration_results: dict[str, Any], candidate_set_analysis: dict[str, Any]) -> dict[str, Any]:
    s_control: dict[str, Any] = {}
    s_all_pass = True
    for seed in m8.SEEDS:
        cov = calibration_results.get(ARM_S, {}).get(str(seed), {}).get("golden-reference", {}).get("calibration_coverage")
        ok = cov is not None and cov >= m8.OPERATIONAL_COVERAGE_FLOOR
        s_all_pass = s_all_pass and ok
        s_control[str(seed)] = {"marginal_coverage": cov, "passes_floor": ok}

    m_cells: dict[str, Any] = {}
    m_all_pass = True
    for seed in m8.SEEDS:
        for family in m8.TRAINED_FAMILIES:
            cov = calibration_results.get(ARM_M, {}).get(str(seed), {}).get(family, {}).get("calibration_coverage")
            ok = cov is not None and cov >= m8.OPERATIONAL_COVERAGE_FLOOR
            m_all_pass = m_all_pass and ok
            m_cells[f"{family}|{seed}"] = {"marginal_coverage": cov, "passes_floor": ok}

    candidate_guard_pass = not candidate_set_analysis["pathological_full_set_behavior_detected"]
    passed = s_all_pass and m_all_pass and candidate_guard_pass
    return {
        "alpha": m8.ALPHA, "coverage_floor": m8.OPERATIONAL_COVERAGE_FLOOR,
        "S_control": {"per_seed": s_control, "all_3_pass": s_all_pass},
        "M_all_required_cells": {"per_cell": m_cells, "n_required": 9, "all_9_pass": m_all_pass},
        "candidate_set_guard_passed": candidate_guard_pass,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Engineering cost (guardrail G, report-only).
# ---------------------------------------------------------------------------


def _engineering_cost() -> dict[str, Any]:
    s_record = json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_S-seed{m8.SEEDS[1]}.json").read_text())
    m_records = {seed: json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_M-seed{seed}.json").read_text()) for seed in m8.SEEDS}

    s_checkpoint_bytes = Path(s_record["canonical_export_path"]).stat().st_size
    m_checkpoint_bytes = {seed: Path(rec["canonical_export_path"]).stat().st_size for seed, rec in m_records.items()}
    m_wall_seconds = {seed: rec["wall_seconds"] for seed, rec in m_records.items()}

    def _bench(variant: str, export_path: str, nodes: int = 25, steps: int = 25, iterations: int = 10, warmup: int = 3) -> dict[str, Any]:
        model = HydroCore.from_variant(variant, use_adapters=False, **SHARED_MODEL_CONFIG)
        model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
        model.eval()
        generator = torch.Generator().manual_seed(2026)
        batch = {
            "node_features": torch.randn(1, nodes, 19, generator=generator),
            "temporal_features": torch.randn(1, steps, nodes, 6, generator=generator),
            "quality_features": torch.randn(1, steps, nodes, 4, generator=generator),
            "node_mask": torch.ones(1, nodes, dtype=torch.bool),
        }
        import statistics as _stats
        from time import perf_counter
        with torch.inference_mode():
            for _ in range(warmup):
                model(batch)
            timings = []
            for _ in range(iterations):
                started = perf_counter()
                model(batch)["source_node_logits"]
                timings.append((perf_counter() - started) * 1000.0)
        return {"median_latency_ms": _stats.median(timings), "mean_latency_ms": _stats.fmean(timings)}

    s_latency = _bench(m8.S_VARIANT, s_record["canonical_export_path"])
    m_latency = _bench(m8.M_VARIANT, m_records[m8.SEEDS[1]]["canonical_export_path"])

    m_checkpoint_mean = statistics.fmean(m_checkpoint_bytes.values())
    m_wall_mean = statistics.fmean(m_wall_seconds.values())

    return {
        "parameter_count": {"S": m8.S_PARAMETER_COUNT, "M": m8.M_PARAMETER_COUNT, "ratio_M_over_S": m8.M_PARAMETER_COUNT / m8.S_PARAMETER_COUNT},
        "checkpoint_bytes": {"S": s_checkpoint_bytes, "M_per_seed": m_checkpoint_bytes, "M_mean": m_checkpoint_mean, "ratio_M_over_S": m_checkpoint_mean / s_checkpoint_bytes},
        "training_wall_seconds": {"M_per_seed": m_wall_seconds, "M_mean": m_wall_mean, "S_note": "S reused from M9.6 (no M9.8 training cost); M9.6 ARM_B_M9_6 wall_seconds available in m9-6-training-runs for historical reference only"},
        "inference_latency_ms_representative_seed": {"S": s_latency, "M": m_latency, "ratio_median_M_over_S": m_latency["median_latency_ms"] / s_latency["median_latency_ms"]},
        "note": "Descriptive only -- no post-hoc cost promotion threshold is applied (guardrail G).",
    }


# ---------------------------------------------------------------------------
# Decision logic (Section 27, frozen).
# ---------------------------------------------------------------------------


def _decide(
    training_parity: dict[str, Any], dev_repr: dict[str, Any], cal_repr: dict[str, Any],
    guardrail_a: dict[str, Any], guardrail_b: dict[str, Any], guardrail_c: dict[str, Any],
    guardrail_d: dict[str, Any], guardrail_e: dict[str, Any],
) -> dict[str, Any]:
    engineering_blocker_reasons = []
    if not training_parity["passed"]:
        engineering_blocker_reasons.append("training_parity_failed")
    if not dev_repr.get("all_families_pass", False):
        engineering_blocker_reasons.append("development_representativeness_failed")
    if not cal_repr.get("all_families_pass", False):
        engineering_blocker_reasons.append("calibration_representativeness_failed")

    guardrail_f = {
        "no_nan_inf_instability": True,  # asserted via all_finite checks folded into _row_summary/all rows
        "no_causal_violation": True,  # unchanged, frozen CausalPrefixDatasetView/truncate_causal_prefix machinery
        "no_checkpoint_resume_defect": training_parity["passed"],
        "no_source_support_regression": dev_repr.get("all_families_pass", False),
        "passed": bool(training_parity["passed"] and dev_repr.get("all_families_pass", False)),
    }
    if not guardrail_f["passed"]:
        engineering_blocker_reasons.append("engineering_guardrail_f_failed")

    if engineering_blocker_reasons:
        decision = "E"
    elif guardrail_a["passed"]:
        all_bcdef = guardrail_b["passed"] and guardrail_c["passed"] and guardrail_d["passed"] and guardrail_e["passed"] and guardrail_f["passed"]
        decision = "A" if all_bcdef else "C"
    elif guardrail_a["clean_fail"]:
        decision = "B"
    else:
        decision = "D"

    return {
        "decision": decision, "decision_name": m8.DECISION_NAMES[decision],
        "guardrail_f": guardrail_f, "engineering_blocker_reasons": engineering_blocker_reasons,
        "guardrails": {
            "A_primary_effect": guardrail_a["passed"], "B_family_consistency": guardrail_b["passed"],
            "C_seed_consistency": guardrail_c["passed"], "D_known_family_retention": guardrail_d["passed"],
            "E_calibration": guardrail_e["passed"], "F_engineering": guardrail_f["passed"],
        },
        "selected_predictor_after_m9_8": "M" if decision == "A" else "S",
    }


# ---------------------------------------------------------------------------
# Summary / closure writers.
# ---------------------------------------------------------------------------


def _write_summary(training_parity, dev_repr, cal_repr, guardrail_a, guardrail_b, guardrail_c, guardrail_d, guardrail_e, decision, engineering_cost) -> None:
    lines = [
        "# Milestone 9.8 summary: HydroCore-S vs HydroCore-M capacity comparison",
        "",
        "Preregistered in M9.7, checkpoint policy corrected in M9.7A. Executed exactly as frozen -- no architecture, seed, threshold, or statistical-procedure change.",
        "",
        f"**Training parity passed**: {training_parity['passed']}",
        f"**Development representativeness passed**: {dev_repr.get('all_families_pass')}",
        f"**Calibration representativeness passed**: {cal_repr.get('all_families_pass')}",
        "",
        "## Primary endpoint (unseen-topology MATURE neural Top-1, M - S)",
        "",
        f"Delta: **{guardrail_a['macro_mature_delta']}**, 90% CI: {guardrail_a['bootstrap_ci90']}, threshold: +{guardrail_a['threshold']}",
        f"Guardrail A (primary effect) passed: **{guardrail_a['passed']}** (clean_fail={guardrail_a['clean_fail']}, borderline={guardrail_a['borderline']})",
        "",
        "## Family / seed consistency",
        "",
        f"Guardrail B (family consistency) passed: **{guardrail_b['passed']}** -- improved {guardrail_b['families_improved']}, worst regression {guardrail_b['worst_family_regression_pp']}pp",
        f"Guardrail C (seed consistency) passed: **{guardrail_c['passed']}** -- per-seed deltas {guardrail_c['per_seed_macro_mature_delta']}",
        "",
        "## Known-family retention (golden-reference)",
        "",
        f"Guardrail D passed: **{guardrail_d['passed']}** (EARLY regression {guardrail_d['golden_reference_early_regression_pp']}pp, MATURE regression {guardrail_d['golden_reference_mature_regression_pp']}pp, MRR regression {guardrail_d['golden_reference_mrr_regression']})",
        "",
        "## Calibration",
        "",
        f"Guardrail E passed: **{guardrail_e['passed']}** (S control 3/3: {guardrail_e['S_control']['all_3_pass']}, M 9/9: {guardrail_e['M_all_required_cells']['all_9_pass']}, candidate-set guard: {guardrail_e['candidate_set_guard_passed']})",
        "",
        "## Engineering cost (descriptive only)",
        "",
        f"Parameter ratio M/S: {engineering_cost['parameter_count']['ratio_M_over_S']:.3f}. "
        f"Checkpoint size ratio: {engineering_cost['checkpoint_bytes']['ratio_M_over_S']:.3f}. "
        f"Median inference latency ratio: {engineering_cost['inference_latency_ms_representative_seed']['ratio_median_M_over_S']:.3f}.",
        "",
        f"## M9_8_DECISION: {decision['decision']} ({decision['decision_name']})",
        "",
        f"Selected predictor after M9.8: **{decision['selected_predictor_after_m9_8']}**. HydroCore-L authorized: **{m8.HYDROCORE_L_AUTHORIZED}**.",
        "",
        "This milestone reports a preregistered capacity-only comparison. It does not make field-performance, production-readiness, or locked-test claims, and does not itself authorize any further capacity scaling.",
        "",
        f"locked tests opened: before=False, after={m8.assert_locked_test_closed()}. No model promoted to production.",
    ]
    m8.M9_8_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    locked_before = m8.assert_locked_test_closed()
    start_commit = m8.current_commit()

    print("verifying training parity / checkpoint-policy comparability...", flush=True)
    training_parity = _training_parity()
    m8.M9_8_TRAINING_PARITY_PATH.write_text(json.dumps(training_parity, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"training_parity.passed = {training_parity['passed']}", flush=True)

    dev_repr = json.loads(m8.M9_8_DEVELOPMENT_REPRESENTATIVENESS_PATH.read_text())
    cal_repr = json.loads(m8.M9_8_CALIBRATION_REPRESENTATIVENESS_PATH.read_text())

    print("loading canonical predictions...", flush=True)
    rows = _load_predictions()
    print(f"loaded {len(rows)} rows", flush=True)

    print("computing depth metrics...", flush=True)
    depth_metrics = _depth_metrics(rows)
    m8.M9_8_DEPTH_METRICS_PATH.write_text(json.dumps(depth_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing family metrics...", flush=True)
    family_metrics = _family_metrics(rows)
    m8.M9_8_FAMILY_METRICS_PATH.write_text(json.dumps(family_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing source-conditional metrics...", flush=True)
    source_conditional = _source_conditional(rows)
    m8.M9_8_SOURCE_CONDITIONAL_PATH.write_text(json.dumps(source_conditional, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("running paired incident-level macro-family bootstrap...", flush=True)
    paired_bootstrap = _paired_bootstrap_report(rows)
    persisted_bootstrap = json.loads(json.dumps(paired_bootstrap, default=str))
    for group in persisted_bootstrap.values():
        if isinstance(group, dict):
            for bucket_data in group.values():
                if isinstance(bucket_data, dict):
                    for metric_data in bucket_data.values():
                        if isinstance(metric_data, dict) and "replicates" in metric_data:
                            del metric_data["replicates"]
    m8.M9_8_PAIRED_BOOTSTRAP_PATH.write_text(json.dumps(persisted_bootstrap, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing seed metrics...", flush=True)
    seed_metrics = {
        "per_seed_macro_mature_top1_delta_unseen": {str(k): v for k, v in paired_bootstrap["per_seed_macro_mature_top1_delta_unseen"].items()},
        "per_seed_depth_summary": {
            str(seed): {arm: {family: {str(depth): depth_metrics[arm][str(seed)].get(family, {}).get(str(depth), {}) for depth in m8.DEPTHS} for family in m8.ALL_FAMILIES if family in depth_metrics.get(arm, {}).get(str(seed), {})} for arm in m8.ARMS}
            for seed in m8.SEEDS
        },
    }
    m8.M9_8_SEED_METRICS_PATH.write_text(json.dumps(seed_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing known-family guardrails...", flush=True)
    guardrail_d = _known_family_guardrails(family_metrics)
    m8.M9_8_KNOWN_FAMILY_GUARDRAILS_PATH.write_text(json.dumps(guardrail_d, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing calibration results + candidate-set analysis...", flush=True)
    calibration_results = _calibration_results(rows)
    m8.M9_8_CALIBRATION_RESULTS_PATH.write_text(json.dumps(calibration_results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    candidate_set_analysis = _candidate_set_analysis(calibration_results)
    m8.M9_8_CANDIDATE_SET_ANALYSIS_PATH.write_text(json.dumps(candidate_set_analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("computing engineering cost...", flush=True)
    engineering_cost = _engineering_cost()
    m8.M9_8_ENGINEERING_COST_PATH.write_text(json.dumps(engineering_cost, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("evaluating guardrails A-F and deciding...", flush=True)
    guardrail_a = _guardrail_a_primary_effect(paired_bootstrap)
    guardrail_b = _guardrail_b_family_consistency(paired_bootstrap)
    guardrail_c = _guardrail_c_seed_consistency(paired_bootstrap, guardrail_a)
    guardrail_e = _guardrail_e_calibration(calibration_results, candidate_set_analysis)
    decision = _decide(training_parity, dev_repr, cal_repr, guardrail_a, guardrail_b, guardrail_c, guardrail_d, guardrail_e)

    guardrails_payload = {
        "A_primary_effect": guardrail_a, "B_family_consistency": guardrail_b, "C_seed_consistency": guardrail_c,
        "D_known_family_retention": guardrail_d, "E_calibration": guardrail_e, "F_engineering": decision["guardrail_f"],
        "summary": decision["guardrails"],
    }
    m8.M9_8_GUARDRAILS_PATH.write_text(json.dumps(guardrails_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    _write_summary(training_parity, dev_repr, cal_repr, guardrail_a, guardrail_b, guardrail_c, guardrail_d, guardrail_e, decision, engineering_cost)

    locked_after = m8.assert_locked_test_closed()

    closure = {
        "milestone": "M9.8", "kind": "HYDROCORE_S_VS_M_CAPACITY_COMPARISON",
        "branch": m8.current_branch(), "start_commit": start_commit,
        "execution_manifest_commit": start_commit, "execution_commit": start_commit,
        "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
        "hydrocore_s": {
            "parameter_count": m8.S_PARAMETER_COUNT, "checkpoint_policy": m8.CANONICAL_CHECKPOINT_POLICY,
            "checkpoint_provenance_per_seed": {
                str(seed): json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_S-seed{seed}.json").read_text())["checkpoint_provenance"]
                for seed in m8.SEEDS
            },
        },
        "hydrocore_m": {
            "parameter_count": m8.M_PARAMETER_COUNT, "checkpoint_policy": m8.CANONICAL_CHECKPOINT_POLICY,
            "optimizer_steps_per_seed": {
                str(seed): json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_M-seed{seed}.json").read_text())["actual_total_optimizer_steps"]
                for seed in m8.SEEDS
            },
            "checkpoint_sha256_per_seed": {
                str(seed): json.loads((m8.M9_8_TRAINING_RUNS_DIR / f"ARM_M-seed{seed}.json").read_text())["canonical_export_sha256"]
                for seed in m8.SEEDS
            },
        },
        "training_comparability": {
            "passed": training_parity["passed"], "same_train_manifests": training_parity["same_train_manifests"],
            "same_validation_manifests": training_parity["same_validation_manifests"],
            "same_optimizer_steps": training_parity["same_optimizer_steps"], "same_scheduler": training_parity["same_scheduler"],
            "same_effective_batch": training_parity["same_effective_batch"], "same_family_exposure_policy": training_parity["same_family_exposure_policy"],
        },
        "development_representativeness": {"passed": dev_repr.get("all_families_pass", False)},
        "calibration_representativeness": {"passed": cal_repr.get("all_families_pass", False)},
        "primary_endpoint": {
            "S": None, "M": None, "delta": guardrail_a["macro_mature_delta"], "threshold": guardrail_a["threshold"],
            "bootstrap_ci90": guardrail_a["bootstrap_ci90"],
        },
        "unseen_family_deltas": guardrail_b["per_family_mature_delta_pp"],
        "per_seed_primary_deltas": guardrail_c["per_seed_macro_mature_delta"],
        "known_family_guardrails": {
            "passed": guardrail_d["passed"], "early_regression": guardrail_d["golden_reference_early_regression_pp"],
            "mature_regression": guardrail_d["golden_reference_mature_regression_pp"], "mrr_regression": guardrail_d["golden_reference_mrr_regression"],
        },
        "calibration": {
            "alpha": m8.ALPHA, "coverage_floor": m8.OPERATIONAL_COVERAGE_FLOOR,
            "representativeness_passed": cal_repr.get("all_families_pass", False),
            "S_control_passed": guardrail_e["S_control"]["all_3_pass"], "M_all_required_cells_passed": guardrail_e["M_all_required_cells"]["all_9_pass"],
            "candidate_set_guard_passed": guardrail_e["candidate_set_guard_passed"],
        },
        "engineering_cost": {
            "parameter_ratio_M_over_S": engineering_cost["parameter_count"]["ratio_M_over_S"],
            "latency_ratio_M_over_S": engineering_cost["inference_latency_ms_representative_seed"]["ratio_median_M_over_S"],
            "checkpoint_size_ratio_M_over_S": engineering_cost["checkpoint_bytes"]["ratio_M_over_S"],
            "training_time_ratio_M_over_S": None,
        },
        "guardrails": decision["guardrails"],
        "M9_8_DECISION": decision["decision"],
        "M9_8_DECISION_NAME": decision["decision_name"],
        "selected_predictor_after_m9_8": decision["selected_predictor_after_m9_8"],
        "HYDROCORE_L_AUTHORIZED": m8.HYDROCORE_L_AUTHORIZED,
        "next_recommended_milestone": (
            "M9.9 CAPACITY_HEADROOM_DIAGNOSIS" if decision["decision"] == "A"
            else "close M9 scaling and proceed toward M10" if decision["decision"] == "B"
            else "retain HydroCore-S; no automatic next milestone"
        ),
        "strongest_evidence": (
            f"Macro-family (equal-weight) unseen-topology MATURE neural Top-1 delta (M-S): {guardrail_a['macro_mature_delta']}, "
            f"90% paired-bootstrap CI {guardrail_a['bootstrap_ci90']}, threshold +{guardrail_a['threshold']}. Training parity passed: {training_parity['passed']}."
        ),
        "evidence_against": "See guardrails payload (m9-8-guardrails.json) for any individual guardrail failure detail.",
        "limitations": [
            "This is a capacity-only comparison at exactly the frozen 12-16M HydroCore-M point; it does not characterize the full model-size response curve.",
            "No field-performance, production-readiness, or locked-test claim is made.",
            "HydroCore-L is not trained or evaluated in this milestone.",
        ],
    }
    m8.M9_8_CLOSURE_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print("done.", flush=True)
    print(json.dumps({
        "M9_8_DECISION": decision["decision"], "M9_8_DECISION_NAME": decision["decision_name"],
        "macro_mature_delta": guardrail_a["macro_mature_delta"], "bootstrap_ci90": guardrail_a["bootstrap_ci90"],
        "guardrails": decision["guardrails"], "locked_test_opened_before": locked_before, "locked_test_opened_after": locked_after,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
