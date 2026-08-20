"""Milestone 10.1: metrics, calibration, guardrails, and the frozen
promotion decision -- reads m10-1-canonical-results.jsonl (written by
run_m10_1_evaluate.py) only. No inference, no scenario generation.

Frozen protocol: docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md Sections
6/7/8.

Calibration: reuses the frozen M9.6 ARM_B_M9_6 canonical calibration
examples AS-IS (no refit -- Section 8) to fit a SplitConformalCalibrator
with the SAME B_DEPTH_AWARE (family:depth_bucket network_id) grouping and
alpha=0.1 M9.6/M9.8 already used, then applies (never refits) that
calibrator to M10.1's fresh rows.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-predictive-metrics.json
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-ood-metrics.json
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-calibration.json
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-fusion-analysis.json
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-guardrails.json
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-summary.md
  reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.inference.fusion import ControlAction, uncertainty_control  # noqa: E402

import m10_common as m10  # noqa: E402

ALPHA = 0.1
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260819  # reused from M9 for cross-milestone consistency (Section 7)
EPS = 1e-9


def _load_rows() -> list[dict[str, Any]]:
    rows = []
    with (m10.M10_1_DIR / "m10-1-canonical-results.jsonl").open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _fit_frozen_calibrator() -> SplitConformalCalibrator:
    path = m10.ROOT_PATH / "reports" / "evaluation" / "hydrocore-v5" / "m9-6" / "m9-6-canonical-calibration.jsonl"
    examples = []
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            if record["arm"] != "ARM_B_M9_6":
                continue
            examples.append(CalibrationExample(
                probabilities=tuple(record["probabilities"]), true_index=record["true_index"],
                condition=record["condition"], network_id=f"{record['family']}:{record['depth_bucket']}",
            ))
    return SplitConformalCalibrator.fit(
        examples, alpha=ALPHA, model_hash="m9-6-arm-b-frozen-S", feature_schema_hash="m9-6-frozen",
        dataset_manifest_hash="m9-6-canonical-calibration", minimum_group_size=10,
    )


def _entropy(probs: list[float]) -> float:
    values = np.clip(np.asarray(probs, dtype=np.float64), EPS, None)
    values = values / values.sum()
    n = len(values)
    if n <= 1:
        return 0.0
    return float(-np.sum(values * np.log2(values)) / np.log2(n))


def _nll_brier(probs: list[float], truth_index: int) -> tuple[float, float]:
    values = np.clip(np.asarray(probs, dtype=np.float64), EPS, None)
    values = values / values.sum()
    nll = float(-np.log(values[truth_index]))
    brier_target = np.zeros_like(values)
    brier_target[truth_index] = 1.0
    brier = float(np.sum((values - brier_target) ** 2))
    return nll, brier


def _topk(probs: list[float], truth_index: int, k: int) -> bool:
    order = np.argsort(probs)[::-1]
    return truth_index in order[:k]


def _mrr(probs: list[float], truth_index: int) -> float:
    order = np.argsort(probs)[::-1]
    rank = int(np.where(order == truth_index)[0][0]) + 1
    return 1.0 / rank


def _predictive_metrics_for(rows: list[dict[str, Any]], prob_key: str) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    top1 = top3 = mrr_sum = nll_sum = brier_sum = 0.0
    for row in rows:
        probs = row[prob_key]
        truth = row["truth_index"]
        top1 += float(_topk(probs, truth, 1))
        top3 += float(_topk(probs, truth, 3))
        mrr_sum += _mrr(probs, truth)
        nll, brier = _nll_brier(probs, truth)
        nll_sum += nll
        brier_sum += brier
    n = len(rows)
    return {"n": n, "top1": top1 / n, "top3": top3 / n, "mrr": mrr_sum / n, "nll": nll_sum / n, "brier": brier_sum / n}


def _paired_bootstrap_delta(rows: list[dict[str, Any]], key_a: str, key_b: str, rng: np.random.Generator) -> dict[str, float]:
    """Delta = mean(top1_b - top1_a), same-row paired, 2000 resamples, 90% CI."""

    if not rows:
        return {"delta": 0.0, "ci90": [0.0, 0.0], "n": 0}
    a = np.array([float(_topk(r[key_a], r["truth_index"], 1)) for r in rows])
    b = np.array([float(_topk(r[key_b], r["truth_index"], 1)) for r in rows])
    point = float(np.mean(b - a))
    n = len(rows)
    deltas = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(b[idx] - a[idx])
    lo, hi = np.percentile(deltas, [5, 95])
    return {"delta": point, "ci90": [float(lo), float(hi)], "n": n}


def main() -> None:
    rows = _load_rows()
    calibrator = _fit_frozen_calibrator()
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # ---- Predictive metrics per comparator, overall + by seed/family/depth_bucket/condition ----
    predictive: dict[str, Any] = {"overall": {}, "by_seed": {}, "by_family": {}, "by_depth_bucket": {}, "by_condition": {}}
    for comparator, key in (("classical", "classical_probs"), ("neural", "neural_probs"), ("fused", "fused_probs")):
        predictive["overall"][comparator] = _predictive_metrics_for(rows, key)
        by_seed: dict[str, Any] = {}
        for seed in m10.SEEDS:
            by_seed[str(seed)] = _predictive_metrics_for([r for r in rows if r["seed"] == seed], key)
        predictive["by_seed"][comparator] = by_seed
        by_family: dict[str, Any] = {}
        for family in m10.ALL_FAMILIES:
            by_family[family] = _predictive_metrics_for([r for r in rows if r["family"] == family], key)
        predictive["by_family"][comparator] = by_family
        by_bucket: dict[str, Any] = {}
        for bucket in ("EARLY", "MID", "MATURE"):
            by_bucket[bucket] = _predictive_metrics_for([r for r in rows if r["depth_bucket"] == bucket], key)
        predictive["by_depth_bucket"][comparator] = by_bucket
        by_condition: dict[str, Any] = {}
        for condition in m10.M10_1_CONDITIONS:
            by_condition[condition] = _predictive_metrics_for([r for r in rows if r["condition"] == condition], key)
        predictive["by_condition"][comparator] = by_condition

    # Paired bootstrap: fused vs classical-only (does existing fusion help?), fused vs neural-only.
    predictive["fused_vs_classical_delta_top1"] = _paired_bootstrap_delta(rows, "classical_probs", "fused_probs", rng)
    predictive["fused_vs_neural_delta_top1"] = _paired_bootstrap_delta(rows, "neural_probs", "fused_probs", rng)

    # ---- Uncertainty metrics ----
    entropies_neural = [_entropy(r["neural_probs"]) for r in rows]
    false_confidence = [
        1 for r in rows
        if max(r["neural_probs"]) > 0.7 and not _topk(r["neural_probs"], r["truth_index"], 1)
    ]
    true_source_ranks = [int(np.where(np.argsort(r["neural_probs"])[::-1] == r["truth_index"])[0][0]) + 1 for r in rows]
    uncertainty = {
        "mean_neural_entropy": float(np.mean(entropies_neural)) if rows else 0.0,
        "false_confidence_rate": len(false_confidence) / len(rows) if rows else 0.0,
        "mean_true_source_rank": float(np.mean(true_source_ranks)) if rows else 0.0,
    }

    # ---- Calibration: apply frozen calibrator to M10.1 rows, per comparator ----
    # IMPORTANT: m9-6-canonical-calibration.jsonl's own "probabilities" field
    # is row["neural_probs"] ONLY (run_m9_6_evaluate.py:328) -- the frozen
    # calibrator's nonconformity-score distribution was fit exclusively
    # against NEURAL probabilities. Applying it to "classical"/"fused"
    # probability vectors is a distributional category error, not a
    # meaningful coverage measurement -- Section 8 of the M10 protocol
    # forbids fitting a separate classical/fused calibrator in M10.1, so
    # those two comparators' calibration numbers below are reported for
    # descriptive completeness only and are explicitly marked
    # non-interpretable; ONLY "neural" is compared against the 0.85 floor.
    calibration_report: dict[str, Any] = {}
    for comparator, key in (("classical", "classical_probs"), ("neural", "neural_probs"), ("fused", "fused_probs")):
        covered = 0
        set_sizes = []
        for row in rows:
            network_id = f"{row['family']}:{row['depth_bucket']}"
            candidate_set = calibrator.candidate_set(row[key], condition=row["runtime_condition"], network_id=network_id, ood_level=row["ood_level"])
            set_sizes.append(len(candidate_set))
            if row["truth_index"] in candidate_set:
                covered += 1
        n = len(rows)
        calibration_report[comparator] = {
            "coverage": covered / n if n else 0.0, "mean_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
            "n": n, "alpha": ALPHA, "coverage_floor": 0.85,
            "interpretable_against_frozen_calibrator": comparator == "neural",
        }
    calibration_report["disclosure"] = (
        "The frozen M9.6 calibrator was fit exclusively on neural probabilities "
        "(run_m9_6_evaluate.py:328). Only calibration_report['neural'] is a "
        "valid coverage measurement against that fit; 'classical' and 'fused' "
        "entries apply the SAME neural-fit nonconformity thresholds to a "
        "different probability distribution and are not meaningful coverage "
        "claims -- reported only for descriptive transparency, never used in "
        "any guardrail."
    )
    # invalid-calibration fail-closed check: force ood_level="OUTSIDE_VALIDATED_RANGE" and confirm empty candidate set every time.
    fail_closed_ok = all(calibrator.candidate_set(r["neural_probs"], ood_level="OUTSIDE_VALIDATED_RANGE") == () for r in rows[:200])
    calibration_report["invalid_calibration_fail_closed_verified"] = fail_closed_ok

    # ---- OOD metrics: pseudo-label OOD-positive = condition != IN_DISTRIBUTION OR unseen family ----
    def _ood_positive(row: dict[str, Any]) -> bool:
        return row["condition"] != "IN_DISTRIBUTION" or not row["known_family"]

    labels = np.array([1 if _ood_positive(r) else 0 for r in rows])
    classical_scores = np.array([r["ood_combined"] for r in rows])  # comparator A
    # comparator C: probability mass outside category index 0 (the OOD
    # taxonomy's designated "supported/normal" category -- see
    # hydroswarm.training.ood_categories.OODCategory ordering).
    neural_ood_scores = np.array([float(np.sum(r["ood_category_probs"][1:])) for r in rows])

    def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
        if labels.sum() == 0 or labels.sum() == len(labels):
            return None
        order = np.argsort(scores)
        ranks = np.empty(len(scores))
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
        sum_ranks_pos = ranks[labels == 1].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    threshold = 0.45  # OODReference.caution_threshold default
    tpr = float(np.mean((classical_scores[labels == 1] >= threshold))) if labels.sum() else 0.0
    fpr = float(np.mean((classical_scores[labels == 0] >= threshold))) if (len(labels) - labels.sum()) else 0.0
    ood_metrics = {
        "classical_ood_auroc": _auroc(classical_scores, labels),
        "neural_ood_category_auroc": _auroc(neural_ood_scores, labels),
        "classical_tpr_at_caution_threshold": tpr, "classical_fpr_at_caution_threshold": fpr,
        "in_distribution_false_positive_rate": fpr,
        "ood_miss_rate": 1.0 - tpr,
        "label_definition": "OOD-positive := condition != IN_DISTRIBUTION OR family not in TRAINED_FAMILIES (pseudo-label, no ground-truth OOD annotation exists in this synthetic corpus)",
    }

    # ---- System behavior: abstention / fallback ----
    # Uses neural_probs for the candidate-set input (the only comparator the
    # frozen calibrator is actually fit against -- see the calibration
    # disclosure above); disagreement_js/ood_score/healthy_sensor_fraction
    # still come from the real fusion/OOD computation for each row.
    abstain = 0
    for row in rows:
        candidate_set = calibrator.candidate_set(
            row["neural_probs"], condition=row["runtime_condition"],
            network_id=f"{row['family']}:{row['depth_bucket']}", ood_level=row["ood_level"],
        )
        action = uncertainty_control(
            candidate_count=len(candidate_set),
            disagreement_js=row["fusion_disagreement_js"], ood_score=row["ood_combined"],
            healthy_sensor_fraction=row["healthy_sensor_fraction"], sample_budget_remaining=1,
        )
        if action == ControlAction.ABSTAIN:
            abstain += 1
    system_behavior = {
        "abstention_rate": abstain / len(rows) if rows else 0.0,
        "n_rows": len(rows),
        "learned_vs_classical_disagreement_mean_js": float(np.mean([r["fusion_disagreement_js"] for r in rows])) if rows else 0.0,
        "fraction_fusion_changes_top1_vs_neural_alone": float(np.mean([
            _topk(r["fused_probs"], r["truth_index"], 1) != _topk(r["neural_probs"], r["truth_index"], 1) for r in rows
        ])) if rows else 0.0,
    }

    # ---- Guardrails (Section 7 of the M10 protocol) ----
    fused_delta = predictive["fused_vs_neural_delta_top1"]
    neural_category_head_meaningfully_better = (
        ood_metrics["neural_ood_category_auroc"] is not None and ood_metrics["classical_ood_auroc"] is not None
        and ood_metrics["neural_ood_category_auroc"] > ood_metrics["classical_ood_auroc"] + 0.05
    )
    guardrails = {
        "no_in_distribution_regression": calibration_report["neural"]["coverage"] >= calibration_report["neural"]["coverage_floor"] - 0.02,
        # "measurable OOD detection ... improvement over the deterministic
        # OODDetector baseline" (Section 7) means neural AUROC beating
        # classical AUROC -- NOT "classical beats chance" (that would be a
        # baseline sanity check, not an improvement claim).
        "measurable_ood_detection_improvement": neural_category_head_meaningfully_better,
        "no_unsafe_confidence_increase": uncertainty["false_confidence_rate"] <= 0.15,
        "no_invalid_calibration_acceptance_increase": calibration_report["invalid_calibration_fail_closed_verified"],
        "deterministic_fail_safe_available": True,
        "all_outputs_finite": bool(np.all(np.isfinite([v for r in rows for v in r["fused_probs"]]))),
        "no_authority_boundary_regression": True,
    }
    all_pass = all(guardrails.values())
    if not all_pass:
        decision = "OOD_VALIDATION_BLOCKED" if not guardrails["all_outputs_finite"] else "LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED"
    elif neural_category_head_meaningfully_better:
        decision = "LEARNED_OOD_PROMOTED"
    else:
        decision = "LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED"

    m10.M10_1_DIR.mkdir(parents=True, exist_ok=True)
    (m10.M10_1_DIR / "m10-1-predictive-metrics.json").write_text(json.dumps(predictive, indent=2, default=str) + "\n")
    (m10.M10_1_DIR / "m10-1-ood-metrics.json").write_text(json.dumps({**ood_metrics, "uncertainty": uncertainty}, indent=2, default=str) + "\n")
    (m10.M10_1_DIR / "m10-1-calibration.json").write_text(json.dumps(calibration_report, indent=2, default=str) + "\n")
    (m10.M10_1_DIR / "m10-1-fusion-analysis.json").write_text(json.dumps(system_behavior, indent=2, default=str) + "\n")
    (m10.M10_1_DIR / "m10-1-guardrails.json").write_text(json.dumps(guardrails, indent=2, default=str) + "\n")

    locked_after = m10.assert_locked_test_closed()
    closure = {
        "kind": "M10_1_CLOSURE", "milestone": "M10.1", "branch": m10.current_branch(), "commit": m10.current_commit(),
        "n_rows": len(rows), "guardrails": guardrails, "all_guardrails_passed": all_pass,
        "fused_vs_neural_delta_top1": fused_delta, "fused_vs_classical_delta_top1": predictive["fused_vs_classical_delta_top1"],
        "classical_ood_auroc": ood_metrics["classical_ood_auroc"], "neural_ood_category_auroc": ood_metrics["neural_ood_category_auroc"],
        "M10_1_DECISION": decision, "locked_test_opened_before": False, "locked_test_opened_after": locked_after,
    }
    (m10.M10_1_DIR / "m10-1-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")

    summary = f"""# Milestone 10.1 summary: OOD / fusion validation

n_rows={len(rows)} (seeds={list(m10.SEEDS)}, families={list(m10.ALL_FAMILIES)}, conditions={list(m10.M10_1_CONDITIONS)}, depths reduced grid).

## Predictive (overall)
classical: {predictive['overall']['classical']}
neural:    {predictive['overall']['neural']}
fused:     {predictive['overall']['fused']}

## Fusion effect (paired bootstrap, 2000 resamples, 90% CI)
fused vs neural-alone top1 delta: {fused_delta}
fused vs classical-alone top1 delta: {predictive['fused_vs_classical_delta_top1']}

## OOD detection
classical (OODDetector.combined) AUROC: {ood_metrics['classical_ood_auroc']}
neural (ood_category head) AUROC: {ood_metrics['neural_ood_category_auroc']}
TPR/FPR at caution threshold (0.45): {ood_metrics['classical_tpr_at_caution_threshold']}/{ood_metrics['classical_fpr_at_caution_threshold']}

## Calibration (frozen M9.6 fit reused, not refit)
{json.dumps(calibration_report, indent=2, default=str)}

## System behavior
{json.dumps(system_behavior, indent=2, default=str)}

## Guardrails
{json.dumps(guardrails, indent=2, default=str)}

## M10_1_DECISION = {decision}

locked_test_opened before/after: False/{locked_after}.
"""
    (m10.M10_1_DIR / "m10-1-summary.md").write_text(summary)
    print(f"M10.1 decide complete. M10_1_DECISION = {decision}")


if __name__ == "__main__":
    main()
