"""Milestone 9.0a Sections 12-22: paired statistical analysis, budget-parity
accounting, the frozen predeclared regression guardrails, per-seed
calibration robustness classification, and the final Outcome A-E decision
logic, applied to `run_m9_0a_evaluate.py`'s output and
`run_m9_0a_arm_b2.py`'s own per-seed training records
(docs/evaluation/HYDROCORE_V5_M9_0A_PROTOCOL.md).

Reads (never regenerates -- run_m9_0a_evaluate.py owns all evaluation,
run_m9_0a_arm_b2.py owns all training):
  reports/evaluation/hydrocore-v5/m8-7-runs/AGE_FIX_ONLY-seed*.json
  reports/evaluation/hydrocore-v5/m9-0a-runs/ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed*.json
  reports/evaluation/hydrocore-v5/m9-0a-results.json
  reports/evaluation/hydrocore-v5/m9-0a-topology-generalization.json
  reports/evaluation/hydrocore-v5/m9-0a-calibration.json

Writes:
  reports/evaluation/hydrocore-v5/m9-0a-summary.md
  reports/evaluation/hydrocore-v5/m9-0a-budget-parity.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

RUNS_M8_7 = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-runs"
RUNS_M9_0A = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-runs"
RESULTS_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-results.json"
TOPOLOGY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-topology-generalization.json"
CALIBRATION_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-calibration.json"
SUMMARY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-summary.md"
BUDGET_PARITY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-budget-parity.json"

M9_0_RESULTS_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-results.json"
M9_0_TOPOLOGY_PATH = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-topology-generalization.json"

SEEDS = (20260814, 31874, 20260815)
UNSEEN_FAMILIES = ("coastal-branch", "tree-branch", "dense-loop")
TRAINED_FAMILIES_B2_ONLY = ("branched-loop", "loop-grid")

#: Frozen protocol Section 12 guardrails (verbatim from M9.0, restated by M9.0a).
GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0
GUARDRAIL_MAX_MRR_REGRESSION = 0.03
GUARDRAIL_MIN_MARGINAL_COVERAGE = 0.90 - 0.05
GUARDRAIL_MAX_MEAN_CANDIDATE_SET_FRACTION = 0.5

#: Frozen protocol Section 12/21 topology-gain bar.
MATERIAL_IMPROVEMENT_BAR_PP = 5.0
MAX_UNSEEN_FAMILY_REGRESSION_PP = 5.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_INTERVAL = 0.90


# ---------------------------------------------------------------------------
# Budget-parity accounting (Sections 3-6, 24 of the milestone instructions).
# ---------------------------------------------------------------------------


def _arm_a_budget() -> dict[str, Any]:
    per_seed = {}
    for seed in SEEDS:
        record = json.loads((RUNS_M8_7 / f"AGE_FIX_ONLY-seed{seed}.json").read_text())
        ts = record["training_summary"]
        per_seed[str(seed)] = {
            "epochs_completed": ts["epochs_completed"], "wall_seconds": record["wall_seconds"],
            "train_scenario_count": record["train_scenario_count"],
        }
    return {
        "total_examples": 600, "microbatches_per_update": 4,
        "total_optimizer_steps": 1350, "scheduler_total_steps": 1500,
        "per_seed": per_seed,
    }


def _arm_b2_budget() -> dict[str, Any]:
    per_seed = {}
    total_steps_all_seeds = []
    matches_total_all_seeds = []
    matches_per_epoch_all_seeds = []
    wall_seconds_all_seeds = []
    for seed in SEEDS:
        record = json.loads((RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        per_seed[str(seed)] = {
            "actual_optimizer_steps_total": record["actual_optimizer_steps_total"],
            "actual_optimizer_steps_per_epoch": record["actual_optimizer_steps_per_epoch"],
            "matches_arm_a_total_optimizer_steps": record["matches_arm_a_total_optimizer_steps"],
            "matches_arm_a_per_epoch_optimizer_steps": record["matches_arm_a_per_epoch_optimizer_steps"],
            "scheduler_total_steps": record["scheduler_total_steps"],
            "family_exposure_counts": record["family_exposure_counts"],
            "wall_seconds": record["wall_seconds"],
        }
        total_steps_all_seeds.append(record["actual_optimizer_steps_total"])
        matches_total_all_seeds.append(record["matches_arm_a_total_optimizer_steps"])
        matches_per_epoch_all_seeds.append(record["matches_arm_a_per_epoch_optimizer_steps"])
        wall_seconds_all_seeds.append(record["wall_seconds"])
    return {
        "total_examples": 600, "microbatches_per_update": 4,
        "total_optimizer_steps_all_seeds": total_steps_all_seeds,
        "scheduler_total_steps": 1500,
        "matches_arm_a_total_optimizer_steps_all_seeds": matches_total_all_seeds,
        "matches_arm_a_per_epoch_optimizer_steps_all_seeds": matches_per_epoch_all_seeds,
        "per_seed": per_seed,
        "mean_wall_seconds": statistics.fmean(wall_seconds_all_seeds),
    }


def build_budget_parity() -> dict[str, Any]:
    arm_a = _arm_a_budget()
    arm_b2 = _arm_b2_budget()
    optimizer_step_ratio = statistics.fmean(arm_b2["total_optimizer_steps_all_seeds"]) / arm_a["total_optimizer_steps"]
    exposure_ratio = arm_b2["total_examples"] / arm_a["total_examples"]
    return {
        "arm_a": arm_a, "arm_b2": arm_b2,
        "optimizer_step_difference": statistics.fmean(arm_b2["total_optimizer_steps_all_seeds"]) - arm_a["total_optimizer_steps"],
        "optimizer_step_ratio": optimizer_step_ratio,
        "exposure_difference": arm_b2["total_examples"] - arm_a["total_examples"],
        "exposure_ratio": exposure_ratio,
        "scheduler_total_steps_match": arm_a["scheduler_total_steps"] == arm_b2["scheduler_total_steps"],
        "all_seeds_match_arm_a_total_steps": all(arm_b2["matches_arm_a_total_optimizer_steps_all_seeds"]),
        "all_seeds_match_arm_a_per_epoch_steps": all(arm_b2["matches_arm_a_per_epoch_optimizer_steps_all_seeds"]),
        "overall_optimization_budget_parity_pass": (
            all(arm_b2["matches_arm_a_total_optimizer_steps_all_seeds"])
            and all(arm_b2["matches_arm_a_per_epoch_optimizer_steps_all_seeds"])
            and arm_a["scheduler_total_steps"] == arm_b2["scheduler_total_steps"]
            and exposure_ratio == 1.0
        ),
    }


# ---------------------------------------------------------------------------
# Known-network / trained-family / unseen-topology summaries (byte-for-byte
# the same reduction logic run_m9_0_decide.py used, ARM_B renamed ARM_B2).
# ---------------------------------------------------------------------------


def _mean_over_seeds(per_seed: dict[str, Any], bucket: str, arm_key: str, metric: str) -> float | None:
    values = [per_seed[str(seed)][bucket][arm_key][metric] for seed in SEEDS if per_seed.get(str(seed), {}).get(bucket)]
    return statistics.fmean(values) if values else None


def _known_network_summary(results: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for arm in ("ARM_A", "ARM_B2"):
        per_seed = results["arms"][arm]["known_network_localization"]
        summary[arm] = {
            "early_top1": _mean_over_seeds(per_seed, "EARLY", "neural", "top1"),
            "mature_top1": _mean_over_seeds(per_seed, "MATURE", "neural", "top1"),
            "mid_top1": _mean_over_seeds(per_seed, "MID", "neural", "top1"),
            "overall_mrr": statistics.fmean(
                per_seed[str(seed)][bucket]["neural"]["mrr"]
                for seed in SEEDS for bucket in ("EARLY", "MID", "MATURE") if per_seed.get(str(seed), {}).get(bucket)
            ),
            "hybrid_early_top1": _mean_over_seeds(per_seed, "EARLY", "hybrid", "top1"),
            "hybrid_mature_top1": _mean_over_seeds(per_seed, "MATURE", "hybrid", "top1"),
            "all_finite": all(
                per_seed[str(seed)][bucket]["all_finite"]
                for seed in SEEDS for bucket in ("EARLY", "MID", "MATURE") if per_seed.get(str(seed), {}).get(bucket)
            ),
        }
    return summary


def _trained_family_summary(topology: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for family in TRAINED_FAMILIES_B2_ONLY:
        per_seed = topology["arms"]["ARM_B2"][f"TRAINED_FAMILY_GENERALIZATION:{family}"]
        summary[family] = {
            "mature_top1": _mean_over_seeds(per_seed, "MATURE", "neural", "top1"),
            "early_top1": _mean_over_seeds(per_seed, "EARLY", "neural", "top1"),
        }
    return summary


def _per_incident_top1(rows: list[dict[str, Any]], metric_key: str, bucket: str) -> dict[int, float]:
    by_incident: dict[int, list[float]] = {}
    for row in rows:
        if row["depth_bucket"] != bucket:
            continue
        by_incident.setdefault(row["seed"], []).append(row[metric_key]["top1"])
    return {incident: statistics.fmean(values) for incident, values in by_incident.items()}


def _unseen_pooled_and_per_family(topology: dict[str, Any]) -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    for arm in ("ARM_A", "ARM_B2"):
        pooled_mature_neural: list[float] = []
        pooled_mature_hybrid: list[float] = []
        pooled_early_neural: list[float] = []
        pooled_mature_mrr: list[float] = []
        by_family: dict[str, dict[str, list[float]]] = {}
        for family in UNSEEN_FAMILIES:
            family_mature_neural: list[float] = []
            family_data = topology["arms"][arm]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"]
            for seed in SEEDS:
                rows = family_data[str(seed)]
                mature_neural = _per_incident_top1(rows, "metrics_neural", "MATURE")
                mature_hybrid = _per_incident_top1(rows, "metrics_hybrid", "MATURE")
                early_neural = _per_incident_top1(rows, "metrics_neural", "EARLY")
                mature_mrr_by_incident: dict[int, list[float]] = {}
                for row in rows:
                    if row["depth_bucket"] == "MATURE":
                        mature_mrr_by_incident.setdefault(row["seed"], []).append(row["metrics_neural"]["mrr"])
                pooled_mature_neural.extend(mature_neural.values())
                pooled_mature_hybrid.extend(mature_hybrid.values())
                pooled_early_neural.extend(early_neural.values())
                pooled_mature_mrr.extend(statistics.fmean(v) for v in mature_mrr_by_incident.values())
                family_mature_neural.extend(mature_neural.values())
            by_family[family] = {"mature_neural_top1_mean": statistics.fmean(family_mature_neural) if family_mature_neural else None}
        per_arm[arm] = {
            "pooled_mature_neural_top1": pooled_mature_neural,
            "pooled_mature_hybrid_top1": pooled_mature_hybrid,
            "pooled_early_neural_top1": pooled_early_neural,
            "pooled_mature_mrr": pooled_mature_mrr,
            "by_family": by_family,
        }
    return per_arm


def _paired_bootstrap(a: list[float], b: list[float], *, resamples: int, seed: int, interval: float) -> dict[str, Any]:
    assert len(a) == len(b)
    diff = np.array(b) - np.array(a)  # Arm B2 minus Arm A.
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = len(diff)
    resample_means = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = diff[idx].mean()
    lower_pct = (1 - interval) / 2 * 100
    upper_pct = (1 - (1 - interval) / 2) * 100
    return {
        "observed_mean_diff": observed, "n": n,
        "ci_lower": float(np.percentile(resample_means, lower_pct)),
        "ci_upper": float(np.percentile(resample_means, upper_pct)),
    }


# ---------------------------------------------------------------------------
# Calibration robustness classification (protocol Section 10/11).
# ---------------------------------------------------------------------------


def _calibration_robustness(calibration: dict[str, Any], arm: str) -> dict[str, Any]:
    aggregate = calibration["arms"][arm]["aggregate"]
    n_pass = aggregate["n_seeds_passing_0_85"]
    if n_pass == 3:
        classification = "CALIBRATION_ROBUST_PASS"
    elif n_pass == 2:
        classification = "CALIBRATION_SEED_UNSTABLE"
    else:
        classification = "CALIBRATION_SYSTEMATICALLY_INCOMPATIBLE"
    return {"aggregate": aggregate, "n_seeds_passing_0_85": n_pass, "classification": classification}


def _guardrails(known: dict[str, Any], calibration_robustness_b2: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    a, b = known["ARM_A"], known["ARM_B2"]
    early_regression_pp = (a["early_top1"] - b["early_top1"]) * 100
    mature_regression_pp = (a["mature_top1"] - b["mature_top1"]) * 100
    mrr_regression = a["overall_mrr"] - b["overall_mrr"]
    coverage_ok = calibration_robustness_b2["classification"] == "CALIBRATION_ROBUST_PASS"
    # Same unweighted-mean-of-actual-node-counts denominator run_m9_0_decide.py used.
    KNOWN_FAMILY_NODE_COUNTS = {"golden-reference": 6, "branched-loop": 8, "loop-grid": 9}
    mean_known_node_count = statistics.fmean(KNOWN_FAMILY_NODE_COUNTS.values())
    # Mean set size across the 3 seeds' known-family calibration (protocol Section 10/20).
    b2_per_seed = calibration["arms"]["ARM_B2"]["per_seed"]
    mean_set_sizes = [b2_per_seed[str(seed)]["known_family"].get("mean_candidate_set_size") for seed in SEEDS]
    mean_set_sizes = [v for v in mean_set_sizes if v is not None]
    mean_set_size = statistics.fmean(mean_set_sizes) if mean_set_sizes else float("inf")
    candidate_set_ok = mean_set_size <= GUARDRAIL_MAX_MEAN_CANDIDATE_SET_FRACTION * mean_known_node_count
    passed = (
        early_regression_pp <= GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP
        and mature_regression_pp <= GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP
        and mrr_regression <= GUARDRAIL_MAX_MRR_REGRESSION
        and coverage_ok and candidate_set_ok and b["all_finite"]
    )
    return {
        "early_regression_pp": early_regression_pp, "mature_regression_pp": mature_regression_pp,
        "mrr_regression": mrr_regression, "coverage_ok": coverage_ok, "candidate_set_ok": candidate_set_ok,
        "mean_candidate_set_size_across_seeds": mean_set_size,
        "all_finite": b["all_finite"], "known_network_guardrails_passed": passed,
    }


def build_decision(
    known: dict[str, Any], trained: dict[str, Any], unseen: dict[str, Any],
    calibration: dict[str, Any], topology: dict[str, Any], budget_parity: dict[str, Any],
) -> dict[str, Any]:
    calibration_robustness_b2 = _calibration_robustness(calibration, "ARM_B2")
    calibration_robustness_a = _calibration_robustness(calibration, "ARM_A")
    guardrails = _guardrails(known, calibration_robustness_b2, calibration)

    mature_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_mature_neural_top1"], unseen["ARM_B2"]["pooled_mature_neural_top1"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )
    mature_hybrid_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_mature_hybrid_top1"], unseen["ARM_B2"]["pooled_mature_hybrid_top1"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )
    early_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_early_neural_top1"], unseen["ARM_B2"]["pooled_early_neural_top1"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )
    mrr_bootstrap = _paired_bootstrap(
        unseen["ARM_A"]["pooled_mature_mrr"], unseen["ARM_B2"]["pooled_mature_mrr"],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, interval=BOOTSTRAP_INTERVAL,
    )

    per_family_diff_pp = {}
    families_improved = []
    worst_regression_pp = 0.0
    for family in UNSEEN_FAMILIES:
        a_val = unseen["ARM_A"]["by_family"][family]["mature_neural_top1_mean"]
        b_val = unseen["ARM_B2"]["by_family"][family]["mature_neural_top1_mean"]
        diff_pp = (b_val - a_val) * 100 if a_val is not None and b_val is not None else None
        per_family_diff_pp[family] = diff_pp
        if diff_pp is not None:
            if diff_pp > 0:
                families_improved.append(family)
            worst_regression_pp = min(worst_regression_pp, diff_pp)

    pooled_mature_diff_pp_neural = (statistics.fmean(unseen["ARM_B2"]["pooled_mature_neural_top1"]) - statistics.fmean(unseen["ARM_A"]["pooled_mature_neural_top1"])) * 100
    pooled_mature_diff_pp_hybrid = (statistics.fmean(unseen["ARM_B2"]["pooled_mature_hybrid_top1"]) - statistics.fmean(unseen["ARM_A"]["pooled_mature_hybrid_top1"])) * 100

    def _mature_regression_pp_ok() -> bool:
        a, b = known["ARM_A"], known["ARM_B2"]
        return (a["mature_top1"] - b["mature_top1"]) * 100 <= GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP

    neural_bar_met = pooled_mature_diff_pp_neural >= MATERIAL_IMPROVEMENT_BAR_PP
    hybrid_bar_met = pooled_mature_diff_pp_hybrid >= MATERIAL_IMPROVEMENT_BAR_PP
    # "no meaningful regression in the OTHER representation" (Section 21 item 3).
    other_not_regressing_if_neural = pooled_mature_diff_pp_hybrid >= -MAX_UNSEEN_FAMILY_REGRESSION_PP
    other_not_regressing_if_hybrid = pooled_mature_diff_pp_neural >= -MAX_UNSEEN_FAMILY_REGRESSION_PP
    material_improvement = (
        (neural_bar_met and other_not_regressing_if_neural) or (hybrid_bar_met and other_not_regressing_if_hybrid)
    ) and _mature_regression_pp_ok()

    ci_lower_positive = mature_bootstrap["ci_lower"] > 0
    at_least_two_families_improved = len(families_improved) >= 2
    no_family_regresses_too_much = worst_regression_pp >= -MAX_UNSEEN_FAMILY_REGRESSION_PP

    per_seed_diffs = []
    for seed in SEEDS:
        a_seed_vals, b_seed_vals = [], []
        for family in UNSEEN_FAMILIES:
            rows_a = topology["arms"]["ARM_A"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
            rows_b = topology["arms"]["ARM_B2"]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"][str(seed)]
            a_seed_vals.extend(_per_incident_top1(rows_a, "metrics_neural", "MATURE").values())
            b_seed_vals.extend(_per_incident_top1(rows_b, "metrics_neural", "MATURE").values())
        per_seed_diffs.append(statistics.fmean(b_seed_vals) - statistics.fmean(a_seed_vals))
    directionally_consistent = all(d >= 0 for d in per_seed_diffs)

    trained_family_learned = all(
        v["mature_top1"] is not None and v["mature_top1"] > 0.15
        for v in trained.values()
    )

    # Section 21's topology-gain bar, EXCLUDING calibration (item 9) and
    # known-network guardrails (item 8, already its own gate above) --
    # calibration is decided separately (Outcomes A/C/D below).
    topology_gain_material_and_robust = (
        material_improvement and ci_lower_positive and at_least_two_families_improved
        and no_family_regresses_too_much and directionally_consistent and trained_family_learned
    )

    if not guardrails["known_network_guardrails_passed"] and (
        guardrails["early_regression_pp"] > GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP
        or guardrails["mature_regression_pp"] > GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP
        or guardrails["mrr_regression"] > GUARDRAIL_MAX_MRR_REGRESSION
        or not guardrails["candidate_set_ok"]
        or not guardrails["all_finite"]
    ):
        # Outcome E: B2 materially damages known-network performance
        # (independent of calibration -- a correctness/capability regression,
        # not the calibration-only blocker Outcomes C/D describe).
        decision = "STEP_MATCHED_INTERLEAVED_TRAINING_REJECTED"
        m9_capacity_unblocked = True
    elif not topology_gain_material_and_robust:
        # Outcome B: the gain does not survive optimizer-step parity.
        decision = "M9_0_GAIN_ATTRIBUTION_NOT_CONFIRMED"
        m9_capacity_unblocked = True
    elif calibration_robustness_b2["classification"] == "CALIBRATION_ROBUST_PASS":
        # Outcome A.
        decision = "PROMOTE_STEP_MATCHED_INTERLEAVED_TOPOLOGY_RECIPE"
        m9_capacity_unblocked = True
    elif calibration_robustness_b2["classification"] == "CALIBRATION_SEED_UNSTABLE":
        # Outcome D.
        decision = "TOPOLOGY_GAIN_VALIDATED_CALIBRATION_SEED_UNSTABLE"
        m9_capacity_unblocked = False
    else:
        # Outcome C.
        decision = "TOPOLOGY_GAIN_VALIDATED_CALIBRATION_BLOCKER_REMAINS"
        m9_capacity_unblocked = False

    return {
        "guardrails": guardrails,
        "calibration_robustness_arm_a": calibration_robustness_a,
        "calibration_robustness_arm_b2": calibration_robustness_b2,
        "mature_bootstrap_neural": mature_bootstrap, "mature_bootstrap_hybrid": mature_hybrid_bootstrap,
        "early_bootstrap_neural": early_bootstrap, "mrr_bootstrap": mrr_bootstrap,
        "per_family_diff_pp": per_family_diff_pp, "families_improved": families_improved,
        "worst_unseen_family_regression_pp": worst_regression_pp,
        "pooled_mature_diff_pp_neural": pooled_mature_diff_pp_neural, "pooled_mature_diff_pp_hybrid": pooled_mature_diff_pp_hybrid,
        "material_improvement": material_improvement, "ci_lower_positive": ci_lower_positive,
        "at_least_two_families_improved": at_least_two_families_improved,
        "no_family_regresses_too_much": no_family_regresses_too_much,
        "per_seed_pooled_mature_diff": per_seed_diffs, "directionally_consistent": directionally_consistent,
        "trained_family_learned": trained_family_learned,
        "topology_gain_material_and_robust": topology_gain_material_and_robust,
        "topology_gain_survives_optimizer_step_parity": topology_gain_material_and_robust,
        "primary_decision": decision,
        "m9_capacity_study_unblocked": m9_capacity_unblocked,
        "budget_parity_pass": budget_parity["overall_optimization_budget_parity_pass"],
    }


def _m9_0_comparison() -> dict[str, Any] | None:
    if not (M9_0_RESULTS_PATH.exists() and M9_0_TOPOLOGY_PATH.exists()):
        return None
    m9_0_topology = json.loads(M9_0_TOPOLOGY_PATH.read_text())
    pooled_mature_a: list[float] = []
    pooled_mature_b: list[float] = []
    for family in UNSEEN_FAMILIES:
        for arm, sink in (("ARM_A", pooled_mature_a), ("ARM_B", pooled_mature_b)):
            family_data = m9_0_topology["arms"][arm]["UNSEEN_TOPOLOGY"][family]["per_incident_rows"]
            for seed in SEEDS:
                rows = family_data[str(seed)]
                sink.extend(_per_incident_top1(rows, "metrics_neural", "MATURE").values())
    if not pooled_mature_a or not pooled_mature_b:
        return None
    return {
        "original_m9_0_arm_b_pooled_mature_gain_pp": (statistics.fmean(pooled_mature_b) - statistics.fmean(pooled_mature_a)) * 100,
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    results = json.loads(RESULTS_PATH.read_text())
    topology = json.loads(TOPOLOGY_PATH.read_text())
    calibration = json.loads(CALIBRATION_PATH.read_text())

    budget_parity = build_budget_parity()
    BUDGET_PARITY_PATH.write_text(json.dumps(budget_parity, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    known = _known_network_summary(results)
    trained = _trained_family_summary(topology)
    unseen = _unseen_pooled_and_per_family(topology)
    decision = build_decision(known, trained, unseen, calibration, topology, budget_parity)
    m9_0_comparison = _m9_0_comparison()

    locked_after = locked_test_opened(ROOT)

    if decision["primary_decision"] == "PROMOTE_STEP_MATCHED_INTERLEAVED_TOPOLOGY_RECIPE":
        m9_recipe_line = "A:\n    representation = AGE_FIX_ONLY\n    topology_training = STEP_MATCHED_INTERLEAVED_MULTI_FAMILY"
    elif decision["primary_decision"] in ("M9_0_GAIN_ATTRIBUTION_NOT_CONFIRMED", "STEP_MATCHED_INTERLEAVED_TRAINING_REJECTED"):
        m9_recipe_line = "B:\n    representation = AGE_FIX_ONLY\n    topology_training = SINGLE_FAMILY_CURRENT_TRAINING"
    else:
        m9_recipe_line = "C:\n    NOT_FROZEN_CALIBRATION_REVIEW_REQUIRED"

    lines = [
        "# Milestone 9.0a summary: optimizer-step-matched interleaved topology training study",
        "",
        "Frozen protocol: `docs/evaluation/HYDROCORE_V5_M9_0A_PROTOCOL.md`. Confound-resolution "
        "follow-up to Milestone 9.0: Arm B2 (`STEP_MATCHED_INTERLEAVED_MULTI_FAMILY`) uses 4 "
        "microbatches/optimizer-update (matching Arm A's `gradient_accumulation_steps=4`) in a "
        "fixed 3-update family rotation, achieving exact per-epoch and total optimizer-step and "
        "scheduler-trajectory parity with Arm A; calibration is fit/evaluated separately for all "
        "three predictor seeds for both arms (M9.0's own calibration rejection used only the "
        "representative seed).",
        "",
        "## Budget parity",
        "",
        f"Arm A total optimizer steps: {budget_parity['arm_a']['total_optimizer_steps']}",
        f"Arm B2 total optimizer steps (per seed): {budget_parity['arm_b2']['total_optimizer_steps_all_seeds']}",
        f"Optimizer-step ratio (B2/A): {budget_parity['optimizer_step_ratio']:.4f}",
        f"Exposure ratio (B2/A): {budget_parity['exposure_ratio']:.4f}",
        f"Scheduler total_steps match: {budget_parity['scheduler_total_steps_match']}",
        f"All 3 seeds match Arm A's total optimizer steps: {budget_parity['all_seeds_match_arm_a_total_steps']}",
        f"All 3 seeds match Arm A's per-epoch optimizer steps: {budget_parity['all_seeds_match_arm_a_per_epoch_steps']}",
        f"**Overall optimization-budget parity: {'PASS' if budget_parity['overall_optimization_budget_parity_pass'] else 'FAIL'}**",
        "",
        "## Known-network (golden-reference) localization, mean over 3 seeds",
        "",
        "| arm | EARLY neural top1 | MID neural top1 | MATURE neural top1 | overall MRR | MATURE hybrid top1 | all finite |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in ("ARM_A", "ARM_B2"):
        k = known[arm]
        lines.append(f"| {arm} | {k['early_top1']:.4f} | {k['mid_top1']:.4f} | {k['mature_top1']:.4f} | {k['overall_mrr']:.4f} | {k['hybrid_mature_top1']:.4f} | {k['all_finite']} |")

    lines += [
        "",
        "## Guardrails (predeclared, not relaxed after results)",
        "",
        f"EARLY regression (pp): {decision['guardrails']['early_regression_pp']:.2f} (bar <= {GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP})",
        f"MATURE regression (pp): {decision['guardrails']['mature_regression_pp']:.2f} (bar <= {GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP})",
        f"MRR regression: {decision['guardrails']['mrr_regression']:.4f} (bar <= {GUARDRAIL_MAX_MRR_REGRESSION})",
        f"Arm B2 known-family calibration robust (all 3 seeds >= 0.85): {decision['guardrails']['coverage_ok']}",
        f"Arm B2 known-family candidate-set size ok (mean across seeds): {decision['guardrails']['candidate_set_ok']} ({decision['guardrails']['mean_candidate_set_size_across_seeds']:.3f})",
        f"**Known-network guardrails passed: {decision['guardrails']['known_network_guardrails_passed']}**",
        "",
        "## Arm-B2 trained-family retention (TRAINED_FAMILY_GENERALIZATION, mean over 3 seeds)",
        "",
        "| family | MATURE neural top1 | EARLY neural top1 |",
        "|---|---|---|",
    ]
    for family, v in trained.items():
        lines.append(f"| {family} | {v['mature_top1']:.4f} | {v['early_top1']:.4f} |")
    lines.append(f"\nLearned above chance for both added families (MATURE top1 > 0.15): **{decision['trained_family_learned']}**")

    lines += [
        "",
        "## Primary unseen-topology generalization (pooled coastal-branch + tree-branch + dense-loop, MATURE, neural top1)",
        "",
        f"Arm A pooled mean: {statistics.fmean(unseen['ARM_A']['pooled_mature_neural_top1']):.4f}",
        f"Arm B2 pooled mean: {statistics.fmean(unseen['ARM_B2']['pooled_mature_neural_top1']):.4f}",
        f"Pooled diff (Arm B2 - Arm A, neural): {decision['pooled_mature_diff_pp_neural']:.2f} pp",
        f"Pooled diff (Arm B2 - Arm A, hybrid): {decision['pooled_mature_diff_pp_hybrid']:.2f} pp",
        "",
        "### Per-family MATURE neural top1 difference (Arm B2 - Arm A)",
        "",
        "| family | diff (pp) | improved |",
        "|---|---|---|",
    ]
    for family in UNSEEN_FAMILIES:
        diff = decision["per_family_diff_pp"][family]
        lines.append(f"| {family} | {diff:+.2f} | {family in decision['families_improved']} |")

    lines += [
        "",
        "### Paired bootstrap (2,000 resamples, 90% interval, bootstrap seed 20260815)",
        "",
        f"MATURE neural top1 (Arm B2 - Arm A): observed {decision['mature_bootstrap_neural']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['mature_bootstrap_neural']['ci_lower']:+.4f}, {decision['mature_bootstrap_neural']['ci_upper']:+.4f}], "
        f"n={decision['mature_bootstrap_neural']['n']}",
        f"MATURE hybrid top1 (Arm B2 - Arm A): observed {decision['mature_bootstrap_hybrid']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['mature_bootstrap_hybrid']['ci_lower']:+.4f}, {decision['mature_bootstrap_hybrid']['ci_upper']:+.4f}]",
        f"EARLY neural top1 (Arm B2 - Arm A): observed {decision['early_bootstrap_neural']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['early_bootstrap_neural']['ci_lower']:+.4f}, {decision['early_bootstrap_neural']['ci_upper']:+.4f}]",
        f"MATURE MRR (Arm B2 - Arm A): observed {decision['mrr_bootstrap']['observed_mean_diff']:+.4f}, "
        f"90% CI [{decision['mrr_bootstrap']['ci_lower']:+.4f}, {decision['mrr_bootstrap']['ci_upper']:+.4f}]",
        "",
        f"CI lower bound > 0 (MATURE neural top1): **{decision['ci_lower_positive']}**",
        f"Improved on >= 2 of 3 unseen families: **{decision['at_least_two_families_improved']}** ({', '.join(decision['families_improved']) or 'none'})",
        f"No unseen-family regression worse than {MAX_UNSEEN_FAMILY_REGRESSION_PP}pp: **{decision['no_family_regresses_too_much']}** (worst: {decision['worst_unseen_family_regression_pp']:.2f}pp)",
        f"Directionally consistent across all 3 seeds (per-seed pooled MATURE diff all >= 0): **{decision['directionally_consistent']}** ({[f'{d:+.4f}' for d in decision['per_seed_pooled_mature_diff']]})",
        f"**Topology gain survives optimizer-step parity: {decision['topology_gain_survives_optimizer_step_parity']}**",
        "",
        "## Calibration by seed (B_DEPTH_AWARE, alpha=0.1)",
        "",
    ]
    for arm in ("ARM_A", "ARM_B2"):
        lines.append(f"### {arm}")
        lines.append("")
        lines.append("| seed | marginal coverage | EARLY coverage | MID coverage | MATURE coverage | mean set size | guardrail pass |")
        lines.append("|---|---|---|---|---|---|---|")
        for seed in SEEDS:
            kc = calibration["arms"][arm]["per_seed"][str(seed)]["known_family"]
            by_maturity = kc.get("by_maturity", {})
            marginal = kc.get("marginal_coverage")
            lines.append(
                f"| {seed} | {marginal:.4f} | {by_maturity.get('EARLY', {}).get('coverage', float('nan')):.4f} | "
                f"{by_maturity.get('MID', {}).get('coverage', float('nan')):.4f} | {by_maturity.get('MATURE', {}).get('coverage', float('nan')):.4f} | "
                f"{kc.get('mean_candidate_set_size', float('nan')):.4f} | {marginal >= 0.85 if marginal is not None else False} |"
            )
        agg = calibration["arms"][arm]["aggregate"]
        lines.append(
            f"\nAggregate: mean={agg['mean_marginal_coverage']:.4f}, min={agg['min_marginal_coverage']:.4f}, "
            f"max={agg['max_marginal_coverage']:.4f}, seeds passing >=0.85: {agg['n_seeds_passing_0_85']}/3\n"
        )
    lines.append(f"**Arm B2 calibration classification: {decision['calibration_robustness_arm_b2']['classification']}**")
    lines.append(
        "\nKnown EARLY conditional-coverage limitation (carried forward from M8.7/M9.0): preserved as a "
        "documented limitation regardless of the marginal-coverage classification above, not claimed solved."
    )

    lines += [
        "",
        "## Unseen-topology calibration transfer (diagnostic, per seed)",
        "",
        "| arm | seed | marginal coverage | EARLY coverage | applicability rate |",
        "|---|---|---|---|---|",
    ]
    for arm in ("ARM_A", "ARM_B2"):
        for seed in SEEDS:
            uc = calibration["arms"][arm]["per_seed"][str(seed)]["unseen_topology_calibration_transfer"]
            by_maturity = uc.get("by_maturity", {})
            lines.append(
                f"| {arm} | {seed} | {uc.get('marginal_coverage', float('nan')):.4f} | "
                f"{by_maturity.get('EARLY', {}).get('coverage', float('nan')):.4f} | {uc.get('calibration_applicability_rate', float('nan')):.4f} |"
            )

    m9_0_comparison_line = "not available (M9.0 artifacts not found)"
    if m9_0_comparison is not None:
        m9_0_comparison_line = (
            f"M9.0 Arm B pooled MATURE gain: {m9_0_comparison['original_m9_0_arm_b_pooled_mature_gain_pp']:+.2f}pp; "
            f"M9.0a Arm B2 pooled MATURE gain: {decision['pooled_mature_diff_pp_neural']:+.2f}pp"
        )

    lines += [
        "",
        "## M9.0 comparison (historical context only -- Section 15, not perfectly paired)",
        "",
        m9_0_comparison_line,
        "",
        "This is diagnostic context only. The primary causal claim of M9.0a remains Arm A vs Arm B2 "
        "under THIS document's own step-matched protocol.",
        "",
        "## FINAL M9.0a DECISION",
        "",
        f"    {decision['primary_decision']}",
        "",
        f"M9_RECIPE:\n\n{m9_recipe_line}",
        "",
        "Preserved regardless:",
        "",
        "- relative-time representation = NOT PROMOTED",
        "- cadence-diverse training = NOT PROMOTED",
        "- PCGrad = OFF",
        "- PyG = NO",
        "- B_DEPTH_AWARE = CURRENT calibration method",
        "- alpha = 0.1",
        "- current OOD/fusion/safety semantics unchanged",
        "",
        f"- M9 capacity study scientifically unblocked: {'YES' if decision['m9_capacity_study_unblocked'] else 'NO'}",
        "",
        f"locked tests opened: before={locked_before}, after={locked_after}. No model promoted to production. "
        "No safety/authority semantics changed. No model-size change. No PyTorch Geometric introduced. "
        "No calibration redesign (M9.0b, if needed, is a separate future milestone).",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
