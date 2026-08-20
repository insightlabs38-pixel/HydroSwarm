"""TRUE Milestone 10.2 decision: reads `run_m10_2_true_evaluation.py`'s
paired trajectory output, computes the frozen metrics/statistics, and
applies the frozen promotion rule (`docs/evaluation/
HYDROCORE_V5_M10_2_TRUE_EVALUATION_PROTOCOL.md` Section 9,
`m10_2_true_protocol.py` Section F). No inference, no scenario generation,
no threshold is chosen or changed here -- every threshold was frozen before
any of these numbers existed.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-aggregate-metrics.json
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-statistics.json
  reports/evaluation/hydrocore-v5/m10/m10-2/m10-2-closure.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import m10_2_true_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402

M10_2_TRUE_DIR = m10.M10_DIR / "m10-2"


def _load_records(seed: int) -> list[dict[str, Any]]:
    path = M10_2_TRUE_DIR / f"m10-2-trajectories-seed{seed}.jsonl"
    records = []
    with path.open() as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def _paired_bootstrap_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(proto.BOOTSTRAP_SEED)
    n = len(a)
    diffs = np.empty(proto.BOOTSTRAP_RESAMPLES)
    for i in range(proto.BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(a[idx]) - np.mean(b[idx]))
    tail = (1.0 - proto.BOOTSTRAP_CI) / 2.0
    return float(np.percentile(diffs, tail * 100)), float(np.percentile(diffs, (1.0 - tail) * 100))


def _samples_to_actionability(arm: dict[str, Any]) -> float | None:
    resolved = arm["resolved_at_step"]
    return float(resolved) if resolved is not None else None


def _never_actionable(arm: dict[str, Any]) -> bool:
    return arm["resolved_at_step"] is None


def _final_top1(arm: dict[str, Any]) -> bool:
    return bool(arm["rounds"][-1]["top1"])


def _actionable_within_budget(arm: dict[str, Any]) -> bool:
    return arm["resolved_at_step"] is not None and arm["resolved_at_step"] <= proto.MAXIMUM_SAMPLES


def _false_stop(arm_l: dict[str, Any], arm_d: dict[str, Any]) -> bool:
    stopped_early = (
        arm_l["voluntary_stop"]
        and arm_l["resolved_at_step"] is None
        and arm_l["final_samples_taken"] < proto.MAXIMUM_SAMPLES
    )
    if not stopped_early:
        return False
    return arm_d["resolved_at_step"] is not None and arm_d["resolved_at_step"] <= proto.MAXIMUM_SAMPLES


def _unnecessary_sampling(arm_l: dict[str, Any]) -> bool:
    resolved = arm_l["resolved_at_step"]
    if resolved is None:
        return False
    return arm_l["final_samples_taken"] > resolved


def compute_seed_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    d_actionable = np.array([_actionable_within_budget(r["arm_D"]) for r in records], dtype=float)
    l_actionable = np.array([_actionable_within_budget(r["arm_L"]) for r in records], dtype=float)
    d_never = np.array([_never_actionable(r["arm_D"]) for r in records], dtype=float)
    l_never = np.array([_never_actionable(r["arm_L"]) for r in records], dtype=float)
    d_top1_final = np.array([_final_top1(r["arm_D"]) for r in records], dtype=float)
    l_top1_final = np.array([_final_top1(r["arm_L"]) for r in records], dtype=float)

    actionable_lower, actionable_upper = _paired_bootstrap_ci(l_actionable, d_actionable)
    never_lower, never_upper = _paired_bootstrap_ci(l_never, d_never)
    top1_lower, top1_upper = _paired_bootstrap_ci(l_top1_final, d_top1_final)

    d_samples = [_samples_to_actionability(r["arm_D"]) for r in records]
    l_samples = [_samples_to_actionability(r["arm_L"]) for r in records]
    both_resolved = [
        (d_step, l_step) for d_step, l_step in zip(d_samples, l_samples) if d_step is not None and l_step is not None
    ]

    false_stop = [_false_stop(r["arm_L"], r["arm_D"]) for r in records]
    unnecessary = [_unnecessary_sampling(r["arm_L"]) for r in records]
    d_budget_exhausted = [r["arm_D"]["final_samples_taken"] >= proto.MAXIMUM_SAMPLES and not r["arm_D"]["voluntary_stop"] for r in records]
    l_budget_exhausted = [r["arm_L"]["final_samples_taken"] >= proto.MAXIMUM_SAMPLES and not r["arm_L"]["voluntary_stop"] for r in records]

    per_round: dict[int, dict[str, Any]] = {}
    for step in range(proto.MAXIMUM_SAMPLES + 1):
        d_top1_r, d_top3_r, l_top1_r, l_top3_r, d_gate, l_gate = [], [], [], [], [], []
        for r in records:
            for arm_key, top1_list, top3_list, gate_list in (
                ("arm_D", d_top1_r, d_top3_r, d_gate), ("arm_L", l_top1_r, l_top3_r, l_gate),
            ):
                rounds = r[arm_key]["rounds"]
                if step < len(rounds):
                    top1_list.append(rounds[step]["top1"])
                    top3_list.append(rounds[step]["top3"])
                    gate_list.append(rounds[step]["candidate_gate_pass"])
        per_round[step] = {
            "n_with_round": len(d_top1_r),
            "arm_D": {"top1": float(np.mean(d_top1_r)) if d_top1_r else None, "top3": float(np.mean(d_top3_r)) if d_top3_r else None, "candidate_gate_pass": float(np.mean(d_gate)) if d_gate else None},
            "arm_L": {"top1": float(np.mean(l_top1_r)) if l_top1_r else None, "top3": float(np.mean(l_top3_r)) if l_top3_r else None, "candidate_gate_pass": float(np.mean(l_gate)) if l_gate else None},
        }

    return {
        "n": len(records),
        "actionable_within_budget": {
            "arm_D": float(d_actionable.mean()), "arm_L": float(l_actionable.mean()),
            "diff_point_estimate": float(l_actionable.mean() - d_actionable.mean()),
            "diff_ci_lower": actionable_lower, "diff_ci_upper": actionable_upper,
        },
        "never_actionable_fraction": {
            "arm_D": float(d_never.mean()), "arm_L": float(l_never.mean()),
            "diff_point_estimate": float(l_never.mean() - d_never.mean()),
            "diff_ci_lower": never_lower, "diff_ci_upper": never_upper,
        },
        "source_top1_final_round": {
            "arm_D": float(d_top1_final.mean()), "arm_L": float(l_top1_final.mean()),
            "diff_point_estimate": float(l_top1_final.mean() - d_top1_final.mean()),
            "diff_ci_lower": top1_lower, "diff_ci_upper": top1_upper,
        },
        "samples_to_actionability_both_resolved": {
            "n": len(both_resolved),
            "arm_D_mean": float(np.mean([d for d, _ in both_resolved])) if both_resolved else None,
            "arm_L_mean": float(np.mean([l_step for _, l_step in both_resolved])) if both_resolved else None,
        },
        "stopping_quality": {
            "false_stop_rate": float(np.mean(false_stop)),
            "unnecessary_sampling_rate": float(np.mean(unnecessary)),
            "budget_exhaustion_rate_arm_D": float(np.mean(d_budget_exhausted)),
            "budget_exhaustion_rate_arm_L": float(np.mean(l_budget_exhausted)),
        },
        "per_round": per_round,
    }


def apply_promotion_rule(safety_audit: dict[str, Any], per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    hard_gates_passed = bool(safety_audit["all_hard_gates_passed"])

    primary_positive_seeds = sum(
        1 for seed in m10.SEEDS if per_seed[seed]["actionable_within_budget"]["diff_point_estimate"] > 0.0
    )
    primary_ci_excludes_zero_seeds = sum(
        1 for seed in m10.SEEDS if per_seed[seed]["actionable_within_budget"]["diff_ci_lower"] > 0.0
    )
    criterion_2 = (
        primary_positive_seeds >= proto.PROMOTION_MIN_SEEDS_POSITIVE_POINT_ESTIMATE
        and primary_ci_excludes_zero_seeds >= proto.PROMOTION_MIN_SEEDS_CI_EXCLUDES_ZERO
    )

    regressions: dict[str, list[int]] = {metric: [] for metric in proto.NO_REGRESSION_METRICS}
    for seed in m10.SEEDS:
        metrics = per_seed[seed]
        # never_actionable_fraction: regression = learned WORSE (higher), CI-confident if diff_ci_lower > 0.
        if metrics["never_actionable_fraction"]["diff_ci_lower"] > 0.0:
            regressions["never_actionable_fraction"].append(seed)
        # source_top1_final_round: regression = learned WORSE (lower), CI-confident if diff_ci_upper < 0.
        if metrics["source_top1_final_round"]["diff_ci_upper"] < 0.0:
            regressions["source_top1_final_round"].append(seed)
    criterion_3 = all(len(seeds) == 0 for seeds in regressions.values())

    if not hard_gates_passed:
        result = "M10_2_SCIENTIFIC_EVALUATION_BLOCKED"
        reason = "one or more mandatory safety/governance hard gates failed"
    elif criterion_2 and criterion_3:
        result = "M10_2_LEARNED_SCOUT_PROMOTION_SUPPORTED"
        reason = (
            f"primary metric positive in {primary_positive_seeds}/3 seeds, CI excludes zero in "
            f"{primary_ci_excludes_zero_seeds}/3 seeds (>= required {proto.PROMOTION_MIN_SEEDS_CI_EXCLUDES_ZERO}); "
            "no CI-confident regression in any no-regression metric"
        )
    else:
        result = "M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED"
        reasons = []
        if not criterion_2:
            reasons.append(
                f"primary metric consistency bar not met (positive point estimate in {primary_positive_seeds}/3 "
                f"seeds, required {proto.PROMOTION_MIN_SEEDS_POSITIVE_POINT_ESTIMATE}; CI excludes zero in "
                f"{primary_ci_excludes_zero_seeds}/3 seeds, required {proto.PROMOTION_MIN_SEEDS_CI_EXCLUDES_ZERO})"
            )
        if not criterion_3:
            reasons.append(f"CI-confident regression detected: {regressions}")
        reason = "; ".join(reasons)

    return {
        "hard_gates_passed": hard_gates_passed,
        "primary_metric_positive_seeds": primary_positive_seeds,
        "primary_metric_ci_excludes_zero_seeds": primary_ci_excludes_zero_seeds,
        "criterion_2_consistency_passed": criterion_2,
        "regressions": regressions,
        "criterion_3_no_regression_passed": criterion_3,
        "result": result,
        "reason": reason,
    }


def main() -> None:
    locked_before = m10.assert_locked_test_closed()

    safety_audit = json.loads((M10_2_TRUE_DIR / "m10-2-safety-audit.json").read_text())
    checkpoint_verification = json.loads((M10_2_TRUE_DIR / "m10-2-checkpoint-verification.json").read_text())

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in m10.SEEDS:
        records = _load_records(seed)
        per_seed[seed] = compute_seed_metrics(records)

    aggregate = {
        "kind": "M10_2_TRUE_AGGREGATE_METRICS",
        "protocol_hash": proto.protocol_hash(),
        "per_seed": {str(seed): metrics for seed, metrics in per_seed.items()},
    }
    (M10_2_TRUE_DIR / "m10-2-aggregate-metrics.json").write_text(json.dumps(aggregate, indent=2, default=str) + "\n")

    decision = apply_promotion_rule(safety_audit, per_seed)
    statistics_doc = {
        "kind": "M10_2_TRUE_STATISTICS",
        "protocol_hash": proto.protocol_hash(),
        "bootstrap_resamples": proto.BOOTSTRAP_RESAMPLES, "bootstrap_ci": proto.BOOTSTRAP_CI,
        "bootstrap_seed": proto.BOOTSTRAP_SEED,
        "promotion_decision": decision,
    }
    (M10_2_TRUE_DIR / "m10-2-statistics.json").write_text(json.dumps(statistics_doc, indent=2, default=str) + "\n")

    locked_after = m10.assert_locked_test_closed()
    closure = {
        "kind": "M10_2_TRUE_CLOSURE",
        "milestone": "M10.2-true",
        "branch": m10.current_branch(),
        "commit": m10.current_commit(),
        "protocol_hash": proto.protocol_hash(),
        "level_a_refit_checkpoint_sha256": proto.LEVEL_A_REFIT_CHECKPOINT_SHA256,
        "parent_m9_6_teacher_sha256_unchanged": checkpoint_verification["teacher_checkpoints_unchanged"],
        "same_refit_predictor_used_in_both_arms": True,
        "original_m9_6_scout_checkpoint_used_as_learned_policy": False,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "learned_scout_runtime_authority_unchanged": True,
        "deterministic_scout_fallback_preserved": True,
        "learned_ood_not_promoted": True,
        "wntr_epanet_authority_unchanged": True,
        "hard_gates_passed": decision["hard_gates_passed"],
        "safety_audit_counters": safety_audit["counters"],
        "M10_2_RESULT": decision["result"],
        "decision_reason": decision["reason"],
        "next_recommended": (
            "None -- this milestone establishes scientific promotion eligibility only. Runtime "
            "promotion (enabling learned Scout in production) is a separate, later, explicitly "
            "authorized milestone regardless of this result. M10.3 (Strategist) requires its own "
            "separately authorized supervision/candidate-schema amendment before any scientific "
            "evaluation, per HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md."
        ),
    }
    (M10_2_TRUE_DIR / "m10-2-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")
    print(json.dumps(closure, indent=2, default=str))


if __name__ == "__main__":
    main()
