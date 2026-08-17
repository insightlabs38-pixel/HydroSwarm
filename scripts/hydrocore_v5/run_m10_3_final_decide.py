"""M10.3A Strategist refit -- FINAL decision, reading the real, already-
produced Level-A gate, Level-B gate, and Level-B M9-preservation artifacts.
Applies the frozen protocol's Level-B promotion rule mechanically: BOTH (A)
material Strategist-competence improvement over Level A AND (B) M9
preservation must pass. Supersedes the intermediate
`M10_3_LEVEL_B_ESCALATION_TRIGGERED` closure `run_m10_3_refit_decide.py`
wrote before Level B executed.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-closure.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m10_3_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402

M10_3_REFIT_DIR = m10.M10_DIR / "m10-3-refit"


def main() -> None:
    locked_before = m10.assert_locked_test_closed()

    level_a_gate = json.loads((M10_3_REFIT_DIR / "m10-3-refit-level-a-gate.json").read_text())
    level_b_gate = json.loads((M10_3_REFIT_DIR / "m10-3-refit-level-b-gate.json").read_text())
    level_b_training = json.loads((M10_3_REFIT_DIR / "m10-3-refit-level-b.json").read_text())
    preservation = json.loads((M10_3_REFIT_DIR / "m10-3-refit-preservation.json").read_text())

    per_seed: dict[str, Any] = {}
    level_b_mechanically_valid_all_seeds = True
    competence_criterion_a_passed_all_seeds = True
    preservation_criterion_b_passed_all_seeds = True
    for seed_key in level_b_gate["per_seed"]:
        gradient_ok = level_b_training["per_seed"][seed_key]["gradient_coverage_passed"]
        support_ok = level_b_gate["per_seed"][seed_key]["support_ok"]
        finite_ok = level_b_gate["per_seed"][seed_key]["all_finite"]
        mechanically_valid = gradient_ok and support_ok and finite_ok
        materially_improves = level_b_gate["per_seed"][seed_key]["materially_improves_over_level_a"]
        m9_preserved = preservation["per_seed"][seed_key]["m9_preservation_passed"]

        level_b_mechanically_valid_all_seeds = level_b_mechanically_valid_all_seeds and mechanically_valid
        competence_criterion_a_passed_all_seeds = competence_criterion_a_passed_all_seeds and materially_improves
        preservation_criterion_b_passed_all_seeds = preservation_criterion_b_passed_all_seeds and m9_preserved

        per_seed[seed_key] = {
            "level_b_gradient_coverage_passed": gradient_ok,
            "level_b_support_ok": support_ok,
            "level_b_all_finite": finite_ok,
            "level_b_mechanically_valid": mechanically_valid,
            "level_b_gate_criteria_passed": level_b_gate["per_seed"][seed_key]["gate_criteria_passed"],
            "criterion_a_materially_improves_over_level_a": materially_improves,
            "criterion_b_m9_preservation_passed": m9_preserved,
            "calibration_below_floor": preservation["per_seed"][seed_key]["calibration_below_floor"],
            "level_b_promotable": bool(materially_improves and m9_preserved),
        }

    level_b_promotable_all_seeds = all(entry["level_b_promotable"] for entry in per_seed.values())

    if level_b_promotable_all_seeds and level_b_mechanically_valid_all_seeds:
        result = "M10_3_STRATEGIST_REFIT_B_ACCEPTED"
        reason = (
            "Level B is mechanically valid, materially improves Strategist competence over Level A for every "
            "criterion Level A failed, AND preserves M9 Sentinel/calibration behavior (no CI-confident "
            "regression, calibration coverage at/above the 0.85 floor) for all three seeds."
        )
    else:
        result = "M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED"
        reasons = []
        if not preservation_criterion_b_passed_all_seeds:
            reasons.append(
                "Level B damages M9 preservation for at least one seed (calibration coverage drops below the "
                "frozen 0.85 floor and/or at least one of the nine Sentinel tasks shows a CI-confident paired-"
                "bootstrap regression against the unmodified M9.6 teacher on the same development population). "
                "Per the frozen protocol, Level B is rejected regardless of its own Strategist-competence result."
            )
        if not competence_criterion_a_passed_all_seeds:
            reasons.append(
                "Level B does not materially improve over Level A's own validation point estimates for every "
                "criterion Level A failed, for at least one seed."
            )
        reason = (
            "Level A showed genuine (non-target-imbalance-degenerate) Strategist-competence gaps "
            "(plan_value/exposure_proxy/plan_regret_proxy/pairwise-ranking discrimination). Level B (the one "
            "predeclared, bounded backbone[3]+final_norm unfreeze) was executed per the frozen escalation "
            "trigger. " + " ".join(reasons) +
            " Per the frozen protocol, broader/full joint retraining would be required to resolve this "
            "safely, which this task is not authorized to perform. Level A's own checkpoint (never Level B's) "
            "is retained as the task's best available, NOT-PROMOTED artifact."
        )

    locked_after = m10.assert_locked_test_closed()
    closure = {
        "kind": "M10_3_REFIT_CLOSURE",
        "milestone": "M10.3A-refit",
        "branch": m10.current_branch(),
        "commit": m10.current_commit(),
        "protocol_hash": proto.protocol_hash(),
        "parent_m9_6_teacher_sha256": proto.PARENT_M9_6_TEACHER_SHA256,
        "level_a_all_seeds_pass_competence_gate": level_a_gate["all_seeds_pass_competence_gate"],
        "level_b_triggered": True,
        "level_b_per_seed_summary": per_seed,
        "level_b_mechanically_valid_all_seeds": level_b_mechanically_valid_all_seeds,
        "level_b_materially_improves_all_seeds": competence_criterion_a_passed_all_seeds,
        "level_b_preserves_m9_all_seeds": preservation_criterion_b_passed_all_seeds,
        "level_b_promotable_all_seeds": level_b_promotable_all_seeds,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "M10_3_REFIT_RESULT": result,
        "decision_reason": reason,
        "best_available_checkpoint": (
            "Level B (accepted)" if result == "M10_3_STRATEGIST_REFIT_B_ACCEPTED"
            else "Level A (not promoted -- Level B rejected)"
        ),
        "next_recommended": (
            "The true M10.3 learned-vs-deterministic Strategist scientific comparison may now be separately "
            "authorized, using the accepted refit checkpoints recorded here -- NOT executed by this task."
            if result == "M10_3_STRATEGIST_REFIT_B_ACCEPTED"
            else "No Strategist refit checkpoint from this task is promoted for true M10.3 use. A future, "
                 "separately authorized amendment would need to determine whether broader shared-representation "
                 "retraining (out of this task's scope) is scientifically warranted, or whether the underlying "
                 "population's low target variance for plan_value/exposure_proxy (Part 4's own disclosed "
                 "finding) makes this objective inherently hard to learn regardless of capacity."
        ),
    }
    (M10_3_REFIT_DIR / "m10-3-refit-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")
    print(json.dumps(closure, indent=2, default=str))


if __name__ == "__main__":
    main()
