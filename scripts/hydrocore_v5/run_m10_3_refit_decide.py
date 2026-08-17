"""M10.3A Strategist refit decision: reads the real, already-produced
Level-A gradient-coverage + gate artifacts and applies the frozen
representation-sufficiency rule
(`docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md` Part 7,
`m10_3_refit_protocol.py`). No gate THRESHOLD is chosen or changed here.

Additionally applies the authorizing task's own explicit exclusion list for
the Level-B trigger ("Do NOT escalate to B because of: ... severe target
imbalance invalidating metrics ... treat as implementation/data blocker").
This is a governance-level classification ABOVE the frozen mechanical gate
script, not a retroactive change to any gate threshold: a metric is
classified `TARGET_IMBALANCE_DEGENERATE` only via a mechanically-observable,
code-traceable fact already present in the gate artifact --
`baseline_mse < DEGENERATE_BASELINE_MSE_EPSILON` (the constant-train-mean
baseline already achieves near-perfect MSE because the target has
near-zero variance in this population) OR the computed Spearman correlation
is NaN (mathematically undefined for a zero-variance vector). Neither
condition is a judgment call made after "feeling like" a result is
inconvenient -- both are the SAME numbers already written to
`m10-3-refit-level-a-gate.json` before this script ever ran, re-read here
under a classification rule that could equally have been (and was, in the
frozen protocol document) written down before training.

If Level A passes for all three seeds: STOPS at `M10_3_STRATEGIST_REFIT_A_ACCEPTED`.
If Level A is mechanically valid but competence fails ONLY on
target-imbalance-degenerate metrics (across all seeds): reports
`M10_3_STRATEGIST_REFIT_BLOCKED_DATA_OR_SCHEMA` (never Level B). If at
least one GENUINE (non-degenerate) competence failure remains: reports the
Level-B escalation trigger. If Level A shows a genuine implementation/data
defect (gradient coverage, support, or finiteness failure): reports
`M10_3_STRATEGIST_REFIT_BLOCKED_DATA_OR_SCHEMA` directly.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-closure.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m10_3_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402

M10_3_REFIT_DIR = m10.M10_DIR / "m10-3-refit"

#: A constant-baseline MSE this small means the target has (numerically)
#: zero variance in the validation population -- "beat the baseline" is not
#: a meaningful competence test when the baseline already achieves
#: near-perfect MSE trivially. Chosen as a fixed, generous numerical-noise
#: floor (not tuned to make any particular seed pass/fail); the two
#: metrics this actually applies to in practice (`pressure_risk_proxy`
#: baseline_mse=0.0 exactly, `service_loss_proxy` baseline_mse=3.6e-23)
#: are both many orders of magnitude below it.
DEGENERATE_BASELINE_MSE_EPSILON = 1e-6

REGRESSION_METRIC_NAMES = (
    "plan_value", "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy",
    "containment_time_proxy", "plan_regret_proxy",
)


def _classify_seed_failures(metrics: dict[str, Any]) -> dict[str, str]:
    """Per-metric classification for every FAILING criterion in one seed's
    gate metrics: `"TARGET_IMBALANCE_DEGENERATE"` or `"GENUINE_COMPETENCE_GAP"`.
    Only failing metrics are classified; passing metrics are omitted."""

    classification: dict[str, str] = {}
    if not metrics["plan_validity"]["criterion_passed"]:
        classification["plan_validity"] = "GENUINE_COMPETENCE_GAP"
    for name in REGRESSION_METRIC_NAMES:
        entry = metrics[name]
        if entry.get("criterion_passed"):
            continue
        spearman = entry.get("spearman")
        baseline_mse = entry.get("baseline_mse")
        degenerate = (
            spearman is None or (isinstance(spearman, float) and math.isnan(spearman))
            or (baseline_mse is not None and baseline_mse < DEGENERATE_BASELINE_MSE_EPSILON)
        )
        classification[name] = "TARGET_IMBALANCE_DEGENERATE" if degenerate else "GENUINE_COMPETENCE_GAP"
    if not metrics["pairwise_ranking"]["criterion_passed"]:
        classification["pairwise_ranking"] = "GENUINE_COMPETENCE_GAP"
    return classification


def main() -> None:
    locked_before = m10.assert_locked_test_closed()

    level_a_training = json.loads((M10_3_REFIT_DIR / "m10-3-refit-level-a.json").read_text())
    gate = json.loads((M10_3_REFIT_DIR / "m10-3-refit-level-a-gate.json").read_text())

    per_seed_summary: dict[str, Any] = {}
    mechanically_valid_all_seeds = True
    competence_passed_all_seeds = True
    any_genuine_competence_gap = False
    all_failures_are_degenerate = True
    for seed_key, entry in gate["per_seed"].items():
        gradient_ok = level_a_training["per_seed"][seed_key]["gradient_coverage_passed"]
        support_ok = entry["support_ok"]
        finite_ok = entry["all_finite"]
        mechanically_valid = gradient_ok and support_ok and finite_ok
        competence_passed = entry["gate_criteria_passed"]
        mechanically_valid_all_seeds = mechanically_valid_all_seeds and mechanically_valid
        competence_passed_all_seeds = competence_passed_all_seeds and competence_passed

        failure_classification = _classify_seed_failures(entry["metrics"])
        if any(v == "GENUINE_COMPETENCE_GAP" for v in failure_classification.values()):
            any_genuine_competence_gap = True
        if failure_classification and not all(v == "TARGET_IMBALANCE_DEGENERATE" for v in failure_classification.values()):
            all_failures_are_degenerate = False
        if not failure_classification:
            all_failures_are_degenerate = False  # no failures at all for this seed -- vacuous, tracked separately

        per_seed_summary[seed_key] = {
            "gradient_coverage_passed": gradient_ok,
            "support_ok": support_ok,
            "all_finite": finite_ok,
            "mechanically_valid": mechanically_valid,
            "competence_gate_passed": competence_passed,
            "failing_metric_classification": failure_classification,
        }

    if mechanically_valid_all_seeds and competence_passed_all_seeds:
        result = "M10_3_STRATEGIST_REFIT_A_ACCEPTED"
        reason = (
            "Level A is mechanically valid and passes the frozen representation-sufficiency gate "
            "(gradient coverage, support, plan_validity AUROC, plan_value/all five proxies' MSE+Spearman, "
            "and within-incident pairwise ranking) for all three seeds. Level B is NOT run merely because "
            "it is authorized, per the frozen protocol."
        )
        level_b_triggered = False
    elif not mechanically_valid_all_seeds:
        result = "M10_3_STRATEGIST_REFIT_BLOCKED_DATA_OR_SCHEMA"
        reason = (
            "Level A fails a mechanical-validity criterion (gradient coverage, support, or finiteness) for at "
            "least one seed. This is an implementation/data defect, not a representation-capacity finding, and "
            "must not be mislabeled 'full retrain required' or used to trigger Level B."
        )
        level_b_triggered = False
    elif all_failures_are_degenerate and not any_genuine_competence_gap:
        result = "M10_3_STRATEGIST_REFIT_BLOCKED_DATA_OR_SCHEMA"
        reason = (
            "Level A is mechanically valid, and every failing competence criterion across all three seeds is "
            "TARGET_IMBALANCE_DEGENERATE (baseline_mse < 1e-6 and/or Spearman undefined -- the target has "
            "near-zero variance in this validation population, so 'beat the constant baseline' is not a "
            "meaningful competence test). Per the authorizing task's own explicit exclusion ('severe target "
            "imbalance invalidating metrics' must never trigger Level B), this is reported as an "
            "implementation/data blocker, never a representation-capacity finding."
        )
        level_b_triggered = False
    else:
        result = "M10_3_LEVEL_B_ESCALATION_TRIGGERED"
        reason = (
            "Level A is mechanically valid (gradient coverage/support/finiteness all pass) for every seed. At "
            "least one competence criterion fails in a way classified GENUINE_COMPETENCE_GAP (not target-"
            "imbalance-degenerate, not a bug/leakage/masking/schema/optimizer defect this audit can identify). "
            "Per the frozen protocol, this legitimately fires the Level-B escalation trigger."
        )
        level_b_triggered = True

    locked_after = m10.assert_locked_test_closed()
    closure = {
        "kind": "M10_3_REFIT_CLOSURE",
        "milestone": "M10.3A-refit",
        "branch": m10.current_branch(),
        "commit": m10.current_commit(),
        "protocol_hash": proto.protocol_hash(),
        "parent_m9_6_teacher_sha256": proto.PARENT_M9_6_TEACHER_SHA256,
        "per_seed_summary": per_seed_summary,
        "mechanically_valid_all_seeds": mechanically_valid_all_seeds,
        "competence_gate_passed_all_seeds": competence_passed_all_seeds,
        "all_failures_are_target_imbalance_degenerate": all_failures_are_degenerate and not competence_passed_all_seeds,
        "any_genuine_competence_gap": any_genuine_competence_gap,
        "level_b_triggered": level_b_triggered,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "M10_3_REFIT_RESULT": result,
        "decision_reason": reason,
        "next_recommended": (
            "The true M10.3 learned-vs-deterministic Strategist scientific comparison may now be separately "
            "authorized, using the accepted Level-A refit checkpoints recorded here (never the original M9.6 "
            "checkpoints, whose candidate-conditioned pathway remains untrained) -- NOT executed by this task."
            if result == "M10_3_STRATEGIST_REFIT_A_ACCEPTED"
            else "Level B (the one predeclared, bounded partial backbone[3]+final_norm unfreeze) is the next "
                 "step, executed later in this same task under the already-frozen Level-B protocol section."
            if level_b_triggered
            else "Fix the identified implementation/data/population-design limitation; do not change the "
                 "frozen scientific design after inspecting this result."
        ),
    }
    (M10_3_REFIT_DIR / "m10-3-refit-closure.json").write_text(json.dumps(closure, indent=2, default=str) + "\n")
    print(json.dumps(closure, indent=2, default=str))


if __name__ == "__main__":
    main()
