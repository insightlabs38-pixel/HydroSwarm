"""Milestone 9.1 closure: final selection (frozen protocol
`docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, Section 12's final-selection
paragraph, clarified by the 2026-08-16 Section-21(b) addendum: "the arm with
the LARGEST 3-seed paired-bootstrap point estimate" means the Step 3
(seed-20260815 confirmation) bootstrap's own point estimate, exclusively --
not a pooled-across-seeds statistic) and Section 13's predeclared outcomes.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_closure.py

Reads (never regenerates):
  reports/evaluation/hydrocore-v5/m9-1-guardrails.json

Writes:
  reports/evaluation/hydrocore-v5/m9-1-summary.md
  reports/evaluation/hydrocore-v5/m9-1-closure.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m9_1_common as common  # noqa: E402

#: Section 1's own listing order -- the final, all-else-tied tie-break.
ARM_ORDER = ("GRAPH_ODE", "GRAPH_CDE", "GRAPH_SDE")


def _select_winner(confirmed: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    if not confirmed:
        return None, "zero arms reached PROMOTION_CONFIRMED"
    if len(confirmed) == 1:
        arm = next(iter(confirmed))
        return arm, "exactly one arm reached PROMOTION_CONFIRMED"

    def sort_key(arm: str) -> tuple[float, float, int]:
        step3 = confirmed[arm]["step3"]
        point_estimate = step3["observed_mean_diff"]
        ci_width = step3["ci_upper"] - step3["ci_lower"]
        # Larger point estimate first (negate for ascending sort), then
        # narrower CI first, then Section 1's fixed listing order.
        return (-point_estimate, ci_width, ARM_ORDER.index(arm))

    ranked = sorted(confirmed, key=sort_key)
    return ranked[0], f"tie-break among {len(confirmed)} PROMOTION_CONFIRMED arms by largest Step-3 point estimate"


def _per_arm_outcome(arm: str, screening: dict[str, Any], confirmation: dict[str, Any], winner: str | None) -> dict[str, Any]:
    screening_entry = screening.get(arm)
    if screening_entry is None:
        return {"status": "NOT_EVALUATED"}
    screening_outcome = screening_entry["outcome"]
    if screening_outcome == "GUARDRAILS_FAILED":
        return {"status": "GUARDRAILS_FAILED", "section_13_outcome": "C_GUARDRAILS_BLOCKED", "screening": screening_entry}
    if screening_outcome == "GUARDRAILS_PASSED_NO_SIGNIFICANT_GAIN":
        return {"status": screening_outcome, "section_13_outcome": "D_CONTRIBUTES_CURRENT_RETAINED", "screening": screening_entry}
    # screening_outcome == "PROMOTION_CANDIDATE" -> must have a confirmation entry.
    confirmation_entry = confirmation.get(arm)
    if confirmation_entry is None:
        return {"status": "PROMOTION_CANDIDATE_CONFIRMATION_NOT_RUN", "screening": screening_entry}
    confirmation_outcome = confirmation_entry["outcome"]
    if confirmation_outcome == "PROMOTION_NOT_CONFIRMED":
        return {
            "status": "PROMOTION_NOT_CONFIRMED", "section_13_outcome": "B_PARTIAL_GAIN_UNCERTAIN",
            "screening": screening_entry, "confirmation": confirmation_entry,
        }
    # PROMOTION_CONFIRMED.
    if arm == winner:
        return {
            "status": "PROMOTION_CONFIRMED_SELECTED", "section_13_outcome": "A_ARCHITECTURE_GAIN_VALIDATED",
            "screening": screening_entry, "confirmation": confirmation_entry,
        }
    return {
        "status": "PROMOTION_CONFIRMED_NOT_SELECTED", "section_13_outcome": "A_TIE_BREAK_NOT_SELECTED",
        "screening": screening_entry, "confirmation": confirmation_entry,
    }


def main() -> int:
    code_under_test_commit = common.assert_code_under_test_commit()
    locked_before = common.assert_locked_test_closed()

    if not common.GUARDRAILS_PATH.exists():
        raise SystemExit("missing m9-1-guardrails.json -- run run_m9_1_decide.py --stage screening first")
    guardrails = json.loads(common.GUARDRAILS_PATH.read_text())
    screening = guardrails.get("screening", {})
    confirmation = guardrails.get("confirmation", {})

    confirmed = {arm: entry for arm, entry in confirmation.items() if isinstance(entry, dict) and entry.get("outcome") == "PROMOTION_CONFIRMED"}
    winner, selection_reason = _select_winner(confirmed)

    per_arm = {arm: _per_arm_outcome(arm, screening, confirmation, winner) for arm in common.NOVEL_ARMS}
    final_decision = "ARCHITECTURE_GAIN_VALIDATED" if winner is not None else "CURRENT_HYDROCORE_RETAINED"

    locked_after = common.assert_locked_test_closed()

    closure = {
        "schema_version": 1,
        "protocol_frozen_at_commit": common.PROTOCOL_FROZEN_AT_COMMIT,
        "code_under_test_commit": code_under_test_commit,
        "M9_1_FINAL_DECISION": final_decision,
        "selected_arm": winner,
        "selection_reason": selection_reason,
        "per_arm_outcome": per_arm,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    common.CLOSURE_PATH.write_text(json.dumps(closure, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Milestone 9.1 summary: continuous-time temporal-dynamics architecture comparison",
        "",
        "Frozen protocol `docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, including its 2026-08-16 Section-21 addendum.",
        f"`protocol_frozen_at_commit`: `{common.PROTOCOL_FROZEN_AT_COMMIT}`",
        f"`code_under_test_commit`: `{code_under_test_commit}`",
        "",
        "## Per-arm outcome",
        "",
        "| arm | status | Section 13 outcome |",
        "|---|---|---|",
    ]
    for arm, entry in per_arm.items():
        lines.append(f"| {arm} | {entry['status']} | {entry.get('section_13_outcome', 'n/a')} |")
    lines += [
        "",
        f"**M9_1_FINAL_DECISION: {final_decision}**",
        "",
        f"Selected arm: {winner or 'NONE (CURRENT retained)'}",
        f"Selection reason: {selection_reason}",
        "",
        f"locked_test_opened: before={locked_before}, after={locked_after}. No M9 S/M/L capacity scaling begins under "
        "this document's authorization. No field-performance claim is made.",
    ]
    common.SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {common.SUMMARY_PATH} and {common.CLOSURE_PATH}")
    print(json.dumps({"M9_1_FINAL_DECISION": final_decision, "selected_arm": winner, "per_arm": {a: e["status"] for a, e in per_arm.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
