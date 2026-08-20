"""M10.5 blocked-serving-freeze governance tests.

Selection identity is a hard prerequisite: these tests deliberately prove the
blocker and do not construct a v5 serving bundle or inspect locked data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_m10_5_selection_preflight as m105  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402


def test_m10_4_parent_pass_and_protocol_are_required() -> None:
    audit = m105.selection_audit()
    assert audit["m10_4_parent_pass"] is True


def test_all_three_canonical_final_step_candidates_are_hash_verified() -> None:
    audit = m105.selection_audit()
    assert audit["candidate_count"] == 3
    assert audit["per_seed_export_policy"] == "FINAL_STEP_1350"
    assert audit["per_seed_export_policy_verified"] is True
    assert all(row["matches_expected"] for row in audit["candidates"])


def test_per_seed_export_rule_does_not_become_deployment_selection_rule() -> None:
    audit = m105.selection_audit()
    assert audit["selection_resolved"] is False
    assert audit["selection_rule"] is None
    assert audit["explicit_selector_token_hits"] == {}
    assert audit["m10_4_performance_used"] is False


def test_next_step_is_not_m9_6_supervised_and_is_not_release_authorized() -> None:
    audit = m105.output_governance_audit()
    assert "next_step" not in audit["known_supervised_m9_6_outputs"]
    assert audit["next_step_disposition"] == "MUST_BE_SUPPRESSED_IN_ANY_FUTURE_V5_RELEASE"


def test_closure_uses_required_selection_identity_blocker() -> None:
    audit = m105.selection_audit()
    preflight = {"result": "M10_5_PREFLIGHT_BLOCKED_SELECTION_IDENTITY"}
    closure = m105.closure_for(preflight, audit)
    assert closure["closure_state"] == "M10_5_SERVING_FREEZE_BLOCKED_SELECTION_IDENTITY"
    assert closure["m10_5_complete"] is False


def test_locked_test_guard() -> None:
    assert locked_test_opened(ROOT) is False
