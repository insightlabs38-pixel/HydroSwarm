from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hydrocore_v5"))

import run_m11_1_finalist_selection as m11  # noqa: E402


def test_m11_1_preflight_verifies_authoritative_m10_identity() -> None:
    preflight = m11.preflight()
    assert preflight["all_checks_pass"] is True
    assert all(preflight["checks"].values())
    assert preflight["checks"]["m10_5_completion_pass"] is True
    assert preflight["checks"]["selected_checkpoint_identity"] is True
    assert preflight["checks"]["calibration_artifact_identity"] is True


def test_m11_1_candidate_eligibility_excludes_experimental_checkpoints() -> None:
    eligibility = m11.candidate_eligibility()
    assert [candidate["name"] for candidate in eligibility["eligible_candidates"]] == [
        "HydroCore-v5 M10 frozen release", "HydroCore-v4 frozen incumbent"
    ]
    excluded = " ".join(candidate["name"] for candidate in eligibility["excluded_candidates"])
    assert "HydroCore-M" in excluded
    assert "Scout-refit" in excluded
    assert "Strategist-refit" in excluded
    assert eligibility["experimental_checkpoint_promotion_forbidden"] is True


def test_m11_1_evidence_manifest_has_no_locked_source() -> None:
    evidence = m11.evidence_manifest()
    assert evidence["locked_source_count"] == 0
    assert all("locked_final_test" not in row["path"] for row in evidence["sources"])
    assert all("locked_topology_test" not in row["path"] for row in evidence["sources"])


def test_m11_1_artifacts_select_without_freezing_or_authorizing_locked_eval(tmp_path: Path) -> None:
    artifacts = m11.build_artifacts(tmp_path)
    selection = artifacts["final-selection.json"]
    assert selection["selected_finalist_system"] == "HydroCore-v5 M10 frozen release"
    assert selection["finalist_selected"] is True
    assert selection["finalist_frozen"] is False
    assert selection["locked_evaluation_authorized"] is False
    assert selection["locked_test_opened"] is False
    assert all(decision["eligible"] for decision in selection["candidate_eligibility_decisions"])
    assert artifacts["m11-1-closure.json"]["closure_state"] == "M11_1_FINALIST_SELECTED"
    assert (tmp_path / "m11-1-protocol.json").is_file()
    assert json.loads((tmp_path / "m11-1-protocol.json").read_text())["protocol_sha256"] == m11.sha256(m11.PROTOCOL_DOC)
