from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hydrocore_v5"))

import run_m11_5_full_validation as m115  # noqa: E402


def _passing_software() -> dict:
    return {"kind": "M11_5_SOFTWARE_GATES", "all_required_pass": True, "gates": []}


def test_m11_5_preflight_preserves_frozen_parent_and_refuses_post_m11_6_rerun(monkeypatch) -> None:
    """Historical M11.5 PASS evidence stays immutable after M11.6 opens the lock.

    The current preflight must therefore refuse any M11.5 rerun rather than
    reinterpret the historical M11.5 closure as the repository's current state.
    """
    # The helper normally reads the historical pre-lock freeze fixture. Model
    # the already-recorded terminal M11.6 opening here without changing that
    # fixture or any production/preflight implementation.
    monkeypatch.setattr(m115, "locked_test_opened", lambda _root: True)
    preflight = m115.preflight()
    assert preflight["checks"]["m10_complete"] is True
    assert preflight["checks"]["m11_1_selected"] is True
    assert preflight["checks"]["m11_2_frozen"] is True
    assert preflight["checks"]["finalist_identity"] is True
    # The repository is now post-lock: the historical M11.5 current flags and
    # unopened-lock condition are intentionally no longer true.
    assert preflight["checks"]["current_flags"] is False
    assert preflight["checks"]["locked_unopened"] is False
    assert preflight["checks"]["next_authorized"] is False
    assert preflight["all_checks_pass"] is False


def test_m11_5_matrix_covers_all_required_domains_without_fake_m11_3_or_m11_4() -> None:
    definition = m115.matrix_definition()
    assert [row["row_id"] for row in definition["rows"]] == list("ABCDEFGHIJKLMN")
    assert definition["m11_3_m11_4"] == "UNUSED_RESERVED_SUBSUMED_BY_M10_M11_2"
    assert definition["rows"][-2]["hard_gating"] is False


def test_m11_5_closed_evidence_has_the_required_frozen_gate_outcomes() -> None:
    results = m115.reused_results()
    assert all(results[key]["pass"] for key in ("predictive", "calibration", "robustness", "ood", "scout", "planning", "end_to_end", "fail_closed"))
    assert all(value == 0 for value in results["safety"].values())


def test_m11_5_artifact_synthesis_requires_every_hard_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m115, "preflight", lambda: {"all_checks_pass": True})
    records = m115.build_artifacts(_passing_software(), tmp_path)
    matrix = records["m11-5-matrix.json"]
    readiness = records["m11-5-readiness.json"]
    assert matrix["matrix_green"] is True
    assert all(row["finalist_identity_verified"] for row in matrix["rows"])
    assert readiness["m11_6_preconditions_satisfied"] is True
    assert readiness["locked_evaluation_authorized"] is False
    assert records["m11-5-closure.json"]["closure_state"] == "M11_5_FULL_VALIDATION_PASS"
