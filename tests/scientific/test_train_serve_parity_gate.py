"""core-issues5.txt Section 5 (P0 GATE): consolidated train/serve parity
gate regression coverage.

Runs the real `scripts/run_train_serve_parity_gate.py` gate (not a mock)
against the real committed `data/learning-v2/cycle-b2/normalization`
artifact and a small, self-generated, non-locked fixture scenario -- the
same gate a CI/pre-freeze run would execute.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_train_serve_parity_gate as gate  # noqa: E402


def _run() -> dict:
    return gate.run_gate(
        normalization_dir=_REPO_ROOT / "data" / "learning-v2" / "cycle-b2" / "normalization",
        seed_base=920_000,
    )


def test_schema_and_normalization_and_mask_fields_match_exactly() -> None:
    report = _run()
    by_field = {field["field"]: field for field in report["fields"]}
    for name in (
        "node_ids",
        "feature_schema_hash",
        "normalization_hash",
        "node_features_shape",
        "edge_features_shape",
        "node_mask",
        "edge_mask",
        "sensor_mask",
        "source_candidate_mask",
        "edge_features",
    ):
        assert by_field[name]["passed"] is True, by_field[name]["detail"]
        assert by_field[name]["known_finding"] is False


def test_classical_prior_mismatch_is_a_known_documented_finding() -> None:
    """This must keep failing until the underlying algorithm mismatch is
    deliberately resolved -- a silent pass here would hide a real defect,
    not fix one."""

    report = _run()
    by_field = {field["field"]: field for field in report["fields"]}
    assert by_field["classical_prior"]["passed"] is False
    assert by_field["classical_prior"]["known_finding"] is True
    assert by_field["node_features"]["passed"] is False
    assert by_field["node_features"]["known_finding"] is True
    assert "classical_source_prior" in by_field["node_features"]["detail"]


def test_gate_report_is_not_passed_overall() -> None:
    report = _run()
    assert report["passed"] is False


def test_gate_exits_nonzero_by_default_and_zero_with_the_known_mismatch_flag(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    strict_exit = gate.main([
        "--report-output", str(report_path),
        "--seed-base", "930000",
    ])
    assert strict_exit == 1

    lenient_exit = gate.main([
        "--report-output", str(report_path),
        "--seed-base", "930000",
        "--allow-known-classical-prior-mismatch",
    ])
    assert lenient_exit == 0


def test_a_real_unrelated_regression_still_fails_even_with_the_lenient_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """The lenient flag must only ever whitelist the one documented
    classical_prior/node_features finding -- never become a blanket
    override that silently accepts an unrelated new defect."""

    report_path = tmp_path / "report.json"
    original_run_gate = gate.run_gate

    def _broken_run_gate(**kwargs):
        report = original_run_gate(**kwargs)
        report["fields"].append({
            "field": "normalization_hash",
            "comparison": "exact",
            "passed": False,
            "detail": "synthetic regression for this test",
            "known_finding": False,
        })
        report["passed"] = False
        return report

    monkeypatch.setattr(gate, "run_gate", _broken_run_gate)
    exit_code = gate.main([
        "--report-output", str(report_path),
        "--seed-base", "940000",
        "--allow-known-classical-prior-mismatch",
    ])
    assert exit_code == 1
