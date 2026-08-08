"""core-issues5.txt Section 5 (P0 GATE) / delta item 1: consolidated
train/serve parity gate regression coverage.

Runs the real `scripts/run_train_serve_parity_gate.py` gate (not a mock)
against the real committed `data/learning-v2/cycle-b2/normalization` and
`data/learning-v2/cycle-b2/signatures` artifacts across every governed
training topology and more than one operating/fault condition each -- the
same gate a CI/pre-freeze run would execute.

Delta item 1 fixed the classical_prior algorithm mismatch this gate used
to document as a known, accepted failure (governed corpus generation and
live serving previously used two structurally different posterior
algorithms). This gate must now PASS outright, with no accepted
classical_prior failure.
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
        cycle_b2_root=_REPO_ROOT / "data" / "learning-v2" / "cycle-b2",
        seed_base=920_000,
    )


def test_gate_report_passes_overall() -> None:
    report = _run()
    failures = [field for field in report["fields"] if not field["passed"]]
    assert not failures, failures
    assert report["passed"] is True


def test_every_behavior_critical_field_matches_exactly_or_within_tolerance() -> None:
    report = _run()
    by_field: dict[str, list[dict]] = {}
    for field in report["fields"]:
        by_field.setdefault(field["field"], []).append(field)
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
        "node_features",
        "classical_prior",
        "signature_policy_identity",
        "signature_mode_is_governed",
    ):
        assert name in by_field, f"missing field {name!r} from gate report"
        for entry in by_field[name]:
            assert entry["passed"] is True, entry["detail"]


def test_classical_prior_matches_to_float_precision() -> None:
    """The formerly-documented known algorithm mismatch is now a real,
    tight-tolerance match -- both paths compute classical_prior via the
    identical hydroswarm.training.corpus.model_input_classical_prior
    function against the same committed signature library."""

    report = _run()
    prior_fields = [field for field in report["fields"] if field["field"] == "classical_prior"]
    assert prior_fields
    for entry in prior_fields:
        assert entry["passed"] is True
        assert entry["comparison"] == "tolerance"


def test_covers_every_governed_training_topology_and_more_than_one_condition() -> None:
    report = _run()
    families = {fixture["network_family"] for fixture in report["evaluated_fixtures"]}
    conditions = {fixture["condition"] for fixture in report["evaluated_fixtures"]}
    assert families == {"golden-reference", "branched-loop", "loop-grid"}
    assert len(conditions) > 1
    assert all(fixture["passed"] for fixture in report["evaluated_fixtures"])


def test_gate_exits_zero(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    exit_code = gate.main([
        "--report-output", str(report_path),
        "--seed-base", "930000",
    ])
    assert exit_code == 0


def test_gate_fails_closed_on_a_real_regression(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    original_run_gate = gate.run_gate

    def _broken_run_gate(**kwargs):
        report = original_run_gate(**kwargs)
        report["fields"].append({
            "field": "normalization_hash",
            "comparison": "exact",
            "passed": False,
            "detail": "synthetic regression for this test",
            "scenario_id": "synthetic",
        })
        report["passed"] = False
        return report

    monkeypatch.setattr(gate, "run_gate", _broken_run_gate)
    exit_code = gate.main([
        "--report-output", str(report_path),
        "--seed-base", "940000",
    ])
    assert exit_code == 1
