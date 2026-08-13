from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydroswarm.evaluation.robustness_scale import REQUIRED_ROW_FIELDS, aggregate, load_protocol


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {field: None for field in REQUIRED_ROW_FIELDS}
    row.update({
        "run_id": "row-1", "perturbation_level": "nominal", "planning_allowed": False,
        "control_action": "REQUEST_SAMPLE", "top1_correct": True, "top3_correct": True,
        "reciprocal_rank": 1.0, "candidate_set_size": 2, "posterior_entropy": 1.0,
        "inference_ms": 10.0,
    })
    row.update(overrides)
    return row


def test_protocol_is_locked_test_excluding_and_deterministic() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "reports/evaluation/robustness-scale/protocol.json"
    first = load_protocol(protocol_path)
    second = load_protocol(protocol_path)
    assert first == second
    assert first["random_seed"] == 20260813
    assert first["locked_test"]["excluded"] is True
    assert "test" not in first["populations"]


def test_protocol_rejects_locked_population(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({
        "locked_test": {"excluded": True, "forbidden_split_names": ["test"]},
        "populations": ["test"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="locked-test"):
        load_protocol(path)


def test_aggregate_is_null_safe_and_checks_authority_invariant() -> None:
    summary = aggregate([
        _row(),
        _row(run_id="row-2", top1_correct=None, reciprocal_rank=None, inference_ms=None),
        _row(run_id="row-3", planning_allowed=False, control_action="GENERATE_PLANS"),
    ])
    condition = summary["conditions"]["nominal"]
    assert condition["top1"] == pytest.approx(1.0)
    assert condition["mrr"] == pytest.approx(1.0)
    assert summary["authority_invariant_failures"] == ["row-3"]


def test_result_schema_is_complete() -> None:
    row = _row()
    assert tuple(row) == REQUIRED_ROW_FIELDS
