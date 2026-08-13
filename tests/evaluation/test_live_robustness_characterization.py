from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydroswarm.evaluation.live_robustness import (
    REQUIRED_ROW_FIELDS,
    _invariants,
    _metric_fields,
    _reject_locked,
    predeclared_conditions,
    runtime_commit,
    summarize,
    write_artifacts,
)


def test_predeclared_conditions_are_deterministic_and_include_live_matrix() -> None:
    first = predeclared_conditions(repetitions=1)
    second = predeclared_conditions(repetitions=1)
    assert first == second
    assert {item.perturbation_type for item in first} >= {
        "nominal", "missingness", "sensor_coverage", "sensor_health",
        "measurement_noise", "measurement_bias", "hydraulic_mismatch",
        "ambiguity", "topology_familiarity", "scale",
    }
    assert all(item.network_id != "locked_final_test" for item in first)
    assert {item.network_id for item in first if item.topology_class == "development_unseen"} == {"coastal-branch"}


def test_locked_test_guard_rejects_locked_and_test_paths() -> None:
    with pytest.raises(ValueError):
        _reject_locked("data/locked_final_test/example")
    with pytest.raises(ValueError):
        _reject_locked("data/populations/test/example")
    _reject_locked("data/topologies/loop-grid.inp")


def test_metrics_and_invariant_checks_are_null_safe() -> None:
    metrics = _metric_fields({"fused_belief": {"J2": 0.7, "J1": 0.3}, "candidate_nodes": ["J2"]}, "J1")
    assert metrics["top1_correct"] is False
    assert metrics["top3_correct"] is True
    invariant = _invariants(
        analysis={"planning_allowed": False, "control_action": "REQUEST_SAMPLE", "ood_level": "CAUTION"},
        generate_status=409, plans=[], approval_status=None, stale_approval_status=None,
    )
    assert invariant["INV-1"] is True
    assert invariant["INV-2"] is True
    assert invariant["INV-3"] is True


def test_artifacts_preserve_null_measurements_and_schema(tmp_path: Path) -> None:
    row = {field: None for field in REQUIRED_ROW_FIELDS}
    row.update({"run_id": "one", "perturbation_type": "nominal", "perturbation_level": "clean", "top1_correct": True, "planning_allowed": False, "invariants": {"INV-1": True}})
    row.update({"study_baseline_commit": "pre-fix", "runtime_commit": "post-fix", "git_commit": "post-fix"})
    result = write_artifacts(tmp_path, [row], locked_opened_after=False, finding_evidence={"ROB-LIVE-01": {"evaluated": True, "passed": True}, "ROB-LIVE-02": {"evaluated": True, "passed": False}})
    assert result["locked_test_opened"] is False
    stored = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert stored[0]["analysis_ms"] is None
    assert (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()[0].split(",") == list(REQUIRED_ROW_FIELDS)
    assert summarize([row])["invariant_failures"] == []
    assert result["runtime_commits"] == ["post-fix"]
    assert {item["id"]: item["status"] for item in result["findings"]} == {"ROB-LIVE-01": "REMEDIATED", "ROB-LIVE-02": "REGRESSION"}


def test_runtime_commit_is_not_the_frozen_study_baseline(tmp_path: Path) -> None:
    assert runtime_commit(Path.cwd()) != "e45f72cf730d3f12c13dbcb9403c64f185510173"
