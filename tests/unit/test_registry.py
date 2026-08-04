from __future__ import annotations

import json

import pytest

from hydroswarm.training import ExperimentRegistry, RegistryError
from hydroswarm.training.registry import ClosedRunRecord


def _open_kwargs(**overrides):
    base = dict(
        kind="training",
        purpose="unit-test smoke run",
        architecture="hydrocore",
        variant="small",
        seed=2026,
        resolved_config={"lr": 1e-3},
        manifest_hashes={"train": "aaa", "validation": "bbb"},
        feature_schema_hash="feat-hash",
        target_schema_hash="target-hash",
        topology_hashes=("topo-1",),
        workdir=".",
    )
    base.update(overrides)
    return base


def test_open_run_records_all_required_provenance_fields(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    handle = registry.open_run(**_open_kwargs())

    events = registry.all_events()
    assert len(events) == 1
    record = events[0]
    required_fields = {
        "run_id",
        "parent_run_id",
        "purpose",
        "git_commit",
        "dirty_worktree",
        "architecture",
        "variant",
        "resolved_config",
        "seed",
        "manifest_hashes",
        "feature_schema_hash",
        "target_schema_hash",
        "topology_hashes",
        "started_at",
        "host",
        "cpu_count",
        "thread_settings",
    }
    assert required_fields.issubset(record.keys())
    assert record["event"] == "opened"
    assert handle.run_id == record["run_id"]


def test_duplicate_run_id_is_rejected(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    registry.open_run(**_open_kwargs(run_id="fixed-id"))
    with pytest.raises(RegistryError, match="duplicate run_id"):
        registry.open_run(**_open_kwargs(run_id="fixed-id"))


def test_close_run_appends_closed_event_with_exit_status_and_peak_rss(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    handle = registry.open_run(**_open_kwargs())
    closed = handle.close(
        exit_status="success",
        checkpoint_paths=("ckpt/best.safetensors",),
        checkpoint_hashes={"ckpt/best.safetensors": "deadbeef"},
        selected_checkpoint="ckpt/best.safetensors",
        selection_metric={"validation_loss": 1.23},
        locked_test_opened=False,
    )
    assert closed.peak_rss_bytes > 0
    events = registry.all_events()
    assert [event["event"] for event in events] == ["opened", "closed"]
    merged = registry.runs()[handle.run_id]
    assert merged["exit_status"] == "success"
    assert merged["selected_checkpoint"] == "ckpt/best.safetensors"
    assert merged["locked_test_opened"] is False


def test_close_run_rejects_invalid_exit_status(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    handle = registry.open_run(**_open_kwargs())
    with pytest.raises(RegistryError):
        handle.close(exit_status="not-a-real-status")


def test_closing_unknown_run_id_fails(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    handle = registry.open_run(**_open_kwargs())
    # A registry that never saw the corresponding "opened" event must refuse to
    # accept a "closed" event for that run_id, even though the run_id is well-formed.
    other_registry = ExperimentRegistry(tmp_path / "other.jsonl")
    orphan_close = ClosedRunRecord(
        run_id=handle.run_id,
        ended_at="2026-01-01T00:00:00Z",
        exit_status="success",
        peak_rss_bytes=1,
    )
    with pytest.raises(RegistryError, match="unknown run_id"):
        other_registry._append_closed(orphan_close)


def test_resumed_run_preserves_lineage_via_parent_run_id(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    original = registry.open_run(**_open_kwargs(run_id="run-A"))
    original.close(exit_status="partial", notes="interrupted for maintenance")

    resumed = registry.open_run(**_open_kwargs(run_id="run-B", parent_run_id="run-A"))
    resumed.close(exit_status="success")

    lineage = registry.lineage("run-B")
    assert lineage == ["run-A", "run-B"]


def test_open_run_rejects_unknown_parent_run_id(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    with pytest.raises(RegistryError, match="parent_run_id"):
        registry.open_run(**_open_kwargs(parent_run_id="does-not-exist"))


def test_open_run_rejects_unknown_kind(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    with pytest.raises(RegistryError, match="kind"):
        registry.open_run(**_open_kwargs(kind="not-a-real-kind"))


def test_registry_file_is_valid_jsonl_after_multiple_runs(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    registry = ExperimentRegistry(path)
    for index in range(3):
        handle = registry.open_run(**_open_kwargs(run_id=f"run-{index}"))
        handle.close(exit_status="success")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6  # 3 opened + 3 closed
    for line in lines:
        json.loads(line)  # must not raise
