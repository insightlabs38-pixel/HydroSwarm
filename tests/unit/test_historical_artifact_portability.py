from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.historical_artifact_portability import inspect_historical_artifact, resolve_recorded_path


def _recorded(path: str = "experiments/runs/example/model.safetensors") -> str:
    return f"/workspace/HydroSwarm/{path}"


def test_resolves_only_exact_repo_relative_recorded_suffix(tmp_path: Path) -> None:
    assert resolve_recorded_path(_recorded(), repo_root=tmp_path) == tmp_path / "experiments/runs/example/model.safetensors"


def test_materialized_correct_historical_artifact_passes(tmp_path: Path) -> None:
    path = tmp_path / "experiments/runs/example/model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"verified")
    artifact = inspect_historical_artifact(_recorded(), hashlib.sha256(b"verified").hexdigest(), repo_root=tmp_path, tracking_resolver=lambda *_: False)
    assert artifact.exists and artifact.actual_sha256 == artifact.expected_sha256


def test_materialized_wrong_historical_artifact_fails(tmp_path: Path) -> None:
    path = tmp_path / "experiments/runs/example/model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        inspect_historical_artifact(_recorded(), hashlib.sha256(b"expected").hexdigest(), repo_root=tmp_path, tracking_resolver=lambda *_: False)


def test_missing_tracked_historical_artifact_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tracked required"):
        inspect_historical_artifact(_recorded(), "a" * 64, repo_root=tmp_path, tracking_resolver=lambda *_: True)


def test_missing_non_distributed_historical_artifact_is_identified(tmp_path: Path) -> None:
    artifact = inspect_historical_artifact(_recorded(), "a" * 64, repo_root=tmp_path, tracking_resolver=lambda *_: False)
    assert artifact.is_non_distributed_missing
