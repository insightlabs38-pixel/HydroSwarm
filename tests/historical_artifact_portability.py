"""Portable, fail-closed handling for recorded historical experiment weights.

Historical records retain their original absolute paths.  This test-only
helper maps only an exact repository-relative suffix to the current checkout;
it never searches by basename or rewrites provenance.  A Git-tracked missing
artifact is always an error.  An ignored historical experiment artifact may
be absent from a fresh checkout, in which case callers explicitly skip only
the binary verification after their record-to-record checks have run.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


NON_DISTRIBUTED_REASON = "historical non-distributed experiment checkpoint not materialized in this checkout"


@dataclass(frozen=True)
class HistoricalArtifact:
    recorded_path: str
    checkout_path: Path
    expected_sha256: str
    tracked: bool
    exists: bool
    actual_sha256: str | None

    @property
    def is_non_distributed_missing(self) -> bool:
        return not self.exists and not self.tracked


def resolve_recorded_path(recorded_path: str | Path, *, repo_root: Path) -> Path:
    """Map `/.../HydroSwarm/<suffix>` to exactly `<repo_root>/<suffix>`."""
    normalized = str(recorded_path).replace("\\", "/")
    marker = "/HydroSwarm/"
    if marker not in normalized:
        return Path(recorded_path)
    suffix = normalized.split(marker, 1)[1]
    if not suffix or suffix.startswith("/") or ".." in Path(suffix).parts:
        raise ValueError(f"invalid recorded HydroSwarm path: {recorded_path!r}")
    return repo_root / suffix


def _git_tracked(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repo_root).as_posix()
    except ValueError:
        return False
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root, capture_output=True, text=True,
    ).returncode == 0


def inspect_historical_artifact(
    recorded_path: str | Path,
    expected_sha256: str,
    *,
    repo_root: Path,
    tracking_resolver: Callable[[Path, Path], bool] = _git_tracked,
) -> HistoricalArtifact:
    checkout_path = resolve_recorded_path(recorded_path, repo_root=repo_root)
    exists = checkout_path.is_file()
    actual = hashlib.sha256(checkout_path.read_bytes()).hexdigest() if exists else None
    artifact = HistoricalArtifact(
        recorded_path=str(recorded_path), checkout_path=checkout_path,
        expected_sha256=expected_sha256, tracked=tracking_resolver(repo_root, checkout_path),
        exists=exists, actual_sha256=actual,
    )
    if artifact.exists and artifact.actual_sha256 != expected_sha256:
        raise AssertionError(
            f"historical checkpoint SHA-256 mismatch at {artifact.checkout_path}: "
            f"expected {expected_sha256}, got {artifact.actual_sha256}"
        )
    if not artifact.exists and artifact.tracked:
        raise FileNotFoundError(f"tracked required historical checkpoint missing: {artifact.checkout_path}")
    return artifact


def require_historical_artifact(recorded_path: str | Path, expected_sha256: str, *, repo_root: Path) -> Path:
    """Return a verified weight or explicitly skip an intentionally omitted one."""
    artifact = inspect_historical_artifact(recorded_path, expected_sha256, repo_root=repo_root)
    if artifact.is_non_distributed_missing:
        import pytest

        pytest.skip(NON_DISTRIBUTED_REASON)
    return artifact.checkout_path
