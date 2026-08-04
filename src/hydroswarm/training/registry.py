"""Immutable, append-only provenance ledger for training/calibration/benchmark/
data-generation runs (overnight-plan.txt Task 0.2).

Every governed run gets an "opened" record when it starts and a "closed" record
when it finishes (successfully, partially, or with a failure). Records are never
rewritten in place; a resumed run opens a new run_id and points parent_run_id at
the run it continues, so the full lineage is reconstructable by following that
chain. Duplicate run_ids are rejected. Appends are guarded by an advisory file
lock so concurrent processes on the same VM cannot race past the duplicate check
or interleave partial lines.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import resource
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import psutil

SCHEMA_VERSION = 1

_RUN_KINDS = frozenset(
    {"training", "calibration", "benchmark", "data_generation", "evaluation"}
)
_EXIT_STATUSES = frozenset({"success", "failed", "partial"})


class RegistryError(Exception):
    """Raised when a registry invariant would be violated."""


def _git_commit(workdir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _dirty_worktree(workdir: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "status", "--porcelain"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _thread_settings() -> dict[str, str | int | None]:
    settings: dict[str, str | int | None] = {
        variable: os.environ.get(variable)
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    try:
        import torch

        settings["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        settings["torch_num_threads"] = None
    return settings


def _peak_rss_bytes() -> int:
    # ru_maxrss is KiB on Linux, bytes on macOS; this project's execution
    # environment is Linux-only (see plan section 10), so we assume KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


@dataclass(slots=True)
class RunRecord:
    """The fields required by overnight-plan.txt Task 0.2 for a single run."""

    run_id: str
    kind: str
    purpose: str
    architecture: str
    variant: str
    seed: int
    resolved_config: Mapping[str, Any]
    manifest_hashes: Mapping[str, str]
    feature_schema_hash: str
    target_schema_hash: str
    topology_hashes: Sequence[str]
    git_commit: str
    dirty_worktree: bool | None
    host: str
    cpu_count: int | None
    ram_bytes: int
    thread_settings: Mapping[str, Any]
    started_at: str
    parent_run_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    event: str = "opened"

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "kind": self.kind,
            "purpose": self.purpose,
            "architecture": self.architecture,
            "variant": self.variant,
            "seed": self.seed,
            "resolved_config": dict(self.resolved_config),
            "manifest_hashes": dict(self.manifest_hashes),
            "feature_schema_hash": self.feature_schema_hash,
            "target_schema_hash": self.target_schema_hash,
            "topology_hashes": list(self.topology_hashes),
            "git_commit": self.git_commit,
            "dirty_worktree": self.dirty_worktree,
            "host": self.host,
            "cpu_count": self.cpu_count,
            "ram_bytes": self.ram_bytes,
            "thread_settings": dict(self.thread_settings),
            "started_at": self.started_at,
        }


@dataclass(slots=True)
class ClosedRunRecord:
    run_id: str
    ended_at: str
    exit_status: str
    peak_rss_bytes: int
    checkpoint_paths: Sequence[str] = field(default_factory=tuple)
    checkpoint_hashes: Mapping[str, str] = field(default_factory=dict)
    selected_checkpoint: str | None = None
    selection_metric: Mapping[str, Any] | None = None
    locked_test_opened: bool = False
    notes: str | None = None
    schema_version: int = SCHEMA_VERSION
    event: str = "closed"

    def to_json(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "ended_at": self.ended_at,
            "exit_status": self.exit_status,
            "peak_rss_bytes": self.peak_rss_bytes,
            "checkpoint_paths": list(self.checkpoint_paths),
            "checkpoint_hashes": dict(self.checkpoint_hashes),
            "selected_checkpoint": self.selected_checkpoint,
            "selection_metric": (
                dict(self.selection_metric) if self.selection_metric is not None else None
            ),
            "locked_test_opened": self.locked_test_opened,
            "notes": self.notes,
        }


@contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunHandle:
    """Handle returned by :meth:`ExperimentRegistry.open_run`; use to close the run."""

    def __init__(self, registry: "ExperimentRegistry", record: RunRecord) -> None:
        self._registry = registry
        self.record = record

    @property
    def run_id(self) -> str:
        return self.record.run_id

    def close(
        self,
        *,
        exit_status: str,
        checkpoint_paths: Sequence[str] = (),
        checkpoint_hashes: Mapping[str, str] | None = None,
        selected_checkpoint: str | None = None,
        selection_metric: Mapping[str, Any] | None = None,
        locked_test_opened: bool = False,
        notes: str | None = None,
    ) -> ClosedRunRecord:
        if exit_status not in _EXIT_STATUSES:
            raise RegistryError(
                f"exit_status must be one of {sorted(_EXIT_STATUSES)}, got {exit_status!r}"
            )
        closed = ClosedRunRecord(
            run_id=self.record.run_id,
            ended_at=datetime.now(UTC).isoformat(),
            exit_status=exit_status,
            peak_rss_bytes=_peak_rss_bytes(),
            checkpoint_paths=tuple(checkpoint_paths),
            checkpoint_hashes=dict(checkpoint_hashes or {}),
            selected_checkpoint=selected_checkpoint,
            selection_metric=selection_metric,
            locked_test_opened=locked_test_opened,
            notes=notes,
        )
        self._registry._append_closed(closed)
        return closed


class ExperimentRegistry:
    """Append-only JSONL ledger of run provenance.

    One line per event ("opened" or "closed"); a run's full record is the join
    of its opened + closed events by run_id. Never rewritten in place.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _existing_run_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        run_ids: set[str] = set()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                run_ids.add(json.loads(line)["run_id"])
        return run_ids

    def _append_line(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def open_run(
        self,
        *,
        kind: str,
        purpose: str,
        architecture: str,
        variant: str,
        seed: int,
        resolved_config: Mapping[str, Any],
        manifest_hashes: Mapping[str, str],
        feature_schema_hash: str,
        target_schema_hash: str,
        topology_hashes: Sequence[str] = (),
        workdir: str | Path = ".",
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> RunHandle:
        if kind not in _RUN_KINDS:
            raise RegistryError(f"kind must be one of {sorted(_RUN_KINDS)}, got {kind!r}")
        workdir_path = Path(workdir)
        started_at = datetime.now(UTC)
        candidate_id = run_id or f"{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"

        with _locked(self.lock_path):
            existing = self._existing_run_ids()
            if candidate_id in existing:
                raise RegistryError(f"duplicate run_id {candidate_id!r} already exists in registry")
            if parent_run_id is not None and parent_run_id not in existing:
                raise RegistryError(
                    f"parent_run_id {parent_run_id!r} does not exist in registry {self.path}"
                )
            record = RunRecord(
                run_id=candidate_id,
                parent_run_id=parent_run_id,
                kind=kind,
                purpose=purpose,
                architecture=architecture,
                variant=variant,
                seed=seed,
                resolved_config=resolved_config,
                manifest_hashes=manifest_hashes,
                feature_schema_hash=feature_schema_hash,
                target_schema_hash=target_schema_hash,
                topology_hashes=topology_hashes,
                git_commit=_git_commit(workdir_path),
                dirty_worktree=_dirty_worktree(workdir_path),
                host=platform.node(),
                cpu_count=os.cpu_count(),
                ram_bytes=psutil.virtual_memory().total,
                thread_settings=_thread_settings(),
                started_at=started_at.isoformat(),
            )
            self._append_line(record.to_json())
        return RunHandle(self, record)

    def _append_closed(self, closed: ClosedRunRecord) -> None:
        with _locked(self.lock_path):
            existing = self._existing_run_ids()
            if closed.run_id not in existing:
                raise RegistryError(
                    f"cannot close unknown run_id {closed.run_id!r}; no opened record found"
                )
            self._append_line(closed.to_json())

    def all_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def runs(self) -> dict[str, dict[str, Any]]:
        """Join opened/closed events per run_id into one merged record each."""
        merged: dict[str, dict[str, Any]] = {}
        for event in self.all_events():
            run_id = event["run_id"]
            merged.setdefault(run_id, {}).update(event)
        return merged

    def lineage(self, run_id: str) -> list[str]:
        """Return the chain of run_ids from the oldest ancestor to run_id (inclusive)."""
        runs = self.runs()
        if run_id not in runs:
            raise RegistryError(f"unknown run_id {run_id!r}")
        chain = [run_id]
        current = runs[run_id]
        seen = {run_id}
        while current.get("parent_run_id"):
            parent_id = current["parent_run_id"]
            if parent_id in seen or parent_id not in runs:
                break
            chain.append(parent_id)
            seen.add(parent_id)
            current = runs[parent_id]
        return list(reversed(chain))
