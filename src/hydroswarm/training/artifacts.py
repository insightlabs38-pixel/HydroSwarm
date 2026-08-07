"""Run provenance, bounded-resource checks, and atomic status artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Mapping
from uuid import uuid4

import psutil


def _git_commit(workdir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(slots=True)
class RunArtifacts:
    path: Path
    started_at: datetime

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        config: Mapping[str, Any],
        manifest_hash: str,
        workdir: str | Path,
    ) -> RunArtifacts:
        timestamp = datetime.now(UTC)
        path = Path(root) / f"{timestamp:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        (path / "checkpoints").mkdir(parents=True)
        atomic_json(path / "config.json", dict(config))
        metadata = {
            "git_commit": _git_commit(Path(workdir)),
            "dataset_manifest_hash": manifest_hash,
            "started_at": timestamp.isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        }
        atomic_json(path / "metadata.json", metadata)
        artifacts = cls(path, timestamp)
        artifacts.status("RUNNING")
        return artifacts

    def status(self, state: str, **details: Any) -> None:
        atomic_json(
            self.path / "status.json",
            {"state": state, "updated_at": datetime.now(UTC).isoformat(), **details},
        )

    def append_metric(self, metric: Mapping[str, Any]) -> None:
        self.append_jsonl("metrics.jsonl", metric)

    def append_jsonl(self, filename: str, value: Mapping[str, Any]) -> None:
        """Generic accumulating-JSONL append, used for anything that needs
        a full history across the run rather than a single latest-value
        snapshot (e.g. core-issues3.txt Phase 11.4's per-epoch,
        per-task validation-loss history in "validation_history.jsonl" --
        epoch_summary.json only ever keeps the MOST RECENT epoch, since
        atomic_json always overwrites the same path)."""

        with (self.path / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")

    def check_disk(self, minimum_free_gb: float) -> None:
        free_gb = shutil.disk_usage(self.path).free / (1024**3)
        if free_gb < minimum_free_gb:
            raise RuntimeError(
                f"free disk {free_gb:.2f} GiB is below required {minimum_free_gb:.2f} GiB"
            )

