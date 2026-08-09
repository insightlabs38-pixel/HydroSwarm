"""GCP-safe resumable job supervisor (overnight-plan.txt Task 0.3).

This environment has no systemd/tmux available to the agent, so this module
is the "thin job layer" the plan allows as an alternative: a small process
supervisor for long-running dataset-generation, tensor-building, training,
calibration, evaluation, and report-generation jobs. Each job gets its own
run directory containing:

- ``status.json``  -- atomic, machine-readable state (RUNNING/COMPLETED/
  FAILED/TERMINATED/CRASHED), pid, command, start/end time, exit code.
- ``job.log``       -- combined stdout/stderr of the child process.
- ``job.pid``       -- the child's PID (also duplicated in status.json).
- ``job.lock``      -- advisory lock preventing two jobs from starting in the
  same run directory concurrently.

The supervisor does not itself implement checkpointing, disk guards, or
graceful-shutdown handling for the wrapped job -- those are the job's own
responsibility (see ``Trainer.install_signal_handlers`` and
``RunArtifacts.check_disk`` for the training path). What this module adds is
the outer layer: detect an already-running job in the same directory, launch
a new one detached from the launching shell's own signal group (POSIX:
setsid; Windows: a new process group) so it survives the launcher exiting,
record enough to write an exact resume command, and forward a graceful-then-
hard termination request to the job's whole process tree on request so a
supervisor-level "stop the job" always reaches the child (and any
descendants it spawned). Cross-platform: POSIX and Windows both supported,
via ``psutil`` for process signaling/termination and ``_cross_platform`` for
the advisory run-directory lock.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ._cross_platform import IS_WINDOWS, file_lock
from .artifacts import atomic_json


class JobRunnerError(Exception):
    """Raised for job-supervision invariant violations (e.g. overlap)."""


def _pid_alive(pid: int) -> bool:
    # psutil.pid_exists is a real existence check on every platform (reads
    # the process table directly) -- unlike the POSIX os.kill(pid, 0) trick
    # this replaces, it never has to distinguish ESRCH/EPERM, and it is safe
    # to call on Windows: os.kill(pid, 0) on Windows does NOT probe liveness
    # the way POSIX signal 0 does -- any non-CTRL_* signal value, including
    # 0, is passed straight to TerminateProcess(), so the POSIX idiom would
    # actually kill the process being checked.
    return psutil.pid_exists(pid)


def _process_tree(pid: int) -> list[psutil.Process]:
    """``pid`` plus every descendant, most-recently-spawned last -- the
    process-group-via-start_new_session kill this replaces was itself an
    approximation of "kill the whole job, not just its top process"."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    return [parent, *children]


@dataclass(slots=True)
class JobHandle:
    run_dir: Path
    pid: int
    log_path: Path
    status_path: Path
    pid_path: Path

    def status(self) -> dict[str, Any]:
        return read_status(self.run_dir)

    def poll(self) -> str:
        """Refresh and return the current state, detecting a crash if the
        recorded pid died without writing a terminal status."""

        current = self.status()
        if current["state"] == "RUNNING" and not _pid_alive(self.pid):
            current = {
                **current,
                "state": "CRASHED",
                "ended_at": datetime.now(UTC).isoformat(),
            }
            atomic_json(self.status_path, current)
        return current["state"]

    def terminate(self, *, grace_period_seconds: float = 30.0) -> str:
        """Ask the job's whole process tree to exit gracefully (SIGTERM on
        POSIX, TerminateProcess on Windows -- psutil.Process.terminate()
        picks the right one), wait up to grace_period_seconds, then escalate
        to a hard kill of anything still alive."""

        if not _pid_alive(self.pid):
            return self.poll()
        for proc in _process_tree(self.pid):
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        deadline = time.monotonic() + grace_period_seconds
        while time.monotonic() < deadline:
            if not _pid_alive(self.pid):
                break
            time.sleep(0.1)
        else:
            for proc in _process_tree(self.pid):
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
        current = self.status()
        state = "TERMINATED" if current["state"] == "RUNNING" else current["state"]
        current = {**current, "state": state, "ended_at": datetime.now(UTC).isoformat()}
        atomic_json(self.status_path, current)
        return state


def read_status(run_dir: str | Path) -> dict[str, Any]:
    status_path = Path(run_dir) / "status.json"
    if not status_path.exists():
        raise JobRunnerError(f"no job status found in {run_dir}")
    return json.loads(status_path.read_text(encoding="utf-8"))


def attach(run_dir: str | Path) -> JobHandle:
    """Re-attach to a job that was launched in an earlier process (e.g. after
    a controller restart), for polling or termination."""

    run_dir = Path(run_dir)
    status = read_status(run_dir)
    return JobHandle(
        run_dir=run_dir,
        pid=int(status["pid"]),
        log_path=run_dir / "job.log",
        status_path=run_dir / "status.json",
        pid_path=run_dir / "job.pid",
    )


def launch(
    command: Sequence[str],
    *,
    run_dir: str | Path,
    workdir: str | Path = ".",
    min_free_disk_gb: float = 5.0,
    resume_command: Sequence[str] | None = None,
    env: dict[str, str] | None = None,
) -> JobHandle:
    """Start ``command`` as a detached, supervised background job.

    Raises JobRunnerError if a job is already RUNNING (and alive) in
    ``run_dir``, or if free disk on the run_dir's filesystem is already below
    ``min_free_disk_gb`` before the job even starts.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    lock_path = run_dir / "job.lock"

    # Detach the child from the launching process's own termination signal
    # group so it survives the launching shell exiting -- start_new_session
    # (setsid) on POSIX; CREATE_NEW_PROCESS_GROUP is the Windows analog
    # (keeps the child out of the console Ctrl+C signal group).
    detach_kwargs: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if IS_WINDOWS
        else {"start_new_session": True}
    )

    with file_lock(lock_path):
        if status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("state") == "RUNNING" and _pid_alive(int(existing["pid"])):
                raise JobRunnerError(
                    f"a job is already RUNNING in {run_dir} (pid {existing['pid']}); "
                    "stop it or choose a different run_dir before starting a new one"
                )

        free_gb = shutil.disk_usage(run_dir).free / (1024**3)
        if free_gb < min_free_disk_gb:
            raise JobRunnerError(
                f"free disk {free_gb:.2f} GiB in {run_dir} is below the required "
                f"{min_free_disk_gb:.2f} GiB; refusing to start the job"
            )

        log_path = run_dir / "job.log"
        pid_path = run_dir / "job.pid"
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            list(command),
            cwd=str(workdir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
            **detach_kwargs,
        )
        pid_path.write_text(str(process.pid), encoding="utf-8")
        atomic_json(
            status_path,
            {
                "state": "RUNNING",
                "pid": process.pid,
                "command": list(command),
                "resume_command": list(resume_command) if resume_command else None,
                "workdir": str(Path(workdir).resolve()),
                "started_at": datetime.now(UTC).isoformat(),
                "ended_at": None,
                "log_path": str(log_path),
            },
        )

    return JobHandle(
        run_dir=run_dir,
        pid=process.pid,
        log_path=log_path,
        status_path=status_path,
        pid_path=pid_path,
    )


def mark_finished(run_dir: str | Path, *, exit_code: int) -> dict[str, Any]:
    """Called by a waiter (not the child itself) once the child process has
    exited, to record a terminal COMPLETED/FAILED status."""

    run_dir = Path(run_dir)
    status = read_status(run_dir)
    status = {
        **status,
        "state": "COMPLETED" if exit_code == 0 else "FAILED",
        "exit_code": exit_code,
        "ended_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(run_dir / "status.json", status)
    return status
