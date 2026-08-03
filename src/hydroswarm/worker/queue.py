"""Persistent single-worker job queue for model and simulator isolation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
import json
import threading
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from hydroswarm.storage import Database


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: UUID
    incident_id: UUID
    kind: str
    state: JobState
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    result: dict[str, Any] | None = None
    error_code: str | None = None
    created_at: datetime
    updated_at: datetime


class PersistentJobQueue:
    """One-thread queue; simulator/model objects never run concurrently in the API process."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydroswarm-worker")
        self._cancel: dict[UUID, threading.Event] = {}
        self._migrate()

    def _migrate(self) -> None:
        with self.database.connect(write=True) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS worker_jobs(
                    job_id TEXT PRIMARY KEY, incident_id TEXT NOT NULL, kind TEXT NOT NULL,
                    state TEXT NOT NULL, progress REAL NOT NULL, message TEXT NOT NULL,
                    result_json TEXT, error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "UPDATE worker_jobs SET state='FAILED', error_code='WORKER_RESTARTED', "
                "message='worker restarted before completion' WHERE state IN ('QUEUED','RUNNING')"
            )

    def _write(self, status: JobStatus) -> None:
        with self.database.connect(write=True) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO worker_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(status.job_id), str(status.incident_id), status.kind, status.state.value,
                    status.progress, status.message,
                    json.dumps(status.result, sort_keys=True, default=str) if status.result else None,
                    status.error_code, status.created_at.isoformat(), status.updated_at.isoformat(),
                ),
            )

    def get(self, job_id: UUID) -> JobStatus | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM worker_jobs WHERE job_id=?", (str(job_id),)).fetchone()
        if row is None:
            return None
        return JobStatus(
            job_id=row["job_id"], incident_id=row["incident_id"], kind=row["kind"],
            state=row["state"], progress=row["progress"], message=row["message"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_code=row["error_code"], created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def submit(
        self,
        incident_id: UUID,
        kind: str,
        function: Callable[[Callable[[float, str], None], threading.Event], dict[str, Any]],
    ) -> JobStatus:
        now = datetime.now(UTC)
        job_id = uuid4()
        cancel = threading.Event()
        self._cancel[job_id] = cancel
        initial = JobStatus(
            job_id=job_id, incident_id=incident_id, kind=kind, state=JobState.QUEUED,
            progress=0.0, message="queued", created_at=now, updated_at=now,
        )
        self._write(initial)

        def run() -> None:
            def progress(value: float, message: str) -> None:
                current = self.get(job_id) or initial
                self._write(current.model_copy(update={
                    "state": JobState.RUNNING, "progress": max(0.0, min(1.0, value)),
                    "message": message, "updated_at": datetime.now(UTC),
                }))
            try:
                progress(0.01, "started")
                result = function(progress, cancel)
                current = self.get(job_id) or initial
                terminal = current.model_copy(update={
                    "state": JobState.CANCELLED if cancel.is_set() else JobState.COMPLETE,
                    "progress": 1.0, "message": "cancelled" if cancel.is_set() else "complete",
                    "result": None if cancel.is_set() else result, "updated_at": datetime.now(UTC),
                })
            except Exception as exc:
                current = self.get(job_id) or initial
                terminal = current.model_copy(update={
                    "state": JobState.FAILED, "message": "job failed",
                    "error_code": type(exc).__name__, "updated_at": datetime.now(UTC),
                })
            self._write(terminal)
            self._cancel.pop(job_id, None)

        self._executor.submit(run)
        return initial

    def cancel(self, job_id: UUID) -> JobStatus | None:
        status = self.get(job_id)
        if status is None or status.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
            return status
        self._cancel.setdefault(job_id, threading.Event()).set()
        updated = status.model_copy(update={
            "state": JobState.CANCELLED, "message": "cancellation requested",
            "updated_at": datetime.now(UTC),
        })
        self._write(updated)
        return updated

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
