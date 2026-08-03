"""Append-only SQLite event ledger with a tamper-evident hash chain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


GENESIS_HASH = "0" * 64


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    incident_id: UUID
    sequence: int = Field(ge=1)
    timestamp: datetime
    event_type: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    input_state_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    payload: dict[str, Any]
    model_version: str
    simulator_version: str
    previous_event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class AuditLedger:
    """Persist events locally and reject all update/delete operations.

    SQLite triggers protect the append-only invariant even if another local component
    gets a raw database connection. The hash chain detects out-of-band row replacement.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    input_state_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    simulator_version TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    UNIQUE (incident_id, sequence)
                );
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit ledger is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit ledger is append-only'); END;
                """
            )
            connection.commit()

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _hash_event(cls, values: Mapping[str, Any]) -> str:
        return hashlib.sha256(cls._canonical_json(values).encode("utf-8")).hexdigest()

    def append(
        self,
        *,
        incident_id: UUID,
        event_type: str,
        actor: str,
        input_state_hash: str,
        payload: Mapping[str, Any],
        model_version: str,
        simulator_version: str,
        timestamp: datetime | None = None,
    ) -> AuditEvent:
        timestamp = timestamp or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("audit timestamp must be timezone-aware")
        if len(input_state_hash) != 64:
            raise ValueError("input_state_hash must be a SHA-256 hex digest")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            last = connection.execute(
                """SELECT sequence, event_hash FROM audit_events
                   WHERE incident_id = ? ORDER BY sequence DESC LIMIT 1""",
                (str(incident_id),),
            ).fetchone()
            sequence = int(last["sequence"]) + 1 if last else 1
            previous_hash = str(last["event_hash"]) if last else GENESIS_HASH
            event_id = uuid4()
            payload_dict = dict(payload)
            hash_fields = {
                "event_id": str(event_id),
                "incident_id": str(incident_id),
                "sequence": sequence,
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "actor": actor,
                "input_state_hash": input_state_hash,
                "payload": payload_dict,
                "model_version": model_version,
                "simulator_version": simulator_version,
                "previous_event_hash": previous_hash,
            }
            event_hash = self._hash_event(hash_fields)
            connection.execute(
                """INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(event_id),
                    str(incident_id),
                    sequence,
                    timestamp.isoformat(),
                    event_type,
                    actor,
                    input_state_hash,
                    self._canonical_json(payload_dict),
                    model_version,
                    simulator_version,
                    previous_hash,
                    event_hash,
                ),
            )
            connection.commit()

        return AuditEvent(**hash_fields, event_hash=event_hash)

    def events(self, incident_id: UUID) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE incident_id = ? ORDER BY sequence",
                (str(incident_id),),
            ).fetchall()
        return [
            AuditEvent(
                event_id=row["event_id"],
                incident_id=row["incident_id"],
                sequence=row["sequence"],
                timestamp=row["timestamp"],
                event_type=row["event_type"],
                actor=row["actor"],
                input_state_hash=row["input_state_hash"],
                payload=json.loads(row["payload_json"]),
                model_version=row["model_version"],
                simulator_version=row["simulator_version"],
                previous_event_hash=row["previous_event_hash"],
                event_hash=row["event_hash"],
            )
            for row in rows
        ]

    def verify_chain(self, incident_id: UUID) -> bool:
        previous_hash = GENESIS_HASH
        for event in self.events(incident_id):
            hash_fields = {
                "event_id": str(event.event_id),
                "incident_id": str(event.incident_id),
                "sequence": event.sequence,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "actor": event.actor,
                "input_state_hash": event.input_state_hash,
                "payload": event.payload,
                "model_version": event.model_version,
                "simulator_version": event.simulator_version,
                "previous_event_hash": event.previous_event_hash,
            }
            if event.previous_event_hash != previous_hash:
                return False
            if self._hash_event(hash_fields) != event.event_hash:
                return False
            previous_hash = event.event_hash
        return True
