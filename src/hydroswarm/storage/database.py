"""Durable SQLite connection management for the offline runtime."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def default_database_path() -> Path:
    configured = os.environ.get("HYDROSWARM_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    # Submission-readiness SUB-1: HYDROSWARM_DATA_DIR is the container
    # image's writable volume mount (declared by the Dockerfile/
    # docker-compose.yml since before this fix, but never previously read
    # by any application code -- the database would otherwise default
    # under the process cwd, which is the container's read-only root).
    data_dir = os.environ.get("HYDROSWARM_DATA_DIR")
    if data_dir:
        return (Path(data_dir).expanduser() / "hydroswarm.db").resolve()
    return (Path.cwd() / "data" / "generated" / "hydroswarm.db").resolve()


class Database:
    """Small connection factory with consistent safety pragmas."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path else default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except Exception:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def ping(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1
