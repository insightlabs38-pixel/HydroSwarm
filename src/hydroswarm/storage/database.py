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
