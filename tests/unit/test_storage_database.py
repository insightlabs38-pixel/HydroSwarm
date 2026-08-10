"""Submission-readiness SUB-1: HYDROSWARM_DATA_DIR is declared by the
Dockerfile/docker-compose.yml (the container's writable volume mount) but,
before this fix, was never actually read by any application code -- the
database would silently default under the process cwd, which is the
container's read-only root, and fail on first write. HYDROSWARM_DB_PATH
remains the highest-priority explicit override.
"""

from __future__ import annotations

from pathlib import Path

from hydroswarm.storage.database import default_database_path


def test_hydroswarm_db_path_takes_priority(tmp_path: Path, monkeypatch) -> None:
    explicit = tmp_path / "explicit" / "custom.db"
    monkeypatch.setenv("HYDROSWARM_DB_PATH", str(explicit))
    monkeypatch.setenv("HYDROSWARM_DATA_DIR", str(tmp_path / "data-dir"))

    assert default_database_path() == explicit.resolve()


def test_hydroswarm_data_dir_is_honored_when_db_path_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HYDROSWARM_DB_PATH", raising=False)
    data_dir = tmp_path / "data-dir"
    monkeypatch.setenv("HYDROSWARM_DATA_DIR", str(data_dir))

    assert default_database_path() == (data_dir / "hydroswarm.db").resolve()


def test_falls_back_to_cwd_when_neither_is_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HYDROSWARM_DB_PATH", raising=False)
    monkeypatch.delenv("HYDROSWARM_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert default_database_path() == (tmp_path / "data" / "generated" / "hydroswarm.db").resolve()
