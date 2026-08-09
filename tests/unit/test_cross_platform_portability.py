"""Regression coverage for the Windows/POSIX portability fix in
hydroswarm.training (registry.py's former unconditional `fcntl`/`resource`
imports broke collection of anything that transitively imports
hydroswarm.training -- including hydroswarm.api -- on Windows CI).

These tests run for real on whichever platform the matrix CI job is on;
none of them monkeypatch sys.platform or fake away real lock/RSS behavior,
so they exercise the actual POSIX branch on ubuntu-latest and the actual
Windows branch on windows-latest.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from hydroswarm.training import ExperimentRegistry, RegistryError
from hydroswarm.training._cross_platform import IS_WINDOWS, file_lock
from hydroswarm.training.registry import _peak_rss_bytes


def test_hydroswarm_training_imports_cleanly() -> None:
    """hydroswarm.training must import on every CI matrix platform --
    registry.py used to import fcntl/resource unconditionally at module
    level, which broke collection outright on Windows."""
    module = importlib.import_module("hydroswarm.training")
    assert module is not None


def test_hydroswarm_api_imports_cleanly() -> None:
    """hydroswarm.api transitively imports hydroswarm.training (via the
    runtime/v4 checkpoint-identity import chain) -- same Windows collection
    concern as the training import above, verified from the other end of
    the chain."""
    module = importlib.import_module("hydroswarm.api")
    assert module is not None


def test_is_windows_flag_matches_the_real_interpreter() -> None:
    assert IS_WINDOWS == (sys.platform == "win32")


def test_file_lock_is_released_after_a_clean_block(tmp_path) -> None:
    lock_path = tmp_path / "clean.lock"
    with file_lock(lock_path):
        pass
    # Re-acquiring immediately afterwards must not hang/raise -- proves the
    # first lock was actually released, not merely that no exception
    # propagated out of the first `with`.
    with file_lock(lock_path):
        pass


def test_file_lock_is_released_even_when_the_block_raises(tmp_path) -> None:
    lock_path = tmp_path / "raising.lock"

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with file_lock(lock_path):
            raise _Boom("simulated failure inside the critical section")

    # If the lock had leaked, re-acquiring here would hang (POSIX) or raise
    # after msvcrt's own retry budget (Windows) -- bounded by pytest's
    # default test timeout either way, so a leak surfaces as a real failure
    # rather than a silent false pass.
    with file_lock(lock_path):
        pass


def test_registry_duplicate_run_id_error_does_not_leak_the_lock(tmp_path) -> None:
    """The duplicate-run-id check that raises RegistryError happens INSIDE
    the same file_lock critical section open_run uses -- this proves that
    raising there doesn't leave every subsequent run blocked."""

    registry = ExperimentRegistry(tmp_path / "runs.jsonl")
    kwargs = dict(
        kind="training",
        purpose="portability smoke",
        architecture="hydrocore",
        variant="small",
        seed=1,
        resolved_config={},
        manifest_hashes={},
        feature_schema_hash="h",
        target_schema_hash="h",
    )
    registry.open_run(run_id="dup", **kwargs)
    with pytest.raises(RegistryError, match="duplicate run_id"):
        registry.open_run(run_id="dup", **kwargs)

    # A fresh run_id must still succeed -- proves the duplicate check's
    # RegistryError above didn't leave the registry's lock file held.
    handle = registry.open_run(run_id="fresh", **kwargs)
    closed = handle.close(exit_status="success")
    assert closed.peak_rss_bytes > 0

    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # dup-opened, fresh-opened, fresh-closed
    for line in lines:
        json.loads(line)  # registry JSONL remains valid


def test_peak_rss_bytes_is_positive_on_this_platform() -> None:
    assert _peak_rss_bytes() > 0
