"""Stdlib-only cross-platform primitives shared by this package's process/
run supervisors (``registry.py``, ``job_runner.py``).

Both modules need the same thing: an advisory exclusive lock on a small
lock file, held for a brief duplicate-check-then-append critical section
and always released on exception. POSIX provides this natively via
``fcntl.flock``; Windows has no equivalent module, so this file is the one
place that branches on ``sys.platform`` and implements the same contract
with ``msvcrt.locking`` -- callers on either platform just use
:func:`file_lock` and never import ``fcntl``/``msvcrt`` directly.

No new third-party dependency is introduced for this: both platforms use
only their own stdlib.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import msvcrt
else:
    import fcntl

#: msvcrt.locking's own LK_LOCK mode retries internally for ~10s before
#: raising OSError rather than blocking indefinitely like fcntl.flock(LOCK_EX)
#: does. We wrap it in our own retry loop to approximate the same "block
#: until acquired" contract, bounded by a generous timeout instead of a
#: literal infinite wait (safer under CI -- an abandoned lock should
#: eventually surface as a real error, not hang the job forever).
_WINDOWS_LOCK_TIMEOUT_SECONDS = 300.0
_WINDOWS_LOCK_RETRY_SECONDS = 0.05
_WINDOWS_LOCK_NBYTES = 1


def _win32_blocking_lock(fileno: int) -> None:
    deadline = time.monotonic() + _WINDOWS_LOCK_TIMEOUT_SECONDS
    while True:
        try:
            msvcrt.locking(fileno, msvcrt.LK_LOCK, _WINDOWS_LOCK_NBYTES)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_WINDOWS_LOCK_RETRY_SECONDS)


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` for the duration of
    the ``with`` block; always released, including when the block raises.

    POSIX: ``fcntl.flock`` (real, indefinitely-blocking exclusive lock).
    Windows: ``msvcrt.locking`` on a 1-byte region of the same file,
    wrapped in :func:`_win32_blocking_lock` to approximate the same
    blocking contract.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        if IS_WINDOWS:
            # msvcrt.locking locks nbytes starting at the current file
            # position -- pin it to a known, stable offset (0) and make
            # sure that byte actually exists in the file first.
            handle.seek(0, 2)  # SEEK_END
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            _win32_blocking_lock(handle.fileno())
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _WINDOWS_LOCK_NBYTES)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
