"""SUB-12.1 #22: cross-platform native-CI smoke test.

Launches the real `hydroswarm.cli start` server (the exact command every
`start_hydroswarm_*` launcher runs) against the interpreter that invokes
this script, waits for it to report healthy, hits `/api/health`,
`/api/reference-demo`, and `/api/live-example-inputs` (a real, WNTR/EPANET
water-quality-simulated result, not a fixture) for real, and stops the
server cleanly. Deliberately pure Python (no shell-specific job control)
so the identical script runs unmodified on Linux, macOS, and Windows CI
runners -- the cross-platform matrix's whole point is proving native
portability, not proving five different platform-specific smoke scripts
each work. The `/api/live-example-inputs` check is what actually catches
the native linux-arm64 EPANET water-quality gap (wntr ships no
linux-arm64 EPANET binary upstream); it runs identically on every
platform rather than being special-cased to arm64.

This is intentionally a second, independent proof beyond `hydroswarm
self-test --strict`: self-test proves the *components* (model, WNTR,
SQLite, bundle) are ready in isolation; this proves the actual server
process binds, serves real HTTP, and shuts down cleanly -- the same
proof a judge doing `./start_hydroswarm_*.sh` and opening a browser gets,
just automated.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
PORT = 8765
BASE_URL = f"http://{HOST}:{PORT}"
STARTUP_TIMEOUT_SECONDS = 90
POLL_INTERVAL_SECONDS = 1.0
SHUTDOWN_TIMEOUT_SECONDS = 15


def _get(path: str, timeout: float = 5.0) -> tuple[int, bytes]:
    request = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def _wait_for_health(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server process exited early with code {process.returncode} "
                "before ever reporting healthy"
            )
        try:
            status, _ = _get("/api/health")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if status == 200:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"server did not report healthy within {STARTUP_TIMEOUT_SECONDS}s")


def _stop(process: subprocess.Popen[bytes]) -> None:
    """`hydroswarm.cli start` itself launches uvicorn as a grandchild via a
    blocking `subprocess.run` (see `_start_console`), so `process.terminate()`
    alone only signals the immediate child -- uvicorn survives as an
    orphan, keeps the port bound, and keeps this script's inherited stdout
    pipe open (so a later `.read()` on it would hang forever waiting for
    an EOF that never comes). The process was started in its own group
    (see `main()`) specifically so shutdown can target the whole tree."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


def main() -> int:
    print(f"[native-ci-smoke] launching: {sys.executable} -m hydroswarm.cli start")
    # Own process group/session so `_stop()` can terminate the whole tree
    # (uvicorn included), not just this immediate child.
    group_kwargs: dict[str, object] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "hydroswarm.cli", "start", "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **group_kwargs,
    )
    try:
        print("[native-ci-smoke] waiting for /api/health ...")
        _wait_for_health(process)
        print("[native-ci-smoke] server is healthy")

        status, body = _get("/api/health")
        assert status == 200, f"/api/health returned {status}"
        print(f"[native-ci-smoke] GET /api/health -> 200: {body[:200]!r}")

        status, body = _get("/api/reference-demo", timeout=30.0)
        assert status == 200, f"/api/reference-demo returned {status}"
        assert len(body) > 0, "/api/reference-demo returned an empty body"
        print(f"[native-ci-smoke] GET /api/reference-demo -> 200 ({len(body)} bytes)")

        # A real, WNTR-simulated water-quality result -- not merely a 200
        # status -- on every native platform this script runs on. This is
        # the concrete regression check for the native linux-arm64 EPANET
        # gap: wntr ships no linux-arm64 EPANET binary upstream, so
        # without scripts/build_epanet_arm64.sh (now run automatically by
        # setup_hydroswarm_linux.sh) this call fails there with a
        # wrong-ELF-class dlopen error even though `self-test --strict`'s
        # bounded hydraulic simulation does not need that binary and would
        # still pass. Real on every platform, not linux-arm64-specific
        # logic, matching this script's own cross-platform-identical design.
        status, body = _get("/api/live-example-inputs", timeout=60.0)
        assert status == 200, f"/api/live-example-inputs returned {status}"
        live_example = json.loads(body)
        signatures = live_example["candidate_signatures_mg_l"]
        assert signatures, "/api/live-example-inputs returned no candidate signatures"
        assert any(value > 0.0 for value in signatures.values()), (
            "/api/live-example-inputs returned no positive real EPANET water-quality "
            f"concentration -- real simulation did not run: {signatures!r}"
        )
        print(
            "[native-ci-smoke] GET /api/live-example-inputs -> 200 "
            f"(real EPANET water-quality simulation OK, {len(signatures)} candidate nodes)"
        )
    finally:
        print("[native-ci-smoke] stopping server ...")
        _stop(process)
        remaining_output = process.stdout.read() if process.stdout else b""
        if remaining_output:
            print("[native-ci-smoke] server output:")
            print(remaining_output.decode(errors="replace"))

    print("[native-ci-smoke] PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"[native-ci-smoke] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[native-ci-smoke] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
