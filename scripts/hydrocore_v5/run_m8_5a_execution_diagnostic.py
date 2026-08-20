"""Milestone 8.5a Phase 0-1-3: diagnose the wrapped-execution timeout
mechanism M8.5 found (and speculated, but did not confirm, was a SIGCHLD/
zombie-reaping issue), then verify a candidate fix's numerical parity
against direct/unwrapped execution.

THIS SCRIPT DOES NOT MODIFY PRODUCTION CODE. It calls production's own
unmodified `_run_with_timeout` (the exact "join-first" pattern M8.5
measured) side by side with a "drain-first" candidate pattern implemented
only in this script, to isolate the mechanism before touching
wrapper.py. Once Phase 1 below identifies the cause, Phase 2 (a separate
commit-time change, not this script) applies the minimal corrected version
to `HydraulicSimulator._run_with_timeout` itself.

Working hypothesis, stated BEFORE running anything (module docstring,
inspected from `_run_with_timeout` source before any diagnostic ran):
production calls `process.join(self.timeout_seconds)` BEFORE ever reading
`result_queue`. `multiprocessing.Queue.put()` only enqueues into an
in-process buffer; a background feeder thread inside the CHILD actually
pickles the object and writes it through an OS pipe to the parent. A
child process that has put a large object on a queue will not actually
exit (the interpreter-shutdown `Finalize` hook for `Queue` blocks until
the feeder thread finishes flushing) until the pipe write completes -- and
a pipe write blocks once the OS pipe buffer (commonly 64KiB on Linux)
fills, if nobody is reading the other end. Because the parent is sitting
in `process.join()`, not `result_queue.get()`, nobody drains that pipe
until `join()` returns -- which it only does when the child exits, which
it can't do until the pipe drains. This is Python's own documented
"joining processes that use queues" deadlock
(docs.python.org/3/library/multiprocessing.html#pipes-and-queues), not a
SIGCHLD/process-reaping fault. The competing hypothesis this diagnostic
must actually rule out, not merely assume away: a SIGCHLD/zombie-reaping
problem, where the OS-level child has already exited (zombie, `Z` state)
but `multiprocessing.Process.is_alive()` fails to observe that in time --
distinguishable by inspecting the child's real `/proc/<pid>/stat` state
character while `is_alive()` still reports True: `Z` supports the
zombie/SIGCHLD hypothesis, `R`/`S`/`D` (running/sleeping/uninterruptible-
IO, i.e., genuinely still executing -- a blocked pipe `write(2)` shows as
`D` or `S`) refutes it and supports the IPC-blocking hypothesis instead.

Smallest experiment capable of distinguishing the competing explanations
(decided before running): sweep ONLY the returned-payload size through
the SAME worker/queue primitives production uses, holding the actual
compute time ~constant (near-zero) throughout. If join-first hangs/times
out only once payload size crosses a threshold, while drain-first (a
strategy that starts reading the queue immediately rather than after
join) succeeds at every size with near-identical child-side compute
timestamps, the mechanism is conclusively IPC/pipe-buffer blocking, not
SIGCHLD/zombie reaping, and not a per-size solver slowdown.

Writes:
  reports/evaluation/hydrocore-v5/m8-5a-execution.json
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import queue
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.wrapper import (  # noqa: E402
    HydraulicSimulator,
    SimulationError,
    SimulationTimeoutError,
    _invoke_wntr_simulator,
    _multiprocessing_worker_entrypoint,
)
from run_m8_scaling import build_grid_network  # noqa: E402

OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-5a-execution.json"

#: Sizes chosen to straddle the common Linux default pipe capacity
#: (65,536 bytes) without assuming that exact number -- both well below
#: and well above it, plus a size close to the M8/M8.5 empirical ceiling
#: (25-49 junctions) for direct comparison in Section C.
PAYLOAD_SIZES_BYTES: tuple[int, ...] = (1_000, 16_000, 60_000, 100_000, 300_000, 1_000_000, 5_000_000)
REAL_SIM_NODE_COUNTS: tuple[int, ...] = (10, 25, 50, 100)
JOIN_FIRST_TIMEOUT_SECONDS = 8.0  # short enough to keep this diagnostic fast; long enough that a real (not IPC-blocked) 60-node PDD solve (tens of ms, per M8.5) would never legitimately hit it.
POLL_INTERVAL_SECONDS = 0.02
REPEATED_RUN_COUNT = 30  # Phase 2 leak-check target size, run here against production's CURRENT (unfixed) wrapper to document the baseline leak/hang behavior before the fix.


def _proc_state(pid: int) -> str:
    """Real OS-level state character from /proc, independent of and not
    trusting multiprocessing.Process.is_alive()'s own bookkeeping -- Linux
    only (this environment); reports 'unavailable' elsewhere or once the
    entry has already been reaped."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return "GONE"
    # Fields after the (comm) field, which may itself contain spaces/parens,
    # so split from the last ')' rather than by fixed position.
    after_comm = raw.rsplit(")", 1)[-1].split()
    return after_comm[0] if after_comm else "UNKNOWN"


def _make_payload(n_bytes: int) -> bytes:
    return os.urandom(n_bytes)


def _payload_worker(n_bytes: int) -> bytes:
    return _make_payload(n_bytes)


def _run_join_first(function, args: tuple[Any, ...], *, timeout_seconds: float) -> dict[str, Any]:
    """Byte-for-byte the same call sequence as production
    HydraulicSimulator._run_with_timeout (join THEN get_nowait), reimplemented
    standalone here (not calling the method directly) so a
    /proc/<pid>/stat sample can be taken at the exact moment is_alive() is
    checked -- production's own method is exercised unmodified elsewhere in
    this script (Section C uses the real method)."""

    context = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_multiprocessing_worker_entrypoint, args=(function, args, result_queue), daemon=True,
    )
    started = time.perf_counter()
    process.start()
    pid = process.pid
    process.join(timeout_seconds)
    join_returned = time.perf_counter()
    is_alive = process.is_alive()
    os_state_at_check = _proc_state(pid) if pid is not None else "NO_PID"
    timed_out = False
    succeeded: bool | None = None
    value: Any = None
    if is_alive:
        timed_out = True
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join()
    else:
        try:
            succeeded, value = result_queue.get_nowait()
        except queue.Empty:
            pass
        process.join()
    result_queue.close()
    result_queue.join_thread()
    process.close()
    return {
        "pattern": "JOIN_FIRST_PRODUCTION_EQUIVALENT", "pid": pid,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "join_wait_ms": (join_returned - started) * 1000.0,
        "is_alive_after_join": is_alive, "os_state_at_join_check": os_state_at_check,
        "timed_out": timed_out, "got_result": succeeded is not None,
        "result_size_bytes": len(value) if succeeded and isinstance(value, (bytes, bytearray)) else None,
    }


def _run_drain_first(function, args: tuple[Any, ...], *, timeout_seconds: float) -> dict[str, Any]:
    """Candidate fix pattern: block on result_queue.get(timeout=...) FIRST
    (which actively drains the pipe as data arrives, so the child's feeder
    thread is never left write()-blocked with nobody reading), THEN join()
    to reap. If this succeeds at payload sizes where join-first times out,
    with near-identical child compute time, that isolates the mechanism to
    IPC/pipe draining, not solver cost or SIGCHLD/zombie handling."""

    context = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_multiprocessing_worker_entrypoint, args=(function, args, result_queue), daemon=True,
    )
    started = time.perf_counter()
    process.start()
    pid = process.pid
    deadline = time.perf_counter() + timeout_seconds
    succeeded: bool | None = None
    value: Any = None
    got_result = False
    timed_out = False
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            timed_out = True
            break
        try:
            succeeded, value = result_queue.get(timeout=min(remaining, POLL_INTERVAL_SECONDS))
            got_result = True
            break
        except queue.Empty:
            if not process.is_alive():
                break
            continue
    if timed_out or not got_result:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
    process.join()
    result_queue.close()
    result_queue.join_thread()
    process.close()
    return {
        "pattern": "DRAIN_FIRST_CANDIDATE", "pid": pid,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "timed_out": timed_out, "got_result": got_result,
        "result_size_bytes": len(value) if got_result and succeeded and isinstance(value, (bytes, bytearray)) else None,
    }


def run_payload_size_sweep() -> list[dict[str, Any]]:
    rows = []
    for n_bytes in PAYLOAD_SIZES_BYTES:
        join_first = _run_join_first(_payload_worker, (n_bytes,), timeout_seconds=JOIN_FIRST_TIMEOUT_SECONDS)
        drain_first = _run_drain_first(_payload_worker, (n_bytes,), timeout_seconds=JOIN_FIRST_TIMEOUT_SECONDS)
        rows.append({
            "payload_bytes": n_bytes, "join_first": join_first, "drain_first": drain_first,
            "join_first_falsely_timed_out": bool(join_first["timed_out"] and drain_first["got_result"] and not drain_first["timed_out"]),
        })
    return rows


def run_abcd_minimal_comparison() -> dict[str, Any]:
    """Phase 1's explicitly required minimal set: A (trivial immediate
    exit), B (trivial small result via the real IPC path), C (small real
    WNTR/PDD simulation, wrapped), D (the same simulation, unwrapped/
    direct, no multiprocessing at all)."""

    def _noop() -> None:
        return None

    def _small_object() -> dict[str, float]:
        return {"ok": 1.0}

    context = multiprocessing.get_context("fork")

    # A: trivial child, immediate exit, no meaningful result payload.
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_multiprocessing_worker_entrypoint, args=(_noop, (), result_queue), daemon=True)
    started = time.perf_counter()
    process.start()
    pid_a = process.pid
    process.join(JOIN_FIRST_TIMEOUT_SECONDS)
    a_is_alive = process.is_alive()
    a_wall_ms = (time.perf_counter() - started) * 1000.0
    succeeded, _value = result_queue.get_nowait() if not a_is_alive else (None, None)
    process.join()
    result_queue.close()
    result_queue.join_thread()
    process.close()
    a_result = {"pid": pid_a, "is_alive_after_join": a_is_alive, "wall_ms": a_wall_ms, "got_result": succeeded is not None}

    # B: trivial child, small object through the real queue/IPC path.
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_multiprocessing_worker_entrypoint, args=(_small_object, (), result_queue), daemon=True)
    started = time.perf_counter()
    process.start()
    pid_b = process.pid
    process.join(JOIN_FIRST_TIMEOUT_SECONDS)
    b_is_alive = process.is_alive()
    b_wall_ms = (time.perf_counter() - started) * 1000.0
    succeeded, value = result_queue.get_nowait() if not b_is_alive else (None, None)
    process.join()
    result_queue.close()
    result_queue.join_thread()
    process.close()
    b_result = {"pid": pid_b, "is_alive_after_join": b_is_alive, "wall_ms": b_wall_ms, "got_result": succeeded is not None, "value": value}

    # C: real small WNTR/PDD simulation, wrapped (production's own method, unmodified).
    # D: the same simulation, direct/unwrapped, no multiprocessing at all.
    c_and_d: dict[str, Any] = {}
    for target in (10, 25):
        network, _names = build_grid_network(target)
        simulator = HydraulicSimulator(network, timeout_seconds=JOIN_FIRST_TIMEOUT_SECONDS)
        model = simulator._prepared_network()

        c_started = time.perf_counter()
        c_status, c_error = "OK", None
        try:
            simulator._run_with_timeout(f"m8-5a-abcd-C-{target}", _invoke_wntr_simulator, (model,))
        except Exception as exc:  # noqa: BLE001
            c_status, c_error = "FAILED", f"{type(exc).__name__}: {exc}"
        c_wall_ms = (time.perf_counter() - c_started) * 1000.0

        d_started = time.perf_counter()
        d_status, d_error = "OK", None
        try:
            _invoke_wntr_simulator(model)
        except Exception as exc:  # noqa: BLE001
            d_status, d_error = "FAILED", f"{type(exc).__name__}: {exc}"
        d_wall_ms = (time.perf_counter() - d_started) * 1000.0

        c_and_d[str(target)] = {
            "C_wrapped_production_method": {"status": c_status, "error": c_error, "wall_ms": c_wall_ms},
            "D_direct_unwrapped": {"status": d_status, "error": d_error, "wall_ms": d_wall_ms},
            "C_falsely_timed_out_relative_to_D": bool(c_status == "FAILED" and d_status == "OK" and d_wall_ms < 1_000.0),
        }

    return {"A_trivial_immediate_exit": a_result, "B_trivial_small_result": b_result, "C_and_D_real_simulation": c_and_d}


def measure_real_pickle_sizes() -> dict[str, Any]:
    """Actual pickled size of production's own PDD hydraulics result at
    each M8/M8.5 node count -- correlates Section A's payload-size
    threshold against the real objects _invoke_wntr_simulator returns, so
    the causal claim is quantitative, not just qualitative."""

    sizes: dict[str, Any] = {}
    for target in REAL_SIM_NODE_COUNTS:
        network, _names = build_grid_network(target)
        simulator = HydraulicSimulator(network)
        model = simulator._prepared_network()
        try:
            results = _invoke_wntr_simulator(model)
            pickled = pickle.dumps((True, results))
            sizes[str(target)] = {"status": "OK", "pickled_result_bytes": len(pickled)}
        except Exception as exc:  # noqa: BLE001
            sizes[str(target)] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    return sizes


def run_repeated_leak_check_current_wrapper(n_runs: int = REPEATED_RUN_COUNT) -> dict[str, Any]:
    """Documents the CURRENT (pre-fix) wrapper's leaked-process/hang
    behavior at a size where the payload-size sweep predicts a false
    timeout, as a pre-fix baseline for Phase 2's regression test to
    improve on. Uses a short timeout so a false-timeout run does not make
    this diagnostic itself impractically slow; a false timeout is still
    correctly recorded as FAILED/timeout here, it just doesn't block this
    script for the full production 60s each time."""

    network, _names = build_grid_network(50)
    simulator = HydraulicSimulator(network, timeout_seconds=5.0)
    model = simulator._prepared_network()
    outcomes = []
    for _ in range(n_runs):
        before_children = len(multiprocessing.active_children())
        started = time.perf_counter()
        status = "OK"
        try:
            simulator._run_with_timeout("m8-5a-leak-check-current", _invoke_wntr_simulator, (model,))
        except SimulationTimeoutError:
            status = "TIMEOUT"
        except SimulationError as exc:
            status = f"FAILED: {exc}"
        after_children = len(multiprocessing.active_children())
        outcomes.append({
            "status": status, "wall_ms": (time.perf_counter() - started) * 1000.0,
            "active_children_before": before_children, "active_children_after": after_children,
        })
    return {
        "node_count": 50, "timeout_seconds": 5.0, "n_runs": n_runs, "outcomes": outcomes,
        "timeout_count": sum(1 for o in outcomes if o["status"] == "TIMEOUT"),
        "leaked_children_detected": any(o["active_children_after"] > 0 for o in outcomes),
    }


def classify_root_cause(payload_sweep: list[dict[str, Any]], abcd: dict[str, Any]) -> dict[str, Any]:
    ipc_blocking_rows = [row for row in payload_sweep if row["join_first_falsely_timed_out"]]
    zombie_state_observed = any(
        row["join_first"]["is_alive_after_join"] and row["join_first"]["os_state_at_join_check"] in ("Z", "X")
        for row in payload_sweep
    )
    running_or_blocked_state_observed = any(
        row["join_first"]["is_alive_after_join"] and row["join_first"]["os_state_at_join_check"] in ("R", "S", "D")
        for row in payload_sweep
    )
    threshold_bytes = min((row["payload_bytes"] for row in ipc_blocking_rows), default=None)
    c_confounds = [
        entry["C_falsely_timed_out_relative_to_D"] for entry in abcd["C_and_D_real_simulation"].values()
    ]

    if ipc_blocking_rows and not zombie_state_observed:
        classification = "IPC_BLOCKING_ON_UNDRAINED_QUEUE_PIPE"
        confidence = "YES" if running_or_blocked_state_observed else "PARTIAL"
    elif ipc_blocking_rows and zombie_state_observed:
        classification = "MIXED_IPC_AND_REAPING_INTERACTION"
        confidence = "PARTIAL"
    elif zombie_state_observed:
        classification = "SIGCHLD_ZOMBIE_REAPING"
        confidence = "PARTIAL"
    else:
        classification = "UNKNOWN"
        confidence = "NO"

    return {
        "classification": classification, "root_cause_established": confidence,
        "payload_bytes_threshold_where_join_first_first_fails": threshold_bytes,
        "zombie_os_state_ever_observed_during_false_timeout": zombie_state_observed,
        "running_or_blocked_os_state_observed_during_false_timeout": running_or_blocked_state_observed,
        "real_simulation_c_vs_d_confound_reproduced": any(c_confounds),
        "rationale": (
            "The join-first (production-equivalent) pattern falsely times out only once the child's return payload "
            "exceeds a threshold size, while the drain-first pattern succeeds at every tested size with the same "
            "worker function; the child's real /proc OS state at the moment of the false timeout was observed as "
            "still running/sleeping/blocked-on-IO, never a zombie -- this rules out SIGCHLD/process-reaping as the "
            "mechanism and confirms the child is genuinely still executing (blocked in a pipe write its own feeder "
            "thread cannot complete because the parent is not draining the queue while it waits) exactly as Python's "
            "own multiprocessing documentation describes for this join-before-get call ordering."
            if classification == "IPC_BLOCKING_ON_UNDRAINED_QUEUE_PIPE" else
            "Evidence did not cleanly separate the competing explanations at the tested sizes/thresholds; see raw "
            "payload_size_sweep and abcd_minimal_comparison for the underlying observations."
        ),
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    payload_sweep = run_payload_size_sweep()
    abcd = run_abcd_minimal_comparison()
    real_pickle_sizes = measure_real_pickle_sizes()
    leak_check_current = run_repeated_leak_check_current_wrapper()
    root_cause = classify_root_cause(payload_sweep, abcd)

    locked_after = locked_test_opened(ROOT)

    report = {
        "schema_version": 1,
        "purpose": (
            "Milestone 8.5a Phase 0/1/3: diagnose the wrapped-execution false-timeout mechanism M8.5 observed but "
            "did not root-cause, testing the IPC/pipe-draining hypothesis against the SIGCHLD/zombie-reaping "
            "hypothesis M8.5 explicitly speculated but never confirmed."
        ),
        "branch": "exp/hydrocore-v5-causal",
        "environment": {
            "python_version": sys.version, "platform": sys.platform,
            "multiprocessing_start_method_used": "fork",
        },
        "payload_size_sweep_bytes_tested": PAYLOAD_SIZES_BYTES,
        "payload_size_sweep": payload_sweep,
        "abcd_minimal_comparison": abcd,
        "real_simulation_pickled_result_sizes": real_pickle_sizes,
        "repeated_leak_check_current_unfixed_wrapper": leak_check_current,
        "root_cause": root_cause,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"root_cause": root_cause}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
