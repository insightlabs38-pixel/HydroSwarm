"""Milestone 8.5a Phase 3/4/5/6/7: numerical parity, corrected-wrapper
scale characterization, environment-sensitivity note, and the combined
M8.5a summary.

Phase 3 (numerical parity) runs FIRST and gates Phase 4: if corrected
wrapped execution does not agree with direct/unwrapped execution within
the predeclared tolerance, this script stops before interpreting any scale
result (module docstring requirement -- "Do not proceed to interpret scale
results if wrapped/direct numerical parity fails").

Phase 4 reruns the M8/M8.5 node-count matrix (10/25/50, extended to
100/250 since all three base sizes are expected -- per M8.5a's own
instructions -- to now complete comfortably) through FOUR arms:
  1. direct/unwrapped WNTR-native PDD      (_invoke_wntr_simulator, no
                                             multiprocessing at all)
  2. corrected wrapped WNTR-native PDD     (HydraulicSimulator.
                                             _run_with_timeout, the real,
                                             now-fixed production method)
  3. direct/unwrapped EPANET-backed PDD    (_invoke_epanet_simulator, no
                                             multiprocessing at all)
  4. corrected wrapped EPANET-backed PDD   (_run_with_timeout again, same
                                             production method, different
                                             callable)
using the SAME `build_grid_network` generator Milestone 8 introduced
(imported unmodified, not re-derived).

Timing decomposition: production's `_run_with_timeout` is exercised
unmodified for arms 2/4 (the real total_wall_ms number this milestone
cares about), but the FINER solver/IPC/reaping breakdown Phase 4 also
asks for requires an instrumented copy of the same call pattern (this
script's own `_instrumented_wrapped_call`, mirroring
`HydraulicSimulator._run_with_timeout`'s corrected drain-first logic
exactly, with a side-channel `multiprocessing.Pipe` added only to observe
timestamps -- production's own method and source file are never edited or
monkeypatched). Decomposition is necessarily approximate (a single-process
Python program cannot instrument another process's kernel-level pipe
write() without ptrace-level tooling this milestone does not add) and is
reported as such:
  process_startup_ms        -- Process().start() return until the child's
                                first instrumented timestamp (fork on
                                POSIX: near-zero, no interpreter reinit).
  solver_compute_ms          -- the child's own wall-clock time inside
                                `function(*args)` (the real
                                WNTR/EPANET call), from the child's own
                                clock.
  ipc_ms                      -- from the child finishing computation to
                                the parent's `result_queue.get()`
                                returning: this is what M8.5a diagnosed as
                                the mechanism (pickling + pipe transfer +
                                parent drain).
  termination_reaping_ms     -- from the parent receiving the result to
                                the child being join()'d (reaped).
  wrapper_overhead_ms         -- process_startup_ms + ipc_ms +
                                termination_reaping_ms (everything that
                                is not solver_compute_ms).
  total_wall_ms                -- process_startup_ms + solver_compute_ms +
                                ipc_ms + termination_reaping_ms (should
                                closely track the real, unmodified
                                `_run_with_timeout` measurement in arms
                                2/4; both are reported so any drift
                                between the instrumented harness and
                                production is visible, not hidden).

Writes:
  reports/evaluation/hydrocore-v5/m8-5a-scale.json
  reports/evaluation/hydrocore-v5/m8-5a-summary.md
"""

from __future__ import annotations

import json
import multiprocessing
import queue
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.simulation.wrapper import (  # noqa: E402
    HydraulicSimulator,
    SimulationError,
    _invoke_epanet_simulator,
    _invoke_wntr_simulator,
)
from run_m8_scaling import build_grid_network  # noqa: E402

OUTPUT_SCALE = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-5a-scale.json"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-5a-summary.md"
EXECUTION_REPORT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-5a-execution.json"

REQUIRED_NODE_COUNTS: tuple[int, ...] = (10, 25, 50)
EXTENDED_NODE_COUNTS: tuple[int, ...] = (100, 250)
WRAPPER_TIMEOUT_SECONDS = 60.0  # production's own default; unchanged.
REPEATS_PER_ARM = 5  # repeated runs to characterize variability without turning this into an HPO project.

#: Phase 3 predeclared numerical tolerance (decided before any comparison
#: is inspected) -- identical in spirit to M8.5's own Section 4/5
#: thresholds, reused here rather than re-derived.
PARITY_MAX_ABS_PRESSURE_DIFF_M = 1e-6
PARITY_MAX_ABS_DEMAND_DIFF_M3S = 1e-9


def _instrumented_worker(function, args, result_queue, timing_conn) -> None:
    child_started = time.time()
    try:
        solver_start = time.time()
        result = function(*args)
        solver_end = time.time()
        try:
            timing_conn.send((child_started, solver_start, solver_end))
        finally:
            timing_conn.close()
        result_queue.put((True, result))
    except BaseException as exc:  # noqa: BLE001
        try:
            timing_conn.send((child_started, time.time(), time.time()))
            timing_conn.close()
        except Exception:  # noqa: BLE001
            pass
        result_queue.put((False, exc))


def _instrumented_wrapped_call(function, args: tuple[Any, ...], *, timeout_seconds: float) -> dict[str, Any]:
    """Mirrors HydraulicSimulator._run_with_timeout's corrected
    drain-first logic exactly (production's own method is exercised
    separately, unmodified, for the real total_wall_ms number) with a
    side-channel timing Pipe added only for this diagnostic's finer
    breakdown."""

    context = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = context.Queue(maxsize=1)
    timing_parent, timing_child = context.Pipe(duplex=False)
    process = context.Process(
        target=_instrumented_worker, args=(function, args, result_queue, timing_child), daemon=True,
    )
    t0 = time.time()
    process.start()
    timing_child.close()  # parent's own reference to the child's write end.

    deadline = time.monotonic() + timeout_seconds
    got_result = False
    succeeded = False
    value: Any = None
    result_received_ts: float | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            succeeded, value = result_queue.get(timeout=min(remaining, 0.02))
            result_received_ts = time.time()
            got_result = True
            break
        except queue.Empty:
            if not process.is_alive():
                break
            continue

    timed_out = not got_result
    if timed_out:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
        process.join()
    else:
        process.join(5.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()
        else:
            process.join()
    reaped_ts = time.time()

    child_started = solver_start = solver_end = None
    if timing_parent.poll(1.0):
        try:
            child_started, solver_start, solver_end = timing_parent.recv()
        except EOFError:
            pass
    timing_parent.close()
    result_queue.close()
    result_queue.join_thread()
    process.close()

    process_startup_ms = ((child_started - t0) * 1000.0) if child_started is not None else None
    solver_compute_ms = ((solver_end - solver_start) * 1000.0) if (solver_start is not None and solver_end is not None) else None
    ipc_ms = ((result_received_ts - solver_end) * 1000.0) if (result_received_ts is not None and solver_end is not None) else None
    termination_reaping_ms = ((reaped_ts - result_received_ts) * 1000.0) if result_received_ts is not None else None
    total_wall_ms = (reaped_ts - t0) * 1000.0
    wrapper_overhead_ms = None
    if process_startup_ms is not None and ipc_ms is not None and termination_reaping_ms is not None:
        wrapper_overhead_ms = process_startup_ms + ipc_ms + termination_reaping_ms

    return {
        "status": "TIMEOUT" if timed_out else ("OK" if succeeded else "CHILD_ERROR"),
        "error": None if (not timed_out and succeeded) else (str(value) if (not timed_out and not succeeded) else "timeout"),
        "process_startup_ms": process_startup_ms, "solver_compute_ms": solver_compute_ms,
        "ipc_ms": ipc_ms, "termination_reaping_ms": termination_reaping_ms,
        "wrapper_overhead_ms": wrapper_overhead_ms, "total_wall_ms": total_wall_ms,
        "value": value if (not timed_out and succeeded) else None,
    }


def _timed(fn):
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000.0


def run_numerical_parity(sizes: tuple[int, ...]) -> dict[str, Any]:
    rows = {}
    all_pass = True
    for size in sizes:
        network, names = build_grid_network(size)
        simulator = HydraulicSimulator(network, timeout_seconds=WRAPPER_TIMEOUT_SECONDS)

        # Fresh deep copy per call: WNTRSimulator.run_sim() mutates its
        # input WaterNetworkModel in place (confirmed by inspection --
        # reusing one `model` object across both calls made the SECOND
        # call's input already-mutated by the first, producing a spurious
        # "parity failure" that was actually a test-harness bug, not a
        # wrapper defect). `_prepared_network()` deep-copies `self.network`
        # fresh on every call, matching M8.5's own `_arm_model` methodology.
        direct = _invoke_wntr_simulator(simulator._prepared_network())
        wrapped = simulator._run_with_timeout(
            f"m8-5a-parity-{size}", _invoke_wntr_simulator, (simulator._prepared_network(),)
        )

        pressure_direct = direct.node["pressure"][names].to_numpy(dtype=float)
        pressure_wrapped = wrapped.node["pressure"][names].to_numpy(dtype=float)
        demand_direct = direct.node["demand"][names].to_numpy(dtype=float)
        demand_wrapped = wrapped.node["demand"][names].to_numpy(dtype=float)

        max_abs_pressure_diff = float(np.max(np.abs(pressure_direct - pressure_wrapped)))
        max_abs_demand_diff = float(np.max(np.abs(demand_direct - demand_wrapped)))
        passed = (
            max_abs_pressure_diff <= PARITY_MAX_ABS_PRESSURE_DIFF_M
            and max_abs_demand_diff <= PARITY_MAX_ABS_DEMAND_DIFF_M3S
        )
        all_pass = all_pass and passed
        rows[str(size)] = {
            "max_abs_pressure_diff_m": max_abs_pressure_diff, "max_abs_demand_diff_m3s": max_abs_demand_diff,
            "max_relative_pressure_diff": float(
                np.max(np.abs(pressure_direct - pressure_wrapped) / np.maximum(np.abs(pressure_direct), 1e-9))
            ),
            "pass": passed,
        }
    return {
        "thresholds": {
            "max_abs_pressure_diff_m": PARITY_MAX_ABS_PRESSURE_DIFF_M,
            "max_abs_demand_diff_m3s": PARITY_MAX_ABS_DEMAND_DIFF_M3S,
        },
        "sizes_tested": sizes, "results": rows, "overall_pass": all_pass,
    }


def run_scale_matrix(sizes: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for target in sizes:
        network, names = build_grid_network(target)
        simulator = HydraulicSimulator(network, timeout_seconds=WRAPPER_TIMEOUT_SECONDS)
        # A fresh deep copy of the network is built for EVERY invocation
        # below (never one `model` object reused across calls): confirmed
        # by inspection (and by the Phase 3 parity-check bug this fix
        # mirrors) that WNTRSimulator.run_sim() mutates its input
        # WaterNetworkModel in place, so reusing one object across repeats
        # would make later repeats measure an already-mutated input, not
        # the same problem every time.

        entry: dict[str, Any] = {"target_node_count": target, "actual_node_count": len(names)}

        # Arm 1: direct/unwrapped WNTR-native PDD.
        direct_wntr_repeats = []
        for _ in range(REPEATS_PER_ARM):
            try:
                _result, ms = _timed(lambda: _invoke_wntr_simulator(simulator._prepared_network()))
                direct_wntr_repeats.append({"status": "OK", "wall_ms": ms})
            except Exception as exc:  # noqa: BLE001
                direct_wntr_repeats.append({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        entry["direct_unwrapped_wntr_pdd"] = {"repeats": direct_wntr_repeats}

        # Arm 2: corrected wrapped WNTR-native PDD -- production's real,
        # unmodified _run_with_timeout for the authoritative total_wall_ms,
        # plus one instrumented call for the finer decomposition.
        wrapped_wntr_repeats = []
        for i in range(REPEATS_PER_ARM):
            started = time.perf_counter()
            try:
                simulator._run_with_timeout(f"m8-5a-scale-wntr-{target}-{i}", _invoke_wntr_simulator, (simulator._prepared_network(),))
                wrapped_wntr_repeats.append({"status": "OK", "wall_ms": (time.perf_counter() - started) * 1000.0})
            except Exception as exc:  # noqa: BLE001
                wrapped_wntr_repeats.append({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        decomposition_wntr = _instrumented_wrapped_call(_invoke_wntr_simulator, (simulator._prepared_network(),), timeout_seconds=WRAPPER_TIMEOUT_SECONDS)
        entry["corrected_wrapped_wntr_pdd"] = {"repeats": wrapped_wntr_repeats, "decomposition_sample": decomposition_wntr}

        # Arm 3: direct/unwrapped EPANET-backed PDD.
        direct_epanet_repeats = []
        for i in range(REPEATS_PER_ARM):
            try:
                _result, ms = _timed(lambda i=i: _invoke_epanet_simulator(simulator._prepared_network(), f"m8-5a-direct-epanet-{target}-{i}"))
                direct_epanet_repeats.append({"status": "OK", "wall_ms": ms})
            except Exception as exc:  # noqa: BLE001
                direct_epanet_repeats.append({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        entry["direct_unwrapped_epanet_pdd"] = {"repeats": direct_epanet_repeats}

        # Arm 4: corrected wrapped EPANET-backed PDD.
        wrapped_epanet_repeats = []
        for i in range(REPEATS_PER_ARM):
            started = time.perf_counter()
            try:
                simulator._run_with_timeout(f"m8-5a-scale-epanet-{target}-{i}", _invoke_epanet_simulator, (simulator._prepared_network(), f"m8-5a-scale-epanet-{target}-{i}"))
                wrapped_epanet_repeats.append({"status": "OK", "wall_ms": (time.perf_counter() - started) * 1000.0})
            except Exception as exc:  # noqa: BLE001
                wrapped_epanet_repeats.append({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        decomposition_epanet = _instrumented_wrapped_call(
            _invoke_epanet_simulator, (simulator._prepared_network(), f"m8-5a-scale-epanet-decomp-{target}"), timeout_seconds=WRAPPER_TIMEOUT_SECONDS,
        )
        entry["corrected_wrapped_epanet_pdd"] = {"repeats": wrapped_epanet_repeats, "decomposition_sample": decomposition_epanet}

        rows.append(entry)
    return rows


def _leak_check(sizes: tuple[int, ...]) -> dict[str, Any]:
    network, names = build_grid_network(max(sizes))
    simulator = HydraulicSimulator(network, timeout_seconds=WRAPPER_TIMEOUT_SECONDS)
    leaked = 0
    for i in range(REPEATED_RUN_COUNT := 15):
        before = len(multiprocessing.active_children())
        try:
            simulator._run_with_timeout(f"m8-5a-leak-check-fixed-{i}", _invoke_wntr_simulator, (simulator._prepared_network(),))
        except SimulationError:
            pass
        after = len(multiprocessing.active_children())
        if after > before:
            leaked += 1
    return {"node_count": max(sizes), "n_runs": REPEATED_RUN_COUNT, "runs_with_leaked_children": leaked}


def build_decision(parity: dict[str, Any], scale: list[dict[str, Any]], leak_check: dict[str, Any]) -> dict[str, Any]:
    if not parity["overall_pass"]:
        return {
            "primary_scalability_decision": "ROOT_CAUSE_UNRESOLVED_MORE_DIAGNOSTIC_REQUIRED",
            "large_network_experiments_reopened": False,
            "rationale": "Numerical parity between corrected wrapped and direct/unwrapped execution FAILED; scale results are not interpreted per this milestone's own predeclared rule.",
        }
    all_arms_ok = all(
        all(r["status"] == "OK" for r in row["direct_unwrapped_wntr_pdd"]["repeats"])
        and all(r["status"] == "OK" for r in row["corrected_wrapped_wntr_pdd"]["repeats"])
        and all(r["status"] == "OK" for r in row["direct_unwrapped_epanet_pdd"]["repeats"])
        and all(r["status"] == "OK" for r in row["corrected_wrapped_epanet_pdd"]["repeats"])
        for row in scale
    )
    largest_tested = max(row["target_node_count"] for row in scale)
    decision = "WRAPPED_EXECUTION_BLOCKER_RESOLVED" if all_arms_ok and leak_check["runs_with_leaked_children"] == 0 else "WRAPPED_EXECUTION_BLOCKER_REMAINS"
    return {
        "primary_scalability_decision": decision,
        "large_network_experiments_reopened": bool(all_arms_ok and leak_check["runs_with_leaked_children"] == 0),
        "largest_node_count_tested": largest_tested,
        "all_arms_completed_at_all_tested_sizes": all_arms_ok,
        "leaked_children_after_repeated_corrected_runs": leak_check["runs_with_leaked_children"],
        "rationale": (
            f"All four arms (direct/wrapped x WNTR-native/EPANET-backed PDD) completed successfully at every tested "
            f"size up to N={largest_tested} with the corrected wrapper, and repeated runs left zero leaked children. "
            "Milestone 8's original PDD-scalability-blocker attribution is retracted: the corrected measurements show "
            "no genuine solver scaling limitation at the sizes actually tested; the original ceiling was the "
            "wrapped-execution IPC defect M8.5a fixed."
            if decision == "WRAPPED_EXECUTION_BLOCKER_RESOLVED" else
            "At least one arm failed or leaked children at a tested size even with the corrected wrapper; see "
            "per-size results before claiming the blocker is resolved."
        ),
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    parity = run_numerical_parity(REQUIRED_NODE_COUNTS)

    scale_sizes = REQUIRED_NODE_COUNTS
    extended_status: dict[str, str] = {}
    if parity["overall_pass"]:
        scale_sizes = REQUIRED_NODE_COUNTS + EXTENDED_NODE_COUNTS
        for size in EXTENDED_NODE_COUNTS:
            extended_status[str(size)] = "ATTEMPTED"
    else:
        for size in EXTENDED_NODE_COUNTS:
            extended_status[str(size)] = "NOT_RUN: numerical parity failed at required sizes"

    scale = run_scale_matrix(scale_sizes) if parity["overall_pass"] else []
    leak_check = _leak_check(REQUIRED_NODE_COUNTS) if parity["overall_pass"] else {"node_count": None, "n_runs": 0, "runs_with_leaked_children": 0}
    decision = build_decision(parity, scale, leak_check)

    locked_after = locked_test_opened(ROOT)

    report = {
        "schema_version": 1,
        "purpose": "Milestone 8.5a Phase 3/4: numerical parity gate and corrected-wrapper scale characterization.",
        "branch": "exp/hydrocore-v5-causal",
        "required_node_counts": REQUIRED_NODE_COUNTS,
        "extended_node_counts_attempted": EXTENDED_NODE_COUNTS,
        "extended_status": extended_status,
        "repeats_per_arm": REPEATS_PER_ARM,
        "numerical_parity": parity,
        "scale_matrix": scale,
        "repeated_leak_check_corrected_wrapper": leak_check,
        "decision": decision,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    OUTPUT_SCALE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SCALE.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    execution_report = json.loads(EXECUTION_REPORT.read_text(encoding="utf-8")) if EXECUTION_REPORT.exists() else None
    _write_summary(execution_report, report)

    print(json.dumps({"decision": decision, "numerical_parity_pass": parity["overall_pass"]}, indent=2, default=str))
    return 0


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _write_summary(execution_report: dict[str, Any] | None, scale_report: dict[str, Any]) -> None:
    lines = [
        "# Milestone 8.5a summary: corrected wrapped simulation execution and scaling attribution",
        "",
        "**Supersedes** the PDD-scalability-blocker attribution in `m8-summary.md` and the "
        "`PDD_SCALABILITY_BLOCKER_REMAINS` decision token in `m8_5-summary.md`. See "
        "`reports/evaluation/hydrocore-v5/m8-5a-execution.json` and `m8-5a-scale.json` for full raw data.",
        "",
        "## What M8/M8.5 got right and wrong",
        "",
        "1. M8 correctly observed a failure in the measured wrapped execution path at 25-49 grid junctions.",
        "2. M8 did NOT validly establish that PDD/WNTR/EPANET itself was the scaling bottleneck -- that "
        "attribution is **RETRACTED** by this milestone.",
        "3. M8.5 correctly demonstrated that direct/unwrapped PDD completed in milliseconds on cases where the "
        "wrapped path failed, and correctly flagged (without confirming) that the wrapper's own process-completion "
        "detection, not the solver, was implicated.",
        "4. M8.5 explicitly speculated a SIGCHLD/zombie-reaping mechanism. **This milestone tested that hypothesis "
        "directly and REFUTES it**: the child's real `/proc` OS state during every observed false timeout was "
        "`S` (sleeping/blocked-on-IO), never `Z` (zombie).",
        "5. The actual mechanism, established with root_cause_established=YES "
        "(`m8-5a-execution.json:root_cause`): `HydraulicSimulator._run_with_timeout` called `process.join(timeout)` "
        "BEFORE ever reading `result_queue`. A child's return value only finishes crossing the OS pipe once the "
        "parent drains it; nothing drained it until join() returned, and join() could not return until the child "
        "exited, which it could not do until the pipe drained -- Python's own documented "
        "\"joining processes that use queues\" deadlock. Once a real PDD result's pickled size crosses the pipe's "
        "buffered capacity (empirically ~60-100KB on this host; a real 25-node grid's PDD result is ~68KB), the "
        "child blocks mid-write and is misreported as hung.",
        "6. Previously reported wrapped-path timeout values from M8/M8.5 must NOT be interpreted as solver-"
        "performance measurements -- they measured this IPC defect, not PDD/WNTR/EPANET performance.",
        "",
        "## Phase 1: root-cause diagnostic",
        "",
    ]
    if execution_report:
        rc = execution_report["root_cause"]
        lines += [
            f"Classification: **{rc['classification']}** (root cause established: {rc['root_cause_established']})",
            "",
            f"Payload-size threshold where join-first first falsely times out: {rc['payload_bytes_threshold_where_join_first_first_fails']} bytes.",
            f"Zombie OS state ever observed during a false timeout: {rc['zombie_os_state_ever_observed_during_false_timeout']}.",
            f"Running/blocked OS state observed during a false timeout: {rc['running_or_blocked_os_state_observed_during_false_timeout']}.",
            "",
            "| payload bytes | join-first timed out | join-first OS state | drain-first timed out | confound |",
            "|---|---|---|---|---|",
        ]
        for row in execution_report["payload_size_sweep"]:
            lines.append(
                f"| {row['payload_bytes']} | {row['join_first']['timed_out']} | {row['join_first']['os_state_at_join_check']} | "
                f"{row['drain_first']['timed_out']} | {row['join_first_falsely_timed_out']} |"
            )
        lines += [
            "",
            "Real-simulation reproduction (ABCD minimal comparison, C=wrapped production method, D=direct/unwrapped):",
            "",
            "| N | C status | C wall ms | D status | D wall ms | confound reproduced |",
            "|---|---|---|---|---|---|",
        ]
        for size, entry in execution_report["abcd_minimal_comparison"]["C_and_D_real_simulation"].items():
            c, d = entry["C_wrapped_production_method"], entry["D_direct_unwrapped"]
            lines.append(
                f"| {size} | {c['status']} | {_fmt(c['wall_ms'])} | {d['status']} | {_fmt(d['wall_ms'])} | "
                f"{entry['C_falsely_timed_out_relative_to_D']} |"
            )
        lc = execution_report["repeated_leak_check_current_unfixed_wrapper"]
        lines += [
            "",
            f"Pre-fix baseline (current/unfixed wrapper, N={lc['node_count']}, {lc['n_runs']} runs, "
            f"{lc['timeout_seconds']}s timeout): {lc['timeout_count']}/{lc['n_runs']} falsely timed out; "
            f"leaked children detected: {lc['leaked_children_detected']}.",
        ]
    else:
        lines.append("(m8-5a-execution.json not found -- Phase 1 diagnostic must be run before this summary is meaningful.)")

    lines += [
        "",
        "## Phase 2: corrected wrapped execution path",
        "",
        "`HydraulicSimulator._run_with_timeout` (src/hydroswarm/simulation/wrapper.py) now drains `result_queue` "
        "WHILE waiting (bounded polling `result_queue.get(timeout=...)` against the deadline, with an "
        "`is_alive()` liveness check on `queue.Empty` to still fail fast on a genuine crash-without-a-result), "
        "instead of `process.join(timeout)` followed by `result_queue.get_nowait()`. The hard timeout, "
        "terminate/kill escalation, exception propagation, and IPC-resource cleanup are all preserved; a genuine "
        "hang still raises `SimulationTimeoutError` within the configured `timeout_seconds`. Regression tests "
        "added to `tests/scientific/test_simulator_extended.py` (large-result transfer, child exception, genuine "
        "timeout, repeated-run leak check, and a real N=25 PDD run at the exact size that previously falsely "
        "timed out).",
        "",
        "## Phase 3: numerical parity",
        "",
    ]
    parity = scale_report["numerical_parity"]
    lines += [
        f"Predeclared thresholds: max abs pressure diff <= {parity['thresholds']['max_abs_pressure_diff_m']} m, "
        f"max abs demand diff <= {parity['thresholds']['max_abs_demand_diff_m3s']} m3/s.",
        "",
        f"**Overall parity: {'PASS' if parity['overall_pass'] else 'FAIL'}**",
        "",
        "| N | max abs pressure diff (m) | max abs demand diff (m3/s) | pass |",
        "|---|---|---|---|",
    ]
    for size, row in parity["results"].items():
        lines.append(f"| {size} | {row['max_abs_pressure_diff_m']:.3e} | {row['max_abs_demand_diff_m3s']:.3e} | {row['pass']} |")

    lines += [
        "",
        "## Phase 4: scale characterization (corrected wrapper)",
        "",
    ]
    if scale_report["scale_matrix"]:
        lines += [
            "| N | direct WNTR ms (mean) | wrapped WNTR ms (mean) | direct EPANET ms (mean) | wrapped EPANET ms (mean) | "
            "solver ms | IPC ms | reaping ms | wrapper overhead ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in scale_report["scale_matrix"]:

            def _mean(repeats):
                oks = [r["wall_ms"] for r in repeats if r["status"] == "OK"]
                return sum(oks) / len(oks) if oks else None

            decomp_w = row["corrected_wrapped_wntr_pdd"]["decomposition_sample"]
            lines.append(
                f"| {row['target_node_count']} | {_fmt(_mean(row['direct_unwrapped_wntr_pdd']['repeats']))} | "
                f"{_fmt(_mean(row['corrected_wrapped_wntr_pdd']['repeats']))} | "
                f"{_fmt(_mean(row['direct_unwrapped_epanet_pdd']['repeats']))} | "
                f"{_fmt(_mean(row['corrected_wrapped_epanet_pdd']['repeats']))} | "
                f"{_fmt(decomp_w['solver_compute_ms'])} | {_fmt(decomp_w['ipc_ms'])} | "
                f"{_fmt(decomp_w['termination_reaping_ms'])} | {_fmt(decomp_w['wrapper_overhead_ms'])} |"
            )
        for size, status in scale_report["extended_status"].items():
            if status != "ATTEMPTED":
                lines.append("")
                lines.append(f"N={size}: {status}")
        leak = scale_report["repeated_leak_check_corrected_wrapper"]
        lines += [
            "",
            f"Repeated-run leak check (corrected wrapper, N={leak['node_count']}, {leak['n_runs']} runs): "
            f"{leak['runs_with_leaked_children']} runs left a leaked child.",
        ]
    else:
        lines.append("NOT RUN: numerical parity failed at the required sizes (Phase 3 gate).")

    decision = scale_report["decision"]
    lines += [
        "",
        "## Phase 5: environment sensitivity",
        "",
        "Tested in the single execution environment available to this milestone (see VALIDATION note in "
        "`m8-5a-execution.json.environment`); a second, independently provisioned environment was not stood up "
        "per this milestone's own scope constraints (\"do not spend substantial time provisioning unrelated "
        "infrastructure\"). The mechanism identified (a general Python `multiprocessing.Queue` \"join before "
        "drain\" deadlock, gated by the OS pipe buffer size) is documented CPython behavior on any POSIX "
        "\"fork\"-context host, not specific to a sandboxing detail of this container -- but this milestone did "
        "not empirically confirm that in a second environment, so the conclusion is reported as "
        "MIXED_WRAPPER_AND_ENVIRONMENT_INTERACTION: a portable wrapper defect (confirmed here), whose exact "
        "failure threshold (pipe buffer size) is itself an environment-dependent parameter.",
        "",
        "## Phase 7: decision",
        "",
        f"**{decision['primary_scalability_decision']}**",
        "",
        decision["rationale"],
        "",
        f"LARGE_NETWORK_EXPERIMENTS_REOPENED: {'YES' if decision['large_network_experiments_reopened'] else 'NO'}",
        "",
        "PYG_NOT_JUSTIFIED: retained unchanged -- this milestone is a wrapper/IPC correction, not a neural-layer "
        "or graph-batching finding; M8's own synthetic neural-inference-scaling evidence is untouched.",
        "",
        f"locked tests opened: before={scale_report['locked_test_opened_before']}, after={scale_report['locked_test_opened_after']}.",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
