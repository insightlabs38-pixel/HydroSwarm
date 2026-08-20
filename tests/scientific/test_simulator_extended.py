from __future__ import annotations

import functools
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from hydroswarm.domain import ActionType, OperationalAction, OperationalPlan
from hydroswarm.simulation import (
    HydraulicSimulator,
    IncidentSource,
    IncidentSourceProfile,
    SimulationBudgetExceeded,
    SimulationIncompleteError,
    SimulationTimeoutError,
    SimulationUnstableError,
    build_wntr_network,
    calculate_consequences,
)
from hydroswarm.simulation.wrapper import SimulationError, _invoke_wntr_simulator
from hydroswarm.storage.cache import SimulationResultCache

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "hydrocore_v5"))


def _network(hours: int = 4):
    network = build_wntr_network()
    network.options.time.duration = hours * 3600
    return network


def _plan() -> OperationalPlan:
    return OperationalPlan(
        incident_id=uuid4(),
        name="Monitor J1",
        actions=(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J1"),),
        created_at=datetime(2026, 8, 3, tzinfo=UTC),
        model_version="test",
    )


@pytest.mark.real_simulation
def test_structured_hypothesis_sample_diagnostics_and_consequences() -> None:
    simulator = HydraulicSimulator(_network())
    profile = IncidentSourceProfile("J1", duration_minutes=60)

    result = simulator.simulate_hypothesis(profile)
    assert result.complete and not result.unstable
    assert result.source_node_ids == ("J1",)
    assert result.water_age_hours is not None and result.water_age_hours.shape == result.concentration_mg_l.shape
    assert result.tracer_percent is not None and result.tracer_percent.shape == result.concentration_mg_l.shape
    assert result.pump_energy_kwh == 0.0 and result.pump_cost == 0.0

    sample = simulator.simulate_sample("J2", at_minute=60, simulation=result)
    assert sample.timestamp_seconds == 3600
    assert sample.concentration_mg_l >= 0.0
    assert sample.water_age_hours is not None and sample.tracer_percent is not None

    metrics = calculate_consequences(
        simulator,
        result,
        threshold_mg_l=0.001,
        population_by_node={"J1": 100, "J2": 200, "J3": 300, "J4": 400},
    )
    assert metrics.contaminant_mass_consumed_mg > 0
    assert metrics.contaminated_pipe_extent_m > 0
    assert 0.0 <= metrics.service_availability <= 1.0


@pytest.mark.real_simulation
def test_multiple_source_stress_mode_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="stress_mode"):
        IncidentSourceProfile("J1", additional_sources=(IncidentSource("J3"),))

    simulator = HydraulicSimulator(_network())
    result = simulator.simulate_stress(
        (IncidentSource("J1", 10.0, 0, 60), IncidentSource("J3", 5.0, 30, 30))
    )
    assert result.source_node_ids == ("J1", "J3")
    assert result.profile is not None and result.profile.stress_mode is True
    assert result.concentration_mg_l.to_numpy().max() > 0


@pytest.mark.real_simulation
def test_incident_and_plan_cache_hits_do_not_consume_budget_and_corruption_invalidates(tmp_path) -> None:
    cache = SimulationResultCache(tmp_path / "cache")
    hooks: list[tuple[str, int, int | None]] = []
    simulator = HydraulicSimulator(
        _network(2),
        cache=cache,
        exact_simulation_budget=2,
        budget_hook=lambda operation, used, remaining: hooks.append((operation, used, remaining)),
    )
    profile = IncidentSourceProfile("J1", duration_minutes=30)

    first = simulator.simulate_hypothesis(profile, include_diagnostics=False)
    second = simulator.simulate_hypothesis(profile, include_diagnostics=False)
    assert not first.cache_hit and second.cache_hit
    assert simulator.exact_runs == 1

    plan = _plan()
    evaluation = simulator.evaluate_plan(plan)
    cached_evaluation = simulator.evaluate_plan(plan)
    assert not evaluation.cache_hit and cached_evaluation.cache_hit
    assert simulator.exact_runs == 2 and simulator.remaining_exact_runs == 0
    assert [item[0] for item in hooks] == ["hypothesis", "plan-evaluation"]

    with pytest.raises(SimulationBudgetExceeded):
        simulator.simulate_hypothesis(IncidentSourceProfile("J2"), include_diagnostics=False)

    key = simulator._cache_key(
        profile=profile.as_dict(), plan=None, operation="hypothesis-diagnostics-False"
    )
    cache._path(key).write_text("corrupt", encoding="utf-8")
    assert cache.get(key) is None
    assert not cache._path(key).exists()


@pytest.mark.real_simulation
def test_run_with_timeout_worker_and_args_survive_a_real_spawn_context() -> None:
    """Windows' multiprocessing start method is mandatorily "spawn" (no
    fork() at all there); unlike "fork" (the POSIX default this module
    otherwise uses), "spawn" pickles the process target and its arguments
    to send to a brand-new interpreter rather than inheriting the parent's
    memory. This proves _run_with_timeout's real callers
    (_invoke_wntr_simulator / _invoke_epanet_simulator, called with a real
    prepared WNTR model) are actually picklable under spawn -- a lambda or
    nested-closure worker (what this module used before the Windows
    portability fix) would fail here with a real PicklingError, which
    running only under POSIX's default "fork" context would never catch.
    Runs on every platform: "spawn" is available (just not default) on
    POSIX too, so this is real coverage on ubuntu-latest CI for the code
    path Windows CI actually exercises, not merely "does not raise
    ValueError: cannot find context for 'fork'"."""

    import multiprocessing

    from hydroswarm.simulation.wrapper import (
        _invoke_epanet_simulator,
        _invoke_wntr_simulator,
        _multiprocessing_worker_entrypoint,
    )

    simulator = HydraulicSimulator(_network(1))
    model = simulator._prepared_network()
    context = multiprocessing.get_context("spawn")

    for function, args in (
        (_invoke_wntr_simulator, (model,)),
        (_invoke_epanet_simulator, (model, "spawn-regression-check")),
    ):
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_multiprocessing_worker_entrypoint,
            args=(function, args, result_queue),
            daemon=True,
        )
        process.start()
        process.join(30.0)
        assert not process.is_alive(), f"spawn worker for {function.__name__} did not finish in time"
        succeeded, value = result_queue.get_nowait()
        assert succeeded, f"spawn worker for {function.__name__} failed: {value!r}"
        process.join()


@pytest.mark.real_simulation
def test_timeout_terminates_the_child_process_rather_than_orphaning_it() -> None:
    # core-issues.txt: a killable subprocess, not an orphanable daemon
    # thread -- verify the process that was still running past the
    # deadline is actually reaped (terminated/killed and joined), not left
    # running in the background consuming resources indefinitely.
    import multiprocessing

    simulator = HydraulicSimulator(_network(1), timeout_seconds=0.05)
    assert multiprocessing.active_children() == []
    with pytest.raises(SimulationTimeoutError):
        simulator._run_with_timeout("hang-test", functools.partial(time.sleep, 30))
    assert multiprocessing.active_children() == []


@pytest.mark.real_simulation
def test_timeout_and_result_completeness_fail_closed() -> None:
    simulator = HydraulicSimulator(_network(1), timeout_seconds=0.01)
    with pytest.raises(SimulationTimeoutError):
        simulator._run_with_timeout("slow-test", functools.partial(time.sleep, 0.2))

    simulator = HydraulicSimulator(_network(1))
    incomplete = SimpleNamespace(
        error_code=0,
        node={
            "pressure": pd.DataFrame([[20.0]], index=[0], columns=["J1"]),
            "demand": pd.DataFrame([[0.01]], index=[0], columns=["J1"]),
        },
    )
    with pytest.raises(SimulationIncompleteError, match="ended before"):
        simulator._validate_results(incomplete)

    unstable = SimpleNamespace(
        error_code=0,
        node={
            "pressure": pd.DataFrame([[20.0], [np.nan]], index=[0, 3600], columns=["J1"]),
            "demand": pd.DataFrame([[0.01], [0.01]], index=[0, 3600], columns=["J1"]),
        },
    )
    with pytest.raises(SimulationUnstableError, match="non-finite"):
        simulator._validate_results(unstable)


# --- Milestone 8.5a regressions -------------------------------------------
#
# reports/evaluation/hydrocore-v5/m8-5a-execution.json root-caused M8's
# reported "PDD scalability ceiling" as a false timeout in
# HydraulicSimulator._run_with_timeout, not a genuine solver limitation:
# `process.join(timeout)` was called BEFORE ever reading `result_queue`,
# so a child whose pickled return value exceeded the OS pipe's buffered
# capacity (~64KiB on Linux -- crossed by a real PDD result around 25-49
# grid junctions, matching M8's own reported ceiling) would block flushing
# its queue feeder thread with nobody draining the pipe, appear
# `is_alive()` for the entire timeout, and be killed and reported as
# SimulationTimeoutError even though the underlying computation had
# already finished in milliseconds. Diagnosed empirically: the child's
# real /proc OS state during a false timeout was 'S' (sleeping/blocked-on-
# IO), never 'Z' (zombie) -- ruling out the SIGCHLD/process-reaping
# hypothesis M8.5 speculated but did not confirm.


def _large_payload(n_bytes: int) -> bytes:
    """Module-level (picklable under spawn) worker returning an
    arbitrarily large object through the same IPC path production uses."""

    return os.urandom(n_bytes)


def _raise_value_error() -> None:
    raise ValueError("m8-5a-regression: deliberate child exception")


@pytest.mark.real_simulation
def test_immediate_successful_child_returns_promptly() -> None:
    simulator = HydraulicSimulator(_network(1), timeout_seconds=10.0)
    result = simulator._run_with_timeout("m8-5a-immediate", _large_payload, (16,))
    assert isinstance(result, (bytes, bytearray)) and len(result) == 16


@pytest.mark.real_simulation
def test_large_result_transfer_does_not_falsely_time_out() -> None:
    """The core M8.5a regression: a result payload well above a typical OS
    pipe buffer (64KiB) must still be returned successfully within a
    timeout that would only be exceeded by a genuine hang, not by IPC
    transfer of a large-but-fast-to-compute result."""

    simulator = HydraulicSimulator(_network(1), timeout_seconds=10.0)
    started = time.perf_counter()
    result = simulator._run_with_timeout("m8-5a-large-result", _large_payload, (5_000_000,))
    elapsed = time.perf_counter() - started
    assert isinstance(result, (bytes, bytearray)) and len(result) == 5_000_000
    assert elapsed < 5.0, f"large-result transfer took {elapsed:.2f}s -- IPC-blocking regression reintroduced"


@pytest.mark.real_simulation
def test_child_exception_propagates_through_wrapper() -> None:
    simulator = HydraulicSimulator(_network(1), timeout_seconds=10.0)
    with pytest.raises(ValueError, match="m8-5a-regression"):
        simulator._run_with_timeout("m8-5a-exception", _raise_value_error)


@pytest.mark.real_simulation
def test_genuine_timeout_still_raises_and_reaps_cleanly() -> None:
    import multiprocessing

    simulator = HydraulicSimulator(_network(1), timeout_seconds=0.2)
    assert multiprocessing.active_children() == []
    started = time.perf_counter()
    with pytest.raises(SimulationTimeoutError):
        simulator._run_with_timeout("m8-5a-genuine-timeout", functools.partial(time.sleep, 30))
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, "genuine timeout should be bounded near timeout_seconds, not the full hang duration"
    assert multiprocessing.active_children() == []


@pytest.mark.real_simulation
def test_repeated_large_result_runs_leave_no_leaked_children() -> None:
    import multiprocessing

    simulator = HydraulicSimulator(_network(1), timeout_seconds=10.0)
    for _ in range(20):
        result = simulator._run_with_timeout("m8-5a-repeated-large", _large_payload, (200_000,))
        assert len(result) == 200_000
        assert multiprocessing.active_children() == []


@pytest.mark.real_simulation
def test_wntr_pdd_wrapped_execution_completes_at_previously_falsely_timing_out_size() -> None:
    """N=25 on M8's own deterministic grid generator is the exact size
    M8/M8.5's own reported ceiling first fails at; its pickled PDD result
    (~68KB, reports/evaluation/hydrocore-v5/m8-5a-execution.json) is large
    enough to have falsely timed out under the pre-M8.5a join-before-get
    wrapper. Confirms the corrected wrapper both completes AND agrees
    numerically with the direct/unwrapped call."""

    from run_m8_scaling import build_grid_network

    network, names = build_grid_network(25)
    simulator = HydraulicSimulator(network, timeout_seconds=10.0)
    model = simulator._prepared_network()

    started = time.perf_counter()
    wrapped = simulator._run_with_timeout("m8-5a-wrapped-pdd-25", _invoke_wntr_simulator, (model,))
    wrapped_elapsed = time.perf_counter() - started
    assert wrapped_elapsed < 5.0, f"wrapped PDD execution at N=25 took {wrapped_elapsed:.2f}s -- IPC-blocking regression reintroduced"

    direct = _invoke_wntr_simulator(model)
    pressure_wrapped = wrapped.node["pressure"][names].to_numpy(dtype=float)
    pressure_direct = direct.node["pressure"][names].to_numpy(dtype=float)
    assert np.allclose(pressure_wrapped, pressure_direct, atol=1e-9)


@pytest.mark.real_simulation
def test_run_with_timeout_worker_exit_without_result_raises_simulation_error() -> None:
    """A child that dies without ever putting a result onto the queue
    (distinct from a genuine hang) must fail fast as SimulationError, not
    be misreported as SimulationTimeoutError."""

    simulator = HydraulicSimulator(_network(1), timeout_seconds=5.0)
    started = time.perf_counter()
    with pytest.raises(SimulationError) as excinfo:
        simulator._run_with_timeout("m8-5a-hard-exit", functools.partial(os._exit, 1))
    elapsed = time.perf_counter() - started
    assert excinfo.type is SimulationError, "a dead-without-result child must not be misreported as a timeout"
    assert elapsed < 2.0, "a child that already exited should fail fast, not wait out the full timeout"

