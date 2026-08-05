from __future__ import annotations

import functools
import time
from datetime import UTC, datetime
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
from hydroswarm.storage.cache import SimulationResultCache


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

