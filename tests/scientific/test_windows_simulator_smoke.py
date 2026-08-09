"""Dedicated, deliberately small real-simulator smoke group for Windows CI
(backend-ci-portability follow-up: the earlier Windows fcntl/resource and
multiprocessing-spawn fixes made the FULL scientific suite collectible and
runnable on Windows, but running it in full there is pathologically
expensive -- every real HydraulicSimulator call now pays a fresh
interpreter startup under "spawn", and the full suite makes thousands of
such calls (see the full_simulation marker in pyproject.toml and the
audit that produced it)).

This file's job is different from the rest of the scientific suite: prove
NATIVE Windows spawn correctness ONCE, end-to-end, for every real behavior
that matters -- not repeat the same interpreter-startup cost hundreds of
times. None of these tests fake sys.platform or force a particular
multiprocessing context; they call the real public HydraulicSimulator/API
surface, so on a real windows-latest runner HydraulicSimulator.
_run_with_timeout's own `"spawn" if sys.platform == "win32" else "fork"`
selection genuinely picks "spawn" here, with zero test-side faking. On
POSIX (where this file also runs, as part of the ordinary full suite) the
same code genuinely picks "fork" instead -- still real coverage, just not
the Windows-specific path.

Not marked full_simulation: deliberately small (each test makes at most a
handful of real simulate calls), so it is safe and fast to include both in
Ubuntu's full run and as Windows CI's separate, explicitly-reported smoke
step (see .github/workflows/ci.yml) without meaningfully adding to either.

Kept where they already were, not duplicated here (both already unmarked/
cheap, so already part of Windows CI's broad job too):
- test_simulator_extended.py::
  test_run_with_timeout_worker_and_args_survive_a_real_spawn_context --
  the lower-level pickling proof (forces multiprocessing.get_context
  ("spawn") directly, on every platform including POSIX).
- test_simulator_extended.py::
  test_timeout_terminates_the_child_process_rather_than_orphaning_it --
  timeout/termination via functools.partial(time.sleep, ...).
"""

from __future__ import annotations

import multiprocessing
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
import wntr
from fastapi.testclient import TestClient

from hydroswarm.api.app import create_app
from hydroswarm.domain import (
    ActionType,
    CandidateSet,
    ConsequenceMetrics,
    OperationalAction,
    OperationalPlan,
)
from hydroswarm.simulation import (
    HydraulicSimulator,
    IncidentSourceProfile,
    PlanEvaluationContext,
    SimulationError,
    SimulationTimeoutError,
    build_wntr_network,
)
from hydroswarm.storage.cache import SimulationResultCache

pytest.importorskip("wntr")

NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _network(hours: int = 2):
    network = build_wntr_network()
    network.options.time.duration = hours * 3600
    return network


def _plan() -> OperationalPlan:
    return OperationalPlan(
        incident_id=uuid4(),
        name="Monitor J1",
        actions=(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J1"),),
        created_at=NOW,
        model_version="windows-smoke",
    )


def _sleep_worker(seconds: float) -> None:
    """Module-level (picklable-under-spawn) worker for the timeout test
    below -- a lambda/closure would defeat the point of this file."""
    time.sleep(seconds)


def _raise_worker() -> None:
    """Module-level (picklable-under-spawn) worker that deliberately
    raises, for test_real_simulator_exception_propagates_through_the_child_process."""
    raise SimulationError("deliberate windows-smoke failure")


def test_real_wntr_hydraulics_simulation_end_to_end() -> None:
    """WNTR path: HydraulicSimulator.simulate_hypothesis -> _run_hydraulics
    -> _run_with_timeout -> real subprocess (spawn on real Windows, fork on
    POSIX) -> wntr.sim.WNTRSimulator. One real call."""

    simulator = HydraulicSimulator(_network())
    profile = IncidentSourceProfile("J1", duration_minutes=60)

    result = simulator.simulate_hypothesis(profile)

    assert result.complete and not result.unstable
    assert result.source_node_ids == ("J1",)
    assert result.concentration_mg_l is not None


def test_real_epanet_simulation_end_to_end() -> None:
    """EPANET (water-quality) path: HydraulicSimulator.evaluate_plan ->
    _run_epanet -> _run_with_timeout -> real subprocess (spawn on real
    Windows, fork on POSIX) -> wntr.sim.EpanetSimulator, via the canonical
    exposure-aware verification path (evaluate_plan_consequences) so the
    real chemical-transport results are actually used, not just the
    hydraulic-only fields evaluate_plan() alone would report. One real
    EPANET call (single source_profile => one hypothesis)."""

    simulator = HydraulicSimulator(_network())
    context = PlanEvaluationContext(
        contamination_threshold_mg_l=0.001,
        source_profile=IncidentSourceProfile("J1", duration_minutes=60),
    )

    evaluation = simulator.evaluate_plan_consequences(_plan(), context)

    assert evaluation.consequences is not None
    assert evaluation.consequences.exposure_evaluated is True
    assert evaluation.exact_simulation_count == 1
    assert simulator.exact_runs >= 1


def test_real_timeout_terminates_the_spawned_or_forked_child() -> None:
    """Real, killable-subprocess timeout/termination behavior, through
    whichever context (spawn/fork) this platform actually selects -- not
    an orphaned/leaked process either way, and a real SimulationTimeoutError
    (not a hang or a generic failure) out the other side."""

    simulator = HydraulicSimulator(_network(1), timeout_seconds=0.1)
    assert multiprocessing.active_children() == []

    with pytest.raises(SimulationTimeoutError):
        simulator._run_with_timeout("windows-smoke-hang", _sleep_worker, (5.0,))

    assert multiprocessing.active_children() == []


def test_real_simulator_exception_propagates_through_the_child_process() -> None:
    """A real exception raised inside the child (spawn on Windows, fork on
    POSIX) must come back out of _run_with_timeout as the SAME exception,
    not be swallowed, hang, or turn into a generic worker-exited error --
    proves _multiprocessing_worker_entrypoint's (True/False, value) result
    protocol round-trips a real exception object across the process
    boundary (pickling an exception is a distinct concern from pickling a
    plain return value)."""

    simulator = HydraulicSimulator(_network(1))
    with pytest.raises(SimulationError, match="deliberate windows-smoke failure"):
        simulator._run_with_timeout("windows-smoke-raise", _raise_worker, ())


def test_real_exact_plan_verification_path(tmp_path) -> None:
    """One exact consequence/plan-verification path through
    HydraulicSimulator directly (evaluate_plan -> real EPANET run ->
    real ConsequenceMetrics), independent of the API-level test below."""

    simulator = HydraulicSimulator(_network(), cache=SimulationResultCache(tmp_path / "cache"))
    plan = _plan()

    evaluation = simulator.evaluate_plan(plan)

    assert isinstance(evaluation.consequences, ConsequenceMetrics)
    assert evaluation.consequences.minimum_pressure_m is not None
    assert evaluation.cache_hit is False

    # A second evaluation of the identical plan must hit the real cache
    # rather than re-running the simulator -- proves the budget accounting
    # this path depends on is itself platform-independent.
    cached = simulator.evaluate_plan(plan)
    assert cached.cache_hit is True


def test_real_end_to_end_api_incident_workflow_invokes_the_real_simulator(tmp_path) -> None:
    """One representative end-to-end incident workflow through the real
    (non-injected) API stack: import a real network, create an incident,
    generate a bounded candidate plan, and verify it through the live
    `/plans/{id}/verify` endpoint -- which goes through the real
    PlanVerifier/HydraulicSimulator, not a test-only injected verifier.
    Bounded: 2 real EPANET runs (one per source hypothesis), matching this
    project's own documented exact-simulation accounting."""

    app = create_app(
        network_directory=tmp_path / "networks",
        ledger_path=tmp_path / "ledger.db",
        database_path=tmp_path / "db.sqlite",
    )
    client = TestClient(app)

    model = _network()
    network_path = tmp_path / "net.inp"
    wntr.network.write_inpfile(model, str(network_path))
    with open(network_path, "rb") as handle:
        imported = client.post(
            "/api/networks/import", files={"file": ("net.inp", handle, "text/plain")}
        )
    assert imported.status_code == 201, imported.text
    network_id = imported.json()["network_id"]

    created = client.post(
        "/api/incidents",
        json={
            "network_id": network_id,
            "detected_at": NOW.isoformat(),
            "observations": [
                {
                    "sensor_id": "S-J2",
                    "node_id": "J2",
                    "observed_at": NOW.isoformat(),
                    "received_at": NOW.isoformat(),
                    "concentration_mg_l": 0.05,
                    "pressure_m": 20.0,
                }
            ],
            "contamination_threshold_mg_l": 0.001,
        },
    )
    assert created.status_code == 201
    incident_id = UUID(created.json()["incident_id"])
    assert client.post(f"/api/incidents/{incident_id}/analyze").status_code == 200

    #: Force a CALIBRATED candidate set (this test's own deterministic
    #: fixture) so plan generation/verify builds a real bounded hypothesis
    #: set -- matches tests/integration/test_live_exposure_verification.py's
    #: established pattern for exercising this same real endpoint.
    record = app.state.runtime.incidents[incident_id]
    record.state = record.state.model_copy(
        update={
            "candidates": CandidateSet(
                node_probabilities={"J1": 0.7, "J3": 0.3},
                node_ids=("J1", "J3"),
                calibrated=True,
            )
        }
    )

    plans_response = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1})
    assert plans_response.status_code == 200
    plan_id = plans_response.json()[0]["plan_id"]

    verify_response = client.post(f"/api/incidents/{incident_id}/plans/{plan_id}/verify")
    assert verify_response.status_code == 200
    result = verify_response.json()

    assert result["consequences"] is not None
    assert result["consequences"]["exposure_evaluated"] is True
    assert result["evaluation_provenance"]["exact_simulation_count"] == 2
