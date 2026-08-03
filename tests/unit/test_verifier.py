from __future__ import annotations

from uuid import uuid4

import pytest

from hydroswarm.domain import ActionType, OperationalAction, OperationalPlan, PlanDecision
from hydroswarm.simulation import HydraulicSimulator, PlanVerifier, build_wntr_network


pytest.importorskip("wntr")


def _plan(*actions: OperationalAction) -> OperationalPlan:
    return OperationalPlan(
        incident_id=uuid4(),
        name="verifier test plan",
        actions=actions,
        model_version="test",
    )


@pytest.fixture
def verifier() -> PlanVerifier:
    model = build_wntr_network()
    model.options.time.duration = 2 * 3_600
    return PlanVerifier(
        HydraulicSimulator(model, minimum_pressure_m=10.0, minimum_service_availability=0.90)
    )


def test_safe_monitor_and_sample_plan_is_verified(verifier: PlanVerifier) -> None:
    plan = _plan(
        OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J2"),
        OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id="J3", start_minute=5),
    )
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.VERIFIED
    assert result.consequences is not None
    assert result.consequences.minimum_pressure_m >= 10.0
    assert result.consequences.service_availability == pytest.approx(1.0)
    assert result.consequences.operation_count == 2
    assert len(result.state_hash) == 64


def test_unknown_target_is_rejected_before_simulation(verifier: PlanVerifier) -> None:
    plan = _plan(OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="NOT_A_PIPE"))
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.REJECTED
    assert result.consequences is None
    assert result.rejection_codes == ("UNKNOWN_TARGET:NOT_A_PIPE",)


def test_unsafe_supply_isolation_is_rejected_by_exact_simulation(verifier: PlanVerifier) -> None:
    plan = _plan(OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P_R1_J1"))
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.REJECTED
    assert result.consequences is not None
    assert "PRESSURE_BELOW_MINIMUM" in result.rejection_codes
    assert result.consequences.minimum_pressure_m < 10.0


def test_state_and_dynamic_graph_are_hydraulically_derived(verifier: PlanVerifier) -> None:
    state = verifier.simulator.calculate_state(at_time=0)
    graph = verifier.simulator.build_dynamic_graph(state)

    assert set(state.pressure_m) >= {"J1", "J2", "J3", "J4"}
    assert graph.number_of_edges() == 7
    assert all("travel_time_seconds" in data for *_, data in graph.edges(data=True, keys=True))
    assert graph.nodes["J1"]["reservoir_reachable"] is True


def test_incident_uses_epanet_quality_without_cwd_artifacts(
    verifier: PlanVerifier, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    incident = verifier.simulator.simulate_incident(
        "J1", strength_mg_min=2.0, start_minute=0, duration_minutes=30
    )

    assert incident.concentration_mg_l.shape[1] == 6
    assert incident.concentration_mg_l.to_numpy().max() > 0.0
    assert len(incident.state_hash) == 64
    assert not list(tmp_path.glob("temp.*"))
