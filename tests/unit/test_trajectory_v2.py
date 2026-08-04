from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from hydroswarm.domain import (
    ActionType,
    IncidentState,
    OperationalAction,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.training import (
    FullTrajectory,
    RemainingBudgets,
    TrajectoryIntegrityError,
    TrajectoryState,
)


def _incident_state(*, status: str = "SAMPLING") -> IncidentState:
    return IncidentState(
        incident_id=uuid4(),
        network_id="net-1",
        status=status,
        observations=(
            SensorObservation(
                sensor_id="S1",
                node_id="J1",
                observed_at=datetime(2026, 8, 3, tzinfo=UTC),
                received_at=datetime(2026, 8, 3, tzinfo=UTC),
                concentration_mg_l=0.1,
            ),
        ),
    )


def _budgets(**overrides) -> RemainingBudgets:
    base = {"samples": 3, "exact_simulations": 2, "actions": 5}
    base.update(overrides)
    return RemainingBudgets(**base)


def test_remaining_budgets_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        RemainingBudgets(samples=-1, exact_simulations=0, actions=0)


def test_trajectory_state_round_trips_through_json() -> None:
    state = TrajectoryState(
        step_index=0,
        incident_state=_incident_state(),
        remaining_budgets=_budgets(),
        previous_action=OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J1"),
    )
    payload = state.to_json()
    restored = TrajectoryState.from_json(payload)
    assert restored.step_index == state.step_index
    assert restored.incident_state.incident_id == state.incident_state.incident_id
    assert restored.remaining_budgets == state.remaining_budgets
    assert restored.previous_action == state.previous_action
    assert restored.state_hash == state.state_hash


def test_trajectory_state_rejects_incompatible_schema_version() -> None:
    state = TrajectoryState(step_index=0, incident_state=_incident_state(), remaining_budgets=_budgets())
    payload = state.to_json()
    payload["schema_version"] = "hydroswarm-trajectory-v1"
    with pytest.raises(ValueError, match="incompatible trajectory schema version"):
        TrajectoryState.from_json(payload)


def test_full_trajectory_requires_contiguous_zero_based_steps() -> None:
    step0 = TrajectoryState(step_index=0, incident_state=_incident_state(), remaining_budgets=_budgets())
    step2 = TrajectoryState(step_index=2, incident_state=_incident_state(), remaining_budgets=_budgets())
    with pytest.raises(TrajectoryIntegrityError, match="contiguous"):
        FullTrajectory(trajectory_id="t1", scenario_id="s1", steps=(step0, step2))


def test_full_trajectory_verifies_hash_chain_between_steps() -> None:
    second_state = TrajectoryState(step_index=1, incident_state=_incident_state(status="PLANNING"), remaining_budgets=_budgets(samples=2))
    first_state = TrajectoryState(
        step_index=0,
        incident_state=_incident_state(),
        remaining_budgets=_budgets(),
        resulting_next_state_hash=second_state.state_hash,
    )
    trajectory = FullTrajectory(trajectory_id="t1", scenario_id="s1", steps=(first_state, second_state))
    assert len(trajectory.steps) == 2


def test_full_trajectory_rejects_broken_hash_chain() -> None:
    second_state = TrajectoryState(step_index=1, incident_state=_incident_state(status="PLANNING"), remaining_budgets=_budgets(samples=2))
    first_state = TrajectoryState(
        step_index=0,
        incident_state=_incident_state(),
        remaining_budgets=_budgets(),
        resulting_next_state_hash="0" * 64,  # deliberately wrong
    )
    with pytest.raises(TrajectoryIntegrityError, match="does not match"):
        FullTrajectory(trajectory_id="t1", scenario_id="s1", steps=(first_state, second_state))


def test_full_trajectory_round_trips_through_json() -> None:
    step = TrajectoryState(step_index=0, incident_state=_incident_state(), remaining_budgets=_budgets())
    trajectory = FullTrajectory(trajectory_id="t1", scenario_id="s1", steps=(step,))
    restored = FullTrajectory.from_json(trajectory.to_json())
    assert restored.trajectory_id == trajectory.trajectory_id
    assert restored.scenario_id == trajectory.scenario_id
    assert restored.steps[0].incident_state.incident_id == step.incident_state.incident_id


def test_trajectory_state_carries_verifier_feedback_and_selected_action() -> None:
    plan_id = uuid4()
    verification = PlanVerification(
        plan_id=plan_id,
        decision=PlanDecision.REJECTED,
        simulator="WNTRSimulator",
        simulator_version="1.5.0",
        state_hash="a" * 64,
        rejection_codes=("PRESSURE_VIOLATION",),
    )
    selected = OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P1")
    state = TrajectoryState(
        step_index=0,
        incident_state=_incident_state(status="PLANNING"),
        remaining_budgets=_budgets(),
        verifier_feedback=verification,
        selected_next_action=selected,
    )
    restored = TrajectoryState.from_json(state.to_json())
    assert restored.verifier_feedback.decision == PlanDecision.REJECTED
    assert restored.selected_next_action.target_id == "P1"


def test_full_trajectory_requires_at_least_one_step() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        FullTrajectory(trajectory_id="t1", scenario_id="s1", steps=())
