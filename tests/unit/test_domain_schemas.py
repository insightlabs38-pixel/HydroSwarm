from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hydroswarm.domain.schemas import (
    ActionType,
    CandidateSet,
    ConsequenceMetrics,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)


def test_candidate_probabilities_are_normalized() -> None:
    candidates = CandidateSet(
        node_probabilities={"J1": 0.6, "J2": 0.4},
        node_ids=("J1", "J2"),
        calibrated=True,
        measured_coverage=0.91,
    )
    assert sum(candidates.node_probabilities.values()) == pytest.approx(1.0)

    with pytest.raises(ValidationError, match="sum to 1"):
        CandidateSet(node_probabilities={"J1": 0.7}, node_ids=("J1",))


def test_delayed_observation_is_valid_but_time_travel_is_not() -> None:
    observed = datetime(2026, 8, 3, tzinfo=UTC)
    item = SensorObservation(
        sensor_id="S1",
        node_id="J1",
        observed_at=observed,
        received_at=observed + timedelta(minutes=15),
        concentration_mg_l=0.2,
    )
    assert item.received_at > item.observed_at

    with pytest.raises(ValidationError, match="cannot precede"):
        SensorObservation(
            sensor_id="S1",
            node_id="J1",
            observed_at=observed,
            received_at=observed - timedelta(seconds=1),
        )


def test_plan_is_typed_and_verified_plan_requires_metrics() -> None:
    incident_id = uuid4()
    plan = OperationalPlan(
        incident_id=incident_id,
        name="Low-disruption containment",
        model_version="hydrocore-0.1",
        actions=(
            OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P7"),
            OperationalAction(
                action_type=ActionType.FLUSH_NODE,
                target_id="J9",
                flow_rate_lps=3.0,
                duration_minutes=20,
            ),
        ),
    )
    metrics = ConsequenceMetrics(
        minimum_pressure_m=22.0,
        operation_count=2,
        service_availability=0.98,
    )
    result = PlanVerification(
        plan_id=plan.plan_id,
        decision=PlanDecision.VERIFIED,
        simulator="WNTRSimulator",
        simulator_version="1.5.0",
        state_hash="a" * 64,
        consequences=metrics,
    )
    assert result.consequences.minimum_pressure_m == 22.0

    with pytest.raises(ValidationError, match="require consequence"):
        PlanVerification(
            plan_id=plan.plan_id,
            decision=PlanDecision.VERIFIED,
            simulator="WNTRSimulator",
            simulator_version="1.5.0",
            state_hash="a" * 64,
        )


def test_missing_observation_cannot_smuggle_values() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="cannot carry"):
        SensorObservation(
            sensor_id="S1",
            node_id="J1",
            observed_at=now,
            received_at=now,
            missing=True,
            pressure_m=18.0,
        )
