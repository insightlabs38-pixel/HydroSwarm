from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hydroswarm.agents import (
    AgentPermissionError,
    FSMState,
    FallbackMode,
    HydroScout,
    HydroSentinel,
    HydroStrategist,
    SwarmController,
    ToolPermission,
)
from hydroswarm.domain import IncidentState, SensorObservation


def _incident() -> IncidentState:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    return IncidentState(
        network_id="unit",
        status="DETECTED",
        observations=(
            SensorObservation(
                sensor_id="S1",
                node_id="J1",
                observed_at=now,
                received_at=now,
                concentration_mg_l=0.2,
            ),
        ),
    )


def test_role_permissions_and_visibility_are_enforced() -> None:
    sentinel = HydroSentinel()
    visible = sentinel.visible_state(
        {
            "observations": (),
            "hydraulic_state": {"pressure": 20},
            "node_ids": ("J1",),
            "verifier_feedback": ("SECRET",),
        }
    )

    assert "verifier_feedback" not in visible
    assert ToolPermission.READ_OBSERVATIONS in sentinel.permissions
    with pytest.raises(AgentPermissionError):
        sentinel.require(ToolPermission.GENERATE_PLANS)
    assert ToolPermission.READ_VERIFIER_FEEDBACK in HydroStrategist().permissions
    assert ToolPermission.RECOMMEND_SAMPLE in HydroScout().permissions


def test_malformed_agent_output_recovers_to_classical_safe_mode() -> None:
    sentinel = HydroSentinel(inference=lambda _: {"not": "a SentinelOutput"})
    output = sentinel.invoke(
        {"observations": _incident().observations, "node_ids": ("J1", "J2")},
        timeout_seconds=1.0,
    )

    assert output.top_candidates
    assert sentinel.last_mode == FallbackMode.CLASSICAL_SAFE
    assert sentinel.last_error is not None


def test_controller_rejects_invalid_transition() -> None:
    controller = SwarmController(
        sentinel=HydroSentinel(),
        scout=HydroScout(),
        strategist=HydroStrategist(),
        verifier=object(),
    )
    with pytest.raises(ValueError, match="invalid swarm transition"):
        controller.transition(FSMState.PLAN_GENERATION)


def test_scout_stops_when_every_candidate_was_sampled() -> None:
    output = HydroScout().invoke(
        {
            "candidate_region": ("J1",),
            "candidate_probabilities": {"J1": 1.0},
            "sampling_history": ("J1",),
            "node_ids": ("J1",),
        },
        timeout_seconds=1.0,
    )
    assert output.action.value == "STOP"
