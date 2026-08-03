from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from hydroswarm.agents import (
    FSMState,
    HydroScout,
    HydroSentinel,
    HydroStrategist,
    SwarmController,
    SwarmLimits,
)
from hydroswarm.domain import (
    ActionType,
    ConsequenceMetrics,
    IncidentState,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)
from hydroswarm.simulation import build_networkx_network
from hydroswarm.simulation import HydraulicSimulator, PlanVerifier, build_wntr_network


pytest.importorskip("wntr")


def _incident() -> IncidentState:
    timestamp = datetime(2026, 8, 3, tzinfo=UTC)
    return IncidentState(
        network_id="hydroswarm-demo",
        status="DETECTED",
        observations=(
            SensorObservation(
                sensor_id="S-J1",
                node_id="J1",
                observed_at=timestamp,
                received_at=timestamp,
                concentration_mg_l=0.2,
                pressure_m=27.0,
            ),
        ),
    )


def test_complete_swarm_pauses_for_approval_and_replays_deterministically() -> None:
    network = build_wntr_network()
    network.options.time.duration = 2 * 3_600
    verifier = PlanVerifier(HydraulicSimulator(network))
    controller = SwarmController(
        sentinel=HydroSentinel(),
        scout=HydroScout(),
        strategist=HydroStrategist(),
        verifier=verifier,
        limits=SwarmLimits(max_incident_runtime_seconds=30.0),
    )

    controller.start(network, _incident())
    approval = controller.run()
    assert approval.state == FSMState.HUMAN_APPROVAL
    assert approval.selected_plan is not None
    assert approval.verification is not None

    # Running again is idempotent and cannot cross the human boundary.
    assert controller.run().events == approval.events
    completed = controller.approve(approval.selected_plan.plan_id)
    assert completed.state == FSMState.COMPLETE
    assert completed.termination_reason is None

    replay = SwarmController.replay(completed.events)
    assert replay.state == FSMState.COMPLETE
    assert replay.event_count == len(completed.events)
    assert replay.final_hash == completed.events[-1].event_hash
    assert replay.trajectory == completed.events


def test_controller_pauses_for_real_sample_when_evidence_is_insufficient() -> None:
    network = build_wntr_network()
    network.options.time.duration = 3_600
    sentinel = HydroSentinel(
        inference=lambda _: {
            "top_candidates": [
                {"node_id": "J1", "probability": 0.4},
                {"node_id": "J2", "probability": 0.3},
                {"node_id": "J3", "probability": 0.3},
            ],
            "candidate_region": ["J1", "J2", "J3"],
            "evidence_sufficient": False,
            "uncertainty": 0.6,
        }
    )
    controller = SwarmController(
        sentinel=sentinel,
        scout=HydroScout(),
        strategist=HydroStrategist(),
        verifier=PlanVerifier(HydraulicSimulator(network)),
    )
    controller.start(network, _incident())

    result = controller.run()
    assert result.state == FSMState.SAMPLE_SELECTION
    assert result.awaiting_sample is not None
    assert result.awaiting_sample.node_id == "J1"


def test_verifier_feedback_drives_a_bounded_strategist_revision() -> None:
    incident = _incident()

    def plan(target: str, action_type: ActionType, revision: int) -> OperationalPlan:
        return OperationalPlan(
            plan_id=UUID(int=revision + 1),
            incident_id=incident.incident_id,
            name=f"revision-{revision}",
            actions=(OperationalAction(action_type=action_type, target_id=target),),
            model_version="test",
        )

    seen_feedback: list[tuple[str, ...]] = []

    def strategy(state):
        feedback = tuple(state.get("verifier_feedback", ()))
        seen_feedback.append(feedback)
        revision = int(state.get("planning_round", 0))
        candidate = (
            plan("J1", ActionType.MONITOR_NODE, revision)
            if feedback
            else plan("P_BAD", ActionType.CLOSE_PIPE, revision)
        )
        return {"plans": [{"plan": candidate, "estimated_value": 0.5}], "revision_round": revision}

    class FeedbackVerifier:
        def prescreen(self, proposed):
            return ()

        def verify(self, proposed):
            rejected = proposed.actions[0].action_type == ActionType.CLOSE_PIPE
            return PlanVerification(
                plan_id=proposed.plan_id,
                decision=PlanDecision.REJECTED if rejected else PlanDecision.VERIFIED,
                simulator="fake",
                simulator_version="1",
                state_hash=("a" if rejected else "b") * 64,
                rejection_codes=("UNKNOWN_TARGET:P_BAD",) if rejected else (),
                consequences=(
                    None
                    if rejected
                    else ConsequenceMetrics(minimum_pressure_m=20.0, operation_count=1)
                ),
            )

    controller = SwarmController(
        sentinel=HydroSentinel(),
        scout=HydroScout(),
        strategist=HydroStrategist(inference=strategy),
        verifier=FeedbackVerifier(),
        limits=SwarmLimits(max_exact_simulations=3, max_planning_rounds=3),
    )
    controller.start(build_networkx_network(), incident)
    result = controller.run()

    assert result.state == FSMState.HUMAN_APPROVAL
    assert any("UNKNOWN_TARGET:P_BAD" in feedback for feedback in seen_feedback)
    assert any(event.to_state == FSMState.REVISE_PLANS for event in result.events)
    assert sum(event.event_type == "plan_verified" for event in result.events) == 2
