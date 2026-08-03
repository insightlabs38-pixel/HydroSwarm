"""HydroStrategist constrained deterministic response-planning agent."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import UUID, uuid5

from hydroswarm.domain import ActionType, OperationalAction, OperationalPlan

from .base import DeterministicAgent
from .schemas import PlanCandidate, StrategistOutput, ToolPermission


class HydroStrategist(DeterministicAgent[StrategistOutput]):
    output_type = StrategistOutput
    permissions = frozenset(
        {
            ToolPermission.READ_CANDIDATES,
            ToolPermission.READ_HYDRAULICS,
            ToolPermission.READ_VERIFIER_FEEDBACK,
            ToolPermission.GENERATE_PLANS,
            ToolPermission.RUN_INFERENCE,
        }
    )
    visible_fields = frozenset(
        {
            "candidate_region",
            "candidate_probabilities",
            "hydraulic_graph",
            "verifier_feedback",
            "incident_id",
            "planning_round",
        }
    )

    def generate(self, state: Mapping[str, object], *, timeout_seconds: float = 15.0) -> StrategistOutput:
        return self.invoke(state, timeout_seconds=timeout_seconds)

    def revise(self, state: Mapping[str, object], *, timeout_seconds: float = 15.0) -> StrategistOutput:
        return self.invoke(state, timeout_seconds=timeout_seconds)

    @staticmethod
    def should_terminate(output: StrategistOutput) -> bool:
        return not output.plans

    @staticmethod
    def _plan(
        incident_id: UUID,
        actions: tuple[OperationalAction, ...],
        *,
        label: str,
        planning_round: int,
    ) -> OperationalPlan:
        signature = ";".join(
            f"{item.action_type}:{item.target_id}:{item.start_minute}:{item.duration_minutes}"
            for item in actions
        )
        return OperationalPlan(
            plan_id=uuid5(incident_id, f"{planning_round}:{label}:{signature}"),
            incident_id=incident_id,
            name=label,
            actions=actions,
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
            model_version="hydroswarm-strategist-0.1",
        )

    def deterministic_fallback(self, state: Mapping[str, object]) -> StrategistOutput:
        incident_id = UUID(str(state["incident_id"]))
        round_number = int(state.get("planning_round", 0))
        feedback = " ".join(str(item) for item in state.get("verifier_feedback", ()))
        candidates = tuple(str(node) for node in state.get("candidate_region", ()))
        node = candidates[0] if candidates else None
        outputs: list[PlanCandidate] = []
        if node is not None:
            monitor = self._plan(
                incident_id,
                (OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=node),),
                label="Increase monitoring",
                planning_round=round_number,
            )
            outputs.append(PlanCandidate(plan=monitor, estimated_value=0.55, template="increase_monitoring"))
            if "UNKNOWN_TARGET" not in feedback and "INOPERABLE_TARGET" not in feedback:
                sample = self._plan(
                    incident_id,
                    (OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id=node),),
                    label="Collect confirmation sample",
                    planning_round=round_number,
                )
                outputs.append(PlanCandidate(plan=sample, estimated_value=0.50, template="request_sample"))
        if not outputs:
            wait = self._plan(
                incident_id,
                (OperationalAction(action_type=ActionType.WAIT, duration_minutes=15),),
                label="Wait and observe",
                planning_round=round_number,
            )
            outputs.append(PlanCandidate(plan=wait, estimated_value=0.20, template="wait"))
        return StrategistOutput(plans=tuple(outputs), revision_round=round_number)
