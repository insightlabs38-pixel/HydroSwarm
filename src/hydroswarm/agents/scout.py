"""HydroScout deterministic active-sampling agent."""

from __future__ import annotations

from collections.abc import Mapping

from .base import DeterministicAgent
from .schemas import ScoutAction, ScoutOutput, ToolPermission


class HydroScout(DeterministicAgent[ScoutOutput]):
    output_type = ScoutOutput
    permissions = frozenset(
        {
            ToolPermission.READ_CANDIDATES,
            ToolPermission.READ_HYDRAULICS,
            ToolPermission.READ_SAMPLING_HISTORY,
            ToolPermission.RECOMMEND_SAMPLE,
            ToolPermission.RUN_INFERENCE,
        }
    )
    visible_fields = frozenset(
        {"candidate_region", "candidate_probabilities", "hydraulic_graph", "sampling_history", "node_ids"}
    )

    def recommend(self, state: Mapping[str, object], *, timeout_seconds: float = 15.0) -> ScoutOutput:
        return self.invoke(state, timeout_seconds=timeout_seconds)

    @staticmethod
    def should_terminate(output: ScoutOutput) -> bool:
        return output.action == ScoutAction.STOP

    def deterministic_fallback(self, state: Mapping[str, object]) -> ScoutOutput:
        sampled = set(str(value) for value in state.get("sampling_history", ()))
        probabilities = state.get("candidate_probabilities", {})
        region = tuple(str(node) for node in state.get("candidate_region", ()))
        pool = [node for node in region if node not in sampled]
        if not pool:
            pool = [str(node) for node in state.get("node_ids", ()) if str(node) not in sampled]
        if not pool:
            return ScoutOutput(action=ScoutAction.STOP, reason="no unsampled accessible nodes")
        if isinstance(probabilities, Mapping):
            pool.sort(key=lambda node: (-float(probabilities.get(node, 0.0)), node))
        else:
            pool.sort()
        information_gain = min(1.0, 1.0 / max(1, len(region)))
        if information_gain < 0.01:
            return ScoutOutput(action=ScoutAction.STOP, reason="marginal information value below threshold")
        return ScoutOutput(
            action=ScoutAction.SAMPLE,
            node_id=pool[0],
            expected_information_gain=information_gain,
            expected_candidate_reduction=min(1.0, information_gain),
            estimated_delay_minutes=10.0,
            estimated_cost=1.0,
            alternatives=tuple(pool[1:3]),
            reason="highest deterministic posterior value among unsampled nodes",
        )
