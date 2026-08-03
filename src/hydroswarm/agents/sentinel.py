"""HydroSentinel deterministic localization agent."""

from __future__ import annotations

from collections.abc import Mapping

from hydroswarm.domain import OODLevel, SensorObservation

from .base import DeterministicAgent
from .schemas import CandidateScore, SentinelOutput, ToolPermission


class HydroSentinel(DeterministicAgent[SentinelOutput]):
    output_type = SentinelOutput
    permissions = frozenset(
        {ToolPermission.READ_OBSERVATIONS, ToolPermission.READ_HYDRAULICS, ToolPermission.RUN_INFERENCE}
    )
    visible_fields = frozenset({"observations", "hydraulic_state", "node_ids", "prior_candidates"})

    def analyze(self, state: Mapping[str, object], *, timeout_seconds: float = 15.0) -> SentinelOutput:
        return self.invoke(state, timeout_seconds=timeout_seconds)

    @staticmethod
    def should_terminate(output: SentinelOutput) -> bool:
        return output.ood_level == OODLevel.OUTSIDE_VALIDATED_RANGE

    def deterministic_fallback(self, state: Mapping[str, object]) -> SentinelOutput:
        observations = tuple(state.get("observations", ()))
        usable = [
            item
            for item in observations
            if isinstance(item, SensorObservation) and not item.missing and item.quality > 0
        ]
        prior = state.get("prior_candidates")
        if isinstance(prior, Mapping) and prior:
            raw_weights = {str(node): max(0.0, float(weight)) for node, weight in prior.items()}
        else:
            observed_nodes = sorted({item.node_id for item in usable})
            candidate_nodes = observed_nodes or [str(node) for node in state.get("node_ids", ())]
            raw_weights = {node: 1.0 for node in candidate_nodes}
        if not raw_weights:
            raw_weights = {"UNKNOWN": 1.0}
        total = sum(raw_weights.values()) or float(len(raw_weights))
        probabilities = {node: weight / total for node, weight in raw_weights.items()}
        ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        region: list[str] = []
        coverage = 0.0
        for node, probability in ranked:
            region.append(node)
            coverage += probability
            if coverage >= 0.90:
                break
        healthy_fraction = len(usable) / len(observations) if observations else 0.0
        sufficient = len(region) <= 2 and bool(usable)
        ood = OODLevel.NORMAL if healthy_fraction >= 0.5 else OODLevel.CAUTION
        return SentinelOutput(
            top_candidates=tuple(CandidateScore(node_id=node, probability=prob) for node, prob in ranked),
            candidate_region=tuple(region),
            target_coverage=0.90,
            evidence_sufficient=sufficient,
            sensor_fault_nodes=tuple(
                sorted({item.node_id for item in observations if isinstance(item, SensorObservation) and item.missing})
            ),
            uncertainty=min(1.0, 1.0 - ranked[0][1]),
            ood_level=ood,
        )
