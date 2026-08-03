"""Physical source screening from directed travel-time consistency."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping

from .dynamic_graph import DirectedHydraulicGraph, NodeId


@dataclass(frozen=True, slots=True)
class SensorObservation:
    node: NodeId
    time: float
    detected: bool
    value: float | None = None
    healthy: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("sensor time must be finite and non-negative")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("sensor value must be finite when present")

    @property
    def usable(self) -> bool:
        return self.healthy and self.value is not None


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    feasible: bool
    min_travel_time: float
    positive_sensor_consistency: float
    negative_sensor_consistency: float
    arrival_order_consistency: float
    reachable_positive_sensors: int
    usable_positive_sensors: int


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    candidate_mask: Mapping[NodeId, bool]
    features: Mapping[NodeId, CandidateFeatures]
    ignored_sensor_count: int


def _pairwise_order_score(
    predicted: list[tuple[float, float]], *, tolerance: float
) -> float:
    if len(predicted) < 2:
        return 1.0
    matches = 0
    pairs = 0
    for index, (observed_a, predicted_a) in enumerate(predicted):
        for observed_b, predicted_b in predicted[index + 1 :]:
            observed_delta = observed_a - observed_b
            predicted_delta = predicted_a - predicted_b
            pairs += 1
            if abs(observed_delta) <= tolerance or observed_delta * predicted_delta >= 0:
                matches += 1
    return matches / pairs


def screen_candidates(
    graph: DirectedHydraulicGraph,
    observations: Iterable[SensorObservation],
    *,
    candidates: Iterable[NodeId] | None = None,
    incident_start_time: float = 0.0,
    arrival_tolerance: float = 0.0,
) -> ScreeningResult:
    """Screen candidates using only physically decisive positive evidence.

    Unhealthy observations and values marked missing with ``None`` are ignored.
    A negative observation contributes a soft consistency feature rather than a
    hard rejection because uncertain injection strength and thresholds can make
    non-detections ambiguous.
    """

    if not math.isfinite(incident_start_time) or incident_start_time < 0:
        raise ValueError("incident_start_time must be finite and non-negative")
    if not math.isfinite(arrival_tolerance) or arrival_tolerance < 0:
        raise ValueError("arrival_tolerance must be finite and non-negative")

    all_observations = tuple(observations)
    usable = tuple(observation for observation in all_observations if observation.usable)
    ignored = len(all_observations) - len(usable)
    positive = tuple(observation for observation in usable if observation.detected)
    negative = tuple(observation for observation in usable if not observation.detected)
    candidate_nodes = (
        tuple(graph.transport.nodes)
        if candidates is None
        else tuple(dict.fromkeys(candidates))
    )

    mask: dict[NodeId, bool] = {}
    features: dict[NodeId, CandidateFeatures] = {}
    for candidate in candidate_nodes:
        positive_times: list[float] = []
        positive_consistent = 0
        order_pairs: list[tuple[float, float]] = []
        for observation in positive:
            travel_time = graph.shortest_travel_time(candidate, observation.node)
            positive_times.append(travel_time)
            elapsed = observation.time - incident_start_time
            if math.isfinite(travel_time) and travel_time <= elapsed + arrival_tolerance:
                positive_consistent += 1
            if math.isfinite(travel_time):
                order_pairs.append((observation.time, travel_time))

        negative_consistent = 0
        for observation in negative:
            travel_time = graph.shortest_travel_time(candidate, observation.node)
            elapsed = observation.time - incident_start_time
            if not math.isfinite(travel_time) or travel_time > elapsed + arrival_tolerance:
                negative_consistent += 1

        feasible = not positive or positive_consistent == len(positive)
        min_time = min(positive_times, default=math.inf)
        mask[candidate] = feasible
        features[candidate] = CandidateFeatures(
            feasible=feasible,
            min_travel_time=min_time,
            positive_sensor_consistency=(
                positive_consistent / len(positive) if positive else 1.0
            ),
            negative_sensor_consistency=(
                negative_consistent / len(negative) if negative else 1.0
            ),
            arrival_order_consistency=_pairwise_order_score(
                order_pairs, tolerance=arrival_tolerance
            ),
            reachable_positive_sensors=sum(math.isfinite(value) for value in positive_times),
            usable_positive_sensors=len(positive),
        )

    return ScreeningResult(mask, features, ignored)

