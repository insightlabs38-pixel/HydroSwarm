"""Expected-information-gain sampling with operational constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from hydroswarm.classical.metrics import entropy
from hydroswarm.classical.signatures import SignatureArtifact


@dataclass(frozen=True, slots=True)
class SamplingConstraints:
    collection_time_minutes: Mapping[str, float] = field(default_factory=dict)
    operational_cost: Mapping[str, float] = field(default_factory=dict)
    accessible: Mapping[str, bool] = field(default_factory=dict)
    already_sampled: frozenset[str] = frozenset()
    maximum_delay_minutes: float = 120.0
    minimum_information_gain_bits: float = 0.01


@dataclass(frozen=True, slots=True)
class SampleCandidate:
    node_id: str
    score: float
    expected_information_gain_bits: float
    expected_candidate_reduction: float
    leading_hypothesis_separation: float
    detection_probability: float
    collection_time_minutes: float
    operational_cost: float
    redundancy: float
    accessible: bool
    classical_rank: int
    neural_residual_delta: float


@dataclass(frozen=True, slots=True)
class ActiveSamplingResult:
    ranked: tuple[SampleCandidate, ...]
    recommended_node: str | None
    stop: bool
    stop_reason: str | None
    posterior_entropy_bits: float


def _signature_matrix(artifact: SignatureArtifact) -> np.ndarray:
    return np.stack([artifact.library.get(item.identifier) for item in artifact.hypotheses])


def rank_sample_locations(
    artifact: SignatureArtifact,
    posterior: Mapping[str, float],
    *,
    constraints: SamplingConstraints | None = None,
    candidate_nodes: Sequence[str] | None = None,
    noise_scale_mg_l: float = 0.05,
    detection_threshold_mg_l: float = 0.001,
    neural_residual_deltas: Mapping[str, float] | None = None,
    top_k: int = 20,
) -> ActiveSamplingResult:
    """Rank locations by expected discrimination, detectability, delay, cost, and redundancy."""
    constraints = constraints or SamplingConstraints()
    if noise_scale_mg_l <= 0 or top_k <= 0:
        raise ValueError("noise scale and top_k must be positive")
    hypotheses = artifact.hypotheses
    weights = np.asarray([posterior.get(item.identifier, 0.0) for item in hypotheses], dtype=float)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("posterior must assign finite positive mass to signature hypotheses")
    weights /= weights.sum()
    current_entropy = entropy(weights)
    values = _signature_matrix(artifact)
    if values.ndim != 3 or values.shape[2] != len(artifact.sensor_nodes):
        raise ValueError("signature artifact must use [time, sensor] traces")
    possible = set(candidate_nodes or artifact.sensor_nodes)
    top_hypotheses = np.argsort(weights)[::-1][:2]
    candidates: list[SampleCandidate] = []
    prediction_vectors: dict[str, np.ndarray] = {}
    for sensor_index, node in enumerate(artifact.sensor_nodes):
        if node not in possible or node in constraints.already_sampled:
            continue
        traces = values[:, :, sensor_index]
        prediction = traces.max(axis=1)
        prediction_vectors[node] = prediction
        mean = float(np.dot(weights, prediction))
        variance = float(np.dot(weights, (prediction - mean) ** 2))
        information_gain = min(
            current_entropy, 0.5 * math.log2(1.0 + variance / (noise_scale_mg_l**2))
        )
        spread = float(np.ptp(prediction))
        candidate_reduction = min(
            max(0.0, float(np.count_nonzero(weights > 1e-6) - 1)),
            spread / max(noise_scale_mg_l, 1e-9),
        )
        separation = (
            abs(float(prediction[top_hypotheses[0]] - prediction[top_hypotheses[1]]))
            if len(top_hypotheses) > 1 else 0.0
        )
        detection = float(weights[prediction >= detection_threshold_mg_l].sum())
        delay = float(constraints.collection_time_minutes.get(node, 30.0))
        cost = max(0.0, float(constraints.operational_cost.get(node, 1.0)))
        accessible = constraints.accessible.get(node, True) and delay <= constraints.maximum_delay_minutes
        candidates.append(SampleCandidate(
            node_id=node,
            score=0.0,
            expected_information_gain_bits=information_gain,
            expected_candidate_reduction=candidate_reduction,
            leading_hypothesis_separation=separation,
            detection_probability=detection,
            collection_time_minutes=delay,
            operational_cost=cost,
            redundancy=0.0,
            accessible=accessible,
            classical_rank=0,
            neural_residual_delta=float((neural_residual_deltas or {}).get(node, 0.0)),
        ))

    scored: list[SampleCandidate] = []
    reference: list[np.ndarray] = []
    for rank, candidate in enumerate(
        sorted(candidates, key=lambda item: item.expected_information_gain_bits, reverse=True), start=1
    ):
        vector = prediction_vectors[candidate.node_id]
        correlations = []
        for other in reference:
            if np.std(vector) > 1e-12 and np.std(other) > 1e-12:
                correlations.append(abs(float(np.corrcoef(vector, other)[0, 1])))
        redundancy = max(correlations, default=0.0)
        if candidate.accessible:
            score = (
                1.00 * candidate.expected_information_gain_bits
                + 0.15 * candidate.expected_candidate_reduction
                + 0.25 * candidate.leading_hypothesis_separation
                + 0.20 * candidate.detection_probability
                + candidate.neural_residual_delta
                - 0.25 * redundancy
                - 0.002 * candidate.collection_time_minutes
                - 0.05 * candidate.operational_cost
            )
        else:
            score = -math.inf
        scored.append(replace(
            candidate, score=score, redundancy=redundancy, classical_rank=rank
        ))
        if candidate.accessible:
            reference.append(vector)
    scored.sort(key=lambda item: item.score, reverse=True)
    ranked = tuple(scored[:top_k])
    if not ranked or not math.isfinite(ranked[0].score):
        return ActiveSamplingResult(ranked, None, True, "no_accessible_sample", current_entropy)
    if ranked[0].expected_information_gain_bits < constraints.minimum_information_gain_bits:
        return ActiveSamplingResult(ranked, None, True, "marginal_value_below_threshold", current_entropy)
    return ActiveSamplingResult(ranked, ranked[0].node_id, False, None, current_entropy)
