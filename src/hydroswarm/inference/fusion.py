"""Physics-aware fusion and deterministic disagreement control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


EPSILON = 1e-12


class ControlAction(StrEnum):
    CONTINUE_ANALYSIS = "CONTINUE_ANALYSIS"
    REQUEST_SAMPLE = "REQUEST_SAMPLE"
    INSPECT_SENSORS = "INSPECT_SENSORS"
    GENERATE_PLANS = "GENERATE_PLANS"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class TrustFeatures:
    healthy_sensor_fraction: float
    missing_rate: float
    normalized_residual: float
    hydraulic_uncertainty: float
    neural_entropy: float
    classical_entropy: float
    ood_score: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class FusionDiagnostics:
    classical_trust: float
    disagreement_js: float
    masked_nodes: int


def _probabilities(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(result.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return result / total


def jensen_shannon_divergence(left: ArrayLike, right: ArrayLike) -> float:
    """Return base-2 Jensen-Shannon divergence bounded to [0, 1]."""
    p = _probabilities(left, name="left")
    q = _probabilities(right, name="right")
    if p.shape != q.shape:
        raise ValueError("probability vectors must have the same shape")
    midpoint = 0.5 * (p + q)

    def kl(values: NDArray[np.float64]) -> float:
        present = values > 0
        return float(np.sum(values[present] * np.log2(values[present] / midpoint[present])))

    return float(np.clip(0.5 * (kl(p) + kl(q)), 0.0, 1.0))


def dynamic_classical_trust(features: TrustFeatures, disagreement_js: float) -> float:
    """Derive lambda from evidence health instead of using a fixed mixture.

    The classical prior is trusted when sensors are healthy and both hydraulic residual
    and state uncertainty are low. OOD and disagreement reduce fusion aggressiveness.
    """
    positive = 0.45 * features.healthy_sensor_fraction
    penalties = (
        0.15 * features.missing_rate
        + 0.12 * features.normalized_residual
        + 0.10 * features.hydraulic_uncertainty
        + 0.08 * features.ood_score
        + 0.10 * disagreement_js
    )
    entropy_balance = 0.05 * (features.neural_entropy - features.classical_entropy)
    return float(np.clip(0.35 + positive + entropy_balance - penalties, 0.05, 0.95))


def fuse_source_probabilities(
    neural_logits: ArrayLike,
    classical_probabilities: ArrayLike,
    physical_mask: ArrayLike,
    features: TrustFeatures,
) -> tuple[NDArray[np.float64], FusionDiagnostics]:
    """Fuse logits using z_i = z_i^NN + lambda log(p_i^classical + eps)."""
    logits = np.asarray(neural_logits, dtype=np.float64)
    classical = _probabilities(classical_probabilities, name="classical_probabilities")
    mask = np.asarray(physical_mask, dtype=bool)
    if logits.ndim != 1 or logits.shape != classical.shape or mask.shape != logits.shape:
        raise ValueError("logits, probabilities, and mask must be aligned 1D arrays")
    if not np.any(mask):
        raise ValueError("physical mask rejects every source candidate")

    shifted = logits - np.max(logits)
    neural = np.exp(shifted)
    neural = neural / neural.sum()
    divergence = jensen_shannon_divergence(neural, classical)
    trust = dynamic_classical_trust(features, divergence)
    fused_logits = logits + trust * np.log(classical + EPSILON)
    fused_logits[~mask] = -np.inf
    fused_logits -= np.max(fused_logits[mask])
    fused = np.zeros_like(fused_logits)
    fused[mask] = np.exp(fused_logits[mask])
    fused /= fused.sum()
    return fused, FusionDiagnostics(
        classical_trust=trust,
        disagreement_js=divergence,
        masked_nodes=int((~mask).sum()),
    )


def conformal_candidate_set(
    probabilities: ArrayLike,
    calibration_nonconformity_scores: ArrayLike,
    *,
    alpha: float = 0.1,
) -> NDArray[np.int64]:
    """Return indices whose 1-p score is below split-conformal finite-sample quantile."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    probabilities_array = _probabilities(probabilities, name="probabilities")
    scores = np.asarray(calibration_nonconformity_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores must be a finite non-empty 1D array")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("calibration scores must lie in [0, 1]")
    rank = int(np.ceil((scores.size + 1) * (1 - alpha)))
    rank = min(max(rank, 1), scores.size)
    threshold = float(np.partition(scores, rank - 1)[rank - 1])
    return np.flatnonzero((1.0 - probabilities_array) <= threshold)


def uncertainty_control(
    *,
    candidate_count: int,
    disagreement_js: float,
    ood_score: float,
    healthy_sensor_fraction: float,
    sample_budget_remaining: int,
) -> ControlAction:
    """Apply the master plan's deterministic uncertainty-to-behavior policy."""
    if not 0 <= candidate_count:
        raise ValueError("candidate_count cannot be negative")
    values = (disagreement_js, ood_score, healthy_sensor_fraction)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("scores must lie in [0, 1]")
    if ood_score >= 0.8 or healthy_sensor_fraction < 0.25:
        return ControlAction.ABSTAIN
    if disagreement_js >= 0.5:
        return ControlAction.INSPECT_SENSORS
    if candidate_count == 1 and disagreement_js < 0.2 and ood_score < 0.4:
        return ControlAction.GENERATE_PLANS
    if candidate_count > 1 and sample_budget_remaining > 0:
        return ControlAction.REQUEST_SAMPLE
    if candidate_count > 1:
        return ControlAction.ABSTAIN
    return ControlAction.CONTINUE_ANALYSIS
