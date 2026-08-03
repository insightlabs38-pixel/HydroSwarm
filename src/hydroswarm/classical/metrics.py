"""Small deterministic operational metrics used by evaluation and the UI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _ranked(probabilities: Mapping[Hashable, float]) -> list[Hashable]:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    for value in probabilities.values():
        if not math.isfinite(value) or value < 0:
            raise ValueError("probabilities must be finite and non-negative")
    # String tie-breaking makes evaluation independent of insertion order.
    return sorted(probabilities, key=lambda key: (-probabilities[key], str(key)))


def localization_top_k(
    probabilities: Mapping[Hashable, float], truth: Hashable, *, k: int = 1
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return float(truth in _ranked(probabilities)[:k])


def mean_reciprocal_rank(
    predictions: Iterable[Mapping[Hashable, float]], truths: Iterable[Hashable]
) -> float:
    pairs = list(zip(predictions, truths, strict=True))
    if not pairs:
        raise ValueError("at least one prediction is required")
    reciprocal_ranks = []
    for probabilities, truth in pairs:
        ranking = _ranked(probabilities)
        reciprocal_ranks.append(
            0.0 if truth not in ranking else 1.0 / (ranking.index(truth) + 1)
        )
    return float(np.mean(reciprocal_ranks))


@dataclass(frozen=True, slots=True)
class CandidateSetMetrics:
    coverage: float
    average_size: float


def candidate_set_metrics(
    candidate_sets: Iterable[Iterable[Hashable]], truths: Iterable[Hashable]
) -> CandidateSetMetrics:
    sets = [set(candidate_set) for candidate_set in candidate_sets]
    truth_values = list(truths)
    if len(sets) != len(truth_values):
        raise ValueError("candidate_sets and truths must have equal length")
    if not sets:
        raise ValueError("at least one candidate set is required")
    return CandidateSetMetrics(
        coverage=float(np.mean([truth in values for values, truth in zip(sets, truth_values)])),
        average_size=float(np.mean([len(values) for values in sets])),
    )


def entropy(probabilities: ArrayLike) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be non-empty, finite, and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    values = values.ravel() / total
    positive = values > 0
    return float(-np.sum(values[positive] * np.log2(values[positive])))


def information_gain_per_sample(
    posterior_history: Sequence[ArrayLike],
) -> float:
    """Average entropy reduction for each transition in posterior history."""

    if len(posterior_history) < 2:
        raise ValueError("posterior_history needs a prior and at least one posterior")
    gains = [
        entropy(before) - entropy(after)
        for before, after in zip(posterior_history, posterior_history[1:])
    ]
    return float(np.mean(gains))


@dataclass(frozen=True, slots=True)
class PressureViolationMetrics:
    count: int
    duration: float
    minimum_pressure: float


def pressure_violations(
    pressures: ArrayLike, *, minimum_allowed: float, time_step: float = 1.0
) -> PressureViolationMetrics:
    values = np.asarray(pressures, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("pressures must be non-empty and finite")
    if not math.isfinite(minimum_allowed):
        raise ValueError("minimum_allowed must be finite")
    if not math.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    violated = values < minimum_allowed
    return PressureViolationMetrics(
        count=int(violated.sum()),
        duration=float(violated.sum() * time_step),
        minimum_pressure=float(values.min()),
    )


@dataclass(frozen=True, slots=True)
class AbstentionMetrics:
    false_abstention_rate: float
    unsafe_non_abstention_rate: float
    abstention_rate: float
    selective_accuracy: float


def abstention_quality(
    abstained: ArrayLike, *, safe_if_answered: ArrayLike
) -> AbstentionMetrics:
    abstention = np.asarray(abstained, dtype=bool).ravel()
    safe = np.asarray(safe_if_answered, dtype=bool).ravel()
    if abstention.shape != safe.shape or abstention.size == 0:
        raise ValueError("abstained and safe_if_answered must be equal non-empty arrays")
    safe_count = int(safe.sum())
    unsafe_count = int((~safe).sum())
    answered = ~abstention
    return AbstentionMetrics(
        false_abstention_rate=(
            float(np.sum(abstention & safe) / safe_count) if safe_count else 0.0
        ),
        unsafe_non_abstention_rate=(
            float(np.sum(answered & ~safe) / unsafe_count) if unsafe_count else 0.0
        ),
        abstention_rate=float(np.mean(abstention)),
        selective_accuracy=(
            float(np.mean(safe[answered])) if answered.any() else 1.0
        ),
    )

