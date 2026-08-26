"""Source-signature construction (Phase 2) and pairwise distances (Phase 3).

Pure-computation module: no I/O, no simulator calls. `build_signature_library`
lives in `library.py`, where the actual `HydraulicSimulator` calls happen;
everything here operates on already-simulated numpy arrays so it is
unit-testable in isolation, following the repo's own
`*_common.py`/`*_analysis_lib.py` split convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: Below this absolute concentration (mg/L), a sensor reading is treated as
#: "not yet arrived" for the ARRIVAL-ORDER signature. Two orders of
#: magnitude below the M11.6 nominal sensor noise floor (sensor_noise_std
#: = 0.01 raw units on a log1p scale), so it is not itself an artifact of
#: measurement noise.
ARRIVAL_DETECTION_THRESHOLD_MG_L = 1e-4

#: Sentinel arrival time (seconds) for a sensor a candidate signature never
#: reaches within the observation window -- larger than any real window so
#: it sorts as "latest/never" without producing NaN/inf in downstream math.
ARRIVAL_NEVER_SECONDS = 1.0e7


@dataclass(frozen=True, slots=True)
class SignatureSet:
    """RAW/NORMALIZED/ARRIVAL-ORDER signatures for one incident's candidate pool."""

    candidates: tuple[str, ...]
    sensor_nodes: tuple[str, ...]
    timestamps_seconds: tuple[int, ...]
    raw: dict[str, np.ndarray]  # candidate -> (n_times, n_sensors) log1p(mg/L)
    normalized: dict[str, np.ndarray]  # candidate -> L2-normalized `raw`
    arrival_order: dict[str, np.ndarray]  # candidate -> (n_sensors,) seconds-to-detection


def build_signature_set(
    concentrations_mg_l: Mapping[str, np.ndarray],
    *,
    sensor_nodes: Sequence[str],
    timestamps_seconds: Sequence[int],
) -> SignatureSet:
    """Turn raw per-candidate concentration matrices into the three signature views.

    `concentrations_mg_l[candidate]` must already be restricted to
    `sensor_nodes` columns, in that column order, at `timestamps_seconds`.
    """

    candidates = tuple(concentrations_mg_l)
    raw: dict[str, np.ndarray] = {}
    normalized: dict[str, np.ndarray] = {}
    arrival: dict[str, np.ndarray] = {}
    for candidate, matrix in concentrations_mg_l.items():
        matrix = np.asarray(matrix, dtype=np.float64)
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError(f"non-finite or negative concentration for candidate {candidate!r}")
        log_matrix = np.log1p(matrix)
        raw[candidate] = log_matrix
        norm = float(np.linalg.norm(log_matrix))
        normalized[candidate] = log_matrix / norm if norm > 0 else log_matrix
        arrival[candidate] = _arrival_times(matrix, timestamps_seconds)
    return SignatureSet(
        candidates=candidates,
        sensor_nodes=tuple(sensor_nodes),
        timestamps_seconds=tuple(int(t) for t in timestamps_seconds),
        raw=raw,
        normalized=normalized,
        arrival_order=arrival,
    )


def _arrival_times(matrix: np.ndarray, timestamps_seconds: Sequence[int]) -> np.ndarray:
    times = np.asarray(timestamps_seconds, dtype=np.float64)
    arrivals = np.full(matrix.shape[1], ARRIVAL_NEVER_SECONDS, dtype=np.float64)
    for sensor_index in range(matrix.shape[1]):
        exceeded = np.flatnonzero(matrix[:, sensor_index] >= ARRIVAL_DETECTION_THRESHOLD_MG_L)
        if exceeded.size:
            arrivals[sensor_index] = times[exceeded[0]]
    return arrivals


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------


def rmse_distance(a: np.ndarray, b: np.ndarray) -> float:
    residual = a.ravel() - b.ravel()
    return float(np.sqrt(np.mean(residual * residual)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.ravel(), b.ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom <= 0:
        return 1.0
    similarity = float(np.dot(a_flat, b_flat) / denom)
    return 1.0 - max(-1.0, min(1.0, similarity))


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.ravel(), b.ravel()
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        # Two constant (typically all-zero, undetected) signatures are
        # indistinguishable by shape -- distance 0 -- unless only one is
        # constant, in which case they are maximally different in shape.
        return 0.0 if np.allclose(a_flat, b_flat) else 1.0
    correlation = float(np.corrcoef(a_flat, b_flat)[0, 1])
    return 1.0 - max(-1.0, min(1.0, correlation))


def arrival_order_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


DISTANCE_METRICS = {
    "rmse": rmse_distance,
    "cosine": cosine_distance,
    "correlation": correlation_distance,
    "arrival_l1": arrival_order_distance,
}


def pairwise_distance_matrix(
    signatures: Mapping[str, np.ndarray], *, metric: str
) -> tuple[tuple[str, ...], np.ndarray]:
    fn = DISTANCE_METRICS[metric]
    candidates = tuple(signatures)
    n = len(candidates)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distance = fn(signatures[candidates[i]], signatures[candidates[j]])
            matrix[i, j] = matrix[j, i] = distance
    return candidates, matrix


@dataclass(frozen=True, slots=True)
class IdentifiabilityResult:
    true_source: str
    n_candidates: int
    nearest_competitor: str | None
    nearest_competitor_distance: float
    second_nearest_distance: float
    mean_competitor_distance: float
    median_competitor_distance: float
    margin: float  # second_nearest - nearest
    incident_mean_pairwise_distance: float
    identifiability_score: float  # nearest_competitor_distance / incident_mean_pairwise_distance
    ambiguity_count_noise_floor: int
    ambiguity_count_percentile: int
    ambiguity_fraction_percentile: float


def identifiability_metrics(
    candidates: Sequence[str],
    distance_matrix: np.ndarray,
    *,
    true_source: str,
    noise_floor_distance: float,
    ambiguity_percentile: float = 10.0,
) -> IdentifiabilityResult:
    """Phase-3/4 per-incident identifiability metrics for the true source.

    `noise_floor_distance` is a physically motivated absolute threshold
    (derived from the incident's own sensor-noise/quantization parameters,
    see `library.py::noise_floor_distance`), used for one of the two
    ambiguity-count definitions the plan calls for; the other is a
    percentile of THIS incident's own pairwise-distance distribution
    (never tuned against outcome labels).
    """

    if true_source not in candidates:
        raise ValueError("true_source must be one of candidates")
    n = len(candidates)
    if n < 2:
        raise ValueError("identifiability requires at least one competing candidate")
    true_index = candidates.index(true_source)
    competitor_distances = np.array(
        [distance_matrix[true_index, j] for j in range(n) if j != true_index], dtype=np.float64
    )
    competitor_names = [candidates[j] for j in range(n) if j != true_index]
    order = np.argsort(competitor_distances)
    nearest_index = int(order[0])
    nearest = float(competitor_distances[nearest_index])
    second_nearest = float(competitor_distances[order[1]]) if n > 2 else float("nan")

    upper = distance_matrix[np.triu_indices(n, k=1)]
    incident_mean_pairwise = float(np.mean(upper)) if upper.size else float("nan")
    # incident_mean_pairwise == 0 means EVERY candidate produced an
    # identical (often all-undetected-within-window) signature -- the
    # maximally information-limited case, not an undefined ratio. Scored
    # as 0.0 (the minimum of this metric) rather than NaN so it sorts to
    # the least-identifiable end instead of silently dropping out of
    # tercile/mean aggregation.
    score = nearest / incident_mean_pairwise if incident_mean_pairwise > 0 else 0.0

    percentile_threshold = float(np.percentile(upper, ambiguity_percentile)) if upper.size else 0.0
    ambiguity_percentile_count = int(np.sum(competitor_distances <= percentile_threshold))
    ambiguity_noise_floor_count = int(np.sum(competitor_distances <= noise_floor_distance))

    return IdentifiabilityResult(
        true_source=true_source,
        n_candidates=n,
        nearest_competitor=competitor_names[nearest_index],
        nearest_competitor_distance=nearest,
        second_nearest_distance=second_nearest,
        mean_competitor_distance=float(np.mean(competitor_distances)),
        median_competitor_distance=float(np.median(competitor_distances)),
        margin=(second_nearest - nearest) if n > 2 else float("nan"),
        incident_mean_pairwise_distance=incident_mean_pairwise,
        identifiability_score=score,
        ambiguity_count_noise_floor=ambiguity_noise_floor_count,
        ambiguity_count_percentile=ambiguity_percentile_count,
        ambiguity_fraction_percentile=ambiguity_percentile_count / (n - 1),
    )


__all__ = [
    "ARRIVAL_DETECTION_THRESHOLD_MG_L",
    "ARRIVAL_NEVER_SECONDS",
    "SignatureSet",
    "build_signature_set",
    "rmse_distance",
    "cosine_distance",
    "correlation_distance",
    "arrival_order_distance",
    "DISTANCE_METRICS",
    "pairwise_distance_matrix",
    "IdentifiabilityResult",
    "identifiability_metrics",
]
