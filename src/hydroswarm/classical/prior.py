"""Compressed simulator signatures and residual-based Bayesian source priors."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

CandidateId = Hashable


@dataclass(slots=True)
class SignatureLibrary:
    """In-memory float16 fingerprints indexed by source hypothesis.

    Float16 is used only for storage.  Retrieval and posterior calculations use
    float32 to keep residual arithmetic stable on CPU.
    """

    _signatures: dict[CandidateId, NDArray[np.float16]] = field(default_factory=dict)
    _shape: tuple[int, ...] | None = None

    def add(self, candidate: CandidateId, signature: ArrayLike) -> None:
        values = np.asarray(signature, dtype=np.float32)
        if values.size == 0:
            raise ValueError("signature must not be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("signature values must be finite")
        if self._shape is None:
            self._shape = values.shape
        elif values.shape != self._shape:
            raise ValueError(
                f"signature shape {values.shape} does not match {self._shape}"
            )
        self._signatures[candidate] = values.astype(np.float16)

    def get(self, candidate: CandidateId) -> NDArray[np.float32]:
        try:
            return self._signatures[candidate].astype(np.float32, copy=True)
        except KeyError as error:
            raise KeyError(f"unknown signature candidate: {candidate!r}") from error

    @property
    def candidates(self) -> tuple[CandidateId, ...]:
        return tuple(self._signatures)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape

    @property
    def compressed_bytes(self) -> int:
        return sum(signature.nbytes for signature in self._signatures.values())


@dataclass(frozen=True, slots=True)
class BayesianPosterior:
    probabilities: Mapping[CandidateId, float]
    residual_rmse: Mapping[CandidateId, float]
    used_observations: int


def _normalized_prior(
    candidates: Sequence[CandidateId], prior: Mapping[CandidateId, float] | None
) -> NDArray[np.float64]:
    if prior is None:
        return np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    weights = np.asarray([prior.get(candidate, 0.0) for candidate in candidates], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("prior weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("prior must assign positive mass to a library candidate")
    return weights / total


def bayesian_source_posterior(
    observations: ArrayLike,
    library: SignatureLibrary,
    *,
    prior: Mapping[CandidateId, float] | None = None,
    observation_mask: ArrayLike | None = None,
    noise_scale: float = 1.0,
    feasible_mask: Mapping[CandidateId, bool] | None = None,
) -> BayesianPosterior:
    """Calculate ``P(source | observations)`` from Gaussian residual likelihoods.

    Non-finite observations and false entries in ``observation_mask`` are
    treated as missing.  If all observations are missing, the normalized prior
    is returned.  A physical feasibility mask assigns exactly zero posterior
    mass to impossible candidates.
    """

    if not math.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("noise_scale must be finite and positive")
    candidates = library.candidates
    if not candidates:
        raise ValueError("signature library is empty")

    observed = np.asarray(observations, dtype=np.float32)
    if observed.shape != library.shape:
        raise ValueError(
            f"observation shape {observed.shape} does not match {library.shape}"
        )
    valid = np.isfinite(observed)
    if observation_mask is not None:
        supplied_mask = np.asarray(observation_mask, dtype=bool)
        if supplied_mask.shape != observed.shape:
            raise ValueError("observation_mask shape must match observations")
        valid &= supplied_mask

    prior_values = _normalized_prior(candidates, prior)
    feasible = np.asarray(
        [True if feasible_mask is None else feasible_mask.get(candidate, False) for candidate in candidates],
        dtype=bool,
    )
    if not feasible.any():
        raise ValueError("physical feasibility mask excludes every candidate")
    prior_values = np.where(feasible, prior_values, 0.0)
    if prior_values.sum() <= 0:
        prior_values = feasible.astype(float)
    prior_values /= prior_values.sum()

    used = int(valid.sum())
    residual_rmse: dict[CandidateId, float] = {}
    log_weights = np.full(len(candidates), -np.inf, dtype=np.float64)
    for index, candidate in enumerate(candidates):
        if not feasible[index]:
            residual_rmse[candidate] = math.inf
            continue
        if used:
            residual = observed[valid] - library.get(candidate)[valid]
            squared_error = float(np.dot(residual, residual))
            residual_rmse[candidate] = math.sqrt(squared_error / used)
            log_likelihood = -0.5 * squared_error / (noise_scale * noise_scale)
        else:
            residual_rmse[candidate] = math.nan
            log_likelihood = 0.0
        log_weights[index] = math.log(prior_values[index]) + log_likelihood

    finite = np.isfinite(log_weights)
    shifted = np.zeros_like(log_weights)
    shifted[finite] = np.exp(log_weights[finite] - np.max(log_weights[finite]))
    probabilities = shifted / shifted.sum()
    return BayesianPosterior(
        probabilities={candidate: float(probabilities[index]) for index, candidate in enumerate(candidates)},
        residual_rmse=residual_rmse,
        used_observations=used,
    )

