"""Non-learned oracle/template localization baseline (Phase 4).

Deliberately reuses the repo's own existing classical Bayesian
signature-residual localizer (`hydroswarm.classical.prior.
bayesian_source_posterior`) rather than inventing a new ranking rule --
this *is* the closest existing "source-signature comparison" primitive in
HydroSwarm (see docs/evaluation/SOURCE_IDENTIFIABILITY_ANALYSIS_PROTOCOL.md
Section 3). `NON-PROMOTABLE / DIAGNOSTIC ONLY`, matching the convention
established by `reports/evaluation/hydrocore-v5/m10/m10-3b-diagnosis/
m10-3b-oracle-utility.json` -- never a deployable policy, never compared to
HydroCore-v5 as a claim of superiority, only as an identifiability upper
bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydroswarm.classical.prior import SignatureLibrary, bayesian_source_posterior

#: Matches `hydroswarm.classical.signatures.localize_with_signatures`'s own
#: default -- reused rather than re-tuned, since it only reshapes the
#: reported probability margin/entropy, never the ranking itself (ranking
#: is a monotonic function of squared residual regardless of noise_scale).
DEFAULT_NOISE_SCALE = 0.05


@dataclass(frozen=True, slots=True)
class OracleResult:
    candidates: tuple[str, ...]
    true_source: str
    true_source_rank: int
    top1: float
    top3: float
    mrr: float
    posterior_entropy: float
    probability_margin: float  # top1 probability - top2 probability
    residual_margin: float  # 2nd-best RMSE - best RMSE (scale-free, noise_scale-independent)
    probabilities: dict[str, float]
    residual_rmse: dict[str, float]


def rank_candidates(
    raw_signatures: dict[str, np.ndarray],
    observed: np.ndarray,
    *,
    true_source: str,
    observation_mask: np.ndarray | None = None,
    noise_scale: float = DEFAULT_NOISE_SCALE,
) -> OracleResult:
    library = SignatureLibrary()
    for candidate, matrix in raw_signatures.items():
        library.add(candidate, matrix)
    posterior = bayesian_source_posterior(
        observed, library, observation_mask=observation_mask, noise_scale=noise_scale
    )
    candidates = tuple(raw_signatures)
    probs = np.array([posterior.probabilities[c] for c in candidates], dtype=np.float64)
    truth_index = candidates.index(true_source)

    row_metrics = _per_row_metrics(probs, truth_index)

    order = np.argsort(-probs)
    top1_prob = float(probs[order[0]])
    top2_prob = float(probs[order[1]]) if len(order) > 1 else float("nan")
    residual_by_candidate = np.array([posterior.residual_rmse[c] for c in candidates], dtype=np.float64)
    residual_order = np.argsort(residual_by_candidate)
    best_residual = float(residual_by_candidate[residual_order[0]])
    second_residual = float(residual_by_candidate[residual_order[1]]) if len(residual_order) > 1 else float("nan")

    return OracleResult(
        candidates=candidates,
        true_source=true_source,
        true_source_rank=row_metrics["true_source_rank"],
        top1=row_metrics["top1"],
        top3=row_metrics["top3"],
        mrr=row_metrics["mrr"],
        posterior_entropy=row_metrics["posterior_entropy"],
        probability_margin=top1_prob - top2_prob,
        residual_margin=second_residual - best_residual,
        probabilities=dict(zip(candidates, probs.tolist())),
        residual_rmse=dict(posterior.residual_rmse),
    )


def _per_row_metrics(probs: np.ndarray, truth_index: int) -> dict[str, float]:
    """Local re-implementation matching `m9_1_common.per_row_metrics`
    exactly (verified against its verbatim body), operating on a plain
    numpy row rather than a torch tensor since the oracle here never
    touches the neural model. Kept identical in formula/semantics so
    downstream comparisons with HydroCore-v5's own `per_row_metrics`-style
    diagnostics (`m9-2-canonical-diagnostics.jsonl`) are apples-to-apples.
    """

    eps = 1e-9
    p_truth = float(probs[truth_index])
    rank = int(np.sum(probs > p_truth)) + 1
    onehot = np.zeros_like(probs)
    onehot[truth_index] = 1.0
    return {
        "top1": 1.0 if rank == 1 else 0.0,
        "top3": 1.0 if rank <= 3 else 0.0,
        "mrr": 1.0 / rank,
        "nll": -float(np.log(p_truth + eps)),
        "brier": float(np.sum((probs - onehot) ** 2)),
        "posterior_entropy": float(-np.sum(probs * np.log(probs + eps))),
        "true_source_rank": rank,
    }


__all__ = ["DEFAULT_NOISE_SCALE", "OracleResult", "rank_candidates"]
