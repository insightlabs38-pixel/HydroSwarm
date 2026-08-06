"""Governed PlanValuePolicy (core-issues3.txt Phase 3.2/3.3).

The audited strategist_labels.py derived plan_value from
`proposal.predicted_value * proposal.predicted_validity` -- the old
heuristic prescreener's own unverified score, not the governed target
(targets_v2's own definition: "exposure reduction, service availability,
and containment time, minus regret against the best bounded valid plan").

hydroswarm.simulation.consequences already implements a transparent,
tested, governed multi-objective plan comparison (`rank_plan_outcomes`),
but its scoring formula does not weigh containment_time_minutes and is not
directly reusable here as a per-plan value with proxy-target decomposition.
Rather than mutate that shared, independently-tested/used utility (risking
behavior changes for any future consumer expecting its exact formula),
this module is a new, explicit, versioned policy that combines the same
kind of transparent, additive, nonnegative-weighted cost construction with
containment_time_minutes included, and exposes each cost component
separately -- these are exactly targets_v2's exposure_proxy/
pressure_risk_proxy/service_loss_proxy/containment_time_proxy/
plan_regret_proxy definitions, previously specified but never populated
(Phase 3.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hydroswarm.domain import ConsequenceMetrics

#: Bumped whenever the weights, transforms, or formula below change.
PLAN_VALUE_POLICY_VERSION = "plan-value-policy-v1"

#: Containment time beyond which "still not contained" is treated as
#: maximally bad rather than unboundedly bad -- keeps a plan with
#: containment_time_minutes=None (never contained within the simulation
#: window) comparable to, not infinitely worse than, a very slow
#: containment. Train-owned, not fit from any split.
DEFAULT_CONTAINMENT_SCALE_MINUTES = 240.0

#: Same normalization idea as consequences.py's own pressure_violation_minutes
#: / 60.0 term -- a fixed, documented, train-owned scale, not fit from data.
DEFAULT_PRESSURE_SCALE_MINUTES = 60.0


@dataclass(frozen=True, slots=True)
class PlanValueResult:
    """One candidate plan's governed value, regret, and consequence-proxy
    supervision targets -- all derived from the SAME exact WNTR
    consequence vector, so a model trained on the proxies is supervised
    from real verifier output, never a hand-invented label."""

    plan_value: float
    regret: float
    exposure_proxy: float
    pressure_risk_proxy: float
    service_loss_proxy: float
    containment_time_proxy: float
    plan_regret_proxy: float


def _cost(
    metrics: ConsequenceMetrics,
    *,
    no_response: ConsequenceMetrics,
    containment_scale_minutes: float,
    pressure_scale_minutes: float,
) -> tuple[float, float, float, float]:
    """Additive, nonnegative-weighted cost components -- the construction
    every monotonicity guarantee in this module relies on. Each component
    is independently bounded and normalized against a train-owned scale,
    never against another candidate's own outcome (which would make the
    scale itself depend on the candidate pool)."""

    exposure = metrics.contaminant_mass_consumed_mg / max(no_response.contaminant_mass_consumed_mg, 1e-9)
    pressure = metrics.pressure_violation_minutes / pressure_scale_minutes
    service_loss = 1.0 - metrics.service_availability
    containment = metrics.containment_time_minutes
    containment_proxy = (
        containment_scale_minutes if containment is None else containment
    ) / containment_scale_minutes
    return exposure, pressure, service_loss, containment_proxy


def evaluate_plan_value(
    metrics: ConsequenceMetrics,
    *,
    no_response: ConsequenceMetrics,
    valid_candidate_metrics: Sequence[ConsequenceMetrics],
    containment_scale_minutes: float = DEFAULT_CONTAINMENT_SCALE_MINUTES,
    pressure_scale_minutes: float = DEFAULT_PRESSURE_SCALE_MINUTES,
) -> PlanValueResult:
    """Compute `metrics`' governed plan_value/regret/proxy targets.

    `valid_candidate_metrics` must be every exactly-WNTR-verified valid
    candidate's ConsequenceMetrics for this same incident state, INCLUDING
    `no_response` and `metrics` itself -- regret is computed against the
    best (lowest-cost) member of this pool, per targets_v2's own
    definition ("regret against the best bounded valid plan"). Never call
    this for a plan that was not itself exactly verified: plan_value must
    never be derived from an unverified heuristic score.
    """

    if not valid_candidate_metrics:
        raise ValueError("valid_candidate_metrics must be nonempty (at minimum, no_response itself)")

    exposure, pressure, service_loss, containment = _cost(
        metrics,
        no_response=no_response,
        containment_scale_minutes=containment_scale_minutes,
        pressure_scale_minutes=pressure_scale_minutes,
    )
    cost = exposure + pressure + service_loss + containment

    pool_costs = [
        sum(
            _cost(
                candidate,
                no_response=no_response,
                containment_scale_minutes=containment_scale_minutes,
                pressure_scale_minutes=pressure_scale_minutes,
            )
        )
        for candidate in valid_candidate_metrics
    ]
    best_cost = min(pool_costs)
    regret = max(0.0, cost - best_cost)
    # 1/(1+regret): bounded in (0, 1], 1.0 exactly at regret=0 (the best
    # plan in the pool), strictly decreasing in regret. Because `cost` is
    # additive over nonnegative-weighted, independently-normalized
    # components and `best_cost` is a pool minimum that can only move
    # toward (not away from) an individual candidate's own improvement,
    # plan_value is guaranteed non-decreasing whenever any single cost
    # component improves with the others held fixed -- see
    # tests/scientific/test_plan_value_policy.py's monotonicity tests.
    plan_value = 1.0 / (1.0 + regret)

    return PlanValueResult(
        plan_value=plan_value,
        regret=regret,
        exposure_proxy=exposure,
        pressure_risk_proxy=pressure,
        service_loss_proxy=service_loss,
        containment_time_proxy=containment,
        plan_regret_proxy=regret,
    )
