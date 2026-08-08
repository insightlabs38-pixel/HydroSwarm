"""Verified response Pareto frontier (core-issues5.txt Section 14, P1
product feature).

Computes a deterministic non-dominated frontier over EXACT, WNTR/EPANET-
verified `ConsequenceMetrics` only -- never asks a learned model to choose
the operator's tradeoff, never calls one plan "best" without an explicit
decision policy (there isn't one here; ranking is left to the human
operator). A plan dominates another when it is at least as good on every
considered objective and strictly better on at least one; the frontier is
exactly the set of plans no other plan dominates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from hydroswarm.domain import ConsequenceMetrics, PlanDecision, PlanVerification

__all__ = [
    "FrontierEntry",
    "FrontierMode",
    "compute_verified_pareto_frontier",
]

FrontierMode = Literal["posterior_weighted", "worst_case"]

#: Real ConsequenceMetrics exposure fields -- only meaningful when
#: exposure_evaluated=True (important-issues.txt requirement 12: the
#: legacy hydraulic-only path leaves these at their Pydantic zero
#: defaults, which must never be treated as measured).
_MINIMIZE_EXPOSURE_FIELDS = (
    "population_impacted",
    "contaminant_mass_consumed_mg",
    "volume_above_threshold_l",
    "contaminated_pipe_extent_m",
)
#: Always real (hydraulic evaluation is always exact, exposure or not).
_MINIMIZE_HYDRAULIC_FIELDS = ("pressure_violation_minutes", "unserved_demand_l", "operation_count")
_MAXIMIZE_FIELDS = ("minimum_pressure_m", "service_availability")


@dataclass(frozen=True, slots=True)
class FrontierEntry:
    plan_id: UUID
    label: str
    consequences: ConsequenceMetrics
    mode: FrontierMode
    dominated: bool
    #: True for the caller-designated NO_ACTION comparator, if any --
    #: always included in the returned tuple regardless of `dominated`,
    #: per this section's own "include NO_ACTION as a comparator when
    #: exact baseline context exists" requirement.
    is_no_action_comparator: bool


def _consequences_for_mode(verification: PlanVerification, mode: FrontierMode) -> ConsequenceMetrics | None:
    if mode == "posterior_weighted":
        return verification.consequences
    return verification.worst_case_consequences


def _objective_vector(consequences: ConsequenceMetrics) -> dict[str, float]:
    """Every objective normalized to "lower is better" (maximize fields
    negated), so two vectors can be compared with plain <=/< -- excludes
    the exposure fields entirely when they were not actually measured, so
    a hydraulic-only evaluation is compared only on what it actually
    verified, never a fabricated zero."""

    vector: dict[str, float] = {}
    if consequences.exposure_evaluated:
        for field in _MINIMIZE_EXPOSURE_FIELDS:
            vector[field] = float(getattr(consequences, field))
        if consequences.containment_time_minutes is not None:
            vector["containment_time_minutes"] = float(consequences.containment_time_minutes)
    for field in _MINIMIZE_HYDRAULIC_FIELDS:
        vector[field] = float(getattr(consequences, field))
    for field in _MAXIMIZE_FIELDS:
        vector[field] = -float(getattr(consequences, field))
    return vector


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """`a` dominates `b`: at least as good on every objective they share,
    strictly better on at least one. Objectives present in only one of the
    two (e.g. one measured exposure, the other did not) are not compared
    -- domination is decided only on the shared, genuinely comparable
    subset."""

    shared = set(a) & set(b)
    if not shared:
        return False
    at_least_as_good = all(a[key] <= b[key] for key in shared)
    strictly_better = any(a[key] < b[key] for key in shared)
    return at_least_as_good and strictly_better


def compute_verified_pareto_frontier(
    verifications: list[tuple[str, PlanVerification]],
    *,
    mode: FrontierMode = "posterior_weighted",
    no_action_label: str = "NO_ACTION",
) -> tuple[FrontierEntry, ...]:
    """`verifications` is a list of (label, PlanVerification) -- label is
    caller-supplied (typically the plan's action_template or name), since
    PlanVerification itself does not carry template identity. Only exact,
    CURRENT (not stale, core-issues5.txt Section 10), VERIFIED entries
    with real consequences for the requested `mode` participate; everything
    else is silently excluded (never fabricated, never guessed).
    """

    candidates: list[tuple[str, PlanVerification, ConsequenceMetrics]] = []
    for label, verification in verifications:
        if verification.decision != PlanDecision.VERIFIED:
            continue
        if verification.verification_status != "CURRENT":
            continue
        consequences = _consequences_for_mode(verification, mode)
        if consequences is None:
            continue
        candidates.append((label, verification, consequences))

    vectors = [_objective_vector(consequences) for _label, _verification, consequences in candidates]
    entries: list[FrontierEntry] = []
    for index, (label, verification, consequences) in enumerate(candidates):
        dominated = any(
            other_index != index and _dominates(vectors[other_index], vectors[index])
            for other_index in range(len(candidates))
        )
        entries.append(
            FrontierEntry(
                plan_id=verification.plan_id,
                label=label,
                consequences=consequences,
                mode=mode,
                dominated=dominated,
                is_no_action_comparator=(label == no_action_label),
            )
        )
    return tuple(entries)
