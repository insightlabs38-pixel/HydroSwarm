"""Verified response Pareto frontier (core-issues5.txt Section 14, P1
product feature).

Computes a deterministic non-dominated frontier over EXACT, WNTR/EPANET-
verified `ConsequenceMetrics` only -- never asks a learned model to choose
the operator's tradeoff, never calls one plan "best" without an explicit
decision policy (there isn't one here; ranking is left to the human
operator). A plan dominates another when it is at least as good on every
considered objective and strictly better on at least one; the frontier is
exactly the set of plans no other plan dominates.

core-issues5.txt delta item 9 (P1 fix): exposure-aware and hydraulic-only
plans are never compared against each other for domination. A hydraulic-
only evaluation (`ConsequenceMetrics.exposure_evaluated=False`, the legacy
single-profile/no-hypothesis path) simply does not measure exposure at
all -- comparing it against an exposure-evaluated plan on only their
SHARED (hydraulic) dimensions would let an unmeasured exposure look
exactly as good as, or even dominate, a measured one, which is a real
false-equivalence bug: "unknown" must never visually behave like
"favorable". Each verified plan is assigned to exactly one `FrontierGroup`
by its own `consequences.exposure_evaluated`, and non-domination is
computed independently within each group -- an entry's `dominated` flag
only ever reflects comparison against other entries in the SAME group.
NO_ACTION, when present, is exempted from domination within its own group
exactly as before, but never bridges the two groups either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from hydroswarm.domain import ConsequenceMetrics, PlanDecision, PlanVerification

__all__ = [
    "FrontierEntry",
    "FrontierGroup",
    "FrontierMode",
    "compute_verified_pareto_frontier",
]

FrontierMode = Literal["posterior_weighted", "worst_case"]
#: EXPOSURE_AWARE: consequences.exposure_evaluated=True (a real, measured
#: contaminant-exposure evaluation was performed).
#: HYDRAULIC_ONLY: exposure_evaluated=False -- the legacy hydraulic-only
#: verification path, shown as a separate, clearly-labeled group rather
#: than silently merged into the exposure-aware frontier.
FrontierGroup = Literal["EXPOSURE_AWARE", "HYDRAULIC_ONLY"]

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
    #: core-issues5.txt delta item 9: which frontier this entry belongs to
    #: -- `dominated` is computed ONLY against other entries in the SAME
    #: group. An EXPOSURE_AWARE entry can never be dominated by (or
    #: reported as equivalent to) a HYDRAULIC_ONLY one, and vice versa.
    group: FrontierGroup


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
    strictly better on at least one.

    Only ever called between two entries in the SAME `FrontierGroup`
    (core-issues5.txt delta item 9) -- within a group, `_objective_vector`
    produces the same key set for every entry (all EXPOSURE_AWARE entries
    include the exposure fields, all HYDRAULIC_ONLY entries do not), so
    `shared` below is a genuine full-vector comparison, not a
    false-equivalence shortcut across measured and unmeasured exposure."""

    shared = set(a) & set(b)
    if not shared:
        return False
    at_least_as_good = all(a[key] <= b[key] for key in shared)
    strictly_better = any(a[key] < b[key] for key in shared)
    return at_least_as_good and strictly_better


def _frontier_group(consequences: ConsequenceMetrics) -> FrontierGroup:
    return "EXPOSURE_AWARE" if consequences.exposure_evaluated else "HYDRAULIC_ONLY"


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

    core-issues5.txt delta item 9: candidates are partitioned by
    `consequences.exposure_evaluated` into an EXPOSURE_AWARE group and a
    HYDRAULIC_ONLY group (see `FrontierGroup`), and non-domination is
    computed SEPARATELY within each group -- an entry's `dominated` flag
    never reflects comparison against the other group. This applies
    equally to the NO_ACTION comparator: it is exempted from domination
    only within its own group, never used to dominate or get dominated
    across the exposure-aware/hydraulic-only boundary. The two groups are
    returned together (both still real, both still CURRENT/VERIFIED); a
    caller that wants a single "contamination-response frontier" should
    filter to `group == "EXPOSURE_AWARE"` rather than treating the
    combined tuple as one comparable set.
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

    groups: dict[FrontierGroup, list[int]] = {"EXPOSURE_AWARE": [], "HYDRAULIC_ONLY": []}
    vectors = [_objective_vector(consequences) for _label, _verification, consequences in candidates]
    for index, (_label, _verification, consequences) in enumerate(candidates):
        groups[_frontier_group(consequences)].append(index)

    entries: list[FrontierEntry | None] = [None] * len(candidates)
    for group_name, indices in groups.items():
        for index in indices:
            _label, verification, consequences = candidates[index]
            dominated = any(
                other_index != index and _dominates(vectors[other_index], vectors[index])
                for other_index in indices
            )
            entries[index] = FrontierEntry(
                plan_id=verification.plan_id,
                label=_label,
                consequences=consequences,
                mode=mode,
                dominated=dominated,
                is_no_action_comparator=(_label == no_action_label),
                group=group_name,
            )
    return tuple(entry for entry in entries if entry is not None)
