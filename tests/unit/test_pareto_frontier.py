"""core-issues5.txt Section 14 (P1 product feature): verified response
Pareto frontier."""

from __future__ import annotations

from uuid import uuid4

from hydroswarm.domain import AbstentionReason, ConsequenceMetrics, PlanDecision, PlanVerification
from hydroswarm.planning.pareto import compute_verified_pareto_frontier


def _verification(
    *,
    decision: PlanDecision = PlanDecision.VERIFIED,
    consequences: ConsequenceMetrics | None = None,
    worst_case_consequences: ConsequenceMetrics | None = None,
    verification_status: str = "CURRENT",
    **overrides,
) -> PlanVerification:
    kwargs = dict(
        plan_id=uuid4(),
        decision=decision,
        simulator="test",
        simulator_version="1.0",
        state_hash="a" * 64,
        consequences=consequences,
        worst_case_consequences=worst_case_consequences,
        verification_status=verification_status,
    )
    if decision == PlanDecision.REJECTED and "rejection_codes" not in overrides:
        kwargs["rejection_codes"] = ("PRESSURE_BELOW_MINIMUM",)
    if decision == PlanDecision.ABSTAINED and "abstention_reason" not in overrides:
        kwargs["abstention_reason"] = AbstentionReason.SIMULATION_TIMEOUT
    kwargs.update(overrides)
    return PlanVerification(**kwargs)


def _metrics(**overrides) -> ConsequenceMetrics:
    defaults = dict(
        population_impacted=0.0,
        contaminant_mass_consumed_mg=0.0,
        volume_above_threshold_l=0.0,
        contaminated_pipe_extent_m=0.0,
        minimum_pressure_m=20.0,
        pressure_violation_minutes=0.0,
        unserved_demand_l=0.0,
        service_availability=1.0,
        operation_count=1,
        exposure_evaluated=True,
    )
    defaults.update(overrides)
    return ConsequenceMetrics(**defaults)


def test_strictly_better_plan_dominates_a_strictly_worse_one() -> None:
    better = _verification(consequences=_metrics(contaminant_mass_consumed_mg=1.0))
    worse = _verification(consequences=_metrics(contaminant_mass_consumed_mg=5.0))

    entries = compute_verified_pareto_frontier([("A", better), ("B", worse)])
    by_label = {e.label: e for e in entries}
    assert by_label["A"].dominated is False
    assert by_label["B"].dominated is True


def test_genuinely_tradeoff_plans_are_both_non_dominated() -> None:
    lower_exposure_worse_service = _verification(
        consequences=_metrics(contaminant_mass_consumed_mg=1.0, service_availability=0.80)
    )
    higher_exposure_better_service = _verification(
        consequences=_metrics(contaminant_mass_consumed_mg=5.0, service_availability=0.99)
    )
    entries = compute_verified_pareto_frontier(
        [("A", lower_exposure_worse_service), ("B", higher_exposure_better_service)]
    )
    assert all(not entry.dominated for entry in entries)


def test_rejected_and_abstained_plans_never_enter_the_frontier() -> None:
    rejected = _verification(decision=PlanDecision.REJECTED)
    abstained = _verification(decision=PlanDecision.ABSTAINED)
    verified = _verification(consequences=_metrics())

    entries = compute_verified_pareto_frontier(
        [("rejected", rejected), ("abstained", abstained), ("verified", verified)]
    )
    assert {e.label for e in entries} == {"verified"}


def test_stale_verification_never_enters_the_frontier() -> None:
    stale = _verification(consequences=_metrics(), verification_status="STALE")
    current = _verification(consequences=_metrics())

    entries = compute_verified_pareto_frontier([("stale", stale), ("current", current)])
    assert {e.label for e in entries} == {"current"}


def test_no_action_comparator_is_flagged_and_included_even_when_dominated() -> None:
    no_action = _verification(consequences=_metrics(contaminant_mass_consumed_mg=10.0))
    better_plan = _verification(consequences=_metrics(contaminant_mass_consumed_mg=1.0))

    entries = compute_verified_pareto_frontier([("NO_ACTION", no_action), ("ISOLATE_SOURCE", better_plan)])
    by_label = {e.label: e for e in entries}
    assert by_label["NO_ACTION"].is_no_action_comparator is True
    assert by_label["NO_ACTION"].dominated is True
    assert by_label["ISOLATE_SOURCE"].is_no_action_comparator is False


def test_posterior_weighted_and_worst_case_modes_are_labeled_and_independent() -> None:
    verification = _verification(
        consequences=_metrics(contaminant_mass_consumed_mg=1.0),
        worst_case_consequences=_metrics(contaminant_mass_consumed_mg=9.0),
    )
    posterior = compute_verified_pareto_frontier([("A", verification)], mode="posterior_weighted")
    worst_case = compute_verified_pareto_frontier([("A", verification)], mode="worst_case")

    assert posterior[0].mode == "posterior_weighted"
    assert posterior[0].consequences.contaminant_mass_consumed_mg == 1.0
    assert worst_case[0].mode == "worst_case"
    assert worst_case[0].consequences.contaminant_mass_consumed_mg == 9.0


def test_worst_case_mode_excludes_verifications_without_worst_case_consequences() -> None:
    """Single-hypothesis (training-time) verification never has
    worst_case_consequences -- must be excluded from worst_case mode, not
    silently substitute the posterior-weighted value."""

    verification = _verification(consequences=_metrics(), worst_case_consequences=None)
    entries = compute_verified_pareto_frontier([("A", verification)], mode="worst_case")
    assert entries == ()


def test_unmeasured_exposure_is_never_compared_as_a_fabricated_zero() -> None:
    """A hydraulic-only (exposure_evaluated=False) verification's exposure
    fields are Pydantic zero defaults, not measured -- must not make it
    spuriously dominate a real exposure-evaluated plan on those fields."""

    hydraulic_only = _verification(
        consequences=_metrics(exposure_evaluated=False, contaminant_mass_consumed_mg=0.0)
    )
    real_exposure = _verification(
        consequences=_metrics(exposure_evaluated=True, contaminant_mass_consumed_mg=3.0)
    )
    entries = compute_verified_pareto_frontier([("hydraulic_only", hydraulic_only), ("real", real_exposure)])
    by_label = {e.label: e for e in entries}
    # core-issues5.txt delta item 9: hydraulic_only and real belong to
    # different FrontierGroups entirely -- dominated is computed only
    # within a group, so this is not merely "compared on shared fields and
    # found equal", it is never compared for domination at all.
    assert by_label["hydraulic_only"].group == "HYDRAULIC_ONLY"
    assert by_label["real"].group == "EXPOSURE_AWARE"
    assert by_label["hydraulic_only"].dominated is False
    assert by_label["real"].dominated is False


# core-issues5.txt delta item 9 (P1 fix): a hydraulic-only plan must never
# dominate, or be treated as equivalent to, an exposure-evaluated plan on
# the contamination-response (EXPOSURE_AWARE) frontier -- proven with a
# case that WOULD have dominated under the old shared-subset-only
# comparison (strictly better on every hydraulic field both happen to
# have), which must have no effect now.


def test_a_hydraulic_only_plan_strictly_better_on_shared_fields_still_cannot_dominate_an_exposure_evaluated_plan() -> None:
    hydraulic_only_better_everywhere = _verification(
        consequences=_metrics(
            exposure_evaluated=False,
            pressure_violation_minutes=0.0,
            unserved_demand_l=0.0,
            operation_count=1,
            minimum_pressure_m=30.0,
            service_availability=1.0,
        )
    )
    exposure_evaluated_worse_hydraulics = _verification(
        consequences=_metrics(
            exposure_evaluated=True,
            contaminant_mass_consumed_mg=3.0,
            pressure_violation_minutes=5.0,
            unserved_demand_l=10.0,
            operation_count=3,
            minimum_pressure_m=15.0,
            service_availability=0.85,
        )
    )
    entries = compute_verified_pareto_frontier([
        ("hydraulic_only", hydraulic_only_better_everywhere),
        ("exposure", exposure_evaluated_worse_hydraulics),
    ])
    by_label = {e.label: e for e in entries}
    assert by_label["hydraulic_only"].group == "HYDRAULIC_ONLY"
    assert by_label["exposure"].group == "EXPOSURE_AWARE"
    # Neither is dominated: each is alone in its own group.
    assert by_label["hydraulic_only"].dominated is False
    assert by_label["exposure"].dominated is False


def test_hydraulic_only_plans_are_dominated_only_by_other_hydraulic_only_plans() -> None:
    worse_hydraulic_only = _verification(
        consequences=_metrics(exposure_evaluated=False, pressure_violation_minutes=10.0)
    )
    better_hydraulic_only = _verification(
        consequences=_metrics(exposure_evaluated=False, pressure_violation_minutes=0.0)
    )
    unrelated_exposure_plan = _verification(
        consequences=_metrics(exposure_evaluated=True, contaminant_mass_consumed_mg=100.0)
    )
    entries = compute_verified_pareto_frontier([
        ("worse", worse_hydraulic_only),
        ("better", better_hydraulic_only),
        ("exposure", unrelated_exposure_plan),
    ])
    by_label = {e.label: e for e in entries}
    assert by_label["worse"].group == "HYDRAULIC_ONLY"
    assert by_label["worse"].dominated is True
    assert by_label["better"].dominated is False
    # The lone exposure-evaluated plan is never dominated by (or compared
    # against) either hydraulic-only entry.
    assert by_label["exposure"].group == "EXPOSURE_AWARE"
    assert by_label["exposure"].dominated is False


def test_no_action_comparator_never_bridges_hydraulic_only_and_exposure_aware_groups() -> None:
    """NO_ACTION follows the same group-isolation rule as every other
    entry -- it is exempted from domination only within its own group,
    never used to compare across the exposure/hydraulic-only boundary."""

    no_action_hydraulic_only = _verification(
        consequences=_metrics(exposure_evaluated=False, contaminant_mass_consumed_mg=0.0)
    )
    exposure_plan = _verification(
        consequences=_metrics(exposure_evaluated=True, contaminant_mass_consumed_mg=3.0)
    )
    entries = compute_verified_pareto_frontier(
        [("NO_ACTION", no_action_hydraulic_only), ("ISOLATE_SOURCE", exposure_plan)]
    )
    by_label = {e.label: e for e in entries}
    assert by_label["NO_ACTION"].group == "HYDRAULIC_ONLY"
    assert by_label["NO_ACTION"].is_no_action_comparator is True
    assert by_label["NO_ACTION"].dominated is False
    assert by_label["ISOLATE_SOURCE"].group == "EXPOSURE_AWARE"
    assert by_label["ISOLATE_SOURCE"].dominated is False
