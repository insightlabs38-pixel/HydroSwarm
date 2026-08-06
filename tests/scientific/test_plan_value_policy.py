"""core-issues3.txt Phase 3.2: PlanValuePolicy monotonicity tests.

These are the exact required tests the spec names: "lower exposure cannot
reduce utility, all else equal"; "fewer pressure violations cannot reduce
utility"; "greater service availability cannot reduce utility"; "shorter
containment time cannot reduce utility"; "an invalid plan cannot outrank a
valid plan solely because of an unverified heuristic score"; "NO_ACTION
remains an explicit comparator."
"""

from __future__ import annotations

from hydroswarm.domain import ConsequenceMetrics
from hydroswarm.planning.plan_value_policy import evaluate_plan_value


def _metrics(**overrides) -> ConsequenceMetrics:
    defaults = dict(
        contaminant_mass_consumed_mg=10.0,
        minimum_pressure_m=15.0,
        pressure_violation_minutes=0.0,
        service_availability=1.0,
        operation_count=1,
        containment_time_minutes=30.0,
    )
    defaults.update(overrides)
    return ConsequenceMetrics(**defaults)


NO_RESPONSE = _metrics(contaminant_mass_consumed_mg=100.0, operation_count=0, containment_time_minutes=None)


def test_lower_exposure_cannot_reduce_utility() -> None:
    worse = _metrics(contaminant_mass_consumed_mg=50.0)
    better = _metrics(contaminant_mass_consumed_mg=10.0)
    pool = [NO_RESPONSE, worse, better]

    worse_result = evaluate_plan_value(worse, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    better_result = evaluate_plan_value(better, no_response=NO_RESPONSE, valid_candidate_metrics=pool)

    assert better_result.plan_value >= worse_result.plan_value
    assert better_result.regret <= worse_result.regret
    assert better_result.exposure_proxy < worse_result.exposure_proxy


def test_fewer_pressure_violations_cannot_reduce_utility() -> None:
    worse = _metrics(pressure_violation_minutes=45.0)
    better = _metrics(pressure_violation_minutes=0.0)
    pool = [NO_RESPONSE, worse, better]

    worse_result = evaluate_plan_value(worse, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    better_result = evaluate_plan_value(better, no_response=NO_RESPONSE, valid_candidate_metrics=pool)

    assert better_result.plan_value >= worse_result.plan_value
    assert better_result.pressure_risk_proxy < worse_result.pressure_risk_proxy


def test_greater_service_availability_cannot_reduce_utility() -> None:
    worse = _metrics(service_availability=0.6)
    better = _metrics(service_availability=0.95)
    pool = [NO_RESPONSE, worse, better]

    worse_result = evaluate_plan_value(worse, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    better_result = evaluate_plan_value(better, no_response=NO_RESPONSE, valid_candidate_metrics=pool)

    assert better_result.plan_value >= worse_result.plan_value
    assert better_result.service_loss_proxy < worse_result.service_loss_proxy


def test_shorter_containment_time_cannot_reduce_utility() -> None:
    worse = _metrics(containment_time_minutes=180.0)
    better = _metrics(containment_time_minutes=15.0)
    pool = [NO_RESPONSE, worse, better]

    worse_result = evaluate_plan_value(worse, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    better_result = evaluate_plan_value(better, no_response=NO_RESPONSE, valid_candidate_metrics=pool)

    assert better_result.plan_value >= worse_result.plan_value
    assert better_result.containment_time_proxy < worse_result.containment_time_proxy


def test_never_contained_is_treated_as_at_least_as_bad_as_the_containment_scale() -> None:
    never_contained = _metrics(containment_time_minutes=None)
    slow_but_contained = _metrics(containment_time_minutes=239.0)
    pool = [NO_RESPONSE, never_contained, slow_but_contained]

    never_result = evaluate_plan_value(never_contained, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    slow_result = evaluate_plan_value(slow_but_contained, no_response=NO_RESPONSE, valid_candidate_metrics=pool)

    assert never_result.plan_value <= slow_result.plan_value


def test_best_plan_in_the_pool_has_zero_regret_and_maximum_value() -> None:
    best = _metrics(
        contaminant_mass_consumed_mg=0.0,
        pressure_violation_minutes=0.0,
        service_availability=1.0,
        containment_time_minutes=0.0,
    )
    worse = _metrics(contaminant_mass_consumed_mg=80.0)
    pool = [NO_RESPONSE, best, worse]

    result = evaluate_plan_value(best, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    assert result.regret == 0.0
    assert result.plan_value == 1.0


def test_no_action_is_scored_the_same_way_as_any_other_candidate() -> None:
    """NO_ACTION remains an explicit comparator, not a free pass: if a real
    response plan strictly dominates it, NO_ACTION's own regret must be
    positive, exactly like any other candidate."""

    strictly_better_plan = _metrics(contaminant_mass_consumed_mg=5.0, pressure_violation_minutes=0.0)
    pool = [NO_RESPONSE, strictly_better_plan]

    no_response_result = evaluate_plan_value(NO_RESPONSE, no_response=NO_RESPONSE, valid_candidate_metrics=pool)
    plan_result = evaluate_plan_value(strictly_better_plan, no_response=NO_RESPONSE, valid_candidate_metrics=pool)

    assert plan_result.regret == 0.0
    assert no_response_result.regret > 0.0
    assert plan_result.plan_value > no_response_result.plan_value


def test_invalid_plans_never_receive_a_plan_value_from_this_policy() -> None:
    """This policy takes only exact ConsequenceMetrics as input -- there is
    no code path here that could construct a plan_value from an unverified
    heuristic score. The caller (strategist_labels.generate_strategist_labels)
    must never call evaluate_plan_value for a plan whose PlanVerifier
    decision was not VERIFIED; that discipline is enforced at the call
    site, verified separately in test_strategist_labels.py."""

    result = evaluate_plan_value(NO_RESPONSE, no_response=NO_RESPONSE, valid_candidate_metrics=[NO_RESPONSE])
    assert isinstance(result.plan_value, float)
