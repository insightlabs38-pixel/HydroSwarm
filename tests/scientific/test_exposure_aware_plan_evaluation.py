"""important-issues.txt requirement 18: tests for the canonical exposure-
aware plan verification path -- the fix for the exposure-blind plan-
verification defect discovered in Phase 12 Stage E (`HydraulicSimulator.
evaluate_plan()` never computed contamination consequences, so every
Strategist `plan_value` label was mechanically tied at 1.0)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from hydroswarm.domain import ActionType, OperationalAction, OperationalPlan, PlanDecision
from hydroswarm.simulation import (
    HydraulicSimulator,
    PlanEvaluationContext,
    PlanVerifier,
    WeightedSourceHypothesis,
    build_wntr_network,
)
from hydroswarm.simulation.wrapper import IncidentSourceProfile
from hydroswarm.storage.cache import SimulationResultCache

pytest.importorskip("wntr")


def _network():
    model = build_wntr_network()
    model.options.time.duration = 2 * 3_600
    return model


def _simulator(**kwargs) -> HydraulicSimulator:
    return HydraulicSimulator(
        _network(), minimum_pressure_m=10.0, minimum_service_availability=0.90, **kwargs
    )


def _plan(name: str, *actions: OperationalAction) -> OperationalPlan:
    return OperationalPlan(
        incident_id=uuid4(),
        name=name,
        actions=actions or (OperationalAction(action_type=ActionType.END_PLAN),),
        model_version="test",
    )


def _no_action() -> OperationalPlan:
    return _plan("no action", OperationalAction(action_type=ActionType.END_PLAN))


def _flush(node: str = "J4") -> OperationalPlan:
    return _plan(
        "flush",
        OperationalAction(
            action_type=ActionType.FLUSH_NODE, target_id=node, flow_rate_lps=3.0, duration_minutes=45
        ),
    )


def _profile(node: str = "J1", strength: float = 50.0) -> IncidentSourceProfile:
    return IncidentSourceProfile(source_node_id=node, strength_mg_min=strength, start_minute=0, duration_minutes=30)


@pytest.mark.real_simulation
def test_no_action_canonical_evaluator_matches_standalone_incident_simulation() -> None:
    simulator = _simulator()
    profile = _profile()
    ctx = PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=profile)

    canonical = simulator.evaluate_plan_consequences(_no_action(), ctx).consequences

    standalone_simulator = _simulator()
    baseline = standalone_simulator._baseline_requested_demand()
    standalone_simulation = standalone_simulator.simulate_hypothesis(profile, include_diagnostics=False)
    standalone = standalone_simulator.calculate_consequences(
        standalone_simulation, threshold_mg_l=0.001, requested_demand_m3s=baseline, operation_count=0
    )

    assert canonical.exposure_evaluated is True
    # rel=1e-3: simulate_incident_plan runs on _prepared_network() (PDD
    # demand mode), simulate_hypothesis on a plain deepcopy (default demand
    # mode) -- for this fully-supplied fixture network both deliver
    # essentially full demand, so results agree to float32 solver precision,
    # not bit-for-bit.
    assert canonical.contaminant_mass_consumed_mg == pytest.approx(standalone.contaminant_mass_consumed_mg, rel=1e-3)
    assert canonical.volume_above_threshold_l == pytest.approx(standalone.volume_above_threshold_l, rel=1e-3)
    assert canonical.minimum_pressure_m == pytest.approx(standalone.minimum_pressure_m, rel=1e-3)


@pytest.mark.real_simulation
def test_known_flush_plan_changes_exposure_relative_to_no_action() -> None:
    simulator = _simulator()
    verifier = PlanVerifier(simulator)
    ctx = PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile())

    no_action_result = verifier.verify(_no_action(), ctx)
    flush_result = verifier.verify(_flush(), ctx)

    assert no_action_result.consequences.exposure_evaluated
    assert flush_result.consequences.exposure_evaluated
    assert (
        flush_result.consequences.contaminant_mass_consumed_mg
        != no_action_result.consequences.contaminant_mass_consumed_mg
    )
    assert flush_result.consequences.contaminant_mass_consumed_mg < no_action_result.consequences.contaminant_mass_consumed_mg


@pytest.mark.real_simulation
def test_same_plan_under_different_source_profiles_has_distinct_exposure() -> None:
    simulator = _simulator()
    verifier = PlanVerifier(simulator)
    plan = _flush()

    result_j1 = verifier.verify(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile("J1"))
    )
    result_j3 = verifier.verify(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile("J3"))
    )

    assert (
        result_j1.consequences.contaminant_mass_consumed_mg != result_j3.consequences.contaminant_mass_consumed_mg
    )


@pytest.mark.real_simulation
def test_profile_specific_cache_isolation(tmp_path) -> None:
    cache = SimulationResultCache(tmp_path / "cache")
    simulator = _simulator(cache=cache)
    plan = _flush()

    result_j1 = simulator.evaluate_plan_consequences(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile("J1"))
    )
    result_j3 = simulator.evaluate_plan_consequences(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile("J3"))
    )
    assert not result_j1.cache_hit and not result_j3.cache_hit
    assert result_j1.consequences.contaminant_mass_consumed_mg != result_j3.consequences.contaminant_mass_consumed_mg

    # A repeat of the exact same profile hits cache and reproduces the same value.
    result_j1_again = simulator.evaluate_plan_consequences(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile("J1"))
    )
    assert result_j1_again.cache_hit
    assert result_j1_again.consequences.contaminant_mass_consumed_mg == pytest.approx(
        result_j1.consequences.contaminant_mass_consumed_mg
    )


@pytest.mark.real_simulation
def test_threshold_specific_cache_isolation(tmp_path) -> None:
    cache = SimulationResultCache(tmp_path / "cache")
    simulator = _simulator(cache=cache)
    plan = _flush()
    profile = _profile()

    low = simulator.evaluate_plan_consequences(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=0.0001, source_profile=profile)
    )
    high = simulator.evaluate_plan_consequences(
        plan, PlanEvaluationContext(contamination_threshold_mg_l=5.0, source_profile=profile)
    )
    assert low.state_hash != high.state_hash
    assert low.consequences.volume_above_threshold_l >= high.consequences.volume_above_threshold_l


@pytest.mark.real_simulation
def test_pressure_and_service_rejection_still_derives_from_exact_simulator() -> None:
    simulator = _simulator()
    verifier = PlanVerifier(simulator)
    unsafe_plan = _plan(
        "unsafe", OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P_R1_J1", duration_minutes=60)
    )
    ctx = PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile())

    result = verifier.verify(unsafe_plan, ctx)

    assert result.decision == PlanDecision.REJECTED
    assert "PRESSURE_BELOW_MINIMUM" in result.rejection_codes
    assert result.consequences.minimum_pressure_m < 10.0
    assert result.consequences.exposure_evaluated is True


@pytest.mark.real_simulation
def test_simulation_failure_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = _simulator()
    verifier = PlanVerifier(simulator)
    ctx = PlanEvaluationContext(contamination_threshold_mg_l=0.001, source_profile=_profile())

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated EPANET failure")

    monkeypatch.setattr(simulator, "simulate_incident_plan", _boom)

    result = verifier.verify(_no_action(), ctx)

    assert result.decision == PlanDecision.ABSTAINED
    assert result.abstention_reason is not None


@pytest.mark.real_simulation
def test_multi_hypothesis_reports_posterior_weighted_and_worst_case() -> None:
    simulator = _simulator()
    verifier = PlanVerifier(simulator)
    hypotheses = (
        WeightedSourceHypothesis(profile=_profile("J1", strength=50.0), probability=0.5),
        WeightedSourceHypothesis(profile=_profile("J3", strength=5.0), probability=0.5),
    )
    ctx = PlanEvaluationContext(contamination_threshold_mg_l=0.001, hypotheses=hypotheses)

    result = verifier.verify(_flush(), ctx)

    assert result.worst_case_consequences is not None
    assert (
        result.worst_case_consequences.contaminant_mass_consumed_mg
        >= result.consequences.contaminant_mass_consumed_mg
    )
    assert result.evaluation_provenance is not None
    #: important-issues.txt requirement 10: a multi-hypothesis verification
    #: must not silently count as one simulator call.
    assert result.evaluation_provenance["exact_simulation_count"] == 2
    assert result.evaluation_provenance["aggregation_policy"] == "posterior_weighted"


@pytest.mark.real_simulation
def test_worst_case_aggregation_policy_reports_the_conservative_number() -> None:
    simulator = _simulator()
    verifier = PlanVerifier(simulator)
    hypotheses = (
        WeightedSourceHypothesis(profile=_profile("J1", strength=50.0), probability=0.5),
        WeightedSourceHypothesis(profile=_profile("J3", strength=5.0), probability=0.5),
    )
    ctx = PlanEvaluationContext(
        contamination_threshold_mg_l=0.001, hypotheses=hypotheses, aggregation_policy="worst_case"
    )

    result = verifier.verify(_flush(), ctx)

    assert result.consequences.contaminant_mass_consumed_mg == pytest.approx(
        result.worst_case_consequences.contaminant_mass_consumed_mg
    )


@pytest.mark.real_simulation
def test_hydraulic_only_legacy_path_marks_exposure_unevaluated_not_zero_measured() -> None:
    """important-issues.txt requirement 12: a plan verified without any
    evaluation context must never present its Pydantic-default exposure
    fields as if they were measured."""

    simulator = _simulator()
    verifier = PlanVerifier(simulator)

    result = verifier.verify(_no_action())

    assert result.decision == PlanDecision.VERIFIED
    assert result.consequences.exposure_evaluated is False
    assert result.consequences.contaminant_mass_consumed_mg == 0.0


def test_maximum_three_hypotheses_enforced() -> None:
    hypotheses = tuple(
        WeightedSourceHypothesis(profile=_profile(node), probability=0.25) for node in ("J1", "J2", "J3", "J4")
    )
    with pytest.raises(ValueError):
        PlanEvaluationContext(contamination_threshold_mg_l=0.001, hypotheses=hypotheses)


def test_context_requires_exactly_one_of_profile_or_hypotheses() -> None:
    with pytest.raises(ValueError):
        PlanEvaluationContext(contamination_threshold_mg_l=0.001)
    with pytest.raises(ValueError):
        PlanEvaluationContext(
            contamination_threshold_mg_l=0.001,
            source_profile=_profile(),
            hypotheses=(WeightedSourceHypothesis(profile=_profile("J2"), probability=1.0),),
        )
