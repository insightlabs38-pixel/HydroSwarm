from __future__ import annotations

from uuid import uuid4

import pytest

from hydroswarm.domain import (
    AbstentionReason,
    ActionType,
    OperationalAction,
    OperationalPlan,
    PlanDecision,
)
from hydroswarm.simulation import HydraulicSimulator, PlanVerifier, build_wntr_network
from hydroswarm.simulation.wrapper import (
    SimulationBudgetExceeded,
    SimulationIncompleteError,
    SimulationTimeoutError,
    SimulationUnstableError,
)


pytest.importorskip("wntr")


def _plan(*actions: OperationalAction) -> OperationalPlan:
    return OperationalPlan(
        incident_id=uuid4(),
        name="verifier test plan",
        actions=actions,
        model_version="test",
    )


@pytest.fixture
def verifier() -> PlanVerifier:
    model = build_wntr_network()
    model.options.time.duration = 2 * 3_600
    return PlanVerifier(
        HydraulicSimulator(model, minimum_pressure_m=10.0, minimum_service_availability=0.90)
    )


def test_safe_monitor_and_sample_plan_is_verified(verifier: PlanVerifier) -> None:
    plan = _plan(
        OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J2"),
        OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id="J3", start_minute=5),
    )
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.VERIFIED
    assert result.consequences is not None
    assert result.consequences.minimum_pressure_m >= 10.0
    assert result.consequences.service_availability == pytest.approx(1.0)
    assert result.consequences.operation_count == 2
    assert len(result.state_hash) == 64


def test_unknown_target_is_rejected_before_simulation(verifier: PlanVerifier) -> None:
    plan = _plan(OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="NOT_A_PIPE"))
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.REJECTED
    assert result.consequences is None
    assert result.rejection_codes == ("UNKNOWN_TARGET:NOT_A_PIPE",)


def test_unsafe_supply_isolation_is_rejected_by_exact_simulation(verifier: PlanVerifier) -> None:
    plan = _plan(OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P_R1_J1"))
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.REJECTED
    assert result.consequences is not None
    assert "PRESSURE_BELOW_MINIMUM" in result.rejection_codes
    assert result.consequences.minimum_pressure_m < 10.0


def test_state_and_dynamic_graph_are_hydraulically_derived(verifier: PlanVerifier) -> None:
    state = verifier.simulator.calculate_state(at_time=0)
    graph = verifier.simulator.build_dynamic_graph(state)

    assert set(state.pressure_m) >= {"J1", "J2", "J3", "J4"}
    assert graph.number_of_edges() == 7
    assert all("travel_time_seconds" in data for *_, data in graph.edges(data=True, keys=True))
    assert graph.nodes["J1"]["reservoir_reachable"] is True


def test_incident_uses_epanet_quality_without_cwd_artifacts(
    verifier: PlanVerifier, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    incident = verifier.simulator.simulate_incident(
        "J1", strength_mg_min=2.0, start_minute=0, duration_minutes=30
    )

    assert incident.concentration_mg_l.shape[1] == 6
    assert incident.concentration_mg_l.to_numpy().max() > 0.0
    assert len(incident.state_hash) == 64
    assert not list(tmp_path.glob("temp.*"))


# core-issues5.txt Section 9 (P0 safety fix): governed simulator failures
# must persist exact-run consumption, categorize into a specific
# AbstentionReason (not one generic bucket), never fabricate a
# VERIFIED/zero-exposure result, and fail closed.


@pytest.mark.parametrize(
    ("exception_type", "expected_reason"),
    [
        (SimulationTimeoutError, AbstentionReason.SIMULATION_TIMEOUT),
        (SimulationBudgetExceeded, AbstentionReason.SIMULATION_BUDGET_EXCEEDED),
        (SimulationUnstableError, AbstentionReason.SIMULATION_UNSTABLE),
        (SimulationIncompleteError, AbstentionReason.SIMULATION_INCOMPLETE),
        (RuntimeError, AbstentionReason.SIMULATION_FAILURE),
    ],
)
def test_legacy_path_failure_categorizes_and_fails_closed(
    verifier: PlanVerifier, monkeypatch: pytest.MonkeyPatch, exception_type: type, expected_reason
) -> None:
    plan = _plan(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J2"))

    def _raise(_plan):
        verifier.simulator.exact_runs += 1  # the real code path consumes a run before it can fail
        raise exception_type("synthetic failure")

    monkeypatch.setattr(verifier.simulator, "evaluate_plan", _raise)
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.ABSTAINED
    assert result.abstention_reason == expected_reason
    assert result.consequences is None  # never a fabricated zero-exposure result
    assert result.evaluation_provenance is not None
    assert result.evaluation_provenance["exact_simulation_count"] == 1
    assert result.evaluation_provenance["failure_type"] == exception_type.__name__


def test_exposure_aware_path_failure_categorizes_and_records_partial_consumption(
    verifier: PlanVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure partway through a multi-hypothesis loop must still report
    exactly how many of the hypotheses were actually simulated before the
    failure -- not 0, and not the full hypothesis count."""

    from hydroswarm.simulation.wrapper import (
        IncidentSourceProfile,
        PlanEvaluationContext,
        WeightedSourceHypothesis,
    )

    plan = _plan(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J2"))
    context = PlanEvaluationContext(
        contamination_threshold_mg_l=0.01,
        hypotheses=(
            WeightedSourceHypothesis(IncidentSourceProfile(source_node_id="J1"), 0.5),
            WeightedSourceHypothesis(IncidentSourceProfile(source_node_id="J3"), 0.5),
        ),
    )

    def _raise(_plan, _context):
        verifier.simulator.exact_runs += 2  # simulate 2 of the 2 hypotheses having already run
        raise SimulationUnstableError("synthetic mid-loop failure")

    monkeypatch.setattr(verifier.simulator, "evaluate_plan_consequences", _raise)
    result = verifier.verify(plan, context)

    assert result.decision == PlanDecision.ABSTAINED
    assert result.abstention_reason == AbstentionReason.SIMULATION_UNSTABLE
    assert result.consequences is None
    assert result.evaluation_provenance is not None
    assert result.evaluation_provenance["exact_simulation_count"] == 2
    assert result.evaluation_provenance["hypotheses_per_plan"] == 2


def test_budget_exceeded_mid_loop_is_categorized_distinctly(
    verifier: PlanVerifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydroswarm.simulation.wrapper import (
        IncidentSourceProfile,
        PlanEvaluationContext,
        WeightedSourceHypothesis,
    )

    plan = _plan(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J2"))
    context = PlanEvaluationContext(
        contamination_threshold_mg_l=0.01,
        hypotheses=(WeightedSourceHypothesis(IncidentSourceProfile(source_node_id="J1"), 1.0),),
    )
    monkeypatch.setattr(
        verifier.simulator, "evaluate_plan_consequences",
        lambda *_: (_ for _ in ()).throw(SimulationBudgetExceeded("budget exhausted")),
    )
    result = verifier.verify(plan, context)

    assert result.abstention_reason == AbstentionReason.SIMULATION_BUDGET_EXCEEDED
    assert result.decision == PlanDecision.ABSTAINED
    assert result.consequences is None


# core-issues5.txt Section 16 (P1 safety/portability feature): margins to
# critical thresholds, not bitwise equality, are the real operational
# safety contract for a value this close to a boundary.


def test_verified_plan_reports_real_margins(verifier: PlanVerifier) -> None:
    plan = _plan(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J2"))
    result = verifier.verify(plan)

    assert result.consequences is not None
    assert result.consequences.pressure_margin_m == pytest.approx(
        result.consequences.minimum_pressure_m - verifier.simulator.minimum_pressure_m
    )
    assert result.consequences.service_availability_margin == pytest.approx(
        result.consequences.service_availability - verifier.simulator.minimum_service_availability
    )


def test_rejected_plan_also_reports_real_margins(verifier: PlanVerifier) -> None:
    """A REJECTED decision is exact simulator output too -- margins must
    not be silently omitted just because the plan failed."""

    plan = _plan(OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P_R1_J1"))
    result = verifier.verify(plan)

    assert result.decision == PlanDecision.REJECTED
    assert result.consequences is not None
    assert result.consequences.pressure_margin_m is not None
    assert result.consequences.pressure_margin_m < 0  # genuinely failed, not a boundary artifact


def test_numerical_sensitivity_helper_flags_a_margin_inside_the_epsilon_band() -> None:
    from hydroswarm.simulation.verifier import (
        PRESSURE_SENSITIVITY_EPSILON_M,
        _with_numerical_sensitivity,
    )

    just_inside = _metrics_with_pressure(10.0 + PRESSURE_SENSITIVITY_EPSILON_M / 2)
    stamped = _with_numerical_sensitivity(
        just_inside, minimum_pressure_m=10.0, minimum_service_availability=0.90
    )
    assert stamped.numerically_sensitive is True
    assert stamped.pressure_margin_m == pytest.approx(PRESSURE_SENSITIVITY_EPSILON_M / 2)


def test_numerical_sensitivity_helper_does_not_flag_a_comfortable_margin() -> None:
    from hydroswarm.simulation.verifier import (
        PRESSURE_SENSITIVITY_EPSILON_M,
        _with_numerical_sensitivity,
    )

    comfortable = _metrics_with_pressure(10.0 + PRESSURE_SENSITIVITY_EPSILON_M * 10)
    stamped = _with_numerical_sensitivity(
        comfortable, minimum_pressure_m=10.0, minimum_service_availability=0.90
    )
    assert stamped.numerically_sensitive is False


def test_numerical_sensitivity_helper_flags_a_near_miss_below_threshold_too() -> None:
    """A REJECTED plan that barely failed is just as numerically sensitive
    as one that barely passed -- sensitivity is about closeness to the
    boundary, not which side of it the value landed on."""

    from hydroswarm.simulation.verifier import (
        PRESSURE_SENSITIVITY_EPSILON_M,
        _with_numerical_sensitivity,
    )

    near_miss = _metrics_with_pressure(10.0 - PRESSURE_SENSITIVITY_EPSILON_M / 2)
    stamped = _with_numerical_sensitivity(
        near_miss, minimum_pressure_m=10.0, minimum_service_availability=0.90
    )
    assert stamped.numerically_sensitive is True
    assert stamped.pressure_margin_m is not None
    assert stamped.pressure_margin_m < 0


def _metrics_with_pressure(minimum_pressure_m: float):
    from hydroswarm.domain import ConsequenceMetrics

    return ConsequenceMetrics(
        minimum_pressure_m=minimum_pressure_m,
        service_availability=1.0,
        operation_count=0,
    )
