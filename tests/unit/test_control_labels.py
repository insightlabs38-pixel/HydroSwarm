"""core-issues2.txt Phase 5: evidence_sufficiency and next_step control labels."""

from __future__ import annotations

from hydroswarm.training.control_labels import (
    MAXIMUM_SAMPLING_ROUNDS,
    NEXT_STEP_RUNTIME_ENABLED,
    classify_evidence_sufficiency,
    classify_next_step,
    next_step_target,
)
from hydroswarm.training.ood_categories import OODCategory
from hydroswarm.training.targets_v2 import EventCause, NextStep, validate_targets_v2

_SUFFICIENT_KWARGS = dict(
    healthy_fraction=0.9, sensors_ever_healthy=3, posterior_entropy_bits=0.5, ood_category=OODCategory.NONE,
)


def test_all_three_conditions_passing_is_sufficient() -> None:
    assert classify_evidence_sufficiency(**_SUFFICIENT_KWARGS) is True


def test_poor_sensor_health_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, healthy_fraction=0.1)
    assert classify_evidence_sufficiency(**kwargs) is False


def test_too_few_healthy_sensors_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, sensors_ever_healthy=1)
    assert classify_evidence_sufficiency(**kwargs) is False


def test_high_posterior_entropy_is_insufficient() -> None:
    kwargs = dict(_SUFFICIENT_KWARGS, posterior_entropy_bits=10.0)
    assert classify_evidence_sufficiency(**kwargs) is False


def test_ood_category_invalidates_calibration_and_is_insufficient() -> None:
    # SEVERE_MISSINGNESS's own ExpectedBehavior states "evidence_sufficiency
    # should already be false in this state" -- confirm that's exactly what
    # this rule does even when sensor health and entropy otherwise pass.
    kwargs = dict(_SUFFICIENT_KWARGS, ood_category=OODCategory.SEVERE_MISSINGNESS)
    assert classify_evidence_sufficiency(**kwargs) is False


def test_next_step_abstains_when_ood_outside_validated_range() -> None:
    step = classify_next_step(
        ood_level_outside_validated_range=True, evidence_sufficient=False,
        sample_count=0, event_cause=EventCause.CONTAMINATION,
    )
    assert step == NextStep.ABSTAIN


def test_next_step_generates_plans_when_evidence_is_sufficient() -> None:
    step = classify_next_step(
        ood_level_outside_validated_range=False, evidence_sufficient=True,
        sample_count=1, event_cause=EventCause.CONTAMINATION,
    )
    assert step == NextStep.GENERATE_PLANS


def test_next_step_abstains_when_sampling_budget_is_exhausted() -> None:
    step = classify_next_step(
        ood_level_outside_validated_range=False, evidence_sufficient=False,
        sample_count=MAXIMUM_SAMPLING_ROUNDS, event_cause=EventCause.CONTAMINATION,
    )
    assert step == NextStep.ABSTAIN


def test_next_step_inspects_sensor_when_cause_is_a_sensor_fault() -> None:
    step = classify_next_step(
        ood_level_outside_validated_range=False, evidence_sufficient=False,
        sample_count=0, event_cause=EventCause.SENSOR_FAULT,
    )
    assert step == NextStep.INSPECT_SENSOR


def test_next_step_collects_sample_as_the_default_case() -> None:
    step = classify_next_step(
        ood_level_outside_validated_range=False, evidence_sufficient=False,
        sample_count=0, event_cause=EventCause.CONTAMINATION,
    )
    assert step == NextStep.COLLECT_SAMPLE


def test_ood_takes_priority_over_a_sensor_fault_cause() -> None:
    step = classify_next_step(
        ood_level_outside_validated_range=True, evidence_sufficient=False,
        sample_count=0, event_cause=EventCause.SENSOR_FAULT,
    )
    assert step == NextStep.ABSTAIN


def test_next_step_target_is_governed_and_covers_every_class() -> None:
    for step in NextStep:
        target = next_step_target(step)
        validate_targets_v2(target)  # must not raise -- never masked, per its TargetSpec
    indices = {int(next_step_target(step)["next_step"]) for step in NextStep}
    assert len(indices) == len(list(NextStep))


def test_inspect_sensor_is_a_trainable_label_but_not_fsm_runtime_enabled() -> None:
    """core-issues3.txt Phase 8 item 8: INSPECT_SENSOR remains a valid
    training label (classify_next_step still derives it deterministically
    from event_cause == SENSOR_FAULT), but must not be authorized by the
    agent FSM controller at runtime -- no matching FSMState exists to
    fulfill it there (see NEXT_STEP_RUNTIME_ENABLED's docstring). This is
    scoped to the agent-FSM controller specifically:
    hydroswarm.inference.pipeline already has a SEPARATE, already-live
    ControlAction.INSPECT_SENSORS driven by a different signal
    (classical/neural disagreement, not event_cause) -- see
    test_fsm_has_no_inspect_sensor_state_but_the_inference_pipeline_
    already_does below."""

    assert NextStep.INSPECT_SENSOR not in NEXT_STEP_RUNTIME_ENABLED
    assert NEXT_STEP_RUNTIME_ENABLED == frozenset(NextStep) - {NextStep.INSPECT_SENSOR}


def test_fsm_has_no_inspect_sensor_state_but_the_inference_pipeline_already_does() -> None:
    """Tripwire, scoped accurately (not "no inspect-sensor concept exists
    anywhere in the codebase" -- it does, just elsewhere): if a future
    change adds a matching FSMState, this test fails and flags
    NEXT_STEP_RUNTIME_ENABLED for review rather than letting the exclusion
    silently go stale. Also documents, so it cannot silently regress, that
    hydroswarm.inference.fusion.uncertainty_control already has a live
    ControlAction.INSPECT_SENSORS -- a genuinely different, already-
    authoritative signal (disagreement_js-driven) from this module's
    event_cause-driven NextStep.INSPECT_SENSOR, not the same mechanism
    under two names."""

    from hydroswarm.agents.schemas import FSMState
    from hydroswarm.inference.fusion import ControlAction, uncertainty_control

    assert not any("SENSOR" in state.value for state in FSMState)
    assert ControlAction.INSPECT_SENSORS in set(ControlAction)
    assert (
        uncertainty_control(
            candidate_count=3, disagreement_js=0.6, ood_score=0.0,
            healthy_sensor_fraction=1.0, sample_budget_remaining=2,
        )
        == ControlAction.INSPECT_SENSORS
    )
