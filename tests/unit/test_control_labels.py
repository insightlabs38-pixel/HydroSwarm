"""core-issues2.txt Phase 5: evidence_sufficiency and next_step control labels."""

from __future__ import annotations

from hydroswarm.training.control_labels import (
    MAXIMUM_SAMPLING_ROUNDS,
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
