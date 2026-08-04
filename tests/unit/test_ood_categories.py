from __future__ import annotations

import pytest

from hydroswarm.training import (
    OOD_CATEGORY_BEHAVIOR,
    AbstentionOutcome,
    ExpectedBehavior,
    OODCategory,
    classify_abstention_outcome,
    topology_calibration_is_valid,
)


def test_every_category_has_expected_behavior_defined() -> None:
    assert set(OOD_CATEGORY_BEHAVIOR) == set(OODCategory)


def test_category_covers_every_plan_specified_shift() -> None:
    expected = {
        "NONE",
        "UNSEEN_TOPOLOGY",
        "UNSEEN_SENSOR_LAYOUT",
        "EXTREME_DEMAND",
        "TANK_STATE_SHIFT",
        "VALVE_PUMP_MISMATCH",
        "ROUGHNESS_MISMATCH",
        "SEVERE_MISSINGNESS",
        "FROZEN_DRIFTING_SENSOR",
        "TIMING_OUTSIDE_TRAINING_RANGE",
        "UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION",
    }
    assert {member.value for member in OODCategory} == expected


def test_every_real_ood_category_suppresses_planning_and_invalidates_calibration() -> None:
    for category, behavior in OOD_CATEGORY_BEHAVIOR.items():
        if category is OODCategory.NONE:
            continue
        assert behavior.planning_permitted is False, f"{category} must suppress planning"
        assert behavior.calibration_valid is False, f"{category} must invalidate calibration"
        assert behavior.description


def test_in_distribution_permits_planning_with_valid_calibration() -> None:
    behavior = OOD_CATEGORY_BEHAVIOR[OODCategory.NONE]
    assert behavior.planning_permitted is True
    assert behavior.calibration_valid is True


def test_expected_behavior_rejects_planning_permitted_without_valid_calibration() -> None:
    with pytest.raises(ValueError, match="planning may never be permitted"):
        ExpectedBehavior(
            category=OODCategory.UNSEEN_TOPOLOGY,
            planning_permitted=True,
            calibration_valid=False,
            description="invalid combination",
        )


def test_expected_behavior_requires_description() -> None:
    with pytest.raises(ValueError, match="description"):
        ExpectedBehavior(
            category=OODCategory.NONE,
            planning_permitted=True,
            calibration_valid=True,
            description="",
        )


def test_topology_calibration_valid_for_known_hash() -> None:
    assert topology_calibration_is_valid("topo-a", {"topo-a", "topo-b"}) is True


def test_topology_calibration_invalid_for_unknown_hash_by_default() -> None:
    assert topology_calibration_is_valid("topo-unknown", {"topo-a", "topo-b"}) is False


def test_topology_calibration_valid_for_unknown_hash_only_with_explicit_broader_artifact() -> None:
    assert (
        topology_calibration_is_valid(
            "topo-unknown", {"topo-a"}, broader_validated_artifact_exists=True
        )
        is True
    )


@pytest.mark.parametrize(
    ("abstained", "safe_to_proceed", "expected"),
    [
        (True, True, AbstentionOutcome.FALSE_ABSTENTION),
        (True, False, AbstentionOutcome.CORRECT_ABSTENTION),
        (False, True, AbstentionOutcome.CORRECT_PROCEED),
        (False, False, AbstentionOutcome.UNSAFE_NON_ABSTENTION),
    ],
)
def test_classify_abstention_outcome_covers_all_four_combinations(abstained, safe_to_proceed, expected) -> None:
    assert classify_abstention_outcome(abstained=abstained, safe_to_proceed=safe_to_proceed) == expected
