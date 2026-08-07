"""core-issues4.txt Section C: granular output-name governance."""

from __future__ import annotations

import pytest

from hydroswarm.training.output_governance import (
    CANONICAL_OUTPUT_NAMES,
    OutputGovernanceError,
    validate_output_governance,
)


def test_canonical_names_are_disjoint_by_role_and_nonempty() -> None:
    assert len(CANONICAL_OUTPUT_NAMES) > 0
    # every name maps to exactly one role bucket (OUTPUT_ROLE is built from
    # disjoint literal sets, so re-deriving it here would just re-test the
    # module's own construction -- instead assert the property that matters:
    # no name is silently duplicated across roles).
    from hydroswarm.training.output_governance import (
        AUXILIARY_OUTPUTS,
        OOD_CONTROL_OUTPUTS,
        SCOUT_OUTPUTS,
        SENTINEL_OUTPUTS,
        STRATEGIST_OUTPUTS,
    )

    buckets = [SENTINEL_OUTPUTS, SCOUT_OUTPUTS, STRATEGIST_OUTPUTS, OOD_CONTROL_OUTPUTS, AUXILIARY_OUTPUTS]
    total = sum(len(bucket) for bucket in buckets)
    assert total == len(CANONICAL_OUTPUT_NAMES)


def test_empty_governance_sets_are_valid() -> None:
    validate_output_governance(
        trained_outputs=frozenset(),
        validated_outputs=frozenset(),
        runtime_enabled_outputs=frozenset(),
        diagnostic_only_outputs=frozenset(),
        training_only_outputs=frozenset(),
    )


def test_full_valid_chain_is_accepted() -> None:
    validate_output_governance(
        trained_outputs=frozenset({"source_node", "source_region"}),
        validated_outputs=frozenset({"source_node"}),
        runtime_enabled_outputs=frozenset({"source_node"}),
        diagnostic_only_outputs=frozenset({"plan_value"}),
        training_only_outputs=frozenset({"sensor_reconstruction"}),
    )


def test_unknown_output_name_fails_closed() -> None:
    with pytest.raises(OutputGovernanceError, match="unknown output name"):
        validate_output_governance(
            trained_outputs=frozenset({"not_a_real_output"}),
            validated_outputs=frozenset(),
            runtime_enabled_outputs=frozenset(),
            diagnostic_only_outputs=frozenset(),
            training_only_outputs=frozenset(),
        )


def test_runtime_enabled_must_be_subset_of_validated() -> None:
    with pytest.raises(OutputGovernanceError, match="runtime_enabled_outputs is not a subset"):
        validate_output_governance(
            trained_outputs=frozenset({"source_node"}),
            validated_outputs=frozenset(),
            runtime_enabled_outputs=frozenset({"source_node"}),
            diagnostic_only_outputs=frozenset(),
            training_only_outputs=frozenset(),
        )


def test_validated_must_be_subset_of_trained() -> None:
    with pytest.raises(OutputGovernanceError, match="validated_outputs is not a subset"):
        validate_output_governance(
            trained_outputs=frozenset(),
            validated_outputs=frozenset({"source_node"}),
            runtime_enabled_outputs=frozenset(),
            diagnostic_only_outputs=frozenset(),
            training_only_outputs=frozenset(),
        )


def test_diagnostic_only_cannot_overlap_trained() -> None:
    with pytest.raises(OutputGovernanceError, match="diagnostic_only_outputs overlaps"):
        validate_output_governance(
            trained_outputs=frozenset({"plan_value"}),
            validated_outputs=frozenset(),
            runtime_enabled_outputs=frozenset(),
            diagnostic_only_outputs=frozenset({"plan_value"}),
            training_only_outputs=frozenset(),
        )


def test_training_only_cannot_overlap_runtime_enabled() -> None:
    with pytest.raises(OutputGovernanceError, match="training_only_outputs overlaps"):
        validate_output_governance(
            trained_outputs=frozenset({"sensor_reconstruction"}),
            validated_outputs=frozenset({"sensor_reconstruction"}),
            runtime_enabled_outputs=frozenset({"sensor_reconstruction"}),
            diagnostic_only_outputs=frozenset(),
            training_only_outputs=frozenset({"sensor_reconstruction"}),
        )
