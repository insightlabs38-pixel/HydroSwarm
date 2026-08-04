from __future__ import annotations

import pytest
import torch

from hydroswarm.training import (
    TARGETS_BY_CATEGORY,
    TARGETS_V2,
    TARGETS_V2_SCHEMA_VERSION,
    EventCause,
    NextStep,
    TargetSchemaError,
    check_schema_version,
    validate_targets_v2,
)


def test_every_target_has_complete_governance_fields() -> None:
    for name, spec in TARGETS_V2.items():
        assert spec.name == name
        assert spec.category in ("sentinel", "scout", "strategist", "control")
        assert spec.definition
        assert spec.unit
        assert spec.masking_rule
        assert spec.source_of_truth


def test_contract_covers_every_plan_specified_target() -> None:
    expected = {
        "sentinel": {
            "source_node",
            "source_region",
            "event_presence",
            "event_cause",
            "start_time",
            "duration",
            "relative_strength",
            "sensor_fault",
            "evidence_sufficiency",
        },
        "scout": {"sample_node", "information_gain", "candidate_reduction", "should_continue_sampling"},
        "strategist": {"action_template", "target_pointer", "plan_validity", "plan_value", "consequence_vector"},
        "control": {"ood_class", "next_step"},
    }
    for category, names in expected.items():
        assert set(TARGETS_BY_CATEGORY[category]) == names


def test_event_cause_has_exactly_the_specified_classes() -> None:
    assert {member.value for member in EventCause} == {
        "CONTAMINATION",
        "SENSOR_FAULT",
        "HYDRAULIC_MISMATCH",
        "AMBIGUOUS",
        "NORMAL",
    }


def test_next_step_has_exactly_the_specified_classes() -> None:
    assert {member.value for member in NextStep} == {
        "COLLECT_SAMPLE",
        "INSPECT_SENSOR",
        "GENERATE_PLANS",
        "ABSTAIN",
    }


def test_check_schema_version_accepts_current_and_rejects_other() -> None:
    check_schema_version(TARGETS_V2_SCHEMA_VERSION)  # must not raise
    with pytest.raises(TargetSchemaError, match="incompatible"):
        check_schema_version("targets_v1")


def test_validate_rejects_unknown_target_key() -> None:
    with pytest.raises(TargetSchemaError, match="unknown"):
        validate_targets_v2({"not_a_real_target": torch.tensor(0)})


def test_validate_accepts_known_targets_with_valid_masks() -> None:
    validate_targets_v2(
        {
            "source_node": torch.tensor(2),
            "source_node_mask": torch.tensor(True),
            "event_presence": torch.tensor(True),
        }
    )  # must not raise


def test_validate_rejects_mask_without_matching_value() -> None:
    with pytest.raises(TargetSchemaError, match="without its value key"):
        validate_targets_v2({"source_node_mask": torch.tensor(True)})


def test_validate_rejects_mask_on_a_non_maskable_target() -> None:
    with pytest.raises(TargetSchemaError, match="not maskable"):
        validate_targets_v2(
            {"event_presence": torch.tensor(True), "event_presence_mask": torch.tensor(True)}
        )


def test_validate_rejects_non_boolean_mask() -> None:
    with pytest.raises(TargetSchemaError, match="must be boolean"):
        validate_targets_v2(
            {"source_node": torch.tensor(1), "source_node_mask": torch.tensor(1)}
        )


def test_plan_validity_source_of_truth_names_wntr_not_a_neural_prediction() -> None:
    # Governance guardrail: the contract text itself must not permit a
    # neural head to set plan_validity.
    spec = TARGETS_V2["plan_validity"]
    assert "WNTR" in spec.source_of_truth
    assert "neural" in spec.source_of_truth.lower()
