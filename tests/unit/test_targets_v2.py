from __future__ import annotations

import pytest
import torch

from hydroswarm.training import (
    TARGETS_BY_CATEGORY,
    TARGETS_V2,
    TARGETS_V2_SCHEMA_VERSION,
    EventCause,
    NextStep,
    OODCategory,
    TargetSchemaError,
    TopologyMetadata,
    check_schema_version,
    validate_targets_v2,
)
from hydroswarm.training.targets_v2 import TARGET_CLASS_COUNTS


def _topology(*, node_count: int = 4, edge_count: int = 2) -> TopologyMetadata:
    node_ids = tuple(f"J{i}" for i in range(node_count))
    edge_ids = tuple((node_ids[i], node_ids[i + 1]) for i in range(edge_count))
    return TopologyMetadata(
        topology_hash="t" * 64,
        network_hash="n" * 64,
        node_ids=node_ids,
        edge_ids=edge_ids,
        source_candidate_ids=node_ids,
        hydraulic_state_hash="h" * 64,
        signature_library_hash="s" * 64,
        target_schema_version=TARGETS_V2_SCHEMA_VERSION,
        feature_schema_version="feature-schema-v1",
    )


def test_every_target_has_complete_governance_fields() -> None:
    for name, spec in TARGETS_V2.items():
        assert spec.name == name
        assert spec.category in ("sentinel", "scout", "strategist", "control", "auxiliary")
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
        "strategist": {
            "action_template",
            "target_pointer",
            "plan_validity",
            "plan_value",
            "consequence_vector",
            "exposure_proxy",
            "pressure_risk_proxy",
            "service_loss_proxy",
            "containment_time_proxy",
            "plan_regret_proxy",
        },
        "control": {"ood_class", "next_step"},
        "auxiliary": {"sensor_reconstruction", "future_concentration", "travel_time"},
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


def test_validate_rejects_a_maskable_target_present_without_its_mask() -> None:
    with pytest.raises(TargetSchemaError, match="required mask"):
        validate_targets_v2({"source_node": torch.tensor(2)})


def test_ood_class_range_is_the_full_governed_category_taxonomy_not_the_head_width() -> None:
    # Regression: TARGET_CLASS_COUNTS["ood_class"] previously mirrored
    # hydroswarm.model.core's ood_head width (3, a distinct concept --
    # OODLevel severity), not ood_class's own governed definition (the
    # full OODCategory taxonomy). That would have silently rejected any
    # real category index >= 3 as "invalid" -- 8 of the 11 governed
    # categories, including UNSEEN_TOPOLOGY-adjacent ones with high
    # ordinal position.
    assert TARGET_CLASS_COUNTS["ood_class"] == len(OODCategory) == 11
    highest_index = len(OODCategory) - 1
    validate_targets_v2({"ood_class": torch.tensor(highest_index)})  # must not raise
    with pytest.raises(TargetSchemaError, match="valid class range"):
        validate_targets_v2({"ood_class": torch.tensor(len(OODCategory))})


def test_validate_rejects_a_categorical_value_outside_its_class_range() -> None:
    with pytest.raises(TargetSchemaError, match="valid class range"):
        # plan_validity has exactly 2 classes: 0,1
        validate_targets_v2({"plan_validity": torch.tensor(2), "plan_validity_mask": torch.tensor(True)})


def test_validate_ignores_categorical_value_outside_range_at_a_masked_out_position() -> None:
    # A masked-out (invalid) position holds an arbitrary placeholder value by
    # design (see targets_v2's module docstring) and must never be range-checked.
    validate_targets_v2(
        {"source_region": torch.tensor(99), "source_region_mask": torch.tensor(False)}
    )  # must not raise


def test_validate_rejects_a_graph_local_index_outside_the_topology_node_count() -> None:
    topology = _topology(node_count=4)
    with pytest.raises(TargetSchemaError, match="graph-local index"):
        validate_targets_v2(
            {"source_node": torch.tensor(4), "source_node_mask": torch.tensor(True)},
            topology=topology,
        )


def test_validate_accepts_a_graph_local_index_within_the_topology_node_count() -> None:
    topology = _topology(node_count=4)
    validate_targets_v2(
        {"source_node": torch.tensor(3), "source_node_mask": torch.tensor(True)},
        topology=topology,
    )  # must not raise -- 3 is the last valid index for node_count=4


def test_validate_skips_graph_local_index_check_without_topology() -> None:
    # No topology given -- node_count is unknowable, so the check must not
    # fire even for a value that looks implausibly large.
    validate_targets_v2({"source_node": torch.tensor(999), "source_node_mask": torch.tensor(True)})


def test_validate_rejects_node_array_target_whose_length_disagrees_with_topology() -> None:
    topology = _topology(node_count=4)
    with pytest.raises(TargetSchemaError, match="disagreeing with"):
        validate_targets_v2(
            {
                "sensor_fault": torch.zeros(3),  # topology has 4 nodes, not 3
                "sensor_fault_mask": torch.zeros(3, dtype=torch.bool),
            },
            topology=topology,
        )


def test_validate_accepts_node_array_target_matching_topology_node_count() -> None:
    topology = _topology(node_count=4)
    validate_targets_v2(
        {"sensor_fault": torch.zeros(4), "sensor_fault_mask": torch.zeros(4, dtype=torch.bool)},
        topology=topology,
    )  # must not raise


def test_validate_rejects_disagreeing_plan_dimension_targets() -> None:
    with pytest.raises(TargetSchemaError, match="plan-dimension targets disagree"):
        validate_targets_v2(
            {
                "action_template": torch.zeros(2, dtype=torch.long),
                "action_template_mask": torch.ones(2, dtype=torch.bool),
                "plan_value": torch.zeros(3),
                "plan_value_mask": torch.ones(3, dtype=torch.bool),
            }
        )


def test_validate_accepts_agreeing_plan_dimension_targets() -> None:
    validate_targets_v2(
        {
            "action_template": torch.zeros(2, dtype=torch.long),
            "action_template_mask": torch.ones(2, dtype=torch.bool),
            "plan_value": torch.zeros(2),
            "plan_value_mask": torch.ones(2, dtype=torch.bool),
        }
    )  # must not raise -- both agree on 2 plan candidates


def test_validate_rejects_non_finite_value_at_a_valid_position() -> None:
    with pytest.raises(TargetSchemaError, match="non-finite"):
        validate_targets_v2(
            {
                "information_gain": torch.tensor([float("nan")]),
                "information_gain_mask": torch.tensor([True]),
            }
        )


def test_validate_ignores_non_finite_value_at_a_masked_out_position() -> None:
    validate_targets_v2(
        {
            "information_gain": torch.tensor([float("nan")]),
            "information_gain_mask": torch.tensor([False]),
        }
    )  # must not raise -- NaN at a masked-out (invalid/placeholder) position is fine


def test_plan_validity_source_of_truth_names_wntr_not_a_neural_prediction() -> None:
    # Governance guardrail: the contract text itself must not permit a
    # neural head to set plan_validity.
    spec = TARGETS_V2["plan_validity"]
    assert "WNTR" in spec.source_of_truth
    assert "neural" in spec.source_of_truth.lower()
