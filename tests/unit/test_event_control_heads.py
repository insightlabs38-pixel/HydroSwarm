"""Task 4.4: event_presence/event_cause/next_step semantic heads.

Unlike prior_mode/incident_pooling/message_direction -- each an existing
pathway made configurable -- these three heads are net-new parameters
with no prior existence in the promoted checkpoint, so they are gated
behind `event_control_heads` (default False) rather than always
constructed. See EVENT_CONTROL_HEADS_DEFAULT's docstring in core.py.
"""

from __future__ import annotations

import random

import pytest
import torch

from hydroswarm.model import ArchitectureCompatibilityError, HydroCore, verify_architecture_compatibility
from hydroswarm.model.core import EVENT_CAUSE_CLASS_COUNT, NEXT_STEP_CLASS_COUNT
from hydroswarm.training import (
    CurriculumStage,
    ScenarioExample,
    TopologyMetadata,
    collate_variable_topology,
    permute_example,
)
from hydroswarm.training.corpus import EVENT_CAUSE_INDEX
from hydroswarm.training.targets_v2 import EventCause, NextStep


def _tiny_model(**overrides) -> HydroCore:
    base = dict(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )
    base.update(overrides)
    return HydroCore(**base)


def _batch(nodes: int = 4) -> dict:
    generator = torch.Generator().manual_seed(17)
    return {
        "node_features": torch.randn(2, nodes, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, nodes, dtype=torch.bool),
    }


def test_class_counts_match_the_targets_v2_enums() -> None:
    assert EVENT_CAUSE_CLASS_COUNT == len(EventCause)
    assert NEXT_STEP_CLASS_COUNT == len(NextStep)
    # EVENT_CAUSE_INDEX is the actual label encoder used by the corpus
    # generator (Task 2.1) -- its index range must line up with the head's
    # output width, or trained labels would silently point at the wrong
    # logit column.
    assert set(EVENT_CAUSE_INDEX.values()) == set(range(EVENT_CAUSE_CLASS_COUNT))


def test_default_event_control_heads_is_disabled() -> None:
    model = _tiny_model()
    assert model.event_control_heads is False
    assert not hasattr(model, "event_presence_head")
    assert not hasattr(model, "event_cause_head")
    assert not hasattr(model, "next_step_head")


def test_disabled_event_control_heads_omits_new_output_keys() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(_batch())
    assert "event_presence_logits" not in output
    assert "event_cause_logits" not in output
    assert "next_step_logits" not in output


def test_enabling_event_control_heads_adds_no_other_parameters() -> None:
    disabled_params = set(_tiny_model(event_control_heads=False).state_dict())
    enabled_params = set(_tiny_model(event_control_heads=True).state_dict())
    # Disabling must not remove or resize anything the disabled model has.
    assert disabled_params <= enabled_params
    new_params = enabled_params - disabled_params
    assert all(
        name.startswith(("event_presence_head", "event_cause_head", "next_step_head"))
        for name in new_params
    )
    assert any(name.startswith("event_presence_head") for name in new_params)
    assert any(name.startswith("event_cause_head") for name in new_params)
    assert any(name.startswith("next_step_head") for name in new_params)


def test_forward_pass_produces_correctly_shaped_logits_when_enabled() -> None:
    model = _tiny_model(event_control_heads=True).eval()
    with torch.no_grad():
        output = model(_batch())
    batch_size = 2
    assert output["event_presence_logits"].shape == (batch_size,)
    assert output["event_cause_logits"].shape == (batch_size, EVENT_CAUSE_CLASS_COUNT)
    assert output["next_step_logits"].shape == (batch_size, NEXT_STEP_CLASS_COUNT)
    assert torch.isfinite(output["event_presence_logits"]).all()
    assert torch.isfinite(output["event_cause_logits"]).all()
    assert torch.isfinite(output["next_step_logits"]).all()


def test_enabling_does_not_change_other_outputs_given_identical_weights() -> None:
    torch.manual_seed(8)
    disabled = _tiny_model(event_control_heads=False).eval()
    enabled = _tiny_model(event_control_heads=True).eval()
    shared = set(disabled.state_dict()) & set(enabled.state_dict())
    enabled.load_state_dict(
        {key: value for key, value in disabled.state_dict().items() if key in shared}, strict=False
    )
    batch = _batch()
    with torch.no_grad():
        disabled_output = disabled(batch)
        enabled_output = enabled(batch)
    torch.testing.assert_close(disabled_output["hidden_state"], enabled_output["hidden_state"])
    torch.testing.assert_close(disabled_output["source_node_logits"], enabled_output["source_node_logits"])
    torch.testing.assert_close(disabled_output["uncertainty"], enabled_output["uncertainty"])


def test_architecture_config_records_event_control_heads() -> None:
    disabled_config = _tiny_model(event_control_heads=False).architecture_config()
    enabled_config = _tiny_model(event_control_heads=True).architecture_config()
    assert disabled_config["event_control_heads"] is False
    assert enabled_config["event_control_heads"] is True


def test_verify_architecture_compatibility_accepts_matching_metadata() -> None:
    model = _tiny_model(event_control_heads=True)
    verify_architecture_compatibility(model, model.architecture_config())  # must not raise


def test_verify_architecture_compatibility_accepts_missing_field_for_old_checkpoints() -> None:
    model = _tiny_model()
    verify_architecture_compatibility(model, {})  # must not raise


def test_verify_architecture_compatibility_rejects_event_control_heads_mismatch() -> None:
    model = _tiny_model(event_control_heads=False)
    with pytest.raises(ArchitectureCompatibilityError, match="event_control_heads"):
        verify_architecture_compatibility(model, {"event_control_heads": True})


def test_event_control_head_outputs_are_permutation_invariant() -> None:
    # These are incident-level (batch-global) outputs, not per-node
    # outputs, so unlike source_node_logits (which measure_equivariance
    # checks by remapping through the permutation) they must be exactly
    # invariant under a node relabeling rather than merely reordered.
    nodes = 5
    edges = [(i, i + 1) for i in range(nodes - 1)] + [(nodes - 1, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    node_ids = tuple(f"J{i}" for i in range(nodes))
    topology = TopologyMetadata(
        topology_hash="t", network_hash="n", node_ids=node_ids,
        edge_ids=tuple((node_ids[a], node_ids[b]) for a, b in edges),
        source_candidate_ids=node_ids, hydraulic_state_hash="s", signature_library_hash="sig",
        target_schema_version="v1", feature_schema_version="v2",
    )
    generator = torch.Generator().manual_seed(19)
    example = ScenarioExample(
        scenario_id="s1", network_id="n", split="train", seed=1, seed_family="f1",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.randn(nodes, 3, generator=generator),
            "temporal_features": torch.randn(2, nodes, 2, generator=generator),
            "quality_features": torch.randn(2, nodes, 2, generator=generator),
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), 2, generator=generator),
            "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
            "node_mask": torch.ones(nodes, dtype=torch.bool),
        },
        targets={
            "source_node": torch.tensor(2),
            "sensor_fault": torch.rand(nodes, generator=generator) > 0.8,
        },
        topology=topology,
    )
    permutation = list(range(nodes))
    random.Random(23).shuffle(permutation)
    permuted_example = permute_example(example, permutation)

    model = _tiny_model(edge_feature_dim=2, event_control_heads=True).eval()
    with torch.no_grad():
        original_inputs, _ = collate_variable_topology([example])
        permuted_inputs, _ = collate_variable_topology([permuted_example])
        original_output = model(original_inputs)
        permuted_output = model(permuted_inputs)

    torch.testing.assert_close(
        original_output["event_presence_logits"], permuted_output["event_presence_logits"], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        original_output["event_cause_logits"], permuted_output["event_cause_logits"], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        original_output["next_step_logits"], permuted_output["next_step_logits"], atol=1e-4, rtol=1e-4
    )


def test_parameter_report_includes_event_control_heads_only_when_enabled() -> None:
    disabled = _tiny_model(event_control_heads=False)
    enabled = _tiny_model(event_control_heads=True)
    disabled_report = disabled.parameter_report()
    enabled_report = enabled.parameter_report()
    assert enabled_report.heads > disabled_report.heads
    assert enabled_report.total == disabled_report.total + (
        enabled.parameter_count() - disabled.parameter_count()
    )
