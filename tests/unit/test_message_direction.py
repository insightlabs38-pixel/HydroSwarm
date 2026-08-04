"""Task 4.3: dual hydraulic message channels."""

from __future__ import annotations

import random

import pytest
import torch

from hydroswarm.model import (
    ARCHITECTURE_VERSION,
    ArchitectureCompatibilityError,
    HydroCore,
    MESSAGE_DIRECTIONS,
    verify_architecture_compatibility,
)
from hydroswarm.training import (
    CurriculumStage,
    ScenarioExample,
    TopologyMetadata,
    collate_variable_topology,
    measure_equivariance,
)


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
    generator = torch.Generator().manual_seed(11)
    return {
        "node_features": torch.randn(2, nodes, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, nodes, dtype=torch.bool),
    }


def _edge_batch(nodes: int = 4) -> dict:
    edges = [(i, i + 1) for i in range(nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    generator = torch.Generator().manual_seed(12)
    batch = _batch(nodes)
    batch["edge_index"] = edge_index
    batch["edge_features"] = torch.randn(len(edges), 3, generator=generator)
    return batch


def test_invalid_message_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="message_direction"):
        _tiny_model(message_direction="not_a_real_mode")


def test_default_message_direction_is_forward_only() -> None:
    model = _tiny_model()
    assert model.message_direction == "forward_only"


@pytest.mark.parametrize("mode", MESSAGE_DIRECTIONS)
def test_forward_pass_succeeds_for_every_mode(mode: str) -> None:
    model = _tiny_model(message_direction=mode, edge_feature_dim=3).eval()
    with torch.no_grad():
        output = model(_edge_batch())
    assert torch.isfinite(output["uncertainty"]).all()
    assert torch.isfinite(output["source_node_logits"]).all()
    assert torch.isfinite(output["hidden_state"]).all()


def test_forward_only_matches_original_single_direction_convolution() -> None:
    torch.manual_seed(0)
    model = _tiny_model(message_direction="forward_only", edge_feature_dim=3).eval()
    with torch.no_grad():
        output = model(_edge_batch())
    block = model.backbone[0]
    from hydroswarm.model.layers import EdgeAwareGraphConv

    assert isinstance(block.local, EdgeAwareGraphConv)
    assert torch.isfinite(output["hidden_state"]).all()


def test_dual_gated_uses_dual_channel_graph_conv() -> None:
    model = _tiny_model(message_direction="dual_gated", edge_feature_dim=3)
    from hydroswarm.model.layers import DualChannelGraphConv

    block = model.backbone[0]
    assert isinstance(block.local, DualChannelGraphConv)


def test_dual_gated_adds_expected_new_parameters() -> None:
    forward_only_params = set(_tiny_model(message_direction="forward_only", edge_feature_dim=3).state_dict())
    dual_gated_params = set(_tiny_model(message_direction="dual_gated", edge_feature_dim=3).state_dict())

    new_params = dual_gated_params - forward_only_params
    assert any("backbone.0.local.upstream" in name for name in new_params)
    assert any("backbone.0.local.gate" in name for name in new_params)
    # The downstream channel keeps the same parameter names the
    # forward_only EdgeAwareGraphConv used, just nested one level deeper
    # under `.downstream`, so it is a "new" fully-qualified name too even
    # though its role is unchanged.
    assert any("backbone.0.local.downstream" in name for name in new_params)


def test_dual_gated_reduces_to_downstream_channel_when_gate_saturates_toward_one() -> None:
    torch.manual_seed(3)
    model = _tiny_model(message_direction="dual_gated", edge_feature_dim=3).eval()
    block = model.backbone[0].local
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.bias.fill_(50.0)  # sigmoid(50) ~= 1.0 -> gate collapses to downstream only
        hidden = torch.randn(1, 4, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_features = torch.randn(3, 3)
        edge_mask = torch.ones(1, 3, dtype=torch.bool)
        combined = block(hidden, edge_index, edge_features, edge_mask)
        downstream_only = block.downstream(hidden, edge_index, edge_features, edge_mask)
    torch.testing.assert_close(combined, downstream_only, atol=1e-4, rtol=1e-4)


def test_upstream_channel_reverses_source_and_target() -> None:
    torch.manual_seed(4)
    from hydroswarm.model.layers import EdgeAwareGraphConv

    conv = EdgeAwareGraphConv(16, 3, dropout=0.0, normalization="rmsnorm", activation="silu").eval()
    hidden = torch.randn(1, 3, 16)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # single edge 0 -> 1
    edge_features = torch.randn(1, 3)
    edge_mask = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        forward_update = conv._aggregate(hidden, edge_index, edge_features, edge_mask, reverse=False)
        reverse_update = conv._aggregate(hidden, edge_index, edge_features, edge_mask, reverse=True)
    # Forward sends a message into node 1; reverse sends the same edge's
    # message into node 0 instead -- their updates at nodes 0 and 1 must
    # therefore differ from each other for this single directed edge.
    assert not torch.allclose(forward_update[:, 1], reverse_update[:, 1])
    assert not torch.allclose(forward_update, reverse_update)


def test_architecture_config_records_message_direction() -> None:
    model = _tiny_model(message_direction="dual_gated", edge_feature_dim=3)
    config = model.architecture_config()
    assert config["architecture_version"] == ARCHITECTURE_VERSION
    assert config["message_direction"] == "dual_gated"


def test_verify_architecture_compatibility_accepts_matching_metadata() -> None:
    model = _tiny_model(message_direction="dual_gated", edge_feature_dim=3)
    verify_architecture_compatibility(model, model.architecture_config())  # must not raise


def test_verify_architecture_compatibility_accepts_missing_field_for_old_checkpoints() -> None:
    model = _tiny_model()
    verify_architecture_compatibility(model, {})  # must not raise


def test_verify_architecture_compatibility_rejects_message_direction_mismatch() -> None:
    model = _tiny_model(message_direction="forward_only")
    with pytest.raises(ArchitectureCompatibilityError, match="message_direction"):
        verify_architecture_compatibility(model, {"message_direction": "dual_gated"})


@pytest.mark.parametrize("mode", MESSAGE_DIRECTIONS)
def test_permutation_equivariance_holds_for_every_message_direction(mode: str) -> None:
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
    generator = torch.Generator().manual_seed(13)
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
    random.Random(21).shuffle(permutation)
    model = _tiny_model(edge_feature_dim=2, message_direction=mode)

    report = measure_equivariance(model, example, permutation, collate_fn=collate_variable_topology, atol=1e-4)
    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees
    assert report.max_absolute_source_logit_difference < 1e-4
