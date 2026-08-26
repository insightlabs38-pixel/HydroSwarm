"""Graph-structural-encoder-v2 experiment (EXPERIMENTAL, NON-RELEASE):
backward compatibility and correctness of the `GraphStructuralEncoder` /
`HydroCore` extension itself (arms B/C/D/E/capacity-control). See
docs/evaluation/experimental/GRAPH_STRUCTURAL_ENCODER_V2_PLAN.md Section 2.
"""

from __future__ import annotations

import torch

from hydroswarm.model.core import HydroCore
from hydroswarm.model.encoders import GraphStructuralEncoder


def test_default_encoder_matches_original_four_scalar_behavior() -> None:
    torch.manual_seed(0)
    encoder = GraphStructuralEncoder(16).eval()
    batch, nodes = 2, 5
    travel = torch.rand(batch, nodes)
    reach = torch.randint(0, 2, (batch, nodes)).float()
    demand = torch.rand(batch, nodes)
    with torch.no_grad():
        default_call = encoder(travel, reach, demand)
        explicit_none_call = encoder(
            travel, reach, demand, structural_features=None, edge_index=None, edge_mask=None, node_mask=None
        )
    assert torch.equal(default_call, explicit_none_call)
    assert encoder.structural_feature_dim == 0
    assert encoder.use_edge_aggregation is False
    # original module had exactly one Linear(4, hidden) as its first layer
    assert encoder.network[0].in_features == 4


def test_structural_feature_dim_widens_input_and_requires_features() -> None:
    encoder = GraphStructuralEncoder(16, structural_feature_dim=3).eval()
    assert encoder.network[0].in_features == 7
    batch, nodes = 2, 4
    travel = torch.rand(batch, nodes)
    reach = torch.rand(batch, nodes)
    demand = torch.rand(batch, nodes)
    try:
        encoder(travel, reach, demand)
        raised = False
    except ValueError:
        raised = True
    assert raised
    extra = torch.rand(batch, nodes, 3)
    with torch.no_grad():
        output = encoder(travel, reach, demand, structural_features=extra)
    assert output.shape == (batch, nodes, 16)
    assert torch.isfinite(output).all()


def test_edge_aggregation_changes_output_vs_graph_agnostic_baseline() -> None:
    torch.manual_seed(0)
    encoder = GraphStructuralEncoder(16, use_edge_aggregation=True).eval()
    batch, nodes = 1, 5
    travel = torch.rand(batch, nodes)
    reach = torch.rand(batch, nodes)
    demand = torch.rand(batch, nodes)
    node_mask = torch.ones(batch, nodes, dtype=torch.bool)
    edge_index = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 4]]], dtype=torch.long)
    edge_mask = torch.ones(batch, 4, dtype=torch.bool)
    with torch.no_grad():
        no_edges = encoder(travel, reach, demand, edge_index=None, node_mask=node_mask)
        with_edges = encoder(
            travel, reach, demand, edge_index=edge_index, edge_mask=edge_mask, node_mask=node_mask
        )
    assert not torch.allclose(no_edges, with_edges)


def test_self_only_capacity_control_ignores_edge_index() -> None:
    torch.manual_seed(0)
    encoder = GraphStructuralEncoder(
        16, use_edge_aggregation=True, edge_aggregation_source="self_only"
    ).eval()
    batch, nodes = 1, 5
    travel = torch.rand(batch, nodes)
    reach = torch.rand(batch, nodes)
    demand = torch.rand(batch, nodes)
    node_mask = torch.ones(batch, nodes, dtype=torch.bool)
    edge_index_a = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    edge_index_b = torch.tensor([[[4, 3, 2], [3, 2, 1]]], dtype=torch.long)
    with torch.no_grad():
        out_a = encoder(travel, reach, demand, edge_index=edge_index_a, node_mask=node_mask)
        out_b = encoder(travel, reach, demand, edge_index=edge_index_b, node_mask=node_mask)
    # self_only mode never reads edge_index, so two different graphs must
    # produce identical output -- this is the entire point of the control.
    assert torch.equal(out_a, out_b)


def test_self_only_control_is_exactly_parameter_matched_to_edge_aggregation() -> None:
    graph_mode = GraphStructuralEncoder(16, use_edge_aggregation=True, edge_aggregation_source="graph")
    self_only_mode = GraphStructuralEncoder(16, use_edge_aggregation=True, edge_aggregation_source="self_only")
    graph_params = sum(p.numel() for p in graph_mode.parameters())
    self_params = sum(p.numel() for p in self_only_mode.parameters())
    assert graph_params == self_params


def _minimal_batch(node_dim: int, structural_dim: int = 0) -> dict[str, torch.Tensor]:
    batch_size, nodes, edges = 2, 6, 8
    batch = {
        "node_features": torch.randn(batch_size, nodes, node_dim),
        "temporal_features": torch.randn(batch_size, 4, nodes, 6),
        "quality_features": torch.randn(batch_size, 4, nodes, 4),
        "node_mask": torch.ones(batch_size, nodes, dtype=torch.bool),
        "edge_index": torch.randint(0, nodes, (batch_size, 2, edges)),
        "edge_features": torch.randn(batch_size, edges, 13),
        "edge_mask": torch.ones(batch_size, edges, dtype=torch.bool),
        "travel_time": torch.rand(batch_size, nodes),
        "reservoir_reachability": torch.randint(0, 2, (batch_size, nodes)).float(),
        "demand_centrality": torch.rand(batch_size, nodes),
        "source_candidate_mask": torch.ones(batch_size, nodes, dtype=torch.bool),
    }
    if structural_dim:
        batch["structural_features"] = torch.rand(batch_size, nodes, structural_dim)
    return batch


def test_hydrocore_default_construction_is_unaffected() -> None:
    model = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13)
    assert model.graph_structural_feature_dim == 0
    assert model.graph_structural_edge_aggregation is False
    config = model.architecture_config()
    assert config["graph_structural_feature_dim"] == 0
    assert config["graph_structural_edge_aggregation"] is False
    output = model(_minimal_batch(19))
    assert torch.isfinite(output["source_node_logits"]).all()


def test_hydrocore_combined_arm_end_to_end_with_gradient() -> None:
    model = HydroCore.from_variant(
        "small",
        node_feature_dim=19,
        edge_feature_dim=13,
        graph_structural_feature_dim=13,
        graph_structural_edge_aggregation=True,
    )
    output = model(_minimal_batch(19, structural_dim=13))
    assert torch.isfinite(output["source_node_logits"]).all()
    output["source_node_logits"].sum().backward()
    graph_encoder_params = list(model.graph_encoder.parameters())
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in graph_encoder_params)


def test_parameter_report_reflects_arm_capacity_deltas() -> None:
    control = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13).parameter_report()
    centrality = HydroCore.from_variant(
        "small", node_feature_dim=19, edge_feature_dim=13, graph_structural_feature_dim=6
    ).parameter_report()
    structural_agg = HydroCore.from_variant(
        "small", node_feature_dim=19, edge_feature_dim=13, graph_structural_edge_aggregation=True
    ).parameter_report()
    assert centrality.total > control.total
    assert structural_agg.total > control.total
    # backbone/heads/adapters must be untouched by this encoder-only change
    assert centrality.backbone == control.backbone == structural_agg.backbone
    assert centrality.heads == control.heads == structural_agg.heads
    assert centrality.adapters == control.adapters == structural_agg.adapters
