from __future__ import annotations

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.training import CurriculumStage, ScenarioExample, collate_variable_topology


def _tiny_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        edge_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        latent_tokens=64,
        adapter_dims=(32, 32, 32),
        dropout=0.0,
    )


def _example(scenario_id: str, *, nodes: int, edges: list[tuple[int, int]], source_local_index: int, seed: int) -> ScenarioExample:
    generator = torch.Generator().manual_seed(seed)
    steps = 2
    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)
    return ScenarioExample(
        scenario_id=scenario_id,
        network_id="net",
        split="train",
        seed=seed,
        seed_family=f"family-{scenario_id}",
        stage=CurriculumStage.CLEAN,
        inputs={
            "node_features": torch.randn(nodes, 3, generator=generator),
            "temporal_features": torch.randn(steps, nodes, 2, generator=generator),
            "quality_features": torch.randn(steps, nodes, 2, generator=generator),
            "edge_index": edge_index,
            "edge_features": torch.randn(len(edges), 2, generator=generator) if edges else torch.zeros(0, 2),
            "travel_time": torch.rand(nodes, generator=generator),
            "reservoir_reachability": torch.rand(nodes, generator=generator),
            "demand_centrality": torch.rand(nodes, generator=generator),
            "source_candidate_mask": torch.ones(nodes, dtype=torch.bool),
            "node_mask": torch.ones(nodes, dtype=torch.bool),
        },
        targets={
            "source_node": torch.tensor(source_local_index),
            "sensor_fault": torch.zeros(nodes),
        },
    )


def _small_example(seed: int = 1) -> ScenarioExample:
    return _example("small", nodes=3, edges=[(0, 1), (1, 2)], source_local_index=1, seed=seed)


def _large_example(seed: int = 2) -> ScenarioExample:
    return _example("large", nodes=6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], source_local_index=4, seed=seed)


def test_batch_two_graphs_of_different_sizes() -> None:
    inputs, targets = collate_variable_topology([_small_example(), _large_example()])
    assert inputs["node_features"].shape == (2, 6, 3)  # padded to max_nodes=6
    assert inputs["node_mask"].tolist() == [
        [True, True, True, False, False, False],
        [True, True, True, True, True, True],
    ]
    assert inputs["source_candidate_mask"].shape == (2, 6)
    assert inputs["source_candidate_mask"][0].tolist() == [True, True, True, False, False, False]
    assert targets["source_node"].tolist() == [1, 4]
    assert targets["sensor_fault"].shape == (2, 6)


def test_forward_pass_through_hydrocore_with_padded_batch() -> None:
    model = _tiny_model().eval()
    inputs, _ = collate_variable_topology([_small_example(), _large_example()])
    with torch.no_grad():
        output = model(inputs)
    assert output["source_node_logits"].shape == (2, 6)


def test_source_logits_are_masked_correctly_for_padded_nodes() -> None:
    model = _tiny_model().eval()
    inputs, _ = collate_variable_topology([_small_example(), _large_example()])
    with torch.no_grad():
        output = model(inputs)
    padded_logits = output["source_node_logits"][0, 3:]  # small example's padded positions
    assert torch.all(padded_logits <= torch.finfo(padded_logits.dtype).min / 2)


def test_padding_does_not_affect_valid_node_predictions() -> None:
    model = _tiny_model().eval()
    small = _small_example()

    alone_inputs, _ = collate_variable_topology([small])
    with torch.no_grad():
        alone_output = model(alone_inputs)

    padded_inputs, _ = collate_variable_topology([small, _large_example()])
    with torch.no_grad():
        padded_output = model(padded_inputs)

    torch.testing.assert_close(
        alone_output["source_node_logits"][0, :3],
        padded_output["source_node_logits"][0, :3],
        atol=1e-5,
        rtol=1e-4,
    )


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_variable_topology([])


def test_collate_rejects_mismatched_target_keys() -> None:
    small = _small_example()
    other = ScenarioExample(
        scenario_id="other",
        network_id="net",
        split="train",
        seed=3,
        seed_family="family-other",
        stage=CurriculumStage.CLEAN,
        inputs=dict(small.inputs),
        targets={"source_node": torch.tensor(0)},  # missing sensor_fault
    )
    with pytest.raises(ValueError, match="identical target keys"):
        collate_variable_topology([small, other])
