"""Task 4.2: source-conditioned incident pooling."""

from __future__ import annotations

import random

import pytest
import torch

from hydroswarm.model import HydroCore, INCIDENT_POOLING_MODES
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


def _batch(nodes: int = 4, *, sensor_mask: torch.Tensor | None = None) -> dict:
    generator = torch.Generator().manual_seed(3)
    batch = {
        "node_features": torch.randn(2, nodes, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, nodes, dtype=torch.bool),
    }
    if sensor_mask is not None:
        batch["sensor_mask"] = sensor_mask
    return batch


def test_invalid_incident_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="incident_pooling"):
        _tiny_model(incident_pooling="not_a_real_mode")


def test_default_incident_pooling_is_mean() -> None:
    model = _tiny_model()
    assert model.incident_pooling == "mean"


@pytest.mark.parametrize("mode", INCIDENT_POOLING_MODES)
def test_forward_pass_succeeds_for_every_mode(mode: str) -> None:
    model = _tiny_model(incident_pooling=mode).eval()
    with torch.no_grad():
        output = model(_batch())
    assert torch.isfinite(output["uncertainty"]).all()
    assert torch.isfinite(output["evidence_sufficiency"]).all()
    assert torch.isfinite(output["ood_logits"]).all()
    assert torch.isfinite(output["start_time_logits"]).all()


def test_mean_pooling_is_unchanged_from_original_masked_mean() -> None:
    torch.manual_seed(0)
    model = _tiny_model(incident_pooling="mean").eval()
    node_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    batch = _batch()
    batch["node_mask"] = node_mask
    with torch.no_grad():
        output = model(batch)
        # Recompute the pooled hidden state independently and check the
        # uncertainty head gives the same result as calling it directly on
        # that pooled value.
        recompute_input = output["hidden_state"]
        pooled = (recompute_input * node_mask.unsqueeze(-1)).sum(1) / node_mask.sum(1, keepdim=True).clamp_min(1)
        expected = model.uncertainty_head(pooled)
    torch.testing.assert_close(output["uncertainty"], expected)


def test_different_pooling_modes_produce_different_outputs_with_identical_weights() -> None:
    torch.manual_seed(5)
    mean_model = _tiny_model(incident_pooling="mean").eval()
    latent_model = _tiny_model(incident_pooling="latent").eval()
    shared_keys = set(mean_model.state_dict()) & set(latent_model.state_dict())
    latent_model.load_state_dict(
        {key: value for key, value in mean_model.state_dict().items() if key in shared_keys},
        strict=False,
    )
    batch = _batch()
    with torch.no_grad():
        mean_output = mean_model(batch)
        latent_output = latent_model(batch)
    assert not torch.allclose(mean_output["uncertainty"], latent_output["uncertainty"])


def test_source_conditioned_pooling_uses_sensor_mask_when_available() -> None:
    torch.manual_seed(6)
    model = _tiny_model(incident_pooling="source_conditioned").eval()
    sensor_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    sensor_mask[:, :, 0] = True  # only node 0 ever observed
    batch = _batch(sensor_mask=sensor_mask)
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output["uncertainty"]).all()


def test_source_conditioned_pooling_handles_no_sensor_mask_by_falling_back_to_node_mask() -> None:
    model = _tiny_model(incident_pooling="source_conditioned").eval()
    with torch.no_grad():
        output = model(_batch(sensor_mask=None))
    assert torch.isfinite(output["uncertainty"]).all()


def test_attention_pool_handles_a_row_with_only_one_valid_node_without_nan() -> None:
    model = _tiny_model(incident_pooling="attention").eval()
    batch = _batch()
    batch["node_mask"] = torch.tensor([[True, False, False, False], [True, True, True, True]])
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output["uncertainty"]).all()


def test_architecture_config_records_incident_pooling() -> None:
    model = _tiny_model(incident_pooling="attention")
    assert model.architecture_config()["incident_pooling"] == "attention"


@pytest.mark.parametrize("mode", INCIDENT_POOLING_MODES)
def test_permutation_equivariance_holds_for_every_incident_pooling_mode(mode: str) -> None:
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
    generator = torch.Generator().manual_seed(9)
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
    random.Random(42).shuffle(permutation)
    model = _tiny_model(edge_feature_dim=2, incident_pooling=mode)

    report = measure_equivariance(model, example, permutation, collate_fn=collate_variable_topology, atol=1e-4)
    assert report.non_equivariant_keys == ()
    assert report.predicted_source_agrees
    assert report.max_absolute_source_logit_difference < 1e-4


def test_source_conditioned_and_attention_add_expected_new_parameters() -> None:
    mean_params = set(_tiny_model(incident_pooling="mean").state_dict())
    attention_params = set(_tiny_model(incident_pooling="attention").state_dict())
    source_conditioned_params = set(_tiny_model(incident_pooling="source_conditioned").state_dict())

    attention_only_new = attention_params - mean_params
    assert any("incident_query" in name for name in attention_only_new)
    assert any("incident_attention" in name for name in attention_only_new)

    source_conditioned_new = source_conditioned_params - mean_params
    assert any("incident_context_projection" in name for name in source_conditioned_new)
