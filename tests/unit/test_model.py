from __future__ import annotations

import pytest
import torch

from hydroswarm.model import (
    ArchitectureCompatibilityError,
    HydroCore,
    HydroMono,
    NoAdapterHydroCore,
    verify_architecture_compatibility,
)


def _tiny_model() -> HydroCore:
    return HydroCore(
        node_feature_dim=3,
        temporal_feature_dim=2,
        quality_feature_dim=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        num_layers=1,
        modality_layers=1,
        adapter_dims=(32, 32, 32),
    )


def test_forward_handles_missing_sensors_and_node_masks() -> None:
    model = _tiny_model().eval()
    temporal = torch.randn(2, 3, 4, 2)
    temporal[0, 1, 1] = torch.nan
    quality = torch.randn(2, 3, 4, 2)
    quality[1, :, 2] = torch.nan
    node_mask = torch.tensor([[True, True, False, True], [True, True, True, False]])
    sensor_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    sensor_mask[0, :, 3] = False
    source_mask = torch.tensor([[True, False, False, True], [True, True, False, False]])

    with torch.no_grad():
        output = model(
            {
                "node_features": torch.randn(2, 4, 3),
                "temporal_features": temporal,
                "quality_features": quality,
                "travel_time": torch.tensor([[0.0, 5.0, 10.0, 20.0], [0.0, 1.0, 2.0, 3.0]]),
                "reservoir_reachability": torch.tensor([[1.0, 1.0, 0.8, 0.5], [1.0, 0.5, 0.0, 0.0]]),
                "demand_centrality": torch.rand(2, 4),
                "node_mask": node_mask,
                "sensor_mask": sensor_mask,
                "source_candidate_mask": source_mask,
                "classical_prior": torch.tensor(
                    [[0.8, 0.1, 0.0, 0.1], [0.2, 0.8, 0.0, 0.0]]
                ),
            }
        )

    assert output["hidden_state"].shape == (2, 4, 32)
    assert output["sentinel"].shape == (2, 4, 2)
    assert output["scout"].shape == (2, 4, 2)
    assert output["strategist"].shape == (2, 4, 3)
    assert torch.isfinite(output["hidden_state"]).all()
    assert torch.count_nonzero(output["hidden_state"][~node_mask]) == 0
    assert torch.count_nonzero(output["strategist"][~node_mask]) == 0
    assert torch.all(output["source_node_logits"][~source_mask] < -1e20)
    floor = torch.finfo(torch.float32).min
    assert torch.all(output["start_time_logits"][..., 4:] == floor)
    assert torch.all(output["duration_logits"][..., 3:] == floor)
    assert torch.all(output["relative_strength_logits"][..., 3:] == floor)


def test_all_missing_observations_are_safe() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(
            {
                "node_features": torch.zeros(1, 2, 3),
                "temporal_features": torch.full((1, 2, 2, 2), torch.nan),
                "quality_features": torch.full((1, 2, 2, 2), torch.nan),
                "travel_time": torch.zeros(1, 2),
                "reservoir_reachability": torch.zeros(1, 2),
                "demand_centrality": torch.zeros(1, 2),
                "sensor_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
                "quality_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
            }
        )
    assert torch.isfinite(output["sentinel"]).all()


def test_default_model_is_approximately_24_5m_parameters() -> None:
    model = HydroCore()
    report = model.parameter_report()
    assert report.total == model.parameter_count()
    assert report.trainable == report.total
    assert abs(report.total - 24_500_000) / 24_500_000 < 0.05


def test_model_variants_meet_parameter_bands() -> None:
    counts = {
        name: HydroCore.from_variant(name).parameter_count()
        for name in ("small", "medium", "large")
    }
    assert 4_000_000 <= counts["small"] <= 6_000_000
    assert 12_000_000 <= counts["medium"] <= 15_000_000
    assert 24_000_000 <= counts["large"] <= 25_000_000


def test_from_variant_records_the_variant_name_in_architecture_config() -> None:
    # core-issues.txt repair item 9: a checkpoint's architecture_config()
    # must declare which named variant it was built from -- a runtime
    # loader needs this to reconstruct the model rather than assume a
    # hardcoded default.
    small = HydroCore.from_variant("small")
    assert small.architecture_config()["variant"] == "small"
    assert small.architecture_config()["use_adapters"] is True
    # A model constructed directly (not via from_variant) has no named
    # variant -- never fabricated as if it matched one of MODEL_VARIANTS.
    direct = HydroCore(d_model=32, nhead=4, dim_feedforward=64, num_layers=1)
    assert direct.architecture_config()["variant"] is None


def test_verify_architecture_compatibility_rejects_use_adapters_mismatch() -> None:
    kwargs = {
        "node_feature_dim": 3,
        "temporal_feature_dim": 2,
        "quality_feature_dim": 2,
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "num_layers": 1,
        "adapter_dims": (32, 32, 32),
    }
    model = HydroCore(**kwargs, use_adapters=True)
    with pytest.raises(ArchitectureCompatibilityError, match="use_adapters"):
        verify_architecture_compatibility(model, {"use_adapters": False})
    verify_architecture_compatibility(model, {})  # missing field: accepted for old checkpoints


def test_edge_inputs_context_and_semantic_heads() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(
            {
                "node_features": torch.randn(1, 3, 3),
                "temporal_features": torch.randn(1, 2, 3, 2),
                "quality_features": torch.randn(1, 2, 3, 2),
                "travel_time": torch.zeros(1, 3),
                "reservoir_reachability": torch.ones(1, 3),
                "demand_centrality": torch.ones(1, 3),
                "edge_index": torch.tensor([[[0, 1], [1, 2]]]),
                "edge_features": torch.randn(1, 2, 13),
                "edge_mask": torch.ones(1, 2, dtype=torch.bool),
                "timestamps": torch.tensor([[0.0, 300.0]]),
                "role_features": torch.zeros(1, 8),
                "previous_actions": torch.zeros(1, 2, 8),
                "verifier_feedback": torch.zeros(1, 8),
                "residual_features": torch.zeros(1, 3, 4),
                "classical_prior": torch.tensor([[0.7, 0.2, 0.1]]),
            }
        )
    assert output["latent_state"].shape == (1, 96, 32)
    assert output["source_node_logits"].shape == (1, 3)
    assert output["sample_node_logits"].shape == (1, 3)
    assert output["action_logits"].shape == (1, 8, 8)
    assert output["action_pointer_logits"].shape == (1, 8, 3)
    assert output["plan_value"].shape == (1, 8)
    assert output["plan_validity_logits"].shape == (1, 8, 2)
    assert output["uncertainty"].shape == (1, 1)
    assert output["ood_logits"].shape == (1, 3)
    assert torch.isfinite(output["action_logits"]).all()


def test_ablation_models_remove_role_adapters() -> None:
    kwargs = {
        "node_feature_dim": 3,
        "temporal_feature_dim": 2,
        "quality_feature_dim": 2,
        "d_model": 32,
        "nhead": 4,
        "dim_feedforward": 64,
        "num_layers": 1,
        "modality_layers": 1,
        "adapter_dims": (32, 32, 32),
    }
    full = HydroCore(**kwargs)
    no_adapter = NoAdapterHydroCore(**kwargs)
    mono = HydroMono(**kwargs)
    assert no_adapter.parameter_count() < full.parameter_count()
    assert mono.use_adapters is False


def _edge_batch(**overrides) -> dict:
    batch = {
        "node_features": torch.randn(1, 3, 3),
        "temporal_features": torch.randn(1, 2, 3, 2),
        "quality_features": torch.randn(1, 2, 3, 2),
        "travel_time": torch.zeros(1, 3),
        "reservoir_reachability": torch.ones(1, 3),
        "demand_centrality": torch.ones(1, 3),
        "edge_index": torch.tensor([[[0, 1], [1, 2]]]),
        "edge_features": torch.randn(1, 2, 13),
        "edge_mask": torch.ones(1, 2, dtype=torch.bool),
        "timestamps": torch.tensor([[0.0, 300.0]]),
    }
    batch.update(overrides)
    return batch


def test_strategist_mode_defaults_to_anonymous_queries_with_no_new_parameters() -> None:
    # core-issues3.txt Phase 4: the default must have zero new parameters
    # and no candidate_plan_encoder submodule at all -- this is what keeps
    # the promoted checkpoint's state_dict keys unchanged (see the reverted
    # action_vocabulary_size default-bump earlier this pass for what
    # happens when a shared default silently changes checkpoint-loading
    # behavior).
    model = _tiny_model()
    assert model.strategist_mode == "anonymous_queries"
    assert not hasattr(model, "candidate_plan_encoder")
    assert "candidate_plan_encoder" not in dict(model.named_children())


def test_candidate_conditioned_mode_produces_plan_hidden_from_real_candidates() -> None:
    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, modality_layers=1,
        adapter_dims=(32, 32, 32), strategist_mode="candidate_conditioned",
    ).eval()
    assert hasattr(model, "candidate_plan_encoder")
    plans = 4
    batch = _edge_batch(
        plan_template_ids=torch.tensor([[0, 1, 2, 3]]),
        plan_target_type=torch.tensor([[0, 1, 2, 1]]),  # NONE, NODE, LINK, NODE
        plan_target_node_index=torch.tensor([[-1, 1, -1, 2]]),
        plan_target_link_index=torch.tensor([[-1, -1, 0, -1]]),
        plan_features=torch.randn(1, plans, 6),
        plan_mask=torch.ones(1, plans, dtype=torch.bool),
    )
    with torch.no_grad():
        output = model(batch)
    assert output["action_logits"].shape == (1, plans, 8)
    assert output["plan_value"].shape == (1, plans)
    assert output["plan_validity_logits"].shape == (1, plans, 2)
    assert torch.isfinite(output["action_logits"]).all()
    assert torch.isfinite(output["plan_value"]).all()


def test_candidate_conditioned_mode_is_sensitive_to_which_candidates_are_passed() -> None:
    """The core point of Phase 4: action_logits for a given plan slot must
    depend on that plan's own identity, not just incident context (the old
    anonymous-query design's failure mode -- same incident, same query
    position, always the same representation regardless of which candidate
    a caller claims occupies it)."""

    torch.manual_seed(0)
    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, modality_layers=1,
        adapter_dims=(32, 32, 32), strategist_mode="candidate_conditioned",
    ).eval()
    base = _edge_batch(
        plan_template_ids=torch.tensor([[0, 1]]),
        plan_target_type=torch.tensor([[1, 1]]),
        plan_target_node_index=torch.tensor([[0, 1]]),
        plan_target_link_index=torch.tensor([[-1, -1]]),
        plan_features=torch.zeros(1, 2, 6),
        plan_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    swapped = dict(base, plan_template_ids=torch.tensor([[1, 0]]))
    with torch.no_grad():
        first = model(base)["action_logits"]
        second = model(swapped)["action_logits"]
    assert not torch.allclose(first[:, 0], second[:, 0])


def test_verify_architecture_compatibility_rejects_strategist_mode_mismatch() -> None:
    model = HydroCore(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, adapter_dims=(32, 32, 32),
    )
    with pytest.raises(ArchitectureCompatibilityError, match="strategist_mode"):
        verify_architecture_compatibility(model, {"strategist_mode": "candidate_conditioned"})
    verify_architecture_compatibility(model, {})  # missing field: accepted for old checkpoints


def test_strategist_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="strategist_mode"):
        HydroCore(
            node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
            d_model=32, nhead=4, dim_feedforward=64, num_layers=1,
            strategist_mode="not_a_real_mode",
        )
