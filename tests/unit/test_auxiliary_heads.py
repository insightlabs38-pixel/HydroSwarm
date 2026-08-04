"""Task 4.5: masked sensor reconstruction, future concentration
prediction, and travel-time prediction -- exactly three auxiliary
objectives, configuration-controlled and never authoritative.
"""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model import ArchitectureCompatibilityError, HydroCore, verify_architecture_compatibility
from hydroswarm.training import compute_multitask_loss
from hydroswarm.training.losses import AUXILIARY_TASK_DEFAULT_WEIGHT, AUXILIARY_TASKS
from hydroswarm.training.targets_v2 import TARGETS_BY_CATEGORY, TARGETS_V2


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
    generator = torch.Generator().manual_seed(29)
    return {
        "node_features": torch.randn(2, nodes, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, nodes, dtype=torch.bool),
    }


def test_governed_targets_v2_registers_exactly_these_three_auxiliary_targets() -> None:
    assert set(TARGETS_BY_CATEGORY["auxiliary"]) == AUXILIARY_TASKS
    for name in AUXILIARY_TASKS:
        assert TARGETS_V2[name].maskable, f"{name} must be maskable (Task 4.5: missing targets are masked)"


def test_default_auxiliary_heads_is_disabled() -> None:
    model = _tiny_model()
    assert model.auxiliary_heads is False
    assert not hasattr(model, "sensor_reconstruction_head")
    assert not hasattr(model, "future_concentration_head")
    assert not hasattr(model, "travel_time_head")


def test_disabled_auxiliary_heads_omits_new_output_keys() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(_batch())
    assert "sensor_reconstruction_prediction" not in output
    assert "future_concentration_prediction" not in output
    assert "travel_time_prediction" not in output


def test_enabling_auxiliary_heads_adds_only_the_three_expected_heads() -> None:
    disabled_params = set(_tiny_model(auxiliary_heads=False).state_dict())
    enabled_params = set(_tiny_model(auxiliary_heads=True).state_dict())
    assert disabled_params <= enabled_params
    new_params = enabled_params - disabled_params
    assert all(
        name.startswith(("sensor_reconstruction_head", "future_concentration_head", "travel_time_head"))
        for name in new_params
    )
    assert any(name.startswith("sensor_reconstruction_head") for name in new_params)
    assert any(name.startswith("future_concentration_head") for name in new_params)
    assert any(name.startswith("travel_time_head") for name in new_params)


def test_forward_pass_produces_correctly_shaped_masked_predictions_when_enabled() -> None:
    nodes = 4
    model = _tiny_model(auxiliary_heads=True).eval()
    batch = _batch(nodes)
    batch["node_mask"] = torch.tensor([[True, True, True, False], [True, True, False, False]])
    with torch.no_grad():
        output = model(batch)
    for key in (
        "sensor_reconstruction_prediction",
        "future_concentration_prediction",
        "travel_time_prediction",
    ):
        assert output[key].shape == (2, nodes)
        assert torch.isfinite(output[key]).all()
        # Padded/masked-out node positions are zeroed, not left as
        # arbitrary uninformative-but-nonzero garbage a downstream
        # consumer could mistake for a real prediction.
        assert torch.equal(output[key][~batch["node_mask"]], torch.zeros_like(output[key][~batch["node_mask"]]))


def test_enabling_does_not_change_other_outputs_given_identical_weights() -> None:
    torch.manual_seed(14)
    disabled = _tiny_model(auxiliary_heads=False).eval()
    enabled = _tiny_model(auxiliary_heads=True).eval()
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


def test_architecture_config_records_auxiliary_heads() -> None:
    assert _tiny_model(auxiliary_heads=False).architecture_config()["auxiliary_heads"] is False
    assert _tiny_model(auxiliary_heads=True).architecture_config()["auxiliary_heads"] is True


def test_verify_architecture_compatibility_rejects_auxiliary_heads_mismatch() -> None:
    model = _tiny_model(auxiliary_heads=False)
    with pytest.raises(ArchitectureCompatibilityError, match="auxiliary_heads"):
        verify_architecture_compatibility(model, {"auxiliary_heads": True})


def test_verify_architecture_compatibility_accepts_missing_field_for_old_checkpoints() -> None:
    model = _tiny_model()
    verify_architecture_compatibility(model, {})  # must not raise


def test_missing_auxiliary_targets_are_masked_out_of_the_loss_not_treated_as_zero() -> None:
    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "sensor_reconstruction_prediction": torch.tensor([[1.5, float("nan")]], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "sensor_reconstruction": torch.tensor([[1.5, float("nan")]]),
    }
    result = compute_multitask_loss(outputs, targets)
    # Only the finite (unmasked) position contributes; a prediction that
    # exactly matches the one valid target gives ~zero reconstruction loss
    # rather than being penalized for the masked NaN position.
    torch.testing.assert_close(result.tasks["sensor_reconstruction"], torch.tensor(0.0), atol=1e-6, rtol=0)


def test_auxiliary_tasks_get_a_reduced_default_weight_relative_to_primary_tasks() -> None:
    # Two identical-magnitude losses: one primary (source_node), one
    # auxiliary (travel_time). Task 4.5 requires loss weights be explicit
    # rather than implicitly equal-weighted with primary objectives.
    outputs = {
        "source_node_logits": torch.tensor([[0.0, 0.0]], requires_grad=True),
        "travel_time_prediction": torch.tensor([[1.0]], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "travel_time": torch.tensor([[2.0]]),
    }
    result = compute_multitask_loss(outputs, targets)
    expected_total = result.tasks["source_node"] + AUXILIARY_TASK_DEFAULT_WEIGHT * result.tasks["travel_time"]
    torch.testing.assert_close(result.total, expected_total, atol=1e-6, rtol=1e-5)


def test_explicit_task_weights_override_the_auxiliary_default() -> None:
    outputs = {
        "source_node_logits": torch.tensor([[0.0, 0.0]], requires_grad=True),
        "travel_time_prediction": torch.tensor([[1.0]], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "travel_time": torch.tensor([[2.0]]),
    }
    result = compute_multitask_loss(outputs, targets, task_weights={"travel_time": 1.0})
    expected_total = result.tasks["source_node"] + 1.0 * result.tasks["travel_time"]
    torch.testing.assert_close(result.total, expected_total, atol=1e-6, rtol=1e-5)
