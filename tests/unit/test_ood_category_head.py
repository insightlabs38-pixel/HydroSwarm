"""core-issues3.txt Phase 6: the governed ood_class target (11 categories,
hydroswarm.training.ood_categories.OODCategory) was mapped in
compute_multitask_loss to the pre-existing 3-logit `ood_head` -- a
different, deterministic-severity-adjacent concept, not the governed
category taxonomy. Training this task with a real label index >= 3
(SEVERE_MISSINGNESS=7, FROZEN_DRIFTING_SENSOR=8) would raise. Adds a
correctly-sized, separately-gated `ood_category_head` and fixes the loss
mapping to use it.
"""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model import ArchitectureCompatibilityError, HydroCore, verify_architecture_compatibility
from hydroswarm.model.core import OOD_CATEGORY_COUNT
from hydroswarm.training import compute_multitask_loss
from hydroswarm.training.ood_categories import OODCategory


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
    generator = torch.Generator().manual_seed(31)
    return {
        "node_features": torch.randn(2, nodes, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, nodes, dtype=torch.bool),
    }


def test_ood_category_count_matches_the_governed_taxonomy() -> None:
    assert OOD_CATEGORY_COUNT == len(OODCategory)


def test_default_ood_category_head_is_disabled() -> None:
    model = _tiny_model()
    assert model.ood_category_head_enabled is False
    assert not hasattr(model, "ood_category_head")


def test_disabled_ood_category_head_omits_new_output_key() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(_batch())
    assert "ood_category_logits" not in output
    assert "ood_logits" in output  # the old 3-logit head is untouched either way


def test_enabling_adds_exactly_the_ood_category_head_parameters() -> None:
    disabled_params = set(_tiny_model(ood_category_head=False).state_dict())
    enabled_params = set(_tiny_model(ood_category_head=True).state_dict())
    assert disabled_params <= enabled_params
    new_params = enabled_params - disabled_params
    assert new_params and all(name.startswith("ood_category_head.") for name in new_params)


def test_forward_pass_produces_the_full_11_class_output_when_enabled() -> None:
    model = _tiny_model(ood_category_head=True).eval()
    with torch.no_grad():
        output = model(_batch())
    assert output["ood_category_logits"].shape == (2, OOD_CATEGORY_COUNT)
    assert torch.isfinite(output["ood_category_logits"]).all()


def test_enabling_does_not_change_other_outputs_given_identical_weights() -> None:
    torch.manual_seed(18)
    disabled = _tiny_model(ood_category_head=False).eval()
    enabled = _tiny_model(ood_category_head=True).eval()
    shared = set(disabled.state_dict()) & set(enabled.state_dict())
    enabled.load_state_dict(
        {key: value for key, value in disabled.state_dict().items() if key in shared}, strict=False
    )
    batch = _batch()
    with torch.no_grad():
        disabled_output = disabled(batch)
        enabled_output = enabled(batch)
    torch.testing.assert_close(disabled_output["hidden_state"], enabled_output["hidden_state"])
    torch.testing.assert_close(disabled_output["ood_logits"], enabled_output["ood_logits"])


def test_architecture_config_records_ood_category_head() -> None:
    assert _tiny_model(ood_category_head=False).architecture_config()["ood_category_head"] is False
    assert _tiny_model(ood_category_head=True).architecture_config()["ood_category_head"] is True


def test_verify_architecture_compatibility_rejects_mismatch() -> None:
    model = _tiny_model(ood_category_head=False)
    with pytest.raises(ArchitectureCompatibilityError, match="ood_category_head"):
        verify_architecture_compatibility(model, {"ood_category_head": True})


def test_verify_architecture_compatibility_accepts_missing_field_for_old_checkpoints() -> None:
    model = _tiny_model()
    verify_architecture_compatibility(model, {})  # must not raise


def test_loss_no_longer_maps_ood_class_to_the_old_3_logit_head() -> None:
    """The actual bug: training ood_class with a real out-of-range category
    index against the old 3-logit ood_head would raise. Confirms the loss
    now silently skips ood_class when only the old head is present (no
    ood_category_logits in outputs), rather than crashing against it."""

    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "ood_logits": torch.zeros(1, 3, requires_grad=True),  # old head, still present
    }
    targets = {
        "source_node": torch.tensor([0]),
        "ood_class": torch.tensor([7]),  # SEVERE_MISSINGNESS -- out of range for a 3-logit head
    }
    result = compute_multitask_loss(outputs, targets)
    assert "ood_class" not in result.tasks
    assert torch.isfinite(result.total)


def test_loss_trains_ood_class_against_the_new_11_class_head_without_crashing() -> None:
    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "ood_category_logits": torch.zeros(1, OOD_CATEGORY_COUNT, requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "ood_class": torch.tensor([7]),  # SEVERE_MISSINGNESS
    }
    result = compute_multitask_loss(outputs, targets)
    assert "ood_class" in result.tasks
    assert torch.isfinite(result.tasks["ood_class"])
    result.total.backward()
