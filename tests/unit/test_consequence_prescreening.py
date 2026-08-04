"""Task 4.6: optional, non-authoritative plan consequence-prescreening
proxies -- exposure, pressure-risk, service-loss, containment-time, and
plan-regret. WNTR/PlanVerifier remains the sole authoritative source of
plan_validity/consequence_vector; these proxies exist only to rank
candidate plans before deciding which are worth exactly simulating.
"""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model import ArchitectureCompatibilityError, HydroCore, verify_architecture_compatibility
from hydroswarm.model.core import CONSEQUENCE_PROXY_NAMES
from hydroswarm.training import compute_multitask_loss
from hydroswarm.training.targets_v2 import TARGETS_V2


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


def test_all_five_proxies_are_registered_as_governed_strategist_targets() -> None:
    for name in CONSEQUENCE_PROXY_NAMES:
        spec = TARGETS_V2[name]
        assert spec.category == "strategist"
        assert spec.maskable
        assert "non-authoritative" in spec.definition or "proxy" in spec.unit


def test_default_consequence_prescreening_heads_is_disabled() -> None:
    model = _tiny_model()
    assert model.consequence_prescreening_heads is False
    assert not hasattr(model, "consequence_proxy_heads")


def test_disabled_consequence_prescreening_heads_omits_new_output_keys() -> None:
    model = _tiny_model().eval()
    with torch.no_grad():
        output = model(_batch())
    for name in CONSEQUENCE_PROXY_NAMES:
        assert name not in output


def test_enabling_adds_exactly_the_five_expected_proxy_heads() -> None:
    disabled_params = set(_tiny_model(consequence_prescreening_heads=False).state_dict())
    enabled_params = set(_tiny_model(consequence_prescreening_heads=True).state_dict())
    assert disabled_params <= enabled_params
    new_params = enabled_params - disabled_params
    for name in CONSEQUENCE_PROXY_NAMES:
        assert any(f"consequence_proxy_heads.{name}" in param_name for param_name in new_params)
    assert all(param_name.startswith("consequence_proxy_heads.") for param_name in new_params)


def test_forward_pass_produces_one_proxy_value_per_plan_query_when_enabled() -> None:
    plan_queries = 3
    model = _tiny_model(consequence_prescreening_heads=True, plan_queries=plan_queries).eval()
    with torch.no_grad():
        output = model(_batch())
    for name in CONSEQUENCE_PROXY_NAMES:
        assert output[name].shape == (2, plan_queries)
        assert torch.isfinite(output[name]).all()


def test_enabling_does_not_change_other_outputs_given_identical_weights() -> None:
    torch.manual_seed(18)
    disabled = _tiny_model(consequence_prescreening_heads=False).eval()
    enabled = _tiny_model(consequence_prescreening_heads=True).eval()
    shared = set(disabled.state_dict()) & set(enabled.state_dict())
    enabled.load_state_dict(
        {key: value for key, value in disabled.state_dict().items() if key in shared}, strict=False
    )
    batch = _batch()
    with torch.no_grad():
        disabled_output = disabled(batch)
        enabled_output = enabled(batch)
    torch.testing.assert_close(disabled_output["hidden_state"], enabled_output["hidden_state"])
    torch.testing.assert_close(disabled_output["plan_value"], enabled_output["plan_value"])
    torch.testing.assert_close(disabled_output["plan_validity_logits"], enabled_output["plan_validity_logits"])


def test_architecture_config_records_consequence_prescreening_heads() -> None:
    assert _tiny_model(consequence_prescreening_heads=False).architecture_config()[
        "consequence_prescreening_heads"
    ] is False
    assert _tiny_model(consequence_prescreening_heads=True).architecture_config()[
        "consequence_prescreening_heads"
    ] is True


def test_verify_architecture_compatibility_rejects_mismatch() -> None:
    model = _tiny_model(consequence_prescreening_heads=False)
    with pytest.raises(ArchitectureCompatibilityError, match="consequence_prescreening_heads"):
        verify_architecture_compatibility(model, {"consequence_prescreening_heads": True})


def test_verify_architecture_compatibility_accepts_missing_field_for_old_checkpoints() -> None:
    model = _tiny_model()
    verify_architecture_compatibility(model, {})  # must not raise


def test_losses_are_computed_only_for_proxies_with_both_output_and_target() -> None:
    plan_queries = 2
    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "exposure_proxy": torch.zeros(1, plan_queries, requires_grad=True),
        "plan_regret_proxy": torch.zeros(1, plan_queries, requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        "exposure_proxy": torch.ones(1, plan_queries),
        # plan_regret_proxy target intentionally absent: no plan candidates
        # generated for this incident, per the target's masking_rule.
    }
    result = compute_multitask_loss(outputs, targets)
    assert "exposure_proxy" in result.tasks
    assert "plan_regret_proxy" not in result.tasks
    assert torch.isfinite(result.total)
    result.total.backward()


def test_proxy_loss_ignores_masked_nan_plan_slots() -> None:
    outputs = {
        "source_node_logits": torch.tensor([[3.0, 0.0]], requires_grad=True),
        "containment_time_proxy": torch.tensor([[5.0, 5.0]], requires_grad=True),
    }
    targets = {
        "source_node": torch.tensor([0]),
        # Second plan slot has no exact WNTR consequence to supervise from
        # (e.g. it was never simulated) -- masked with NaN, per the
        # target's documented masking_rule.
        "containment_time_proxy": torch.tensor([[5.0, float("nan")]]),
    }
    result = compute_multitask_loss(outputs, targets)
    torch.testing.assert_close(result.tasks["containment_time_proxy"], torch.tensor(0.0), atol=1e-6, rtol=0)
