"""M10.2 Scout refit amendment, Part 2: adversarial tests for
`hydroswarm.training.gradient_coverage`.
"""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.training.gradient_coverage import (
    GradientCoverageError,
    compute_gradient_coverage,
    require_gradient_coverage,
)


def _tiny_model(**overrides) -> HydroCore:
    base = dict(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, modality_layers=1,
        adapter_dims=(32, 32, 32), dropout=0.0, scout_control_heads=True,
    )
    base.update(overrides)
    return HydroCore(**base)


def _batch(nodes: int = 4, batch: int = 2) -> dict:
    generator = torch.Generator().manual_seed(31)
    return {
        "node_features": torch.randn(batch, nodes, 3, generator=generator),
        "temporal_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(batch, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(batch, nodes, dtype=torch.bool),
    }


def _scout_parameter_groups(model: HydroCore) -> dict[str, list[str]]:
    return {
        "sample_node": [name for name, _ in model.sample_node_head.named_parameters(prefix="sample_node_head")],
        "information_gain": [name for name, _ in model.information_gain_head.named_parameters(prefix="information_gain_head")],
        "candidate_reduction": [name for name, _ in model.candidate_reduction_head.named_parameters(prefix="candidate_reduction_head")],
        "should_continue_sampling": [
            name for name, _ in model.should_continue_sampling_head.named_parameters(prefix="should_continue_sampling_head")
        ],
    }


def _real_targets(nodes: int = 4, batch: int = 2) -> dict:
    return {
        "sample_node": torch.tensor([0, 1]),
        "sample_node_mask": torch.tensor([True, True]),
        "information_gain": torch.rand(batch, nodes),
        "information_gain_mask": torch.ones(batch, nodes, dtype=torch.bool),
        "candidate_reduction": torch.rand(batch, nodes),
        "candidate_reduction_mask": torch.ones(batch, nodes, dtype=torch.bool),
        "should_continue_sampling": torch.tensor([1.0, 0.0]),
    }


def test_all_four_scout_tasks_pass_with_real_targets_and_correct_parameter_groups() -> None:
    model = _tiny_model()
    groups = _scout_parameter_groups(model)
    weights = {task: 1.0 for task in groups}
    targets = _real_targets()
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights=weights, parameter_groups=groups
    )
    require_gradient_coverage(certs)  # must not raise
    for task, cert in certs.items():
        assert cert.passed is True
        assert cert.gradient_observed is True
        assert cert.parameter_changed is True
        assert cert.gradient_norm_finite is True


def test_task_weight_with_no_target_fails_closed() -> None:
    """The exact defect this module exists to catch: a nonzero task_weight
    with no real target must never be reported as passing."""

    model = _tiny_model()
    groups = _scout_parameter_groups(model)
    weights = {task: 1.0 for task in groups}
    targets_missing_sample_node = _real_targets()
    del targets_missing_sample_node["sample_node"]
    del targets_missing_sample_node["sample_node_mask"]
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets_missing_sample_node, task_weights=weights, parameter_groups=groups
    )
    assert certs["sample_node"].passed is False
    assert certs["sample_node"].target_present is False
    assert "no target present" in certs["sample_node"].failure_reasons[0]
    with pytest.raises(GradientCoverageError):
        require_gradient_coverage(certs)


def test_output_absent_cannot_be_falsely_reported_trained() -> None:
    """A task whose output the model never produces (scout_control_heads
    disabled) must fail closed, not silently pass with output_present=False."""

    model = _tiny_model(scout_control_heads=False)
    weights = {"candidate_reduction": 1.0}
    targets = {"candidate_reduction": torch.rand(2, 4), "candidate_reduction_mask": torch.ones(2, 4, dtype=torch.bool)}
    with pytest.raises(ValueError, match="no compatible model outputs"):
        compute_gradient_coverage(
            model, lambda m: m(_batch()), targets, task_weights=weights, parameter_groups={"candidate_reduction": []}
        )


def test_all_masked_target_cannot_be_falsely_reported_as_real_supervision() -> None:
    model = _tiny_model()
    groups = _scout_parameter_groups(model)
    weights = {"candidate_reduction": 1.0}
    targets = {
        "candidate_reduction": torch.rand(2, 4),
        "candidate_reduction_mask": torch.zeros(2, 4, dtype=torch.bool),  # every position masked out
    }
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights=weights,
        parameter_groups={"candidate_reduction": groups["candidate_reduction"]},
    )
    cert = certs["candidate_reduction"]
    assert cert.valid_target_count == 0
    assert cert.passed is False
    assert any("valid_target_count" in reason for reason in cert.failure_reasons)


def test_disconnected_head_cannot_be_falsely_reported_as_trained() -> None:
    """Declares a parameter group belonging to a DIFFERENT head than the one
    the task's loss actually flows through -- must fail closed (no
    gradient), not pass just because task/target/output all look fine."""

    model = _tiny_model()
    groups = _scout_parameter_groups(model)
    weights = {"sample_node": 1.0}
    targets = {"sample_node": torch.tensor([0, 1]), "sample_node_mask": torch.tensor([True, True])}
    wrong_group = {"sample_node": groups["information_gain"]}  # disconnected from sample_node's own loss term
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights=weights, parameter_groups=wrong_group,
        verify_parameter_update=False,
    )
    cert = certs["sample_node"]
    assert cert.gradient_observed is False
    assert cert.passed is False


def test_task_weight_zero_is_recorded_disabled_and_never_fails() -> None:
    model = _tiny_model()
    groups = _scout_parameter_groups(model)
    weights = {"sample_node": 1.0, "should_continue_sampling": 0.0}
    targets = {"sample_node": torch.tensor([0, 1]), "sample_node_mask": torch.tensor([True, True])}
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights=weights,
        parameter_groups={"sample_node": groups["sample_node"]},
    )
    assert certs["should_continue_sampling"].task_enabled is False
    assert certs["should_continue_sampling"].passed is True
    require_gradient_coverage(certs)  # must not raise despite should_continue_sampling having no target


def test_unknown_parameter_name_in_group_is_reported_as_missing() -> None:
    model = _tiny_model()
    weights = {"sample_node": 1.0}
    targets = {"sample_node": torch.tensor([0, 1]), "sample_node_mask": torch.tensor([True, True])}
    certs = compute_gradient_coverage(
        model, lambda m: m(_batch()), targets, task_weights=weights,
        parameter_groups={"sample_node": ["not_a_real_parameter.weight"]},
        verify_parameter_update=False,
    )
    assert certs["sample_node"].passed is False
    assert any("unknown parameter" in reason for reason in certs["sample_node"].failure_reasons)


def test_parameter_update_verification_uses_a_deep_copy_and_does_not_mutate_the_caller_model() -> None:
    model = _tiny_model()
    groups = _scout_parameter_groups(model)
    before = {name: value.clone() for name, value in model.named_parameters()}
    weights = {task: 1.0 for task in groups}
    targets = _real_targets()
    compute_gradient_coverage(model, lambda m: m(_batch()), targets, task_weights=weights, parameter_groups=groups)
    for name, value in model.named_parameters():
        torch.testing.assert_close(before[name], value)
