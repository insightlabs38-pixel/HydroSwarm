"""core-issues3.txt Phase 11.3: scale-safety preflight check."""

from __future__ import annotations

import math

import pytest
import torch

from hydroswarm.model import HydroCore
from hydroswarm.training.scale_safety import ScaleSafetyError, run_scale_safety_check


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


def test_passes_and_returns_a_report_for_a_healthy_batch() -> None:
    model = _tiny_model()
    nodes = 4
    targets = {
        "source_node": torch.tensor([0, 1]),
        "sensor_fault": torch.zeros(2, nodes),
    }
    report = run_scale_safety_check(
        model, _batch(nodes), targets, required_tasks=frozenset({"source_node", "sensor_fault"})
    )
    assert report.tasks_checked == ("sensor_fault", "source_node")
    assert report.valid_counts["source_node"] > 0
    assert report.valid_counts["sensor_fault"] > 0
    assert report.gradient_norms["source_node"] > 0.0
    assert report.gradient_norms["sensor_fault"] > 0.0
    assert math.isfinite(report.total_loss)


def test_raises_when_a_required_task_has_zero_valid_count() -> None:
    model = _tiny_model()
    nodes = 4
    targets = {
        "source_node": torch.tensor([0, 1]),
        # Every position invalid (negative) -- sensor_fault reaches
        # compute_multitask_loss (present in both outputs and targets) but
        # contributes nothing real.
        "sensor_fault": torch.full((2, nodes), -1.0),
    }
    with pytest.raises(ScaleSafetyError, match="sensor_fault"):
        run_scale_safety_check(
            model, _batch(nodes), targets, required_tasks=frozenset({"source_node", "sensor_fault"})
        )


def test_does_not_raise_on_zero_valid_count_for_a_task_that_is_not_required() -> None:
    # future_concentration-style case: a task that structurally never has
    # real coverage must not make the preflight impossible to ever pass,
    # as long as it is not declared required.
    model = _tiny_model()
    nodes = 4
    targets = {
        "source_node": torch.tensor([0, 1]),
        "sensor_fault": torch.full((2, nodes), -1.0),  # zero valid count, NOT required
    }
    report = run_scale_safety_check(model, _batch(nodes), targets, required_tasks=frozenset({"source_node"}))
    assert report.valid_counts["sensor_fault"] == 0
    assert report.gradient_norms["sensor_fault"] == 0.0
    assert report.valid_counts["source_node"] > 0
    assert report.gradient_norms["source_node"] > 0.0


def test_raises_when_a_required_task_never_reaches_the_loss_at_all() -> None:
    model = _tiny_model()  # scout_control_heads=False (default): no candidate_reduction_prediction output
    targets = {
        "source_node": torch.tensor([0, 1]),  # reaches the loss, so compute_multitask_loss itself succeeds
        "candidate_reduction": torch.zeros(2, 4),  # never reaches it -- no matching model output
    }
    with pytest.raises(ScaleSafetyError, match="candidate_reduction"):
        run_scale_safety_check(
            model, _batch(), targets, required_tasks=frozenset({"source_node", "candidate_reduction"})
        )


def test_masked_regression_zero_valid_count_task_receives_exactly_zero_gradient() -> None:
    # Direct proof of the "padded/masked positions contribute zero"
    # property for a regression-style (not classification-style) task.
    model = _tiny_model(consequence_prescreening_heads=True, plan_queries=2)
    targets = {
        "source_node": torch.tensor([0, 1]),
        "exposure_proxy": torch.full((2, 2), float("nan")),  # fully masked out (non-finite)
    }
    report = run_scale_safety_check(model, _batch(), targets, required_tasks=frozenset({"source_node"}))
    assert report.valid_counts["exposure_proxy"] == 0
    assert report.gradient_norms["exposure_proxy"] == 0.0
