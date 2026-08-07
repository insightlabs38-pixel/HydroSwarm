"""core-issues3.txt Phase 11.1: explicit task weights, per-task valid-target
counts, and weighted-contribution diagnostics on MultiTaskLoss, plus the
ALL_TASK_NAMES/validate_task_weights_complete completeness machinery that
lets a real training entry point fail closed on a config with a hidden
default instead of silently falling back to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hydroswarm.training import TrainingConfig
from hydroswarm.training.losses import (
    ALL_TASK_NAMES,
    IncompleteTaskWeightsError,
    compute_multitask_loss,
    validate_task_weights_complete,
)

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_YAML = _ROOT / "configs" / "training.yaml"


def _complete_batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """One synthetic (outputs, targets) pair exercising every task in
    ALL_TASK_NAMES -- shapes are deliberately simplified (not real
    per-node/per-plan widths), since this test only exercises
    compute_multitask_loss's own bookkeeping, not real model geometry
    (already covered by tests/integration/test_full_output_gradient_smoke.py
    and friends)."""

    generator = torch.Generator().manual_seed(7)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, requires_grad=True)

    outputs = {
        "source_node_logits": randn(2, 4),
        "source_region_logits": randn(2, 3),
        "sample_node_logits": randn(2, 4),
        "action_logits": randn(2, 9),
        "action_pointer_logits": randn(2, 4),
        "plan_validity_logits": randn(2, 2),
        "ood_category_logits": randn(2, 11),
        "event_cause_logits": randn(2, 5),
        "next_step_logits": randn(2, 4),
        "start_time_logits": randn(2, 4),
        "duration_logits": randn(2, 3),
        "relative_strength_logits": randn(2, 3),
        "plan_value": randn(2),
        "expected_information_gain": randn(2),
        "candidate_reduction_prediction": randn(2),
        "sensor_reconstruction_prediction": randn(2),
        "future_concentration_prediction": randn(2),
        "travel_time_prediction": randn(2),
        "exposure_proxy": randn(2),
        "pressure_risk_proxy": randn(2),
        "service_loss_proxy": randn(2),
        "containment_time_proxy": randn(2),
        "plan_regret_proxy": randn(2),
        "sensor_fault_logits": randn(2),
        "event_presence_logits": randn(2),
        "evidence_sufficiency": torch.sigmoid(randn(2)),
        "should_continue_sampling_logits": randn(2),
    }
    targets = {
        "source_node": torch.tensor([0, 1]),
        "source_region": torch.tensor([0, 2]),
        "sample_node": torch.tensor([1, 3]),
        "action_template": torch.tensor([2, 5]),
        "target_pointer": torch.tensor([0, 3]),
        "plan_validity": torch.tensor([0, 1]),
        "ood_class": torch.tensor([0, 7]),
        "event_cause": torch.tensor([0, 1]),
        "next_step": torch.tensor([0, 2]),
        "start_time": torch.tensor([0, 3]),
        "duration": torch.tensor([1, 2]),
        "relative_strength": torch.tensor([0, 1]),
        "plan_value": torch.tensor([0.5, 0.2]),
        "information_gain": torch.tensor([0.1, 0.3]),
        "candidate_reduction": torch.tensor([0.2, 0.4]),
        "sensor_reconstruction": torch.tensor([0.0, 1.0]),
        "future_concentration": torch.tensor([0.0, 0.0]),
        "travel_time": torch.tensor([1.0, 2.0]),
        "exposure_proxy": torch.tensor([0.1, 0.2]),
        "pressure_risk_proxy": torch.tensor([0.1, 0.2]),
        "service_loss_proxy": torch.tensor([0.1, 0.2]),
        "containment_time_proxy": torch.tensor([0.1, 0.2]),
        "plan_regret_proxy": torch.tensor([0.1, 0.2]),
        "sensor_fault": torch.tensor([0.0, 1.0]),
        "event_presence": torch.tensor([1.0, 0.0]),
        "evidence_sufficiency": torch.tensor([1.0, 0.0]),
        "should_continue_sampling": torch.tensor([1.0, 0.0]),
    }
    return outputs, targets


def test_all_task_names_matches_every_task_a_full_batch_actually_produces() -> None:
    outputs, targets = _complete_batch()
    result = compute_multitask_loss(outputs, targets)
    assert set(result.tasks) == ALL_TASK_NAMES


def test_valid_counts_weights_and_weighted_are_populated_for_every_task() -> None:
    outputs, targets = _complete_batch()
    weights = {name: float(index + 1) for index, name in enumerate(sorted(ALL_TASK_NAMES))}
    result = compute_multitask_loss(outputs, targets, task_weights=weights)
    assert set(result.valid_counts) == ALL_TASK_NAMES
    assert set(result.weights) == ALL_TASK_NAMES
    assert set(result.weighted) == ALL_TASK_NAMES
    for name in ALL_TASK_NAMES:
        assert result.valid_counts[name] >= 1, name
        assert result.weights[name] == weights[name]
        torch.testing.assert_close(
            result.weighted[name], result.tasks[name] * weights[name], atol=1e-6, rtol=1e-5
        )
    total = torch.stack(list(result.weighted.values())).sum()
    torch.testing.assert_close(result.total, total, atol=1e-6, rtol=1e-5)


def test_unweighted_tasks_fall_back_to_the_documented_defaults() -> None:
    outputs, targets = _complete_batch()
    result = compute_multitask_loss(outputs, targets)
    # AUXILIARY_TASKS default to 0.1; everything else defaults to 1.0.
    assert result.weights["sensor_reconstruction"] == pytest.approx(0.1)
    assert result.weights["travel_time"] == pytest.approx(0.1)
    assert result.weights["future_concentration"] == pytest.approx(0.1)
    assert result.weights["source_node"] == pytest.approx(1.0)
    assert result.weights["plan_value"] == pytest.approx(1.0)


def test_masked_out_task_reports_zero_valid_count() -> None:
    outputs, targets = _complete_batch()
    targets["source_node"] = torch.tensor([-100, -100])
    result = compute_multitask_loss(outputs, targets)
    assert result.valid_counts["source_node"] == 0
    torch.testing.assert_close(result.tasks["source_node"], torch.tensor(0.0), atol=1e-6, rtol=0)


def test_validate_task_weights_complete_accepts_a_complete_mapping() -> None:
    validate_task_weights_complete({name: 1.0 for name in ALL_TASK_NAMES})  # must not raise


def test_validate_task_weights_complete_rejects_a_missing_task() -> None:
    incomplete = {name: 1.0 for name in ALL_TASK_NAMES if name != "source_region"}
    with pytest.raises(IncompleteTaskWeightsError, match="source_region"):
        validate_task_weights_complete(incomplete)


def test_configs_training_yaml_declares_every_retained_task_weight_explicitly() -> None:
    config = TrainingConfig.from_yaml(_TRAINING_YAML, require_complete_task_weights=True)
    assert ALL_TASK_NAMES <= set(config.task_weights)


def test_from_yaml_raises_on_an_incomplete_config_when_required(tmp_path: Path) -> None:
    incomplete_yaml = tmp_path / "incomplete.yaml"
    incomplete_yaml.write_text(
        "training:\n  task_weights:\n    source_node: 1.0\n",
        encoding="utf-8",
    )
    TrainingConfig.from_yaml(incomplete_yaml)  # default: not required, must not raise
    with pytest.raises(IncompleteTaskWeightsError):
        TrainingConfig.from_yaml(incomplete_yaml, require_complete_task_weights=True)
