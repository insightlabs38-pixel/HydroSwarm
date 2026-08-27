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

from hydroswarm.model import HydroCore
from hydroswarm.training import TrainingConfig
from hydroswarm.training.losses import (
    ALL_TASK_NAMES,
    PRIMARY_TASKS,
    IncompleteTaskWeightsError,
    compute_multitask_loss,
    task_gradient_conflict,
    validate_task_weights_complete,
)

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_YAML = _ROOT / "configs" / "training.yaml"
_TRAINING_V5_CAUSAL_YAML = _ROOT / "configs" / "training-v5-causal.yaml"


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


def test_masked_out_classification_task_with_extreme_logits_is_finite_not_nan() -> None:
    """Regression test (physics-informed-localizer-full-data-gate,
    full-data-9000 stage): CandidateConditionedLocalizer/default localizer
    heads write a large-magnitude sentinel (observed: -3.4028e38, float32's
    near-max-magnitude) into masked-out candidate positions of
    `source_node_logits` before softmax/cross-entropy ever sees them. A
    real scenario with source_node_mask all False (no real source in this
    example -- overwhelmingly common once training uses the FULL, unfiltered
    Cycle-B2 corpus rather than a has_real_source-filtered subsample) and
    two or more masked-out candidates hits `not valid.any()`'s zero-loss
    fallback. Summing two such sentinels overflows float32's representable
    range to -inf; -inf * 0.0 is NaN, poisoning `compute_multitask_loss`'s
    `total` and crashing `Trainer._train_epoch`'s `torch.isfinite(result.total)`
    fail-closed check. The fallback must zero every element BEFORE summing
    (never after) so it is always exactly 0.0, regardless of how extreme the
    unmasked logits are."""

    outputs, targets = _complete_batch()
    sentinel = torch.finfo(torch.float32).min  # -3.4028235e38, the observed real-world sentinel magnitude
    outputs["source_node_logits"] = torch.tensor(
        [[0.5, sentinel, sentinel, sentinel], [0.1, sentinel, sentinel, sentinel]], requires_grad=True
    )
    targets["source_node"] = torch.tensor([-100, -100])  # every example masked out -- count == 0
    result = compute_multitask_loss(outputs, targets)
    assert result.valid_counts["source_node"] == 0
    assert torch.isfinite(result.total), f"total loss must stay finite, got {result.total}"
    torch.testing.assert_close(result.tasks["source_node"], torch.tensor(0.0), atol=1e-6, rtol=0)
    result.total.backward()  # must not raise; confirms the zero stays graph-connected


def test_validate_task_weights_complete_accepts_a_complete_mapping() -> None:
    validate_task_weights_complete({name: 1.0 for name in ALL_TASK_NAMES})  # must not raise


def test_validate_task_weights_complete_rejects_a_missing_task() -> None:
    incomplete = {name: 1.0 for name in ALL_TASK_NAMES if name != "source_region"}
    with pytest.raises(IncompleteTaskWeightsError, match="source_region"):
        validate_task_weights_complete(incomplete)


def test_configs_training_yaml_declares_every_retained_task_weight_explicitly() -> None:
    config = TrainingConfig.from_yaml(_TRAINING_YAML, require_complete_task_weights=True)
    assert ALL_TASK_NAMES <= set(config.task_weights)


def test_configs_training_v5_causal_yaml_declares_every_retained_task_weight_explicitly() -> None:
    # Milestone 0.3 (experiments.txt): every promotion-quality v5 run must
    # load this config with require_complete_task_weights=True. This proves
    # the file itself satisfies that gate rather than relying on a future
    # runner to discover an omission at run time.
    config = TrainingConfig.from_yaml(_TRAINING_V5_CAUSAL_YAML, require_complete_task_weights=True)
    assert ALL_TASK_NAMES <= set(config.task_weights)
    assert config.pcgrad_enabled is False


def test_from_yaml_raises_on_an_incomplete_config_when_required(tmp_path: Path) -> None:
    incomplete_yaml = tmp_path / "incomplete.yaml"
    incomplete_yaml.write_text(
        "training:\n  task_weights:\n    source_node: 1.0\n",
        encoding="utf-8",
    )
    TrainingConfig.from_yaml(incomplete_yaml)  # default: not required, must not raise
    with pytest.raises(IncompleteTaskWeightsError):
        TrainingConfig.from_yaml(incomplete_yaml, require_complete_task_weights=True)


# --- Phase 11.1/11.4: gradient-conflict diagnostic -------------------------


def _two_task_batch(nodes: int = 4) -> dict:
    generator = torch.Generator().manual_seed(41)
    return {
        "node_features": torch.randn(2, nodes, 3, generator=generator),
        "temporal_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "quality_features": torch.randn(2, 3, nodes, 2, generator=generator),
        "source_candidate_mask": torch.ones(2, nodes, dtype=torch.bool),
    }


def test_task_gradient_conflict_reports_every_primary_vs_other_pair() -> None:
    # source_node and sensor_fault are both in PRIMARY_TASKS and share the
    # backbone, so this exercises real primary-vs-primary pairs (both
    # directions) through a real model, not a synthetic disconnected
    # tensor pair.
    model = _tiny_model()
    model.train()
    nodes = 4
    output = model(_two_task_batch(nodes))
    targets = {
        "source_node": torch.tensor([0, 1]),
        "sensor_fault": torch.zeros(2, nodes),
    }
    result = compute_multitask_loss(output, targets)
    assert set(result.tasks) == {"source_node", "sensor_fault"}
    conflict = task_gradient_conflict(result.tasks, model)
    assert set(conflict) == {"source_node|sensor_fault", "sensor_fault|source_node"}
    for value in conflict.values():
        assert -1.0 - 1e-6 <= value <= 1.0 + 1e-6


def test_task_gradient_conflict_is_empty_when_no_primary_task_is_present() -> None:
    model = _tiny_model()
    model.train()
    output = model(_two_task_batch())
    targets = {"sensor_fault": torch.zeros(2, 4)}
    result = compute_multitask_loss(output, targets)
    assert task_gradient_conflict(result.tasks, model, primary_tasks=PRIMARY_TASKS) == {}


def test_task_gradient_conflict_omits_a_pair_with_only_one_present_task() -> None:
    model = _tiny_model()
    model.train()
    output = model(_two_task_batch())
    targets = {"source_node": torch.tensor([0, 1])}
    result = compute_multitask_loss(output, targets)
    assert set(result.tasks) == {"source_node"}
    assert task_gradient_conflict(result.tasks, model) == {}


class _DisjointHeadModule(torch.nn.Module):
    """Three independent, unequal-sized parameter groups: `a` (3,) and `b`
    (2,) are each used by exactly one of two tasks below; `shared` (1,) is
    used by both. Registration order (a, b, shared) matters: it fixes the
    per-parameter coordinate system `model.parameters()` iterates in."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.b = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
        self.shared = torch.nn.Parameter(torch.tensor([6.0]))


def test_task_gradient_conflict_aligns_disjoint_parameter_subsets_by_position() -> None:
    # Milestone 0.4 regression: "primary" only touches `a` and `shared`;
    # "other" only touches `b` and `shared` -- disjoint, unequal-sized
    # per-task parameter subsets (3+1=4 vs 2+1=3 parameters if None
    # gradients were dropped instead of zero-filled by shape). Before the
    # fix, `flattened["primary"]` and `flattened["other"]` had different
    # lengths and `torch.dot` raised a shape-mismatch RuntimeError here --
    # this test would not even run under the old implementation, let alone
    # pass.
    module = _DisjointHeadModule()
    loss_primary = (module.a**2).sum() + 2.0 * module.shared.sum()
    loss_other = (module.b**3).sum() + 3.0 * module.shared.sum()

    conflict = task_gradient_conflict(
        {"primary": loss_primary, "other": loss_other},
        module,
        primary_tasks=frozenset({"primary"}),
    )

    parameters = tuple(module.parameters())
    grads_primary = torch.autograd.grad(loss_primary, parameters, retain_graph=True, allow_unused=True)
    grads_other = torch.autograd.grad(loss_other, parameters, retain_graph=True, allow_unused=True)
    expected_primary = torch.cat(
        [
            g.detach().float().reshape(-1) if g is not None else torch.zeros(p.numel())
            for g, p in zip(grads_primary, parameters)
        ]
    )
    expected_other = torch.cat(
        [
            g.detach().float().reshape(-1) if g is not None else torch.zeros(p.numel())
            for g, p in zip(grads_other, parameters)
        ]
    )
    # Both vectors span every trainable parameter (3 + 2 + 1 = 6), proving
    # `a`'s and `b`'s zero segments were kept rather than dropped -- a
    # common per-parameter coordinate system, not two independently
    # shrunk/reordered vectors.
    assert expected_primary.shape == expected_other.shape == (6,)
    expected_cosine = float(
        torch.dot(expected_primary, expected_other) / (expected_primary.norm() * expected_other.norm())
    )
    assert conflict["primary|other"] == pytest.approx(expected_cosine)
    # `a` and `b` never share a gradient component, so only the shared
    # parameter's [12.0] . [18.0] contributes to the numerator: the cosine
    # is strictly between 0 and 1, not the 0.0 a fully-orthogonal
    # (shape-mismatched-then-truncated) comparison would silently produce.
    assert 0.0 < conflict["primary|other"] < 1.0


def test_task_gradient_conflict_zero_fills_unused_parameters_not_drops_them() -> None:
    # Same disjoint setup, but isolates the zero-fill claim directly: the
    # "other" task's contribution to the flattened vector at `a`'s position
    # must be an explicit zero of `a`'s shape (matching Milestone 0.4's
    # "parameter unused by that task -> zero vector with same parameter
    # shape"), not simply absent from the vector.
    module = _DisjointHeadModule()
    loss_primary = (module.a**2).sum()
    loss_other = (module.b**3).sum()

    conflict = task_gradient_conflict(
        {"primary": loss_primary, "other": loss_other},
        module,
        primary_tasks=frozenset({"primary"}),
    )
    # `a` and `b` share no parameter, and neither touches `shared`, so the
    # zero-filled vectors are exactly orthogonal: cosine is exactly 0.
    assert conflict["primary|other"] == pytest.approx(0.0, abs=1e-6)


def test_pcgrad_and_gradient_conflict_logging_default_off() -> None:
    # core-issues3.txt Phase 11.4: "Do not enable PCGrad or a complex
    # automatic weighting scheme by default. Enable it only after measured
    # conflict and compare against the simpler baseline." A guard against
    # either default silently flipping in a future edit.
    assert TrainingConfig().pcgrad_enabled is False
    assert TrainingConfig().gradient_conflict_logging is False
    config = TrainingConfig.from_yaml(_TRAINING_YAML)
    assert config.pcgrad_enabled is False
