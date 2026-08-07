"""core-issues3.txt Phase 11.2: class-imbalance reporting and train-owned
class weights."""

from __future__ import annotations

import pytest
import torch

from hydroswarm.training.class_balance import (
    CLASS_WEIGHT_POLICY_VERSION,
    class_prevalence,
    class_weights_tensor,
    merge_prevalence,
    train_owned_class_weights,
)
from hydroswarm.training.losses import compute_multitask_loss


def test_class_prevalence_counts_a_scalar_per_example_label() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2])
    assert class_prevalence(labels) == {0: 3, 1: 1, 2: 1}


def test_class_prevalence_honors_ignore_index_and_mask() -> None:
    labels = torch.tensor([0, -100, 1, 1])
    assert class_prevalence(labels) == {0: 1, 1: 2}  # -100 excluded by default

    labels2 = torch.tensor([0, 0, 1, 1])
    mask = torch.tensor([True, False, True, False])
    assert class_prevalence(labels2, mask=mask) == {0: 1, 1: 1}


def test_class_prevalence_flattens_a_per_position_array_and_handles_bool() -> None:
    # plan_validity's real shape: [batch, plan_slots], bool.
    labels = torch.tensor([[True, False, True], [False, False, True]])
    assert class_prevalence(labels) == {1: 3, 0: 3}


def test_merge_prevalence_sums_across_shards() -> None:
    merged = merge_prevalence({0: 3, 1: 1}, {0: 2, 1: 5}, {2: 1})
    assert merged == {0: 5, 1: 6, 2: 1}


def test_train_owned_class_weights_gives_rarer_classes_higher_weight() -> None:
    prevalence = {0: 90, 1: 10}
    weights = train_owned_class_weights(prevalence)
    assert weights[1] > weights[0]
    # Mean-rescaled to 1.0 across the 2 observed classes.
    assert sum(weights.values()) / len(weights) == pytest.approx(1.0)


def test_train_owned_class_weights_caps_extreme_imbalance() -> None:
    prevalence = {0: 9999, 1: 1}
    weights = train_owned_class_weights(prevalence, maximum_weight=5.0)
    # class 1's raw inverse-frequency weight (10000/(2*1) = 5000) gets
    # capped to 5.0; class 0's raw weight (10000/(2*9999) ~= 0.5) is far
    # below the cap and untouched. Rescaling by the mean is a uniform
    # multiply, so the ratio between the two rescaled weights equals the
    # ratio between their pre-rescale (capped) values exactly.
    raw_class_0 = 10000 / (2 * 9999)
    assert weights[1] / weights[0] == pytest.approx(5.0 / raw_class_0, rel=1e-6)
    # The raw (uncapped) inverse-frequency ratio would have been ~10000x;
    # confirm the cap brought it down to a sane range.
    assert weights[1] / weights[0] < 20.0


def test_train_owned_class_weights_rejects_empty_prevalence() -> None:
    with pytest.raises(ValueError, match="empty"):
        train_owned_class_weights({})


def test_class_weights_tensor_defaults_to_one_for_unlisted_classes() -> None:
    dense = class_weights_tensor({1: 3.0}, num_classes=4)
    torch.testing.assert_close(dense, torch.tensor([1.0, 3.0, 1.0, 1.0]))


def test_class_weights_tensor_rejects_out_of_range_class() -> None:
    with pytest.raises(ValueError, match="out of range"):
        class_weights_tensor({5: 2.0}, num_classes=3)


def test_policy_version_is_a_stable_string_constant() -> None:
    assert isinstance(CLASS_WEIGHT_POLICY_VERSION, str) and CLASS_WEIGHT_POLICY_VERSION


# --- wiring into compute_multitask_loss ------------------------------------


def test_class_weights_change_the_loss_but_not_the_valid_count() -> None:
    outputs = {"source_node_logits": torch.tensor([[3.0, 0.0], [0.1, 2.0]], requires_grad=True)}
    targets = {"source_node": torch.tensor([0, 1])}

    unweighted = compute_multitask_loss(outputs, targets)
    weighted = compute_multitask_loss(
        outputs, targets, class_weights={"source_node": torch.tensor([1.0, 5.0])}
    )
    assert unweighted.valid_counts == weighted.valid_counts  # counts unaffected by class weighting
    assert not torch.allclose(unweighted.tasks["source_node"], weighted.tasks["source_node"])


def test_class_weights_match_f_cross_entropy_directly() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.1, 2.0], [1.0, 1.0]], requires_grad=True)
    target = torch.tensor([0, 1, 1])
    weight = torch.tensor([1.0, 4.0])
    result = compute_multitask_loss(
        {"source_node_logits": logits}, {"source_node": target}, class_weights={"source_node": weight}
    )
    import torch.nn.functional as F

    expected = F.cross_entropy(logits, target, weight=weight)
    torch.testing.assert_close(result.tasks["source_node"], expected)
