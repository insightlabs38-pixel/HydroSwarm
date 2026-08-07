"""core-issues4.txt Section E: candidate/vocabulary contract validation.

Before this pass, HydroCore's candidate-conditioned forward pass read
plan_template_ids/plan_target_type/plan_target_node_index/
plan_target_link_index directly into torch.gather/nn.Embedding with only a
`clamp(min=0)` (no upper bound) and no range checks at all -- an
out-of-range value would either raise an opaque low-level indexing error
or, for an in-bounds-but-wrong gather index, silently read a real,
unrelated node/link's embedding. These tests exercise the new fail-closed
validation added to that path.
"""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model.core import HydroCore


def _model(**overrides) -> HydroCore:
    base = dict(
        node_feature_dim=3, temporal_feature_dim=2, quality_feature_dim=2,
        d_model=32, nhead=4, dim_feedforward=64, num_layers=1, modality_layers=1,
        adapter_dims=(32, 32, 32), strategist_mode="candidate_conditioned",
    )
    base.update(overrides)
    return HydroCore(**base).eval()


def _edge_batch(**overrides) -> dict:
    batch = {
        "node_features": torch.randn(1, 3, 3),
        "temporal_features": torch.randn(1, 2, 3, 2),
        "quality_features": torch.randn(1, 2, 3, 2),
        "edge_index": torch.tensor([[[0, 1], [1, 2]]]),  # 2 edges, 3 nodes
        "edge_features": torch.randn(1, 2, 13),
        "edge_mask": torch.ones(1, 2, dtype=torch.bool),
        "timestamps": torch.tensor([[0.0, 300.0]]),
    }
    batch.update(overrides)
    return batch


def _valid_plan_batch(**overrides) -> dict:
    plans = 2
    batch = _edge_batch(
        plan_template_ids=torch.tensor([[0, 1]]),
        plan_target_type=torch.tensor([[1, 2]]),  # NODE, LINK
        plan_target_node_index=torch.tensor([[1, -1]]),
        plan_target_link_index=torch.tensor([[-1, 0]]),
        plan_features=torch.zeros(1, plans, 6),
        plan_mask=torch.ones(1, plans, dtype=torch.bool),
    )
    batch.update(overrides)
    return batch


def test_valid_batch_does_not_raise() -> None:
    model = _model()
    with torch.no_grad():
        model(_valid_plan_batch())  # must not raise


def test_out_of_range_template_id_at_real_position_fails_closed() -> None:
    model = _model()
    batch = _valid_plan_batch(plan_template_ids=torch.tensor([[0, 99]]))
    with pytest.raises(ValueError, match="plan_template_ids"):
        with torch.no_grad():
            model(batch)


def test_negative_template_id_at_real_position_fails_closed() -> None:
    model = _model()
    batch = _valid_plan_batch(plan_template_ids=torch.tensor([[-1, 1]]))
    with pytest.raises(ValueError, match="plan_template_ids"):
        with torch.no_grad():
            model(batch)


def test_out_of_range_template_id_at_padded_position_is_tolerated() -> None:
    """Padded plans (plan_mask=False) may carry sentinel/garbage template
    ids -- CandidatePlanEncoder's own plan_mask handling already zeroes
    their contribution, so range-checking them would reject legitimate
    padding, not a real defect."""

    model = _model()
    batch = _valid_plan_batch(
        plan_template_ids=torch.tensor([[0, 99]]),
        plan_mask=torch.tensor([[True, False]]),
    )
    with torch.no_grad():
        model(batch)  # must not raise


def test_out_of_range_target_type_fails_closed() -> None:
    model = _model()
    batch = _valid_plan_batch(plan_target_type=torch.tensor([[1, 5]]))
    with pytest.raises(ValueError, match="plan_target_type"):
        with torch.no_grad():
            model(batch)


def test_out_of_range_node_target_index_fails_closed() -> None:
    model = _model()
    batch = _valid_plan_batch(
        plan_target_type=torch.tensor([[1, 0]]),
        plan_target_node_index=torch.tensor([[999, -1]]),
    )
    with pytest.raises(ValueError, match="plan_target_node_index"):
        with torch.no_grad():
            model(batch)


def test_negative_sentinel_node_index_is_tolerated_when_type_is_not_node() -> None:
    """The established -1 "no target" sentinel at a non-NODE position must
    not be flagged -- only a real (target_type=NODE) position is
    range-checked."""

    model = _model()
    batch = _valid_plan_batch(
        plan_target_type=torch.tensor([[0, 2]]),  # NONE, LINK
        plan_target_node_index=torch.tensor([[-1, -1]]),
    )
    with torch.no_grad():
        model(batch)  # must not raise


def test_out_of_range_link_target_index_fails_closed() -> None:
    model = _model()
    # Only 2 edges exist (edge_index has 2 columns); index 7 is invalid.
    batch = _valid_plan_batch(
        plan_target_type=torch.tensor([[2, 0]]),
        plan_target_link_index=torch.tensor([[7, -1]]),
    )
    with pytest.raises(ValueError, match="plan_target_link_index"):
        with torch.no_grad():
            model(batch)


def test_zero_candidate_plans_does_not_raise() -> None:
    model = _model()
    batch = _edge_batch(
        plan_template_ids=torch.zeros(1, 0, dtype=torch.long),
        plan_target_type=torch.zeros(1, 0, dtype=torch.long),
        plan_target_node_index=torch.zeros(1, 0, dtype=torch.long),
        plan_target_link_index=torch.zeros(1, 0, dtype=torch.long),
        plan_features=torch.zeros(1, 0, 6),
        plan_mask=torch.zeros(1, 0, dtype=torch.bool),
    )
    with torch.no_grad():
        output = model(batch)
    assert output["plan_value"].shape == (1, 0)


def test_wrong_plan_features_width_fails_closed() -> None:
    model = _model()
    batch = _valid_plan_batch(plan_features=torch.zeros(1, 2, 5))  # 5, not plan_feature_dim=6
    with pytest.raises(ValueError, match="plan_features"):
        with torch.no_grad():
            model(batch)


def test_mismatched_plan_mask_shape_fails_closed() -> None:
    model = _model()
    batch = _valid_plan_batch(plan_mask=torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="plan_mask"):
        with torch.no_grad():
            model(batch)
