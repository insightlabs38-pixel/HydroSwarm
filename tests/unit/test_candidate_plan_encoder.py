"""core-issues3.txt Phase 4: CandidatePlanEncoder, standalone."""

from __future__ import annotations

import pytest
import torch

from hydroswarm.model.candidate_plan_encoder import TARGET_TYPE_INDEX, CandidatePlanEncoder


def _encoder(d_model: int = 16, action_vocabulary_size: int = 9, plan_feature_dim: int = 6) -> CandidatePlanEncoder:
    return CandidatePlanEncoder(
        d_model, action_vocabulary_size=action_vocabulary_size, plan_feature_dim=plan_feature_dim
    )


def test_output_shape_and_finiteness() -> None:
    encoder = _encoder().eval()
    batch, plans, d_model = 2, 5, 16
    with torch.no_grad():
        output = encoder(
            template_ids=torch.randint(0, 9, (batch, plans)),
            target_type=torch.randint(0, 3, (batch, plans)),
            target_embedding=torch.randn(batch, plans, d_model),
            plan_features=torch.randn(batch, plans, 6),
            plan_mask=torch.ones(batch, plans, dtype=torch.bool),
            incident_context=torch.randn(batch, d_model),
        )
    assert output.shape == (batch, plans, d_model)
    assert torch.isfinite(output).all()


def test_padded_plans_are_zeroed() -> None:
    encoder = _encoder().eval()
    batch, plans, d_model = 2, 4, 16
    plan_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    with torch.no_grad():
        output = encoder(
            template_ids=torch.randint(0, 9, (batch, plans)),
            target_type=torch.randint(0, 3, (batch, plans)),
            target_embedding=torch.randn(batch, plans, d_model),
            plan_features=torch.randn(batch, plans, 6),
            plan_mask=plan_mask,
            incident_context=torch.randn(batch, d_model),
        )
    assert torch.count_nonzero(output[~plan_mask]) == 0


def test_different_templates_produce_different_representations() -> None:
    """The core point of this module: two candidates that differ only in
    action_template must NOT collapse to the same representation (the old
    anonymous-query design's failure mode)."""

    encoder = _encoder().eval()
    d_model = 16
    shared_target = torch.randn(1, 2, d_model)
    shared_features = torch.randn(1, 2, 6)
    with torch.no_grad():
        output = encoder(
            template_ids=torch.tensor([[0, 1]]),
            target_type=torch.tensor([[int(TARGET_TYPE_INDEX["NODE"])] * 2]),
            target_embedding=shared_target,
            plan_features=shared_features,
            plan_mask=torch.ones(1, 2, dtype=torch.bool),
            incident_context=torch.zeros(1, d_model),
        )
    assert not torch.allclose(output[0, 0], output[0, 1])


def test_different_targets_produce_different_representations() -> None:
    encoder = _encoder().eval()
    d_model = 16
    with torch.no_grad():
        output = encoder(
            template_ids=torch.zeros(1, 2, dtype=torch.long),
            target_type=torch.full((1, 2), int(TARGET_TYPE_INDEX["NODE"]), dtype=torch.long),
            target_embedding=torch.stack([torch.zeros(d_model), torch.ones(d_model)]).unsqueeze(0),
            plan_features=torch.zeros(1, 2, 6),
            plan_mask=torch.ones(1, 2, dtype=torch.bool),
            incident_context=torch.zeros(1, d_model),
        )
    assert not torch.allclose(output[0, 0], output[0, 1])


def test_deterministic_in_eval_mode() -> None:
    encoder = _encoder().eval()
    batch, plans, d_model = 1, 3, 16
    inputs = dict(
        template_ids=torch.randint(0, 9, (batch, plans)),
        target_type=torch.randint(0, 3, (batch, plans)),
        target_embedding=torch.randn(batch, plans, d_model),
        plan_features=torch.randn(batch, plans, 6),
        plan_mask=torch.ones(batch, plans, dtype=torch.bool),
        incident_context=torch.randn(batch, d_model),
    )
    with torch.no_grad():
        first = encoder(**inputs)
        second = encoder(**inputs)
    assert torch.equal(first, second)


def test_rejects_mismatched_shapes() -> None:
    encoder = _encoder()
    with pytest.raises(ValueError):
        encoder(
            template_ids=torch.zeros(2, 3, dtype=torch.long),
            target_type=torch.zeros(2, 4, dtype=torch.long),  # wrong plan count
            target_embedding=torch.zeros(2, 3, 16),
            plan_features=torch.zeros(2, 3, 6),
            plan_mask=torch.ones(2, 3, dtype=torch.bool),
            incident_context=torch.zeros(2, 16),
        )
