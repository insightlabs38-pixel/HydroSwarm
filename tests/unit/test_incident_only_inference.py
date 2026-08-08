"""core-issues5.txt Section 2 (P0 blocker): incident-only inference must not
require candidate-plan tensors.

Before this pass, HydroCore.forward() raised KeyError whenever
strategist_mode="candidate_conditioned" and any of the four required plan
tensors (plan_template_ids/plan_target_type/plan_mask/plan_features) was
absent -- unconditionally, even when NONE of them were supplied. Live
incident analysis (HydroSentinel localization) runs before any response
plan exists, so a real HybridInferencePipeline.analyze() call against a
candidate-conditioned V4 checkpoint always hit this KeyError inside
_run_model, which is wrapped in a broad `except Exception`, silently
downgrading the whole analysis to HybridRuntimeMode.CLASSICAL_SAFE instead
of surfacing a genuine architecture/train-serve mismatch.

These tests exercise the fix: plan tensors fully absent now means
incident-only inference (plan_hidden stays None, no plan-scoring keys are
produced), a partially-supplied set of plan tensors remains a hard
failure (a real caller defect, not "no plan yet"), and full plan tensors
still produce real plan-scoring outputs exactly as before.
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


def _incident_batch(**overrides) -> dict:
    """A batch with no candidate-plan tensors at all -- the real shape of a
    live PASS 1 (Sentinel/localization) call before any response plan
    exists."""
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


def _plan_batch(**overrides) -> dict:
    plans = 2
    batch = _incident_batch(
        plan_template_ids=torch.tensor([[0, 1]]),
        plan_target_type=torch.tensor([[1, 2]]),  # NODE, LINK
        plan_target_node_index=torch.tensor([[1, -1]]),
        plan_target_link_index=torch.tensor([[-1, 0]]),
        plan_features=torch.zeros(1, plans, 6),
        plan_mask=torch.ones(1, plans, dtype=torch.bool),
    )
    batch.update(overrides)
    return batch


def test_incident_only_inference_does_not_raise() -> None:
    """A candidate-conditioned model must run PASS 1 without any plan
    tensors supplied."""
    model = _model()
    with torch.no_grad():
        model(_incident_batch())  # must not raise


def test_incident_only_inference_returns_sentinel_outputs() -> None:
    model = _model()
    with torch.no_grad():
        output = model(_incident_batch())
    assert output["source_node_logits"].shape == (1, 3)
    assert output["source_region_logits"].shape[0] == 1
    assert output["evidence_sufficiency"].shape == (1,)
    assert output["sensor_fault_logits"].shape == (1, 3)
    assert output["sample_node_logits"].shape == (1, 3)


def test_incident_only_inference_omits_plan_scoring_outputs() -> None:
    """No fabricated Strategist plan scores may appear when no plan
    tensors were supplied -- HydroOutput is total=False, so the correct
    representation of "not scored yet" is key absence, not a
    zero/garbage tensor."""
    model = _model()
    with torch.no_grad():
        output = model(_incident_batch())
    assert "action_logits" not in output
    assert "action_pointer_logits" not in output
    assert "plan_value" not in output
    assert "plan_validity_logits" not in output


def test_incident_only_inference_omits_consequence_proxies() -> None:
    model = _model(consequence_prescreening_heads=True)
    with torch.no_grad():
        output = model(_incident_batch())
    for key in (
        "exposure_proxy",
        "pressure_risk_proxy",
        "service_loss_proxy",
        "containment_time_proxy",
        "plan_regret_proxy",
    ):
        assert key not in output


def test_candidate_scoring_still_works_when_plan_tensors_supplied() -> None:
    """PASS 2: once real deterministic candidate-plan tensors are
    supplied, plan-scoring outputs are produced exactly as before."""
    model = _model()
    with torch.no_grad():
        output = model(_plan_batch())
    assert output["plan_value"].shape == (1, 2)
    assert output["plan_validity_logits"].shape[:2] == (1, 2)
    assert output["action_logits"].shape[:2] == (1, 2)
    assert output["action_pointer_logits"].shape[:2] == (1, 2)


@pytest.mark.parametrize(
    "missing_field",
    ["plan_template_ids", "plan_target_type", "plan_mask", "plan_features"],
)
def test_partially_supplied_plan_tensors_still_fail_closed(missing_field: str) -> None:
    """A partially-supplied set of plan tensors is a real caller defect
    (not "no plan yet") and must still raise, exactly as full absence
    used to before this fix -- only *complete* absence is treated as
    incident-only inference."""
    model = _model()
    batch = _plan_batch()
    del batch[missing_field]
    with pytest.raises(KeyError, match=missing_field):
        with torch.no_grad():
            model(batch)


def test_anonymous_queries_mode_is_unaffected() -> None:
    """The default (non-candidate-conditioned) forward_only path must
    keep producing plan-scoring outputs from its anonymous learned query
    tokens exactly as before -- this fix only changes
    strategist_mode="candidate_conditioned" behavior."""
    model = _model(strategist_mode="anonymous_queries")
    with torch.no_grad():
        output = model(_incident_batch())
    assert "plan_value" in output
    assert "action_logits" in output
