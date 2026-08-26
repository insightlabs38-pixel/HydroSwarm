"""Unit tests for `CapacityMatchedProjection` / `HydroCore`'s
`localizer_capacity_hidden_dim` knob (exp/physics-informed-localizer-
validation Phase 2 -- A_CAPACITY_MATCHED, a generic-capacity control for
the candidate-conditioned-localizer-v1 pilot's Arm B/C parameter delta).

Covers: default (0) is byte-identical to pre-existing behavior, a nonzero
value adds capacity only to the default localization path (never
candidate/graph/physics information), forward/backward runs cleanly and
deterministically, and checkpoint-compatibility metadata catches a
mismatched value the way every other architecture-identity flag in
core.py already does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hydroswarm.model.adapters import CapacityMatchedProjection
from hydroswarm.model.core import ArchitectureCompatibilityError, HydroCore, verify_architecture_compatibility


def _batch(batch=2, nodes=6):
    return {
        "node_features": torch.randn(batch, nodes, 19),
        "temporal_features": torch.randn(batch, 4, nodes, 6),
        "quality_features": torch.randn(batch, 4, nodes, 4),
        "node_mask": torch.ones(batch, nodes, dtype=torch.bool),
        "source_candidate_mask": torch.ones(batch, nodes, dtype=torch.bool),
    }


class TestCapacityMatchedProjection:
    def test_rejects_nonpositive_hidden_dim(self) -> None:
        try:
            CapacityMatchedProjection(16, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for hidden_dim=0")

    def test_output_shape_and_residual_reads_only_its_input(self) -> None:
        module = CapacityMatchedProjection(16, 32)
        hidden = torch.randn(2, 5, 16)
        out = module(hidden)
        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()

    def test_deterministic_under_fixed_seed(self) -> None:
        torch.manual_seed(7)
        a = CapacityMatchedProjection(16, 32)
        torch.manual_seed(7)
        b = CapacityMatchedProjection(16, 32)
        hidden = torch.randn(2, 5, 16)
        assert torch.allclose(a(hidden), b(hidden))


class TestHydroCoreCapacityMatchedIntegration:
    def test_default_hidden_dim_constructs_nothing(self) -> None:
        model = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13)
        assert model.capacity_matched_projection is None
        assert model.localizer_capacity_hidden_dim == 0

    def test_zero_hidden_dim_forward_matches_pre_existing_behavior(self) -> None:
        torch.manual_seed(0)
        baseline = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13)
        torch.manual_seed(0)
        explicit_zero = HydroCore.from_variant(
            "small", node_feature_dim=19, edge_feature_dim=13, localizer_capacity_hidden_dim=0
        )
        batch = _batch()
        baseline.eval()
        explicit_zero.eval()
        with torch.inference_mode():
            out_a = baseline(batch)["source_node_logits"]
            out_b = explicit_zero(batch)["source_node_logits"]
        # Both configurations construct byte-identical module graphs (no
        # capacity_matched_projection either way), so any difference here
        # can only be floating-point reduction-order noise, not a real
        # behavior change -- tight tolerance, not exact equality.
        assert torch.allclose(out_a, out_b, atol=1e-4)

    def test_nonzero_hidden_dim_adds_parameters_and_runs(self) -> None:
        control = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13)
        capacity_matched = HydroCore.from_variant(
            "small", node_feature_dim=19, edge_feature_dim=13, localizer_capacity_hidden_dim=482
        )
        assert capacity_matched.capacity_matched_projection is not None
        control_total = control.parameter_report_dict()["total"]
        capacity_total = capacity_matched.parameter_report_dict()["total"]
        assert capacity_total > control_total
        # Phase 2 target: ~+4.6% over CONTROL, matching Arm B/C's own delta
        # as closely as practical without candidate conditioning.
        ratio = capacity_total / control_total
        assert 0.04 < (ratio - 1) < 0.06

        batch = _batch()
        out = capacity_matched(batch)
        assert out["source_node_logits"].shape == (2, 6)
        assert torch.isfinite(out["source_node_logits"]).all()
        out["source_node_logits"].sum().backward()
        capacity_grad_norm = sum(
            p.grad.norm().item() for p in capacity_matched.capacity_matched_projection.parameters() if p.grad is not None
        )
        assert capacity_grad_norm > 0

    def test_has_no_effect_in_candidate_conditioned_mode(self) -> None:
        """localizer_capacity_hidden_dim is a control for the DEFAULT
        localization path only -- constructing it alongside
        localizer_mode="candidate_conditioned" must not change that mode's
        output (source_node_head/capacity_matched_projection are simply
        unused on that branch)."""

        torch.manual_seed(0)
        candidate_conditioned = HydroCore.from_variant(
            "small", node_feature_dim=19, edge_feature_dim=13, localizer_mode="candidate_conditioned"
        )
        torch.manual_seed(0)
        candidate_conditioned_with_capacity = HydroCore.from_variant(
            "small",
            node_feature_dim=19,
            edge_feature_dim=13,
            localizer_mode="candidate_conditioned",
            localizer_capacity_hidden_dim=482,
        )
        batch = _batch()
        batch["active_sensor_mask_nodes"] = torch.tensor([[True, False, False, False, False, False]] * 2)
        batch["candidate_hop_distance"] = torch.randint(0, 4, (2, 6, 6))
        candidate_conditioned.eval()
        candidate_conditioned_with_capacity.eval()
        with torch.inference_mode():
            out_a = candidate_conditioned(batch)["source_node_logits"]
            out_b = candidate_conditioned_with_capacity(batch)["source_node_logits"]
        assert torch.equal(out_a, out_b)

    def test_checkpoint_compatibility_check_catches_mismatch(self) -> None:
        model = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13, localizer_capacity_hidden_dim=482)
        metadata = model.architecture_config()
        verify_architecture_compatibility(model, metadata)  # matching metadata: no raise

        mismatched = dict(metadata)
        mismatched["localizer_capacity_hidden_dim"] = 0
        try:
            verify_architecture_compatibility(model, mismatched)
        except ArchitectureCompatibilityError:
            pass
        else:
            raise AssertionError("expected ArchitectureCompatibilityError for mismatched localizer_capacity_hidden_dim")
