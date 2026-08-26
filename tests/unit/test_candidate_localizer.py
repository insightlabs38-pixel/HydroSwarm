"""Unit tests for CandidateConditionedLocalizer (exp/candidate-conditioned-
localizer-v1, EXPERIMENTAL). Covers the mandatory invariance/shape/backward-
compatibility checks the experiment plan requires: candidate relabeling
invariance, sensor-ordering invariance, candidate-list-ordering invariance,
variable candidate/network sizes, tensor shapes, candidate masks, zero/
invalid candidates, deterministic behavior under a fixed seed, and
backward compatibility of the default HydroCore path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hydroswarm.model.candidate_localizer import (
    NUM_HOP_BUCKETS,
    UNREACHABLE_HOP_SENTINEL,
    CandidateConditionedLocalizer,
    bucket_hop_distance,
)
from hydroswarm.model.core import HydroCore


def _random_hop_distance(batch: int, nodes: int, *, unreachable_prob: float = 0.1, generator: torch.Generator) -> torch.Tensor:
    hop = torch.randint(0, 6, (batch, nodes, nodes), generator=generator)
    hop = torch.minimum(hop, hop.transpose(1, 2))  # symmetric, matches a real undirected hop matrix
    unreachable = torch.rand(batch, nodes, nodes, generator=generator) < unreachable_prob
    unreachable = unreachable | unreachable.transpose(1, 2)
    hop = hop.masked_fill(unreachable, UNREACHABLE_HOP_SENTINEL)
    diag = torch.arange(nodes)
    hop[:, diag, diag] = 0
    return hop


class TestBucketing:
    def test_negative_maps_to_unreachable_bucket(self) -> None:
        hop = torch.tensor([-1, 0, 1, 2, 3, 5, 100])
        buckets = bucket_hop_distance(hop)
        assert int(buckets[0]) == NUM_HOP_BUCKETS - 1
        assert int(buckets[-1]) == NUM_HOP_BUCKETS - 2  # largest finite bucket, not unreachable
        assert buckets.min() >= 0
        assert buckets.max() < NUM_HOP_BUCKETS


class TestShapesAndValidation:
    def _make(self, **kwargs) -> CandidateConditionedLocalizer:
        return CandidateConditionedLocalizer(16, **kwargs)

    def _batch(self, batch=2, nodes=5, generator=None):
        generator = generator or torch.Generator().manual_seed(0)
        sentinel = torch.randn(batch, nodes, 16, generator=generator)
        candidate_mask = torch.ones(batch, nodes, dtype=torch.bool)
        sensor_mask = torch.zeros(batch, nodes, dtype=torch.bool)
        sensor_mask[:, 0] = True
        hop = _random_hop_distance(batch, nodes, generator=generator)
        return sentinel, candidate_mask, sensor_mask, hop

    def test_output_shape(self) -> None:
        module = self._make()
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        out = module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop)
        assert out.shape == (2, 5)
        assert torch.isfinite(out).all()

    def test_wrong_sentinel_last_dim_rejected(self) -> None:
        module = self._make()
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        with pytest.raises(ValueError):
            module(sentinel[..., :8], candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop)

    def test_wrong_mask_shape_rejected(self) -> None:
        module = self._make()
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        with pytest.raises(ValueError):
            module(sentinel, candidate_mask=candidate_mask[:, :-1], sensor_mask_nodes=sensor_mask, hop_distance=hop)

    def test_wrong_hop_distance_shape_rejected(self) -> None:
        module = self._make()
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        with pytest.raises(ValueError):
            module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop[:, :, :-1])

    def test_missing_structural_features_when_configured_rejected(self) -> None:
        module = self._make(structural_feature_dim=3)
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        with pytest.raises(ValueError):
            module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop)

    def test_missing_physics_features_when_configured_rejected(self) -> None:
        module = self._make(physics_feature_dim=2)
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        with pytest.raises(ValueError):
            module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop)

    def test_non_candidate_positions_masked_to_min(self) -> None:
        module = self._make()
        sentinel, candidate_mask, sensor_mask, hop = self._batch()
        candidate_mask = candidate_mask.clone()
        candidate_mask[:, -1] = False
        out = module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop)
        assert torch.all(out[:, -1] == torch.finfo(out.dtype).min)

    def test_zero_active_sensors_stays_finite(self) -> None:
        """A degenerate example with no active sensor anywhere must not
        produce NaN (an all-masked softmax row) -- see the module's own
        `no_valid_sensor` handling."""

        module = self._make()
        sentinel, candidate_mask, _, hop = self._batch()
        empty_sensor_mask = torch.zeros_like(candidate_mask)
        out = module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=empty_sensor_mask, hop_distance=hop)
        assert torch.isfinite(out).all()

    def test_single_candidate_single_sensor(self) -> None:
        module = self._make()
        sentinel = torch.randn(1, 1, 16)
        candidate_mask = torch.ones(1, 1, dtype=torch.bool)
        sensor_mask = torch.ones(1, 1, dtype=torch.bool)
        hop = torch.zeros(1, 1, 1, dtype=torch.long)
        out = module(sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask, hop_distance=hop)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all()


class TestInvariance:
    def _random_module(self, seed: int = 0) -> CandidateConditionedLocalizer:
        torch.manual_seed(seed)
        return CandidateConditionedLocalizer(16, structural_feature_dim=2, dropout=0.0)

    def _random_example(self, nodes: int, generator: torch.Generator):
        sentinel = torch.randn(1, nodes, 16, generator=generator)
        candidate_mask = torch.rand(1, nodes, generator=generator) > 0.3
        candidate_mask[0, 0] = True  # guarantee at least one real candidate
        sensor_mask = torch.rand(1, nodes, generator=generator) > 0.5
        hop = _random_hop_distance(1, nodes, generator=generator)
        structural = torch.randn(1, nodes, 2, generator=generator)
        return sentinel, candidate_mask, sensor_mask, hop, structural

    def test_node_relabeling_invariance(self) -> None:
        """Permuting node identity consistently across every tensor must
        leave each real candidate's score unchanged (up to floating-point
        tolerance) -- the module has zero per-node-index parameters."""

        module = self._random_module()
        module.eval()
        generator = torch.Generator().manual_seed(42)
        nodes = 7
        sentinel, candidate_mask, sensor_mask, hop, structural = self._random_example(nodes, generator)

        with torch.no_grad():
            baseline = module(
                sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask,
                hop_distance=hop, structural_features=structural,
            )

        perm = torch.randperm(nodes, generator=generator)
        inv_perm = torch.argsort(perm)
        with torch.no_grad():
            permuted = module(
                sentinel[:, perm],
                candidate_mask=candidate_mask[:, perm],
                sensor_mask_nodes=sensor_mask[:, perm],
                hop_distance=hop[:, perm][:, :, perm],
                structural_features=structural[:, perm],
            )
        unpermuted = permuted[:, inv_perm]
        mask = candidate_mask[0]
        assert torch.allclose(baseline[0, mask], unpermuted[0, mask], atol=1e-5)

    def test_multiple_relabelings_agree(self) -> None:
        module = self._random_module(seed=1)
        module.eval()
        generator = torch.Generator().manual_seed(7)
        nodes = 6
        sentinel, candidate_mask, sensor_mask, hop, structural = self._random_example(nodes, generator)
        with torch.no_grad():
            baseline = module(
                sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask,
                hop_distance=hop, structural_features=structural,
            )
        for trial in range(3):
            perm = torch.randperm(nodes, generator=generator)
            inv_perm = torch.argsort(perm)
            with torch.no_grad():
                permuted = module(
                    sentinel[:, perm], candidate_mask=candidate_mask[:, perm],
                    sensor_mask_nodes=sensor_mask[:, perm], hop_distance=hop[:, perm][:, :, perm],
                    structural_features=structural[:, perm],
                )
            unpermuted = permuted[:, inv_perm]
            mask = candidate_mask[0]
            assert torch.allclose(baseline[0, mask], unpermuted[0, mask], atol=1e-5), f"trial {trial} mismatch"

    def test_variable_network_sizes(self) -> None:
        """Same module instance must handle different node counts across
        calls without any shape-dependent parameter."""

        module = self._random_module(seed=2)
        module.eval()
        generator = torch.Generator().manual_seed(3)
        for nodes in (2, 4, 9, 12):
            sentinel, candidate_mask, sensor_mask, hop, structural = self._random_example(nodes, generator)
            with torch.no_grad():
                out = module(
                    sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask,
                    hop_distance=hop, structural_features=structural,
                )
            assert out.shape == (1, nodes)
            assert torch.isfinite(out[candidate_mask]).all()

    def test_deterministic_under_fixed_seed(self) -> None:
        generator = torch.Generator().manual_seed(11)
        sentinel, candidate_mask, sensor_mask, hop, structural = self._random_example(6, generator)

        torch.manual_seed(99)
        module_a = CandidateConditionedLocalizer(16, structural_feature_dim=2, dropout=0.0)
        module_a.eval()
        torch.manual_seed(99)
        module_b = CandidateConditionedLocalizer(16, structural_feature_dim=2, dropout=0.0)
        module_b.eval()

        with torch.no_grad():
            out_a = module_a(
                sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask,
                hop_distance=hop, structural_features=structural,
            )
            out_b = module_b(
                sentinel, candidate_mask=candidate_mask, sensor_mask_nodes=sensor_mask,
                hop_distance=hop, structural_features=structural,
            )
        assert torch.equal(out_a, out_b)


class TestHydroCoreIntegration:
    def _batch(self, batch=2, nodes=6):
        return {
            "node_features": torch.randn(batch, nodes, 19),
            "temporal_features": torch.randn(batch, 4, nodes, 6),
            "quality_features": torch.randn(batch, 4, nodes, 4),
            "node_mask": torch.ones(batch, nodes, dtype=torch.bool),
            "source_candidate_mask": torch.ones(batch, nodes, dtype=torch.bool),
            "active_sensor_mask_nodes": torch.tensor([[True, False] + [False] * (nodes - 2)] * batch),
            "candidate_hop_distance": torch.randint(0, 4, (batch, nodes, nodes)),
        }

    def test_default_mode_has_no_candidate_localizer(self) -> None:
        model = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13)
        assert model.candidate_localizer is None
        assert model.localizer_mode == "default"

    def test_default_mode_unaffected_by_missing_localizer_batch_fields(self) -> None:
        """Backward compatibility: a batch with none of the new optional
        fields must still work in default mode, exactly as before this
        experiment existed."""

        model = HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13)
        batch = self._batch()
        del batch["active_sensor_mask_nodes"]
        del batch["candidate_hop_distance"]
        out = model(batch)
        assert out["source_node_logits"].shape == (2, 6)

    def test_candidate_conditioned_mode_forward_backward(self) -> None:
        model = HydroCore.from_variant(
            "small", node_feature_dim=19, edge_feature_dim=13,
            localizer_mode="candidate_conditioned",
        )
        assert model.candidate_localizer is not None
        batch = self._batch()
        out = model(batch)
        assert out["source_node_logits"].shape == (2, 6)
        assert torch.isfinite(out["source_node_logits"]).all()
        out["source_node_logits"].sum().backward()
        localizer_grad_norm = sum(
            p.grad.norm().item() for p in model.candidate_localizer.parameters() if p.grad is not None
        )
        assert localizer_grad_norm > 0.0
        # source_node_head exists (checkpoint-shape stability) but its
        # output is not read in this mode, so it receives no gradient.
        assert all(p.grad is None for p in model.source_node_head.parameters())

    def test_candidate_conditioned_mode_requires_new_fields(self) -> None:
        model = HydroCore.from_variant(
            "small", node_feature_dim=19, edge_feature_dim=13,
            localizer_mode="candidate_conditioned",
        )
        batch = self._batch()
        del batch["candidate_hop_distance"]
        with pytest.raises(KeyError):
            model(batch)

    def test_variable_candidate_count_via_mask(self) -> None:
        model = HydroCore.from_variant(
            "small", node_feature_dim=19, edge_feature_dim=13,
            localizer_mode="candidate_conditioned",
        )
        batch = self._batch(nodes=8)
        batch["source_candidate_mask"] = torch.tensor(
            [[True, True, False, True, False, False, True, True]] * 2
        )
        out = model(batch)
        masked_out = out["source_node_logits"][~batch["source_candidate_mask"]]
        assert torch.all(masked_out == torch.finfo(masked_out.dtype).min)

    def test_invalid_localizer_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            HydroCore.from_variant("small", node_feature_dim=19, edge_feature_dim=13, localizer_mode="bogus")
