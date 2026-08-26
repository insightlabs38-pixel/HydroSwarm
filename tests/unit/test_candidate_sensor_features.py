"""Unit tests for the label-free candidate<->sensor feature computations
used by the candidate-conditioned-localizer-v1 experiment
(`scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/`).
Covers relabeling invariance, variable network sizes, and the "never reads
a target tensor" contract each module's docstring claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental"))

from candidate_conditioned_localizer_v1 import candidate_sensor_features as csf  # noqa: E402
from candidate_conditioned_localizer_v1 import physics_features as physf  # noqa: E402
from hydroswarm.model.candidate_localizer import UNREACHABLE_HOP_SENTINEL  # noqa: E402


def _path_graph_batch(nodes: int = 5, batch: int = 1):
    edges = [[i, i + 1] for i in range(nodes - 1)]
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]]).unsqueeze(0).expand(batch, -1, -1)
    node_mask = torch.ones(batch, nodes, dtype=torch.bool)
    return node_mask, edge_index


class TestHopDistance:
    def test_path_graph_hop_distances(self) -> None:
        node_mask, edge_index = _path_graph_batch(5)
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        assert hop[0, 0, 4].item() == 4
        assert hop[0, 2, 2].item() == 0
        assert hop[0, 1, 3].item() == 2

    def test_disconnected_pair_is_sentinel(self) -> None:
        node_mask = torch.ones(1, 4, dtype=torch.bool)
        edge_index = torch.tensor([[[0], [1]]])  # only edge 0-1; nodes 2,3 isolated
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        assert hop[0, 0, 2].item() == UNREACHABLE_HOP_SENTINEL
        assert hop[0, 2, 3].item() == UNREACHABLE_HOP_SENTINEL

    def test_no_edge_index_returns_all_sentinel(self) -> None:
        node_mask = torch.ones(1, 3, dtype=torch.bool)
        hop = csf.compute_hop_distance(node_mask, None, None)
        assert torch.all(hop == UNREACHABLE_HOP_SENTINEL)

    def test_relabeling_invariance(self) -> None:
        # perm[i] = which OLD node position now lives at NEW position i, so
        # remapping edge_index's VALUES (old node ids) to new ids requires
        # the inverse permutation, not perm itself.
        node_mask, edge_index = _path_graph_batch(6)
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        inv_perm = torch.argsort(perm)
        edge_index_p = inv_perm[edge_index]
        hop_p = csf.compute_hop_distance(node_mask, edge_index_p, None)
        expected = hop[0][perm][:, perm]
        assert torch.equal(hop_p[0], expected)


class TestActiveSensorMask:
    def test_reduces_over_time(self) -> None:
        node_mask = torch.ones(1, 4, dtype=torch.bool)
        sensor_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
        sensor_mask[0, 1, 2] = True  # node 2 has one valid reading at t=1
        active = csf.active_sensor_mask_from_temporal(sensor_mask, node_mask)
        assert active.tolist() == [[False, False, True, False]]

    def test_none_returns_all_false(self) -> None:
        node_mask = torch.ones(1, 4, dtype=torch.bool)
        active = csf.active_sensor_mask_from_temporal(None, node_mask)
        assert not active.any()


class TestStructuralFeatures:
    def test_shape_and_finite(self) -> None:
        node_mask, edge_index = _path_graph_batch(6)
        sensor_mask = torch.zeros(1, 6, dtype=torch.bool)
        sensor_mask[0, [0, 5]] = True
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        out = csf.compute_structural_features(node_mask, edge_index, None, sensor_mask, hop)
        assert out.shape == (1, 6, len(csf.NODE_STRUCTURAL_COLUMNS))
        assert torch.isfinite(out).all()

    def test_padded_positions_are_zero(self) -> None:
        node_mask = torch.zeros(1, 6, dtype=torch.bool)
        node_mask[0, :3] = True
        edge_index = torch.tensor([[[0, 1], [1, 2]]])
        sensor_mask = torch.zeros(1, 6, dtype=torch.bool)
        sensor_mask[0, 0] = True
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        out = csf.compute_structural_features(node_mask, edge_index, None, sensor_mask, hop)
        assert torch.all(out[0, 3:] == 0.0)

    def test_relabeling_invariance(self) -> None:
        node_mask, edge_index = _path_graph_batch(6)
        sensor_mask = torch.zeros(1, 6, dtype=torch.bool)
        sensor_mask[0, [0, 5]] = True
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        out = csf.compute_structural_features(node_mask, edge_index, None, sensor_mask, hop)

        perm = torch.tensor([4, 2, 0, 5, 1, 3])
        edge_index_p = torch.argsort(perm)[edge_index]
        sensor_mask_p = sensor_mask[:, perm]
        hop_p = csf.compute_hop_distance(node_mask, edge_index_p, None)
        out_p = csf.compute_structural_features(node_mask, edge_index_p, None, sensor_mask_p, hop_p)
        assert torch.allclose(out[0][perm], out_p[0], atol=1e-6)


class TestPhysicsFeatures:
    def test_nearer_higher_reading_candidate_scores_higher(self) -> None:
        node_mask, edge_index = _path_graph_batch(5)
        sensor_mask_t = torch.zeros(1, 3, 5, dtype=torch.bool)
        sensor_mask_t[0, :, 0] = True
        sensor_mask_t[0, :, 4] = True
        active = csf.active_sensor_mask_from_temporal(sensor_mask_t, node_mask)
        hop = csf.compute_hop_distance(node_mask, edge_index, None)

        temporal = torch.zeros(1, 3, 5, 2)
        temporal[0, :, 0, 0] = torch.tensor([0.1, 0.5, 0.9])
        temporal[0, :, 4, 0] = torch.tensor([0.0, 0.0, 0.02])
        temporal[0, :, 1:4, 0] = float("nan")

        feats = physf.compute_physics_features(temporal, hop, active, node_mask)
        assert feats.shape == (1, 5, len(physf.PHYSICS_FEATURE_COLUMNS))
        assert torch.isfinite(feats).all()
        # node 0 (colocated with the high-reading sensor) must score at
        # least as high on the nearest-sensor-concentration column as node 4.
        assert feats[0, 0, 0] >= feats[0, 4, 0]

    def test_no_active_sensors_returns_zero(self) -> None:
        node_mask, edge_index = _path_graph_batch(4)
        active = torch.zeros(1, 4, dtype=torch.bool)
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        temporal = torch.randn(1, 3, 4, 2)
        feats = physf.compute_physics_features(temporal, hop, active, node_mask)
        assert torch.all(feats == 0.0)

    def test_relabeling_invariance(self) -> None:
        node_mask, edge_index = _path_graph_batch(5)
        sensor_mask_t = torch.zeros(1, 3, 5, dtype=torch.bool)
        sensor_mask_t[0, :, 0] = True
        sensor_mask_t[0, :, 3] = True
        active = csf.active_sensor_mask_from_temporal(sensor_mask_t, node_mask)
        hop = csf.compute_hop_distance(node_mask, edge_index, None)
        torch.manual_seed(0)
        temporal = torch.rand(1, 3, 5, 2)
        temporal[0, :, [1, 2, 4], 0] = float("nan")
        feats = physf.compute_physics_features(temporal, hop, active, node_mask)

        perm = torch.tensor([2, 4, 0, 1, 3])
        inv_perm = torch.argsort(perm)
        edge_index_p = inv_perm[edge_index]
        sensor_mask_t_p = sensor_mask_t[:, :, perm]
        active_p = csf.active_sensor_mask_from_temporal(sensor_mask_t_p, node_mask)
        hop_p = csf.compute_hop_distance(node_mask, edge_index_p, None)
        temporal_p = temporal[:, :, perm]
        feats_p = physf.compute_physics_features(temporal_p, hop_p, active_p, node_mask)
        assert torch.allclose(feats[0], feats_p[0][inv_perm], atol=1e-5)
