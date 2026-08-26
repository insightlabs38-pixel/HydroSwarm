"""Graph-structural-encoder-v2 experiment (EXPERIMENTAL, NON-RELEASE):
label-free-ness, determinism, node-relabeling invariance, and basic
correctness of the two new candidate feature modules
(`scripts/hydrocore_v5_experimental/graph_structural_encoder_v2/
structural_features.py` / `observability_features.py`). See
docs/evaluation/experimental/GRAPH_STRUCTURAL_ENCODER_V2_PLAN.md Section 5
for the leakage/invariance requirements these tests enforce.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts" / "hydrocore_v5_experimental"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from graph_structural_encoder_v2 import (  # noqa: E402
    observability_features as obs,
)
from graph_structural_encoder_v2 import (  # noqa: E402
    structural_features as struct,
)

#: A 5-node "lollipop": triangle {0,1,2} plus a pendant tail 2-3-4.
#: Deterministic degree/dead-end/reservoir-hop structure used across
#: several tests below.
LOLLIPOP_EDGES = ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4))
LOLLIPOP_NODES = 5


def _batch_from_edges(
    edges: tuple[tuple[int, int], ...],
    node_count: int,
    *,
    non_candidate_nodes: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    node_mask = torch.zeros(1, node_count, dtype=torch.bool)
    node_mask[0, :node_count] = True
    if edges:
        source = torch.tensor([edge[0] for edge in edges], dtype=torch.long)
        target = torch.tensor([edge[1] for edge in edges], dtype=torch.long)
        edge_index = torch.stack((source, target)).unsqueeze(0)
    else:
        edge_index = torch.zeros(1, 2, 0, dtype=torch.long)
    edge_mask = torch.ones(1, edge_index.shape[-1], dtype=torch.bool)
    candidate_mask = torch.ones(1, node_count, dtype=torch.bool)
    for node in non_candidate_nodes:
        candidate_mask[0, node] = False
    return {
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_mask": edge_mask,
        "source_candidate_mask": candidate_mask,
    }


# --------------------------------------------------------------------------
# No-label-usage: forbidden by construction, not just by convention.
# --------------------------------------------------------------------------


def test_structural_features_signature_has_no_label_inputs() -> None:
    signature = inspect.signature(struct.compute_structural_features)
    forbidden = {"source_node", "source_node_mask", "target", "label", "outcome"}
    for name in signature.parameters:
        assert not any(bad in name.lower() for bad in forbidden), name


def test_observability_features_signature_has_no_label_inputs() -> None:
    signature = inspect.signature(obs.compute_observability_features)
    forbidden = {"source_node", "source_node_mask", "target", "label", "outcome"}
    for name in signature.parameters:
        assert not any(bad in name.lower() for bad in forbidden), name


def test_modules_do_not_import_training_targets() -> None:
    for module in (struct, obs):
        source = inspect.getsource(module)
        assert "targets_v2" not in source
        assert "hydroswarm.training.losses" not in source


# --------------------------------------------------------------------------
# Basic correctness on a known, hand-checkable topology.
# --------------------------------------------------------------------------


def test_structural_features_degree_and_dead_end_on_lollipop() -> None:
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES, non_candidate_nodes=(0,))
    features = struct.compute_structural_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], batch["source_candidate_mask"]
    )
    columns = {name: index for index, name in enumerate(struct.NODE_STRUCTURAL_COLUMNS)}
    degree_col = features[0, :, columns["degree_normalized"]]
    # raw degrees: node0=2, node1=2, node2=3, node3=2, node4=1; normalized by (n-1)=4
    expected_degree = torch.tensor([2, 2, 3, 2, 1], dtype=torch.float32) / 4.0
    assert torch.allclose(degree_col, expected_degree, atol=1e-6)

    dead_end_col = features[0, :, columns["hop_to_dead_end_normalized"]]
    assert dead_end_col[4].item() == pytest.approx(0.0)  # node 4 is itself a dead end
    # node 0 is 3 hops from the only dead end (node 4): 0-2-3-4
    diameter = 3  # 0/1 to 4 is the longest shortest path in this graph
    assert dead_end_col[0].item() == pytest.approx(3 / diameter)

    reservoir_col = features[0, :, columns["hop_to_reservoir_normalized"]]
    assert reservoir_col[0].item() == pytest.approx(0.0)  # node 0 IS the reservoir/tank here
    assert reservoir_col[4].item() == pytest.approx(3 / diameter)


def test_structural_features_zero_at_padded_positions() -> None:
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES)
    padded_node_mask = torch.zeros(1, 8, dtype=torch.bool)
    padded_node_mask[0, :LOLLIPOP_NODES] = True
    padded_edge_index = torch.zeros(1, 2, batch["edge_index"].shape[-1], dtype=torch.long)
    padded_edge_index[0] = batch["edge_index"][0]
    features = struct.compute_structural_features(
        padded_node_mask, padded_edge_index, batch["edge_mask"], None
    )
    assert torch.equal(features[0, LOLLIPOP_NODES:], torch.zeros(3, len(struct.NODE_STRUCTURAL_COLUMNS)))


def test_observability_no_active_sensors_saturates() -> None:
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES)
    sensor_mask = torch.zeros(1, 4, LOLLIPOP_NODES, dtype=torch.bool)  # nobody reporting
    features = obs.compute_observability_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], sensor_mask
    )
    columns = {name: index for index, name in enumerate(obs.NODE_OBSERVABILITY_COLUMNS)}
    assert torch.all(features[0, :, columns["hop_to_nearest_sensor_normalized"]] == 1.0)
    assert torch.all(features[0, :, columns["fraction_sensors_within_1hop"]] == 0.0)
    assert torch.all(features[0, :, columns["local_sensor_coverage_density"]] == 0.0)


def test_observability_nearest_hop_matches_hand_computed_distance() -> None:
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES)
    sensor_mask = torch.zeros(1, 2, LOLLIPOP_NODES, dtype=torch.bool)
    sensor_mask[0, 0, 4] = True  # only node 4 ever reports
    features = obs.compute_observability_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], sensor_mask
    )
    columns = {name: index for index, name in enumerate(obs.NODE_OBSERVABILITY_COLUMNS)}
    diameter = 3
    # hops to node 4 (the only active sensor): 0->3 (via 0-2-3-4), 1->3
    # (via 1-2-3-4), 2->2 (2-3-4), 3->1, 4->0.
    expected_hops = torch.tensor([3, 3, 2, 1, 0], dtype=torch.float32) / diameter
    actual = features[0, :, columns["hop_to_nearest_sensor_normalized"]]
    assert torch.allclose(actual, expected_hops, atol=1e-6)
    # node 4 is the sensor itself: within both the 1-hop and (vacuously) the
    # 2-hop bin.
    assert features[0, 4, columns["fraction_sensors_within_1hop"]].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Node-relabeling / permutation invariance (plan doc Section 5).
# --------------------------------------------------------------------------


def _permute_batch(batch: dict[str, torch.Tensor], permutation: list[int], sensor_mask: torch.Tensor | None = None):
    node_count = len(permutation)
    inverse = [0] * node_count
    for new_index, old_index in enumerate(permutation):
        inverse[old_index] = new_index
    remap = torch.tensor(inverse, dtype=torch.long)

    edge_index = batch["edge_index"].clone()
    edge_index[0] = remap[edge_index[0]]

    node_mask = torch.zeros_like(batch["node_mask"])
    candidate_mask = torch.zeros_like(batch["source_candidate_mask"]) if batch["source_candidate_mask"] is not None else None
    for old_index in range(node_count):
        new_index = inverse[old_index]
        node_mask[0, new_index] = batch["node_mask"][0, old_index]
        if candidate_mask is not None:
            candidate_mask[0, new_index] = batch["source_candidate_mask"][0, old_index]

    permuted = dict(batch)
    permuted["edge_index"] = edge_index
    permuted["node_mask"] = node_mask
    permuted["source_candidate_mask"] = candidate_mask

    permuted_sensor_mask = None
    if sensor_mask is not None:
        permuted_sensor_mask = torch.zeros_like(sensor_mask)
        for old_index in range(node_count):
            permuted_sensor_mask[0, :, inverse[old_index]] = sensor_mask[0, :, old_index]
    return permuted, permuted_sensor_mask, inverse


@pytest.mark.parametrize(
    "permutation",
    [
        [4, 3, 2, 1, 0],
        [1, 0, 3, 4, 2],
        [2, 4, 0, 3, 1],
    ],
)
def test_structural_features_are_relabeling_invariant(permutation: list[int]) -> None:
    struct._compute_topology_features.cache_clear()
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES, non_candidate_nodes=(0,))
    original = struct.compute_structural_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], batch["source_candidate_mask"]
    )
    permuted_batch, _, inverse = _permute_batch(batch, permutation)
    permuted = struct.compute_structural_features(
        permuted_batch["node_mask"],
        permuted_batch["edge_index"],
        permuted_batch["edge_mask"],
        permuted_batch["source_candidate_mask"],
    )
    for old_index in range(LOLLIPOP_NODES):
        new_index = inverse[old_index]
        assert torch.allclose(original[0, old_index], permuted[0, new_index], atol=1e-6), (old_index, new_index)


@pytest.mark.parametrize(
    "permutation",
    [
        [4, 3, 2, 1, 0],
        [1, 0, 3, 4, 2],
    ],
)
def test_observability_features_are_relabeling_invariant(permutation: list[int]) -> None:
    obs._topology_hop_matrix.cache_clear()
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES)
    sensor_mask = torch.zeros(1, 2, LOLLIPOP_NODES, dtype=torch.bool)
    sensor_mask[0, 0, 4] = True
    sensor_mask[0, 1, 1] = True
    original = obs.compute_observability_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], sensor_mask
    )
    permuted_batch, permuted_sensor_mask, inverse = _permute_batch(batch, permutation, sensor_mask)
    permuted = obs.compute_observability_features(
        permuted_batch["node_mask"],
        permuted_batch["edge_index"],
        permuted_batch["edge_mask"],
        permuted_sensor_mask,
    )
    for old_index in range(LOLLIPOP_NODES):
        new_index = inverse[old_index]
        assert torch.allclose(original[0, old_index], permuted[0, new_index], atol=1e-6), (old_index, new_index)


def test_features_are_deterministic_across_repeated_calls() -> None:
    struct._compute_topology_features.cache_clear()
    batch = _batch_from_edges(LOLLIPOP_EDGES, LOLLIPOP_NODES, non_candidate_nodes=(0,))
    first = struct.compute_structural_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], batch["source_candidate_mask"]
    )
    second = struct.compute_structural_features(
        batch["node_mask"], batch["edge_index"], batch["edge_mask"], batch["source_candidate_mask"]
    )
    assert torch.equal(first, second)
