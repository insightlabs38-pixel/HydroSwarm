"""Pure, deterministic, label-free candidate<->sensor relational features
for the candidate-conditioned-localizer-v1 experiment.

EXPERIMENTAL / NON-RELEASE. Standalone batch-augmentation module, same
convention as `exp/graph-structural-encoder-v2`'s `structural_features.py`/
`observability_features.py` (not imported -- that branch is not merged into
this one; the all-pairs-hop-distance/topology-caching approach below
reimplements the same well-tested idea rather than re-deriving it from
scratch). Every function reads only `node_mask`/`edge_index`/`edge_mask`/
`source_candidate_mask`/`sensor_mask` -- structural/observability
information available at inference time. Nothing here reads `source_node`,
`source_node_mask`, or any other target/evaluation-outcome tensor.

Produces the three `HydroBatch` fields
`CandidateConditionedLocalizer.forward` needs
(`candidate_hop_distance`, `active_sensor_mask_nodes`,
`candidate_structural_features`) plus the arm-labeling columns
(`NODE_STRUCTURAL_COLUMNS`) used for subgroup reporting.
"""

from __future__ import annotations

import functools

import networkx as nx
import torch
from torch import Tensor

from hydroswarm.model.candidate_localizer import UNREACHABLE_HOP_SENTINEL

NODE_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "degree_normalized",
    "betweenness_centrality",
    "closeness_centrality",
    "hop_to_nearest_sensor_normalized",
    "mean_hop_to_sensors_normalized",
    "fraction_sensors_within_2hop",
)


def _edges_from_row(edge_index_row: Tensor, edge_mask_row: Tensor | None, edge_count: int) -> tuple[tuple[int, int], ...]:
    if edge_count == 0:
        return ()
    real = edge_index_row[:, :edge_count]
    if edge_mask_row is not None:
        valid = edge_mask_row[:edge_count].bool()
        real = real[:, valid]
    return tuple(zip(real[0].tolist(), real[1].tolist()))


@functools.lru_cache(maxsize=256)
def _topology_hop_matrix(node_count: int, edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    """All-pairs unweighted shortest-path hop counts for one topology,
    cached per (node_count, edges) since this corpus reuses a small handful
    of topologies across thousands of examples that differ only in
    demand/sensor/incident state. `UNREACHABLE_HOP_SENTINEL` (-1) for any
    unreachable pair -- never `inf`/a large finite magic number, matching
    `CandidateConditionedLocalizer`'s own convention exactly so no separate
    "is this really unreachable or just far" translation is needed
    downstream.
    """

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    return tuple(
        tuple(lengths.get(source, {}).get(target, UNREACHABLE_HOP_SENTINEL) for target in range(node_count))
        for source in range(node_count)
    )


@functools.lru_cache(maxsize=256)
def _topology_centrality(node_count: int, edges: tuple[tuple[int, int], ...]) -> tuple[tuple[float, float, float], ...]:
    """(degree_normalized, betweenness, closeness) per node, cached the
    same way as `_topology_hop_matrix`."""

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)
    if graph.number_of_edges() == 0:
        return tuple((0.0, 0.0, 0.0) for _ in range(node_count))
    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    return tuple(
        (
            degree.get(node, 0) / max(node_count - 1, 1),
            betweenness.get(node, 0.0),
            closeness.get(node, 0.0),
        )
        for node in range(node_count)
    )


def _edges_for_index(
    index: int,
    node_mask_cpu: Tensor,
    edge_index_cpu: Tensor,
    edge_mask_cpu: Tensor | None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    node_count = int(node_mask_cpu[index].sum().item())
    edge_count = edge_index_cpu.shape[-1]
    edges = _edges_from_row(edge_index_cpu[index], edge_mask_cpu[index] if edge_mask_cpu is not None else None, edge_count)
    edges = tuple((s, t) for s, t in edges if s < node_count and t < node_count)
    return node_count, edges


def active_sensor_mask_from_temporal(sensor_mask: Tensor | None, node_mask: Tensor) -> Tensor:
    """`[batch, time, nodes]` (or all-True fallback) -> `[batch, nodes]`
    bool, True iff that node has at least one valid reading anywhere in the
    window -- the same "active sensor" definition
    `hydroswarm.preprocessing.builder.HydraulicFeatureBuilder` already uses
    for `distance_to_sensor`."""

    if sensor_mask is None:
        return torch.zeros_like(node_mask, dtype=torch.bool)
    mask = sensor_mask.bool()
    if mask.ndim == node_mask.ndim:
        return mask & node_mask.bool()
    return mask.any(dim=1) & node_mask.bool()


def compute_hop_distance(node_mask: Tensor, edge_index: Tensor | None, edge_mask: Tensor | None) -> Tensor:
    """Returns `[batch, nodes, nodes]` long, `UNREACHABLE_HOP_SENTINEL` (-1)
    at any pair involving a padded node or with no path between them."""

    batch, nodes = node_mask.shape
    out = torch.full((batch, nodes, nodes), UNREACHABLE_HOP_SENTINEL, dtype=torch.long)
    if edge_index is None:
        return out
    edge_index_cpu = edge_index.detach().to("cpu", dtype=torch.long)
    edge_mask_cpu = edge_mask.detach().to("cpu").bool() if edge_mask is not None else None
    node_mask_cpu = node_mask.detach().to("cpu").bool()
    for index in range(batch):
        node_count, edges = _edges_for_index(index, node_mask_cpu, edge_index_cpu, edge_mask_cpu)
        if node_count == 0:
            continue
        matrix = _topology_hop_matrix(node_count, edges)
        out[index, :node_count, :node_count] = torch.tensor(matrix, dtype=torch.long)
    return out


def compute_structural_features(
    node_mask: Tensor,
    edge_index: Tensor | None,
    edge_mask: Tensor | None,
    active_sensor_mask_nodes: Tensor,
    hop_distance: Tensor | None = None,
) -> Tensor:
    """Returns `[batch, nodes, len(NODE_STRUCTURAL_COLUMNS)]`, zero at
    padded node positions. Label-free: only reads structural/observability
    tensors, never a target."""

    batch, nodes = node_mask.shape
    out = torch.zeros(batch, nodes, len(NODE_STRUCTURAL_COLUMNS), dtype=torch.float32)
    if edge_index is None:
        return out
    edge_index_cpu = edge_index.detach().to("cpu", dtype=torch.long)
    edge_mask_cpu = edge_mask.detach().to("cpu").bool() if edge_mask is not None else None
    node_mask_cpu = node_mask.detach().to("cpu").bool()
    sensor_cpu = active_sensor_mask_nodes.detach().to("cpu").bool()
    hop_cpu = hop_distance.detach().to("cpu") if hop_distance is not None else None

    for index in range(batch):
        node_count, edges = _edges_for_index(index, node_mask_cpu, edge_index_cpu, edge_mask_cpu)
        if node_count == 0:
            continue
        centrality = _topology_centrality(node_count, edges)
        hop_matrix = hop_cpu[index] if hop_cpu is not None else torch.tensor(_topology_hop_matrix(node_count, edges))
        sensors = [n for n in range(node_count) if bool(sensor_cpu[index, n])]
        diameter = max(
            (value for row in hop_matrix[:node_count, :node_count].tolist() for value in row if value >= 0),
            default=1,
        )
        diameter = max(diameter, 1)
        for node in range(node_count):
            deg, bet, clo = centrality[node]
            if sensors:
                distances = [int(hop_matrix[node, s].item()) for s in sensors]
                finite = [d for d in distances if d >= 0]
                nearest = min(finite) / diameter if finite else 1.0
                mean_d = (sum(finite) / len(finite)) / diameter if finite else 1.0
                within_2 = sum(1 for d in distances if 0 <= d <= 2) / len(sensors)
            else:
                nearest = mean_d = 1.0
                within_2 = 0.0
            out[index, node] = torch.tensor(
                [deg, bet, clo, min(nearest, 1.0), min(mean_d, 1.0), within_2], dtype=torch.float32
            )
    return out


__all__ = [
    "NODE_STRUCTURAL_COLUMNS",
    "active_sensor_mask_from_temporal",
    "compute_hop_distance",
    "compute_structural_features",
]
