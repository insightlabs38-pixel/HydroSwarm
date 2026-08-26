"""Pure, deterministic, label-free per-candidate sensor-observability
features (Arm C -- OBSERVABILITY/DISTANCE) for the
graph-structural-encoder-v2 experiment.

EXPERIMENTAL / NON-RELEASE. Standalone batch-augmentation module, same
convention as `structural_features.py` in this directory (see this
experiment's plan doc, Section 0/2/5).

"Active sensor" here means the same thing
`hydroswarm.preprocessing.builder.HydraulicFeatureBuilder.build` already
means by it when it derives the existing `distance_to_sensor` node-feature
column (`builder.py`'s own `sensor_distance = _distance(graph, sensors)`
where `sensors = {item.node_id for item in sensor_series}`): a node with at
least one non-missing temporal reading in the current incident window --
recovered here from `sensor_mask` (`[batch, time, nodes]`, already computed
and attached to every batch by `pad_graph_batch`/`_example_to_graph_sample`)
via `sensor_mask.any(dim=time)`, not re-derived from raw sensor placement,
so this module never disagrees with the model's own existing notion of
"sensor." This is a genuinely different signal from the existing
`distance_to_sensor` column, though: that column is a pipe-length-weighted
Dijkstra distance, globally normalized; every feature below is an unweighted
hop-count, normalized per-graph (diameter-relative) -- see the plan doc's
Section 2 for why that distinction matters (the prior pilot's negative
result was specifically about renormalizing an already-existing scalar, not
about adding new derived signal).

Only reads `sensor_mask`/`edge_index`/`edge_mask`/`node_mask`. Never reads
`source_node`, `source_node_mask`, or any other target/evaluation-outcome
tensor.
"""

from __future__ import annotations

import functools

import networkx as nx
import torch
from torch import Tensor

from .structural_features import _edges_from_row

NODE_OBSERVABILITY_COLUMNS: tuple[str, ...] = (
    "hop_to_nearest_sensor_normalized",
    "mean_hop_to_sensors_normalized",
    "max_hop_to_sensors_normalized",
    "fraction_sensors_within_1hop",
    "fraction_sensors_within_2hop",
    "fraction_sensors_within_3plus_hop",
    "local_sensor_coverage_density",
)

#: Sentinel hop distance (in graph hops) used for a candidate/sensor pair
#: with no path between them or when an example has zero active sensors --
#: deliberately `node_count` (always >= diameter + 1 for a connected graph
#: of that size), never `inf`/`nan`, so downstream normalization stays
#: finite without a special-case branch.
def _unreachable_sentinel(node_count: int) -> int:
    return max(node_count, 1)


@functools.lru_cache(maxsize=256)
def _topology_hop_matrix(node_count: int, edges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], ...]:
    """All-pairs unweighted shortest-path hop counts for one topology,
    cached per (node_count, edges) exactly like
    `structural_features._compute_topology_features` -- this corpus reuses
    the same handful of topologies across thousands of examples that only
    differ in which sensors are currently active, not in graph structure."""

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)
    sentinel = _unreachable_sentinel(node_count)
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    return tuple(
        tuple(lengths.get(source, {}).get(target, sentinel) for target in range(node_count))
        for source in range(node_count)
    )


def _diameter_and_radius(hop_matrix: tuple[tuple[int, ...], ...], node_count: int) -> tuple[int, int]:
    sentinel = _unreachable_sentinel(node_count)
    finite = [value for row in hop_matrix for value in row if value < sentinel]
    diameter = max(finite, default=1)
    diameter = max(diameter, 1)
    radius = max(diameter // 2, 1)
    return diameter, radius


def compute_observability_features(
    node_mask: Tensor,
    edge_index: Tensor | None,
    edge_mask: Tensor | None,
    sensor_mask: Tensor | None,
) -> Tensor:
    """Returns `[batch, nodes, len(NODE_OBSERVABILITY_COLUMNS)]`, zero at
    padded/invalid node positions. `sensor_mask` is `[batch, time, nodes]`
    (the same tensor `TemporalEncoder` already consumes); a node counts as
    an active sensor for this example iff it has at least one valid
    (non-missing) reading anywhere in the window."""

    batch, nodes = node_mask.shape
    out = torch.zeros(batch, nodes, len(NODE_OBSERVABILITY_COLUMNS), dtype=torch.float32)
    if edge_index is None or sensor_mask is None:
        return out
    edge_index_cpu = edge_index.detach().to("cpu", dtype=torch.long)
    edge_mask_cpu = edge_mask.detach().to("cpu").bool() if edge_mask is not None else None
    node_mask_cpu = node_mask.detach().to("cpu").bool()
    active_sensor_cpu = sensor_mask.detach().to("cpu").bool().any(dim=1)  # [batch, nodes]

    for index in range(batch):
        node_count = int(node_mask_cpu[index].sum().item())
        if node_count == 0:
            continue
        edge_count = edge_index_cpu.shape[-1]
        edges = _edges_from_row(
            edge_index_cpu[index], edge_mask_cpu[index] if edge_mask_cpu is not None else None, edge_count
        )
        edges = tuple((source, target) for source, target in edges if source < node_count and target < node_count)
        hop_matrix = _topology_hop_matrix(node_count, edges)
        diameter, radius = _diameter_and_radius(hop_matrix, node_count)
        sentinel = _unreachable_sentinel(node_count)

        sensors = [node for node in range(node_count) if bool(active_sensor_cpu[index, node])]
        for node in range(node_count):
            if not sensors:
                # No active sensor anywhere in this example: maximally
                # unobserved, not zero/undefined -- normalized distance
                # saturates at 1.0 (sentinel/diameter, clamped), all
                # coverage fractions are 0.
                out[index, node] = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                continue
            distances = [hop_matrix[node][sensor] for sensor in sensors]
            finite_distances = [distance for distance in distances if distance < sentinel]
            if not finite_distances:
                nearest = mean_distance = max_distance = float(diameter)
            else:
                nearest = min(finite_distances)
                mean_distance = sum(finite_distances) / len(finite_distances)
                max_distance = max(finite_distances)
            nearest_norm = min(nearest / diameter, 1.0)
            mean_norm = min(mean_distance / diameter, 1.0)
            max_norm = min(max_distance / diameter, 1.0)
            within_1 = sum(1 for distance in distances if distance <= 1) / len(sensors)
            within_2 = sum(1 for distance in distances if distance <= 2) / len(sensors)
            within_3plus = sum(1 for distance in distances if distance >= 3) / len(sensors)
            neighborhood = [other for other in range(node_count) if hop_matrix[node][other] <= radius]
            sensors_in_neighborhood = sum(1 for other in neighborhood if other in sensors)
            coverage_density = sensors_in_neighborhood / max(len(neighborhood), 1)
            out[index, node] = torch.tensor(
                [nearest_norm, mean_norm, max_norm, within_1, within_2, within_3plus, coverage_density]
            )
    return out
