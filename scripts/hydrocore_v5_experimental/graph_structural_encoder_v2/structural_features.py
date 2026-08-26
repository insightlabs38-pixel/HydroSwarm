"""Pure, deterministic, label-free per-candidate graph-structural features
(Arm B -- CENTRALITY) for the graph-structural-encoder-v2 experiment.

EXPERIMENTAL / NON-RELEASE. Not imported by any production module
(`hydroswarm.model`, `hydroswarm.preprocessing`, `hydroswarm.training`) --
this is a standalone batch-augmentation module in the same spirit as
`topology_normalization.py`'s `augment_batch` from `exp/failure-mode-
diagnostics` (not reused directly: see this experiment's plan doc,
`docs/evaluation/experimental/GRAPH_STRUCTURAL_ENCODER_V2_PLAN.md`,
Section 0).

Every function here takes only `edge_index`/`edge_mask`/`node_mask`/
`source_candidate_mask` -- structural, network-only information available
at inference time. Nothing here reads `source_node`, `source_node_mask`,
any other target tensor, or any evaluation outcome. Nothing here reads
sensor data either (that is Arm C, `observability_features.py`) -- this
module answers "where is this candidate in the network," never "what has
been observed."

Definitions mirror `exp/failure-mode-diagnostics`'s own
`graph_features.py` (betweenness/closeness centrality, hop-distance,
normalized graph position) for direct comparability with that branch's
diagnostic findings, generalized here to every candidate node in a batch
(not just the one labeled true source that diagnostic module computed).

Networks in this corpus are tiny (6-9 nodes,
`data/learning-v2/cycle-b2/dataset-report.json`'s own
`topology_node_counts`), so exact networkx computation is cheap.
`pad_graph_batch` (`hydroswarm.preprocessing.batching`) always places a
sample's real nodes at padded positions `[0, node_count)` and its real
edges at padded columns `[0, edge_count)` -- this module relies on that
existing, tested invariant rather than re-deriving a node index mapping.
"""

from __future__ import annotations

import functools

import networkx as nx
import torch
from torch import Tensor

#: Column order of the tensor `compute_structural_features` returns.
NODE_STRUCTURAL_COLUMNS: tuple[str, ...] = (
    "degree_normalized",
    "betweenness_centrality",
    "closeness_centrality",
    "hop_to_reservoir_normalized",
    "hop_to_dead_end_normalized",
    "normalized_graph_position",
)


def _edges_from_row(edge_index_row: Tensor, edge_mask_row: Tensor | None, edge_count: int) -> tuple[tuple[int, int], ...]:
    if edge_count == 0:
        return ()
    real = edge_index_row[:, :edge_count]
    if edge_mask_row is not None:
        valid = edge_mask_row[:edge_count].bool()
        real = real[:, valid]
    source = real[0].tolist()
    target = real[1].tolist()
    return tuple(zip(source, target))


@functools.lru_cache(maxsize=256)
def _compute_topology_features(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    non_candidate_nodes: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    """Pure function of a hashable (node_count, edges, non_candidate_nodes)
    key -- cached because this corpus repeats the same handful of
    topologies (`golden-reference`/`branched-loop`/`loop-grid`/
    `coastal-branch`) across thousands of examples that differ only in
    demand/sensor/incident state, not in graph structure itself. Returns
    one row of `NODE_STRUCTURAL_COLUMNS` per node index in `[0,
    node_count)`."""

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)

    degree = dict(graph.degree())
    rows = [[0.0] * len(NODE_STRUCTURAL_COLUMNS) for _ in range(node_count)]
    if graph.number_of_edges() == 0:
        return tuple(tuple(row) for row in rows)

    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    closeness_values = list(closeness.values())
    low, high = min(closeness_values), max(closeness_values)
    dead_ends = [node for node, value in degree.items() if value == 1]

    if graph.number_of_nodes() > 1 and nx.is_connected(graph):
        diameter = max(nx.diameter(graph), 1)
    else:
        component_diameters = [
            nx.diameter(graph.subgraph(component)) if len(component) > 1 else 1
            for component in nx.connected_components(graph)
        ]
        diameter = max(component_diameters, default=1)
        diameter = max(diameter, 1)

    def _min_hop_to(node: int, targets: list[int]) -> float:
        reachable = [target for target in targets if target != node and nx.has_path(graph, node, target)]
        if not reachable:
            return float(diameter)
        return float(min(nx.shortest_path_length(graph, node, target) for target in reachable))

    for node in range(node_count):
        deg_norm = degree.get(node, 0) / max(node_count - 1, 1)
        bet = betweenness.get(node, 0.0)
        clo = closeness.get(node, 0.0)
        position = (clo - low) / (high - low) if high > low else 0.5
        hop_to_reservoir = (
            0.0 if node in non_candidate_nodes else _min_hop_to(node, list(non_candidate_nodes))
        )
        hop_to_reservoir_norm = hop_to_reservoir / diameter
        if degree.get(node, 0) == 1:
            hop_to_dead_end = 0.0
        else:
            hop_to_dead_end = _min_hop_to(node, dead_ends)
        hop_to_dead_end_norm = hop_to_dead_end / diameter
        rows[node] = [deg_norm, bet, clo, hop_to_reservoir_norm, hop_to_dead_end_norm, position]
    return tuple(tuple(row) for row in rows)


def compute_structural_features(
    node_mask: Tensor,
    edge_index: Tensor | None,
    edge_mask: Tensor | None,
    source_candidate_mask: Tensor | None = None,
) -> Tensor:
    """Returns `[batch, nodes, len(NODE_STRUCTURAL_COLUMNS)]`, zero at
    padded/invalid node positions. `source_candidate_mask` (True for
    junction candidates, False for reservoir/tank nodes -- the same mask
    `hydroswarm.preprocessing.builder.HydraulicFeatureBuilder.build` already
    attaches to every batch) is used only to identify which nodes are
    reservoirs/tanks for the hop-to-reservoir feature; never to read a
    label."""

    batch, nodes = node_mask.shape
    out = torch.zeros(batch, nodes, len(NODE_STRUCTURAL_COLUMNS), dtype=torch.float32)
    if edge_index is None:
        return out
    edge_index_cpu = edge_index.detach().to("cpu", dtype=torch.long)
    edge_mask_cpu = edge_mask.detach().to("cpu").bool() if edge_mask is not None else None
    node_mask_cpu = node_mask.detach().to("cpu").bool()
    candidate_cpu = source_candidate_mask.detach().to("cpu").bool() if source_candidate_mask is not None else None

    for index in range(batch):
        node_count = int(node_mask_cpu[index].sum().item())
        if node_count == 0:
            continue
        edge_count = edge_index_cpu.shape[-1]
        edges = _edges_from_row(
            edge_index_cpu[index], edge_mask_cpu[index] if edge_mask_cpu is not None else None, edge_count
        )
        # Drop any edge referencing a padded node position -- defensive only;
        # pad_graph_batch never produces one, but this keeps the cache key
        # (and the resulting graph) well-formed if a caller ever passes a
        # hand-built batch that does not respect that invariant.
        edges = tuple((source, target) for source, target in edges if source < node_count and target < node_count)
        if candidate_cpu is not None:
            non_candidates = tuple(
                sorted(
                    node
                    for node in range(node_count)
                    if not bool(candidate_cpu[index, node])
                )
            )
        else:
            non_candidates = ()
        rows = _compute_topology_features(node_count, edges, non_candidates)
        out[index, :node_count] = torch.tensor(rows, dtype=torch.float32)
    return out
