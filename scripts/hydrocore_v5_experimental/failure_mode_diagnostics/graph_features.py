"""Pure, read-only graph-structural feature derivation for failure-mode
diagnostics (branch exp/failure-mode-diagnostics).

EXPERIMENTAL / NON-RELEASE / DIAGNOSTIC-ONLY. Nothing here writes to,
mutates, or re-derives labels from any frozen ``data/locked/`` artifact --
it only reads already-materialized ``.inp`` topology files and per-example
``(node_ids, edge_ids)`` metadata that HydroSwarm's own preprocessing
already attaches to every example, and computes standard graph-theoretic
descriptors (degree, centrality, distance, density) with networkx. These
are DERIVED diagnostics, not directly recorded fields -- every consumer of
this module should tag them as such (see build_m11_6_diagnostic_table.py).

Networks in this corpus are tiny (6-13 nodes; see dataset-report.json's
topology_node_counts and the M11.6 design freeze's junction_count range
9-12), so exact centrality/diameter computation is cheap and exact --
no approximation is used or needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx


@dataclass(frozen=True, slots=True)
class GraphLevelFeatures:
    node_count: int
    edge_count: int
    density: float
    diameter: int | None  # None if disconnected (should not happen; connectivity is a generator invariant)
    reservoir_count: int
    dead_end_count: int  # nodes with degree 1 (excluding reservoirs/tanks boundary nodes is not assumed)


@dataclass(frozen=True, slots=True)
class SourceNodeFeatures:
    degree: int
    betweenness_centrality: float
    closeness_centrality: float
    normalized_graph_position: float  # closeness rescaled to [0,1] within this graph's own min/max
    hops_to_reservoir: int | None
    hops_to_nearest_dead_end: int  # 0 if the source node is itself a dead end
    is_boundary_node: bool  # degree == 1
    eccentricity: int


def build_graph(node_ids: list[str], edge_ids: list[tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edge_ids)
    return graph


def graph_level_features(graph: nx.Graph, *, reservoir_ids: list[str]) -> GraphLevelFeatures:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    density = nx.density(graph)
    diameter = nx.diameter(graph) if nx.is_connected(graph) else None
    dead_ends = sum(1 for _, degree in graph.degree() if degree == 1)
    return GraphLevelFeatures(
        node_count=node_count,
        edge_count=edge_count,
        density=density,
        diameter=diameter,
        reservoir_count=len(reservoir_ids),
        dead_end_count=dead_ends,
    )


def source_node_features(
    graph: nx.Graph, source: str, *, reservoir_ids: list[str]
) -> SourceNodeFeatures | None:
    """None if ``source`` is not present in ``graph`` (should not happen for
    a well-formed example; callers must not silently substitute a default)."""

    if source not in graph:
        return None
    degree = graph.degree(source)
    betweenness = nx.betweenness_centrality(graph).get(source, 0.0)
    closeness_all = nx.closeness_centrality(graph)
    closeness = closeness_all[source]
    values = list(closeness_all.values())
    low, high = min(values), max(values)
    normalized_position = (closeness - low) / (high - low) if high > low else 0.5
    dead_ends = [node for node, degree_value in graph.degree() if degree_value == 1]
    hops_to_dead_end = (
        0
        if degree == 1
        else min(nx.shortest_path_length(graph, source, node) for node in dead_ends)
        if dead_ends
        else -1
    )
    reachable_reservoirs = [node for node in reservoir_ids if node in graph]
    hops_to_reservoir = (
        min(nx.shortest_path_length(graph, source, node) for node in reachable_reservoirs)
        if reachable_reservoirs
        else None
    )
    eccentricity = nx.eccentricity(graph, v=source) if nx.is_connected(graph) else -1
    return SourceNodeFeatures(
        degree=degree,
        betweenness_centrality=betweenness,
        closeness_centrality=closeness,
        normalized_graph_position=normalized_position,
        hops_to_reservoir=hops_to_reservoir,
        hops_to_nearest_dead_end=hops_to_dead_end,
        is_boundary_node=(degree == 1),
        eccentricity=eccentricity,
    )


def load_inp_graph(path: str | Path) -> tuple[nx.Graph, list[str]]:
    """Parse a ``.inp`` file with wntr (already a repository dependency,
    used elsewhere for hydraulic simulation) and return its undirected
    junction/reservoir/tank connectivity graph plus its reservoir node IDs.
    Read-only: never calls wntr's hydraulic solver."""

    import wntr  # local import: heavy optional dependency, only needed here

    model = wntr.network.WaterNetworkModel(str(path))
    graph = nx.Graph()
    graph.add_nodes_from(model.node_name_list)
    for link_name in model.link_name_list:
        link = model.get_link(link_name)
        graph.add_edge(link.start_node_name, link.end_node_name)
    return graph, list(model.reservoir_name_list)


def features_to_json(level: GraphLevelFeatures, source: SourceNodeFeatures | None) -> dict:
    payload = {"graph": asdict(level)}
    payload["source"] = asdict(source) if source is not None else None
    return payload
