"""Node-level graph-structural features.

No graph-theoretic centrality (betweenness/closeness/degree) exists
anywhere in HydroSwarm today -- the only "centrality"-named quantity in the
repo (`demand_centrality`, `src/hydroswarm/simulation/wrapper.py`) is a
demand-share feature, not a structural one (confirmed by full-repo survey).
This module is therefore new, built directly on `networkx` (an existing
hard dependency) against the same undirected-graph export
(`WaterNetworkModel.to_graph()`) the rest of the repo already uses for
structural queries -- no new library, no reinvented convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True, slots=True)
class NodeStructuralFeatures:
    node: str
    degree: int
    betweenness_centrality: float
    closeness_centrality: float
    is_leaf: bool
    reservoir_distance_hops: int | None
    sensor_distance_hops: int | None  # min hops to any node in the incident's sensor set
    eccentricity: int | None


def compute_structural_features(
    graph: nx.Graph,
    *,
    reservoir_nodes: tuple[str, ...],
    sensor_nodes: tuple[str, ...],
) -> dict[str, NodeStructuralFeatures]:
    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    connected = nx.is_connected(graph)
    eccentricity = nx.eccentricity(graph) if connected else {}

    reservoir_distances: dict[str, int] = {}
    if reservoir_nodes:
        for node in graph.nodes:
            best = None
            for reservoir in reservoir_nodes:
                if reservoir not in graph:
                    continue
                try:
                    d = nx.shortest_path_length(graph, node, reservoir)
                except nx.NetworkXNoPath:
                    continue
                if best is None or d < best:
                    best = d
            reservoir_distances[node] = best

    sensor_distances: dict[str, int] = {}
    for node in graph.nodes:
        best = None
        for sensor in sensor_nodes:
            if sensor not in graph:
                continue
            try:
                d = nx.shortest_path_length(graph, node, sensor)
            except nx.NetworkXNoPath:
                continue
            if best is None or d < best:
                best = d
        sensor_distances[node] = best

    return {
        node: NodeStructuralFeatures(
            node=node,
            degree=degree.get(node, 0),
            betweenness_centrality=betweenness.get(node, 0.0),
            closeness_centrality=closeness.get(node, 0.0),
            is_leaf=degree.get(node, 0) <= 1,
            reservoir_distance_hops=reservoir_distances.get(node),
            sensor_distance_hops=sensor_distances.get(node),
            eccentricity=eccentricity.get(node),
        )
        for node in graph.nodes
    }


def reservoir_and_tank_nodes(network) -> tuple[str, ...]:
    names = list(network.reservoir_name_list) + list(network.tank_name_list)
    return tuple(sorted(names))


__all__ = ["NodeStructuralFeatures", "compute_structural_features", "reservoir_and_tank_nodes"]
