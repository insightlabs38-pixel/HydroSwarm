"""Hydraulically directed graph construction.

The static network describes how assets are connected.  Transport, however,
follows the sign of the current hydraulic flow.  This module materializes both
the transport graph and its diagnostic reverse without mutating the static
network, making flow reversals explicit and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable

import networkx as nx

NodeId = Hashable


@dataclass(frozen=True, slots=True)
class HydraulicLink:
    """A link in static (``start`` to ``end``) network orientation.

    Flow is expected in volume/time units and may be negative.  A negative
    value means that the current transport direction is ``end`` to ``start``.
    Length and diameter must use compatible length units.
    """

    link_id: str
    start: NodeId
    end: NodeId
    flow: float
    length: float
    diameter: float
    operable: bool = True

    def __post_init__(self) -> None:
        if not self.link_id:
            raise ValueError("link_id must not be empty")
        if self.start == self.end:
            raise ValueError("hydraulic links must connect distinct nodes")
        for name, value in (
            ("flow", self.flow),
            ("length", self.length),
            ("diameter", self.diameter),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")


@dataclass(frozen=True, slots=True)
class DirectedHydraulicGraph:
    """Paired transport and diagnostic views for one hydraulic state."""

    transport: nx.MultiDiGraph
    diagnostic: nx.MultiDiGraph

    def shortest_travel_time(self, source: NodeId, target: NodeId) -> float:
        """Return minimum directed transport time, or infinity if unreachable."""

        if source not in self.transport or target not in self.transport:
            return math.inf
        try:
            return float(
                nx.shortest_path_length(
                    self.transport, source, target, weight="travel_time"
                )
            )
        except nx.NetworkXNoPath:
            return math.inf

    def possible_sources(self, sensor: NodeId) -> set[NodeId]:
        """Return nodes reachable upstream through the diagnostic channel."""

        if sensor not in self.diagnostic:
            return set()
        return {sensor, *nx.descendants(self.diagnostic, sensor)}


def _edge_kinematics(link: HydraulicLink) -> tuple[float, float]:
    area = math.pi * (link.diameter / 2.0) ** 2
    magnitude = abs(link.flow)
    if magnitude == 0.0 or not link.operable:
        return 0.0, math.inf
    velocity = magnitude / area
    return velocity, link.length / velocity


def build_dynamic_graph(
    links: Iterable[HydraulicLink],
    *,
    nodes: Iterable[NodeId] = (),
    flow_epsilon: float = 1e-12,
) -> DirectedHydraulicGraph:
    """Build current transport and reverse-diagnostic graph views.

    Stagnant and inoperable links remain represented as isolated static nodes
    but do not create transport edges.  Parallel pipes are retained via a
    ``MultiDiGraph``.  Iteration order does not affect graph attributes.
    """

    if not math.isfinite(flow_epsilon) or flow_epsilon < 0:
        raise ValueError("flow_epsilon must be finite and non-negative")

    transport = nx.MultiDiGraph(channel="transport")
    transport.add_nodes_from(nodes)
    seen_ids: set[str] = set()

    for link in links:
        if link.link_id in seen_ids:
            raise ValueError(f"duplicate link_id: {link.link_id}")
        seen_ids.add(link.link_id)
        transport.add_nodes_from((link.start, link.end))
        if not link.operable or abs(link.flow) <= flow_epsilon:
            continue

        reversed_flow = link.flow < 0
        upstream, downstream = (
            (link.end, link.start) if reversed_flow else (link.start, link.end)
        )
        velocity, travel_time = _edge_kinematics(link)
        transport.add_edge(
            upstream,
            downstream,
            key=link.link_id,
            link_id=link.link_id,
            flow_magnitude=abs(link.flow),
            signed_static_flow=link.flow,
            velocity=velocity,
            travel_time=travel_time,
            length=link.length,
            diameter=link.diameter,
            flow_reversed=reversed_flow,
            operable=link.operable,
        )

    diagnostic = transport.reverse(copy=True)
    diagnostic.graph["channel"] = "diagnostic"
    return DirectedHydraulicGraph(transport=transport, diagnostic=diagnostic)

