"""Small, deterministic water-distribution networks for simulations and tests.

The NetworkX and WNTR builders use the same declarative topology so synthetic
data does not silently diverge from hydraulic experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

try:  # WNTR is intentionally an optional dependency for data-only workflows.
    import wntr
except ImportError as exc:  # pragma: no cover - exercised by monkeypatch in tests
    wntr = None
    _WNTR_IMPORT_ERROR: ImportError | None = exc
else:
    _WNTR_IMPORT_ERROR = None


DIURNAL_PATTERN = (
    0.62, 0.56, 0.52, 0.50, 0.55, 0.72,
    1.00, 1.22, 1.18, 1.05, 0.98, 1.02,
    1.08, 1.04, 0.98, 0.96, 1.08, 1.28,
    1.38, 1.24, 1.08, 0.94, 0.80, 0.70,
)


@dataclass(frozen=True)
class NetworkDefinition:
    """Parameters for the canonical HydroSwarm demonstration network."""

    name: str = "hydroswarm-demo"
    demand_pattern_name: str = "diurnal"
    reservoir_head_m: float = 132.0
    tank_elevation_m: float = 101.0
    tank_initial_level_m: float = 8.0
    tank_min_level_m: float = 2.0
    tank_max_level_m: float = 14.0
    tank_diameter_m: float = 16.0


# name, x, y, elevation (m), base demand (m^3/s)
_JUNCTIONS = (
    ("J1", 1_000.0, 0.0, 96.0, 0.0080),
    ("J2", 2_000.0, 0.0, 101.0, 0.0100),
    ("J3", 3_000.0, 650.0, 107.0, 0.0065),
    ("J4", 1_750.0, 1_250.0, 103.0, 0.0090),
)

# name, start, end, length (m), diameter (m), roughness
_PIPES = (
    ("P_R1_J1", "R1", "J1", 1_000.0, 0.35, 120.0),
    ("P_J1_J2", "J1", "J2", 1_050.0, 0.30, 115.0),
    ("P_J2_J3", "J2", "J3", 1_200.0, 0.25, 110.0),
    ("P_J3_J4", "J3", "J4", 1_350.0, 0.25, 110.0),
    ("P_J4_J1", "J4", "J1", 1_100.0, 0.30, 115.0),
    ("P_J2_T1", "J2", "T1", 850.0, 0.25, 115.0),
    ("P_T1_J4", "T1", "J4", 900.0, 0.25, 115.0),
)


def build_networkx_network(definition: NetworkDefinition | None = None) -> nx.MultiDiGraph:
    """Build the canonical network as a deterministic directed multigraph."""

    spec = definition or NetworkDefinition()
    graph = nx.MultiDiGraph(name=spec.name, demand_pattern=list(DIURNAL_PATTERN))
    graph.add_node(
        "R1",
        node_type="reservoir",
        elevation_m=0.0,
        base_head_m=spec.reservoir_head_m,
        coordinates=(0.0, 0.0),
        base_demand_m3s=0.0,
    )
    for name, x, y, elevation, demand in _JUNCTIONS:
        graph.add_node(
            name,
            node_type="junction",
            elevation_m=elevation,
            coordinates=(x, y),
            base_demand_m3s=demand,
            demand_pattern=spec.demand_pattern_name,
        )
    graph.add_node(
        "T1",
        node_type="tank",
        elevation_m=spec.tank_elevation_m,
        initial_level_m=spec.tank_initial_level_m,
        min_level_m=spec.tank_min_level_m,
        max_level_m=spec.tank_max_level_m,
        diameter_m=spec.tank_diameter_m,
        coordinates=(2_650.0, 1_450.0),
        base_demand_m3s=0.0,
    )
    for name, start, end, length, diameter, roughness in _PIPES:
        graph.add_edge(
            start,
            end,
            key=name,
            name=name,
            link_type="pipe",
            length_m=length,
            diameter_m=diameter,
            roughness=roughness,
            travel_steps=max(1, round(length / 600.0)),
        )
    return graph


def build_wntr_network(definition: NetworkDefinition | None = None) -> Any:
    """Build an equivalent :class:`wntr.network.WaterNetworkModel`.

    Raises:
        ImportError: with an actionable message when the optional WNTR package
            is not installed.
    """

    if wntr is None:
        raise ImportError(
            "WNTR is required to build a hydraulic model; install the 'wntr' "
            "package or use build_networkx_network() for data-only workflows."
        ) from _WNTR_IMPORT_ERROR

    spec = definition or NetworkDefinition()
    model = wntr.network.WaterNetworkModel()
    model.name = spec.name
    model.add_pattern(spec.demand_pattern_name, list(DIURNAL_PATTERN))
    model.add_reservoir("R1", base_head=spec.reservoir_head_m, coordinates=(0.0, 0.0))
    for name, x, y, elevation, demand in _JUNCTIONS:
        model.add_junction(
            name,
            base_demand=demand,
            demand_pattern=spec.demand_pattern_name,
            elevation=elevation,
            coordinates=(x, y),
        )
    model.add_tank(
        "T1",
        elevation=spec.tank_elevation_m,
        init_level=spec.tank_initial_level_m,
        min_level=spec.tank_min_level_m,
        max_level=spec.tank_max_level_m,
        diameter=spec.tank_diameter_m,
        coordinates=(2_650.0, 1_450.0),
    )
    for name, start, end, length, diameter, roughness in _PIPES:
        model.add_pipe(
            name,
            start,
            end,
            length=length,
            diameter=diameter,
            roughness=roughness,
            minor_loss=0.0,
            initial_status="OPEN",
        )
    model.options.time.pattern_timestep = 3_600
    model.options.time.hydraulic_timestep = 3_600
    model.options.time.quality_timestep = 300
    model.options.time.duration = 24 * 3_600
    return model

