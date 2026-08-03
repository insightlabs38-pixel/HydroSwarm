"""Reproducible contamination and imperfect-sensor time-series generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from hydroswarm.simulation.network import DIURNAL_PATTERN


@dataclass(frozen=True)
class SyntheticConfig:
    """Controls for a deterministic synthetic contamination experiment."""

    seed: int = 2026
    periods: int = 96
    step_minutes: int = 15
    start: str = "2026-01-01T00:00:00Z"
    source_node: str = "J1"
    contamination_start_step: int = 16
    contamination_duration_steps: int = 20
    source_concentration: float = 1.0
    decay_per_step: float = 0.035
    sensor_noise_std: float = 0.012
    drift_per_step: float = 0.00015
    missing_probability: float = 0.06
    sensor_delay_steps: int = 1
    flow_reversal_period_steps: int = 32
    flow_reversal_duration_steps: int = 4

    def __post_init__(self) -> None:
        if self.periods <= 0 or self.step_minutes <= 0:
            raise ValueError("periods and step_minutes must be positive")
        if self.contamination_start_step < 0 or self.contamination_duration_steps <= 0:
            raise ValueError("contamination timing must be non-negative with positive duration")
        if self.sensor_noise_std < 0 or self.drift_per_step < 0:
            raise ValueError("noise and drift must be non-negative")
        if not 0.0 <= self.missing_probability <= 1.0:
            raise ValueError("missing_probability must be between 0 and 1")
        if self.sensor_delay_steps < 0:
            raise ValueError("sensor_delay_steps must be non-negative")
        if self.flow_reversal_period_steps <= 0 or self.flow_reversal_duration_steps < 0:
            raise ValueError("flow-reversal period must be positive and duration non-negative")


@dataclass(frozen=True)
class SyntheticDataset:
    """Aligned truth, observation, and data-quality matrices."""

    contamination: pd.DataFrame
    sensor_readings: pd.DataFrame
    demand: pd.DataFrame
    noise: pd.DataFrame
    drift: pd.DataFrame
    missing_mask: pd.DataFrame
    flow_reversal_mask: pd.DataFrame
    transport_delay_steps: pd.Series

    def to_long_frame(self) -> pd.DataFrame:
        """Return one tidy record per timestamp and sensor node."""

        index_names = ["timestamp", "node_id"]
        pieces = {
            "true_concentration": self.contamination.stack(future_stack=True),
            "sensor_reading": self.sensor_readings.stack(future_stack=True),
            "demand_m3s": self.demand.stack(future_stack=True),
            "noise": self.noise.stack(future_stack=True),
            "drift": self.drift.stack(future_stack=True),
            "is_missing": self.missing_mask.stack(future_stack=True),
            "flow_reversal": self.flow_reversal_mask.stack(future_stack=True),
        }
        frame = pd.DataFrame(pieces)
        frame.index.names = index_names
        frame = frame.reset_index()
        frame["transport_delay_steps"] = frame["node_id"].map(self.transport_delay_steps)
        return frame


def _as_graph(network: Any) -> nx.MultiDiGraph:
    if isinstance(network, nx.Graph):
        return nx.MultiDiGraph(network)
    to_graph = getattr(network, "to_graph", None)
    if callable(to_graph):
        return nx.MultiDiGraph(to_graph())
    raise TypeError("network must be a NetworkX graph or WNTR WaterNetworkModel")


def _node_type(network: Any, graph: nx.Graph, name: str) -> str:
    explicit = graph.nodes[name].get("node_type")
    if explicit:
        return str(explicit).lower()
    getter = getattr(network, "get_node", None)
    if callable(getter):
        return type(getter(name)).__name__.lower()
    return "junction"


def _base_demand(network: Any, graph: nx.Graph, name: str) -> float:
    if "base_demand_m3s" in graph.nodes[name]:
        return float(graph.nodes[name]["base_demand_m3s"])
    getter = getattr(network, "get_node", None)
    if callable(getter):
        node = getter(name)
        demand_timeseries = getattr(node, "demand_timeseries_list", ())
        if demand_timeseries:
            return float(demand_timeseries[0].base_value)
    return 0.0


def _travel_weight(_start: str, _end: str, attributes: dict[str, Any]) -> float:
    # MultiGraph shortest-path callbacks receive a mapping of edge keys.
    edge = next(iter(attributes.values())) if attributes and "travel_steps" not in attributes else attributes
    if "travel_steps" in edge:
        return float(edge["travel_steps"])
    length = edge.get("length_m", edge.get("length", 600.0))
    return max(1.0, round(float(length) / 600.0))


def generate_synthetic_data(
    network: Any,
    config: SyntheticConfig | None = None,
) -> SyntheticDataset:
    """Generate contamination truth and realistically imperfect sensor data.

    Contamination is a bounded pulse propagated by weighted shortest-path delay.
    The observation channel independently applies sensor latency, Gaussian noise,
    signed calibration drift, missing samples, and deterministic reversal windows.
    """

    cfg = config or SyntheticConfig()
    graph = _as_graph(network)
    if cfg.source_node not in graph:
        raise ValueError(f"source node {cfg.source_node!r} is not in the network")

    sensor_nodes = tuple(
        sorted(name for name in graph if _node_type(network, graph, name) != "reservoir")
    )
    if not sensor_nodes:
        raise ValueError("network has no junction or tank sensor nodes")

    timestamps = pd.date_range(
        cfg.start,
        periods=cfg.periods,
        freq=pd.Timedelta(minutes=cfg.step_minutes),
    )
    rng = np.random.default_rng(cfg.seed)
    distances = nx.single_source_dijkstra_path_length(graph, cfg.source_node, weight=_travel_weight)
    delays = pd.Series(
        {node: int(round(distances[node])) if node in distances else cfg.periods + 1 for node in sensor_nodes},
        name="transport_delay_steps",
        dtype="int64",
    )

    shape = (cfg.periods, len(sensor_nodes))
    truth = np.zeros(shape, dtype=float)
    for column, node in enumerate(sensor_nodes):
        delay = int(delays[node])
        begin = cfg.contamination_start_step + delay
        end = min(cfg.periods, begin + cfg.contamination_duration_steps)
        if begin >= cfg.periods:
            continue
        elapsed = np.arange(end - begin, dtype=float) + delay
        truth[begin:end, column] = cfg.source_concentration * np.exp(-cfg.decay_per_step * elapsed)

    # Repeat the hourly pattern across sub-hourly samples without interpolation.
    steps_per_hour = max(1, round(60 / cfg.step_minutes))
    pattern = np.asarray(DIURNAL_PATTERN, dtype=float)
    demand_multiplier = pattern[(np.arange(cfg.periods) // steps_per_hour) % len(pattern)]
    base_demand = np.asarray([_base_demand(network, graph, node) for node in sensor_nodes])
    demand = demand_multiplier[:, None] * base_demand[None, :]

    noise = rng.normal(0.0, cfg.sensor_noise_std, size=shape)
    drift_direction = rng.choice(np.array([-1.0, 1.0]), size=len(sensor_nodes))
    drift = np.arange(cfg.periods, dtype=float)[:, None] * cfg.drift_per_step * drift_direction[None, :]

    observed_truth = np.zeros_like(truth)
    if cfg.sensor_delay_steps == 0:
        observed_truth[:] = truth
    elif cfg.sensor_delay_steps < cfg.periods:
        observed_truth[cfg.sensor_delay_steps :] = truth[: -cfg.sensor_delay_steps]
    readings = np.maximum(0.0, observed_truth + noise + drift)

    missing = rng.random(shape) < cfg.missing_probability
    # Preserve a valid first sample for every sensor and make nonzero probabilities
    # visible in small deterministic fixtures without changing RNG behavior.
    missing[0, :] = False
    if cfg.missing_probability > 0 and not missing.any() and cfg.periods > 1:
        missing[1, 0] = True
    readings[missing] = np.nan

    reversal = np.zeros(shape, dtype=bool)
    duration = min(cfg.flow_reversal_duration_steps, cfg.flow_reversal_period_steps)
    if duration:
        for column in range(len(sensor_nodes)):
            phase = (column * max(1, duration)) % cfg.flow_reversal_period_steps
            cycle_position = (np.arange(cfg.periods) - phase) % cfg.flow_reversal_period_steps
            reversal[:, column] = cycle_position < duration

    columns = pd.Index(sensor_nodes, name="node_id")
    frame = lambda values: pd.DataFrame(values, index=timestamps, columns=columns)  # noqa: E731
    return SyntheticDataset(
        contamination=frame(truth),
        sensor_readings=frame(readings),
        demand=frame(demand),
        noise=frame(noise),
        drift=frame(drift),
        missing_mask=frame(missing),
        flow_reversal_mask=frame(reversal),
        transport_delay_steps=delays,
    )

