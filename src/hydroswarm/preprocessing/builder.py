"""Canonical WNTR/telemetry-to-HydroBatch feature construction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

from hydroswarm.classical.state_estimation import EstimatedHydraulicState
from hydroswarm.preprocessing.alignment import align_edges, canonical_node_order
from hydroswarm.preprocessing.batching import GraphSample, pad_graph_batch
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats


@dataclass(frozen=True, slots=True)
class SensorSeries:
    node_id: str
    timestamps_seconds: tuple[float, ...]
    concentration_mg_l: tuple[float | None, ...]
    pressure_m: tuple[float | None, ...]
    health: tuple[float, ...]
    missing: tuple[bool, ...]
    drift: tuple[bool, ...]
    delayed: tuple[bool, ...]
    #: Operational provenance only: this is deliberately not a new HydroCore
    #: feature column.  Live API ingestion preserves the raw declared state
    #: here so evidence hashing and deterministic authority checks cannot
    #: confuse a frozen sensor with an ordinary low-health reading.
    frozen: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if not self.frozen:
            object.__setattr__(self, "frozen", (False,) * len(self.timestamps_seconds))
        lengths = {
            len(self.timestamps_seconds), len(self.concentration_mg_l), len(self.pressure_m),
            len(self.health), len(self.missing), len(self.drift), len(self.delayed), len(self.frozen),
        }
        if len(lengths) != 1 or not self.timestamps_seconds:
            raise ValueError("sensor series fields must be aligned and non-empty")
        if any(right < left for left, right in zip(self.timestamps_seconds, self.timestamps_seconds[1:])):
            raise ValueError("sensor timestamps must be ordered")


@dataclass(frozen=True, slots=True)
class BuiltHydroBatch:
    node_ids: tuple[str, ...]
    batch: dict[str, torch.Tensor]
    feature_schema_version: str
    feature_schema_hash: str
    #: core-issues.txt repair item 6: identity of whichever governed
    #: node/edge NormalizationStats (if any) this batch was built with --
    #: see HydraulicFeatureBuilder.normalization_fingerprint.
    normalization_hash: str


#: core-issues.txt repair item 6: the always-on hardcoded unit-scaling
#: divisors in build() below are applied regardless of node_normalization/
#: edge_normalization; this sentinel records "no *governed* (fitted)
#: normalization layer is active on top of them" explicitly, rather than
#: leaving that state unrecorded, which is indistinguishable from
#: "nobody remembered to check."
NO_NORMALIZATION_SENTINEL = hashlib.sha256(b"hydroswarm-normalization-none-v1").hexdigest()


def _stable_category(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _distance(graph: nx.Graph, sources: set[str]) -> dict[str, float]:
    if not sources:
        return {str(node): math.inf for node in graph}
    undirected = graph.to_undirected()
    result = nx.multi_source_dijkstra_path_length(undirected, sources, weight="length")
    finite = [float(value) for value in result.values() if math.isfinite(value)]
    fallback = max(finite, default=1.0) * 2.0
    return {str(node): float(result.get(node, fallback)) for node in graph}


class HydraulicFeatureBuilder:
    """Build the versioned native feature set in one canonical node order."""

    def __init__(
        self,
        *,
        node_normalization: NormalizationStats | None = None,
        edge_normalization: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.node_normalization = node_normalization
        self.edge_normalization = edge_normalization
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    def normalization_fingerprint(self) -> str:
        """A stable identity for whichever governed normalization this
        builder applies -- NO_NORMALIZATION_SENTINEL when neither
        node_normalization nor edge_normalization is set, so corpus
        generation and live inference can be compared and a mismatch
        caught (core-issues.txt repair item 6) rather than silently
        drifting."""

        if self.node_normalization is None and self.edge_normalization is None:
            return NO_NORMALIZATION_SENTINEL
        payload = {
            "node": self.node_normalization.fingerprint if self.node_normalization else None,
            "edge": self.edge_normalization.fingerprint if self.edge_normalization else None,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def build(
        self,
        network: Any,
        graph: nx.MultiDiGraph,
        state: EstimatedHydraulicState,
        sensor_series: Sequence[SensorSeries],
        *,
        classical_prior: Mapping[str, float],
        arrival_residual: Mapping[str, float] | None = None,
        positive_consistency: Mapping[str, float] | None = None,
        negative_consistency: Mapping[str, float] | None = None,
        window_steps: int = 12,
    ) -> BuiltHydroBatch:
        node_ids = tuple(str(item) for item in canonical_node_order(network.node_name_list))
        positions = {node: index for index, node in enumerate(node_ids)}
        reservoirs = set(network.reservoir_name_list)
        tanks = set(network.tank_name_list)
        sensors = {item.node_id for item in sensor_series}
        reservoir_distance = _distance(graph, reservoirs)
        sensor_distance = _distance(graph, sensors)
        components = list(nx.connected_components(graph.to_undirected()))
        isolation = {str(node): index for index, component in enumerate(components) for node in component}

        latest = {item.node_id: item for item in sensor_series}
        now = max(item.timestamps_seconds[-1] for item in sensor_series)
        node_rows: list[list[float]] = []
        for node_id in node_ids:
            node = network.get_node(node_id)
            series = latest.get(node_id)
            concentration = (
                series.concentration_mg_l[-1] if series and series.concentration_mg_l[-1] is not None else 0.0
            )
            health = series.health[-1] if series else 0.0
            age = now - series.timestamps_seconds[-1] if series else now
            missing = float(series is None or series.missing[-1])
            node_type = 2.0 if node_id in reservoirs else 3.0 if node_id in tanks else 1.0
            elevation = float(getattr(node, "elevation", getattr(node, "base_head", 0.0)))
            base_demand = 0.0
            if hasattr(node, "demand_timeseries_list") and len(node.demand_timeseries_list):
                base_demand = float(node.demand_timeseries_list[0].base_value)
            current_demand = state.demand_m3s.get(node_id)
            demand_value = current_demand.estimate if current_demand else 0.0
            pressure = state.pressure_m.get(node_id)
            pressure_value = pressure.estimate if pressure else 0.0
            node_rows.append([
                node_type, elevation, base_demand, demand_value, pressure_value,
                elevation + pressure_value, float(concentration), float(series is not None), health,
                age, missing, classical_prior.get(node_id, 0.0),
                (arrival_residual or {}).get(node_id, 0.0),
                (positive_consistency or {}).get(node_id, 1.0),
                (negative_consistency or {}).get(node_id, 1.0),
                _stable_category(str(graph.nodes[node_id].get("pressure_zone", "default"))),
                reservoir_distance.get(node_id, 0.0), sensor_distance.get(node_id, 0.0),
                float(isolation.get(node_id, 0)),
            ])
        node_values = DEFAULT_FEATURE_SCHEMA.validate_node_array(node_rows)
        node_values[:, 6] = np.log1p(np.maximum(node_values[:, 6], 0.0))
        node_values = node_values / np.asarray(
            [
                3.0, 200.0, 0.02, 0.02, 100.0, 200.0, 5.0, 1.0, 1.0,
                86_400.0, 1.0, 1.0, 86_400.0, 1.0, 1.0, 1.0, 10_000.0,
                10_000.0, 10.0,
            ],
            dtype=np.float32,
        )
        if self.node_normalization:
            if self.node_normalization.schema_version != DEFAULT_FEATURE_SCHEMA.version:
                raise ValueError("node normalization schema version is incompatible")
            node_values = self.node_normalization.transform(node_values)

        edge_pairs: list[tuple[str, str]] = []
        edge_rows: list[list[float]] = []
        for start, end, key, attributes in graph.edges(keys=True, data=True):
            link_id = str(attributes.get("link_id", key))
            link = network.get_link(link_id)
            length = float(getattr(link, "length", 0.0))
            diameter = float(getattr(link, "diameter", 0.0))
            flow = float(attributes.get("flow_m3s", 0.0))
            velocity = float(attributes.get("velocity_mps", 0.0))
            travel_time = float(attributes.get("travel_time_seconds", 0.0))
            if not math.isfinite(velocity):
                velocity = 0.0
            if not math.isfinite(travel_time):
                travel_time = 24.0 * 3_600.0
            edge_pairs.append((str(start), str(end)))
            kind = type(link).__name__.lower()
            edge_rows.append([
                1.0 if "pipe" in kind else 2.0 if "pump" in kind else 3.0,
                length, diameter, float(getattr(link, "roughness", 0.0)), abs(flow),
                1.0, velocity, travel_time,
                float("valve" not in kind or str(getattr(link, "status", "Open")).lower() != "closed"),
                float("pump" not in kind or str(getattr(link, "status", "Open")).lower() != "closed"),
                float(getattr(link, "start_node_name", str(start)) != str(start)),
                float(attributes.get("operable", True)), math.pi * (diameter / 2) ** 2 * length,
            ])
        edge_values = DEFAULT_FEATURE_SCHEMA.validate_edge_array(edge_rows)
        edge_values = edge_values / np.asarray(
            [3.0, 10_000.0, 2.0, 200.0, 0.1, 1.0, 5.0, 86_400.0, 1.0,
             1.0, 1.0, 1.0, 100_000.0],
            dtype=np.float32,
        )
        edge_index, edge_values = align_edges(edge_pairs, edge_values, node_ids)
        if self.edge_normalization:
            if self.edge_normalization.schema_version != DEFAULT_FEATURE_SCHEMA.version:
                raise ValueError("edge normalization schema version is incompatible")
            edge_values = self.edge_normalization.transform(edge_values)

        selected_times = sorted({time for item in sensor_series for time in item.timestamps_seconds})[-window_steps:]
        steps = len(selected_times)
        temporal = np.zeros((steps, len(node_ids), 6), dtype=np.float32)
        quality = np.zeros((steps, len(node_ids), 4), dtype=np.float32)
        temporal[:] = np.nan
        quality[:] = np.nan
        for series in sensor_series:
            node_index = positions[series.node_id]
            by_time = {timestamp: index for index, timestamp in enumerate(series.timestamps_seconds)}
            for time_index, timestamp in enumerate(selected_times):
                if timestamp not in by_time:
                    continue
                source_index = by_time[timestamp]
                concentration = series.concentration_mg_l[source_index]
                pressure = series.pressure_m[source_index]
                age = now - timestamp
                temporal[time_index, node_index] = [
                    np.log1p(max(concentration, 0.0)) if concentration is not None else np.nan,
                    pressure / 100.0 if pressure is not None else np.nan,
                    age / 86_400.0, float(series.missing[source_index]), float(series.drift[source_index]),
                    float(series.delayed[source_index]),
                ]
                quality[time_index, node_index] = [
                    series.health[source_index], float(series.missing[source_index]),
                    float(series.drift[source_index]), age / 86_400.0,
                ]
        sample = GraphSample(
            node_features=torch.as_tensor(node_values, dtype=self.dtype, device=self.device),
            temporal_features=torch.as_tensor(temporal, dtype=self.dtype, device=self.device),
            quality_features=torch.as_tensor(quality, dtype=self.dtype, device=self.device),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long, device=self.device),
            edge_features=torch.as_tensor(edge_values, dtype=self.dtype, device=self.device),
            timestamps=torch.as_tensor(selected_times, dtype=self.dtype, device=self.device),
        )
        batch = pad_graph_batch([sample])
        batch["classical_prior"] = torch.as_tensor(
            [[classical_prior.get(node, 0.0) for node in node_ids]], dtype=self.dtype, device=self.device
        )
        batch["source_candidate_mask"] = torch.as_tensor(
            [[node in network.junction_name_list for node in node_ids]],
            dtype=torch.bool,
            device=self.device,
        )
        batch["travel_time"] = torch.as_tensor([[reservoir_distance[node] for node in node_ids]], dtype=self.dtype, device=self.device)
        batch["reservoir_reachability"] = torch.as_tensor(
            [[float(graph.nodes[node].get("reservoir_reachable", False)) for node in node_ids]],
            dtype=self.dtype, device=self.device,
        )
        batch["demand_centrality"] = torch.as_tensor(
            [[float(graph.nodes[node].get("demand_centrality", 0.0)) for node in node_ids]],
            dtype=self.dtype, device=self.device,
        )
        return BuiltHydroBatch(
            node_ids=node_ids, batch=batch, feature_schema_version=DEFAULT_FEATURE_SCHEMA.version,
            feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
            normalization_hash=self.normalization_fingerprint,
        )
