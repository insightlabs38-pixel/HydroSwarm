"""Auxiliary representation-learning target generation (core-issues2.txt
Phase 6): sensor reconstruction, future concentration, and hydraulic
travel time. All three are explicitly non-authoritative per their own
targets_v2.TargetSpec text ("never an authoritative product output") and
share the [node_count] shape/masking convention corpus.py's sensor_fault
target already established -- a full topology-length array, masked to
whatever subset of nodes actually has ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import networkx as nx
import numpy as np
import torch

from hydroswarm.data.scenarios import EventType, GeneratedScenario
from hydroswarm.simulation.wrapper import FEATURE_SNAPSHOT_TIME_SECONDS

#: Predict this far past reference_time_seconds -- a fixed forecasting
#: horizon, not fit from data (matching the plan's "Loss weights are
#: explicit" spirit for auxiliary objectives).
DEFAULT_FUTURE_HORIZON_SECONDS = 3_600


def _nearest_time_index(timestamps_seconds: Sequence[float], target_seconds: float) -> int:
    return int(np.argmin(np.abs(np.asarray(timestamps_seconds, dtype=float) - target_seconds)))


def sensor_reconstruction_target(
    scenario: GeneratedScenario,
    node_ids: Sequence[str],
    *,
    reference_time_seconds: float = FEATURE_SNAPSHOT_TIME_SECONDS,
) -> dict[str, torch.Tensor]:
    """The scenario's own unmasked truth_concentration at reference_time,
    for every sensor node -- ground truth for a denoising/reconstruction
    objective over whatever the input's own (possibly degraded) reading
    looked like at that node. Masked out for every non-sensor node: it was
    never simulated as an observation point at all, so there is no ground
    truth to reconstruct against there."""

    value = np.zeros(len(node_ids), dtype=np.float32)
    mask = np.zeros(len(node_ids), dtype=bool)
    time_index = _nearest_time_index(scenario.timestamps_seconds, reference_time_seconds)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for sensor_index, node_id in enumerate(scenario.sensor_nodes):
        position = positions.get(node_id)
        if position is None:
            continue
        value[position] = scenario.truth_concentration[time_index, sensor_index]
        mask[position] = True
    return {
        "sensor_reconstruction": torch.from_numpy(value),
        "sensor_reconstruction_mask": torch.from_numpy(mask),
    }


def future_concentration_target(
    scenario: GeneratedScenario,
    node_ids: Sequence[str],
    *,
    reference_time_seconds: float = FEATURE_SNAPSHOT_TIME_SECONDS,
    horizon_seconds: float = DEFAULT_FUTURE_HORIZON_SECONDS,
) -> dict[str, torch.Tensor]:
    """truth_concentration at reference_time_seconds + horizon_seconds, for
    every sensor node. Masked out entirely (no future truth available) for
    any scenario whose simulated duration does not reach that far, and
    always masked out for non-sensor nodes (no concentration truth exists
    for them at any time)."""

    value = np.zeros(len(node_ids), dtype=np.float32)
    mask = np.zeros(len(node_ids), dtype=bool)
    target_time = reference_time_seconds + horizon_seconds
    max_time = float(np.max(scenario.timestamps_seconds)) if len(scenario.timestamps_seconds) else -math.inf
    if target_time > max_time:
        return {
            "future_concentration": torch.from_numpy(value),
            "future_concentration_mask": torch.from_numpy(mask),
        }
    time_index = _nearest_time_index(scenario.timestamps_seconds, target_time)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for sensor_index, node_id in enumerate(scenario.sensor_nodes):
        position = positions.get(node_id)
        if position is None:
            continue
        value[position] = scenario.truth_concentration[time_index, sensor_index]
        mask[position] = True
    return {
        "future_concentration": torch.from_numpy(value),
        "future_concentration_mask": torch.from_numpy(mask),
    }


def travel_time_target(
    scenario: GeneratedScenario, graph: Any, node_ids: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Shortest directed hydraulic travel time from the true contamination
    source to every node, via the scenario's own exact dynamic hydraulic
    graph (HydraulicSimulator.build_dynamic_graph's "travel_time_seconds"
    edge weight -- the same structural feature the model already consumes
    as an input, reused here as ground truth for this auxiliary target).
    Masked out for every node unreachable from the source (no directed
    path, or every path traverses a zero-flow/infinite-travel-time edge),
    and for every node on a NORMAL/SENSOR_FAULT_ONLY scenario where no real
    contamination source exists."""

    value = np.zeros(len(node_ids), dtype=np.float32)
    mask = np.zeros(len(node_ids), dtype=bool)
    if scenario.manifest.event_type != EventType.CONTAMINATION.value:
        return {"travel_time": torch.from_numpy(value), "travel_time_mask": torch.from_numpy(mask)}

    source = scenario.manifest.incident.source_nodes[0]
    for index, node_id in enumerate(node_ids):
        try:
            travel_time = nx.shortest_path_length(graph, source, node_id, weight="travel_time_seconds")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if math.isfinite(travel_time):
            value[index] = travel_time
            mask[index] = True
    return {"travel_time": torch.from_numpy(value), "travel_time_mask": torch.from_numpy(mask)}
