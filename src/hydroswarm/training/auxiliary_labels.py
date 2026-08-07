"""Auxiliary representation-learning target generation (core-issues2.txt
Phase 6): sensor reconstruction, future concentration, and hydraulic
travel time. All three are explicitly non-authoritative per their own
targets_v2.TargetSpec text ("never an authoritative product output") and
share the [node_count] shape/masking convention corpus.py's sensor_fault
target already established -- a full topology-length array, masked to
whatever subset of nodes actually has ground truth.

Every `node_ids` parameter below MUST be the canonical full topology node
space (junctions + reservoirs + tanks), matching source_node_logits/
sensor_fault_logits' own index space -- not sorted(network.junction_name_
list). See scout_trajectory.build_scout_trajectory's docstring for the
full rationale; full_trajectory.py's build_incident_trajectory derives
node_ids from example.topology for exactly this reason.
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
    for every sensor node whose OWN input reading was actually degraded
    (missing, frozen, in communication outage) at that instant -- a
    denoising objective over whatever the input's own corrupted reading
    looked like there.

    core-issues3.txt Phase 7.3 / item I: masking every sensor node
    regardless of whether its reading was degraded (the pre-fix behavior)
    supervises mostly trivial identity copying -- corpus.py's
    build_sensor_series feeds the model scenario.observed_concentration
    directly, which equals truth_concentration exactly at every
    healthy (non-degraded) sensor position. A model "reconstructing" a
    value it was already given verbatim as input learns nothing; only a
    position where the input itself was actually corrupted poses a real
    denoising problem. Masked out for every non-sensor node (never
    simulated as an observation point, so there is no ground truth) and
    for every healthy sensor position (nothing to denoise -- the input
    already shows the true value)."""

    value = np.zeros(len(node_ids), dtype=np.float32)
    mask = np.zeros(len(node_ids), dtype=bool)
    time_index = _nearest_time_index(scenario.timestamps_seconds, reference_time_seconds)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for sensor_index, node_id in enumerate(scenario.sensor_nodes):
        position = positions.get(node_id)
        if position is None:
            continue
        degraded = (
            not scenario.observation_mask[time_index, sensor_index]
            or scenario.frozen_mask[time_index, sensor_index]
            or scenario.communication_outage_mask[time_index, sensor_index]
        )
        if not degraded:
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
    """DISABLED (core-issues3.txt Phase 7.4 / item H): always returns an
    all-masked-out placeholder. Do not re-enable by simply removing this
    early return without first giving the base ScenarioExample a genuine
    observation cutoff -- see the leakage explanation below.

    Why this is disabled, not fixed in place: the target this function is
    meant to supervise is truth_concentration at
    reference_time_seconds + horizon_seconds (2h into the scenario by
    default). But hydroswarm.training.corpus.scenario_to_example builds its
    temporal/quality input tensors via HydraulicFeatureBuilder.build(...,
    window_steps=len(scenario.timestamps_seconds)) -- EVERY timestamp the
    scenario simulated (typically a 24h window, per
    hydroswarm.simulation.network's options.time.duration), not a bounded
    lookback from some "current" instant. The specific target timestamp
    this function would supervise is therefore already directly present as
    a raw temporal input feature the model can read off verbatim -- not an
    approximate leak, an exact one. A model "forecasting" a value it was
    already handed as input learns nothing and would report a
    misleadingly perfect validation score.

    A real fix requires a genuinely cutoff-aware feature representation
    (a ScenarioExample variant whose temporal/quality tensors are built
    from only scenario.timestamps_seconds <= reference_time_seconds,
    per core-issues3.txt Phase 5's "explicit observation cutoff" applied
    to the base Sentinel example, not only Scout) -- a materially larger,
    cross-cutting change belonging with Phase 9/10's dataset-and-collator
    work, not a self-contained Phase 7 loss/masking fix. Disabling this
    target here (rather than shipping it broken, or attempting an
    under-tested partial fix) is the fail-closed choice: masked-out
    positions contribute nothing to the loss (losses.masked_regression),
    so downstream training is unaffected either way -- this just prevents
    a real (if low-weighted) leakage channel from silently existing."""

    value = np.zeros(len(node_ids), dtype=np.float32)
    mask = np.zeros(len(node_ids), dtype=bool)
    return {
        "future_concentration": torch.from_numpy(value),
        "future_concentration_mask": torch.from_numpy(mask),
    }


#: core-issues3.txt Phase 7.5: raw travel-time seconds range from 0 to tens
#: of thousands and can dominate a multitask MSE next to bounded [0, 1]-ish
#: regression targets. log1p is a fixed, train-owned, documented transform
#: (not fit from data), applied identically to every split; invert with
#: expm1 to recover physical-unit seconds for reporting. Bumped only if the
#: transform itself changes (never for target/weighting tuning).
TRAVEL_TIME_TRANSFORM = "log1p"


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
    contamination source exists.

    The value is TRAVEL_TIME_TRANSFORM-transformed (log1p(seconds)), not raw
    seconds -- see that constant's docstring. Invert with
    torch.expm1/np.expm1 before reporting a physical-unit metric."""

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
            value[index] = math.log1p(max(0.0, travel_time))
            mask[index] = True
    return {"travel_time": torch.from_numpy(value), "travel_time_mask": torch.from_numpy(mask)}
