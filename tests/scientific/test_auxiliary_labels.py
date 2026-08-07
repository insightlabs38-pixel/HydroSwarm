"""core-issues2.txt Phase 6: auxiliary representation-learning target generation."""

from __future__ import annotations

import numpy as np
import pytest

from hydroswarm.data.scenarios import DatasetSplit, EventType, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.auxiliary_labels import (
    future_concentration_target,
    sensor_reconstruction_target,
    travel_time_target,
)
from hydroswarm.training.corpus import build_feature_context
from hydroswarm.training.targets_v2 import validate_targets_v2


def _scenario(
    network,
    *,
    event_type=EventType.CONTAMINATION,
    source_node="J2",
    seed=10,
    missing_probability=0.05,
    frozen_probability=0.02,
    communication_outage_probability=0.02,
):
    generator = WNTRScenarioGenerator()
    return generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
            event_type=event_type, source_node=source_node, sensor_count=3,
            missing_probability=missing_probability, frozen_probability=frozen_probability,
            communication_outage_probability=communication_outage_probability,
        ),
    )


def test_sensor_reconstruction_masks_nothing_when_every_sensor_is_healthy() -> None:
    """core-issues3.txt Phase 7.3: a healthy sensor's input already equals
    its own truth value verbatim -- 'reconstructing' it is a trivial
    identity mapping, not a real denoising objective, so it must stay
    masked out even though the node IS a sensor."""

    network = build_wntr_network()
    scenario = _scenario(
        network, missing_probability=0.0, frozen_probability=0.0, communication_outage_probability=0.0
    )
    node_ids = tuple(sorted(network.node_name_list))

    target = sensor_reconstruction_target(scenario, node_ids, reference_time_seconds=3600)
    validate_targets_v2(target)

    time_index = int(np.argmin(np.abs(np.asarray(scenario.timestamps_seconds, dtype=float) - 3600)))
    for sensor_index in range(len(scenario.sensor_nodes)):
        assert scenario.observation_mask[time_index, sensor_index]
        assert not scenario.frozen_mask[time_index, sensor_index]
        assert not scenario.communication_outage_mask[time_index, sensor_index]
    assert not bool(target["sensor_reconstruction_mask"].any())


def test_sensor_reconstruction_masks_exactly_the_degraded_sensor_positions() -> None:
    """With degradation forced to certain, every sensor's reference-time
    reading is corrupted, so every sensor node (and no non-sensor node)
    must be unmasked, with the true (undegraded) value as the target."""

    network = build_wntr_network()
    scenario = _scenario(
        network, missing_probability=1.0, frozen_probability=0.0, communication_outage_probability=0.0
    )
    node_ids = tuple(sorted(network.node_name_list))

    target = sensor_reconstruction_target(scenario, node_ids, reference_time_seconds=3600)
    validate_targets_v2(target)

    mask = target["sensor_reconstruction_mask"]
    unmasked_nodes = {node_ids[i] for i in range(len(node_ids)) if bool(mask[i])}
    assert unmasked_nodes == set(scenario.sensor_nodes)

    time_index = int(np.argmin(np.abs(np.asarray(scenario.timestamps_seconds, dtype=float) - 3600)))
    for sensor_index, node_id in enumerate(scenario.sensor_nodes):
        position = node_ids.index(node_id)
        assert float(target["sensor_reconstruction"][position]) == float(
            scenario.truth_concentration[time_index, sensor_index]
        )


def test_future_concentration_is_always_fully_masked_disabled_pending_a_cutoff_aware_representation() -> None:
    """core-issues3.txt Phase 7.4 / item H: future_concentration_target is
    deliberately disabled (always fully masked), not merely masked when the
    horizon exceeds simulated duration. Re-enabling it against the current
    ScenarioExample representation would leak the exact target instant into
    the model's own temporal input features -- see the next test."""

    network = build_wntr_network()
    scenario = _scenario(network)
    node_ids = tuple(sorted(network.node_name_list))

    within_horizon = future_concentration_target(scenario, node_ids, horizon_seconds=3600)
    validate_targets_v2(within_horizon)
    assert not bool(within_horizon["future_concentration_mask"].any())

    beyond_simulated_duration = future_concentration_target(scenario, node_ids, horizon_seconds=10_000_000)
    validate_targets_v2(beyond_simulated_duration)
    assert not bool(beyond_simulated_duration["future_concentration_mask"].any())


def test_future_concentration_disable_is_justified_by_real_input_window_leakage() -> None:
    """Proves the leakage claim in future_concentration_target's docstring
    is real, not hypothetical: the base ScenarioExample's temporal input
    already includes the exact timestamp a future_concentration target at
    the default horizon would supervise."""

    from hydroswarm.simulation.wrapper import FEATURE_SNAPSHOT_TIME_SECONDS
    from hydroswarm.training.auxiliary_labels import DEFAULT_FUTURE_HORIZON_SECONDS
    from hydroswarm.training.corpus import fit_signature_library, scenario_to_example

    network = build_wntr_network()
    node_ids = tuple(sorted(network.junction_name_list))
    generator = WNTRScenarioGenerator()
    fitting_scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=3000 + index * 10, network_id="ref", network_family="reference",
                split=DatasetSplit.TRAIN, source_node=source, sensor_count=3,
            ),
        )
        for index, source in enumerate(node_ids)
    ]
    signature_library = fit_signature_library(fitting_scenarios, node_ids)

    scenario = _scenario(network)
    example = scenario_to_example(scenario, network, signature_library)
    target_time = float(FEATURE_SNAPSHOT_TIME_SECONDS + DEFAULT_FUTURE_HORIZON_SECONDS)

    # The model's own visible temporal input spans the entire simulated
    # window, not a bounded lookback from "now" -- the target instant a
    # future_concentration target would supervise is already directly
    # present in it, confirming the leakage the disable avoids is real.
    max_visible_timestamp = float(np.max(scenario.timestamps_seconds))
    assert target_time <= max_visible_timestamp
    visible_timestamps = example.inputs["timestamps"]
    assert float(visible_timestamps.max()) == max_visible_timestamp
    assert any(abs(float(value) - target_time) < 1e-6 for value in visible_timestamps)


def test_travel_time_is_masked_out_entirely_for_non_contamination_events() -> None:
    network = build_wntr_network()
    scenario = _scenario(network, event_type=EventType.NORMAL)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = travel_time_target(scenario, context.graph, node_ids)
    validate_targets_v2(target)
    assert not bool(target["travel_time_mask"].any())


def test_travel_time_from_a_real_source_covers_reachable_nodes_with_finite_values() -> None:
    network = build_wntr_network()
    scenario = _scenario(network, event_type=EventType.CONTAMINATION, source_node="J2")
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = travel_time_target(scenario, context.graph, node_ids)
    validate_targets_v2(target)
    mask = target["travel_time_mask"]
    assert bool(mask.any())  # at least some nodes are reachable
    values = target["travel_time"]
    for i in range(len(node_ids)):
        if bool(mask[i]):
            assert float(values[i]) >= 0.0
        else:
            assert float(values[i]) == 0.0  # placeholder, never a real travel time


def test_travel_time_is_log1p_transformed() -> None:
    """core-issues3.txt Phase 7.5: raw seconds would dominate a multitask
    MSE; the stored value must be log1p(seconds), not raw seconds."""

    import networkx as nx

    network = build_wntr_network()
    scenario = _scenario(network, event_type=EventType.CONTAMINATION, source_node="J2")
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))
    source = scenario.manifest.incident.source_nodes[0]

    target = travel_time_target(scenario, context.graph, node_ids)
    mask = target["travel_time_mask"]
    for index, node_id in enumerate(node_ids):
        if not bool(mask[index]) or node_id == source:
            continue
        raw_seconds = nx.shortest_path_length(context.graph, source, node_id, weight="travel_time_seconds")
        assert float(target["travel_time"][index]) == pytest.approx(np.log1p(raw_seconds), rel=1e-5)


def test_travel_time_at_the_source_itself_is_zero() -> None:
    network = build_wntr_network()
    scenario = _scenario(network, event_type=EventType.CONTAMINATION, source_node="J2")
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = travel_time_target(scenario, context.graph, node_ids)
    source_index = node_ids.index("J2")
    assert bool(target["travel_time_mask"][source_index])
    assert float(target["travel_time"][source_index]) == 0.0
