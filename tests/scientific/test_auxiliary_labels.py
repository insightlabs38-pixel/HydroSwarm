"""core-issues2.txt Phase 6: auxiliary representation-learning target generation."""

from __future__ import annotations

import numpy as np

from hydroswarm.data.scenarios import DatasetSplit, EventType, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.auxiliary_labels import (
    future_concentration_target,
    sensor_reconstruction_target,
    travel_time_target,
)
from hydroswarm.training.corpus import build_feature_context
from hydroswarm.training.targets_v2 import validate_targets_v2


def _scenario(network, *, event_type=EventType.CONTAMINATION, source_node="J2", seed=10):
    generator = WNTRScenarioGenerator()
    return generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
            event_type=event_type, source_node=source_node, sensor_count=3,
        ),
    )


def test_sensor_reconstruction_masks_exactly_the_sensor_nodes() -> None:
    network = build_wntr_network()
    scenario = _scenario(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = sensor_reconstruction_target(scenario, node_ids)
    validate_targets_v2(target)

    mask = target["sensor_reconstruction_mask"]
    unmasked_nodes = {node_ids[i] for i in range(len(node_ids)) if bool(mask[i])}
    assert unmasked_nodes == set(scenario.sensor_nodes)


def test_sensor_reconstruction_value_matches_truth_at_reference_time() -> None:
    network = build_wntr_network()
    scenario = _scenario(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = sensor_reconstruction_target(scenario, node_ids, reference_time_seconds=3600)
    time_index = int(np.argmin(np.abs(np.asarray(scenario.timestamps_seconds, dtype=float) - 3600)))
    for sensor_index, node_id in enumerate(scenario.sensor_nodes):
        position = node_ids.index(node_id)
        assert float(target["sensor_reconstruction"][position]) == float(
            scenario.truth_concentration[time_index, sensor_index]
        )


def test_future_concentration_is_unmasked_when_the_horizon_is_within_simulated_duration() -> None:
    network = build_wntr_network()
    scenario = _scenario(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = future_concentration_target(scenario, node_ids, horizon_seconds=3600)
    validate_targets_v2(target)
    mask = target["future_concentration_mask"]
    unmasked_nodes = {node_ids[i] for i in range(len(node_ids)) if bool(mask[i])}
    assert unmasked_nodes == set(scenario.sensor_nodes)


def test_future_concentration_is_fully_masked_when_the_horizon_exceeds_simulated_duration() -> None:
    network = build_wntr_network()
    scenario = _scenario(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = future_concentration_target(scenario, node_ids, horizon_seconds=10_000_000)
    validate_targets_v2(target)
    assert not bool(target["future_concentration_mask"].any())


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


def test_travel_time_at_the_source_itself_is_zero() -> None:
    network = build_wntr_network()
    scenario = _scenario(network, event_type=EventType.CONTAMINATION, source_node="J2")
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    target = travel_time_target(scenario, context.graph, node_ids)
    source_index = node_ids.index("J2")
    assert bool(target["travel_time_mask"][source_index])
    assert float(target["travel_time"][source_index]) == 0.0
