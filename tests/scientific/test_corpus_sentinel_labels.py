"""Task 2.2: Sentinel label generation (event_presence/event_cause/
evidence_sufficiency/source_region) built on top of scenario_to_example."""

from __future__ import annotations

import pytest

from hydroswarm.data.scenarios import DatasetSplit, EventType, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import (
    EVENT_CAUSE_INDEX,
    SOURCE_REGION_COUNT,
    assign_source_regions,
    fit_signature_library,
    scenario_to_example,
)
from hydroswarm.training.targets_v2 import EventCause, validate_targets_v2


@pytest.fixture(scope="module")
def network():
    return build_wntr_network()


@pytest.fixture(scope="module")
def signature_library(network):
    generator = WNTRScenarioGenerator()
    node_ids = tuple(sorted(network.junction_name_list))
    scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=1000 + index * 10, network_id="ref", network_family="reference",
                split=DatasetSplit.TRAIN, source_node=source, sensor_count=3,
            ),
        )
        for index, source in enumerate(node_ids)
    ]
    return fit_signature_library(scenarios, node_ids)


def _example(network, signature_library, *, event_type: EventType, seed: int = 2000):
    generator = WNTRScenarioGenerator()
    scenario = generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
            event_type=event_type, sensor_count=3,
        ),
    )
    return scenario_to_example(scenario, network, signature_library)


def test_assign_source_regions_is_deterministic_and_bounded(network) -> None:
    regions_a = assign_source_regions(network)
    regions_b = assign_source_regions(network)
    assert regions_a == regions_b
    assert set(regions_a) == set(network.junction_name_list)
    assert all(0 <= region < SOURCE_REGION_COUNT for region in regions_a.values())


def test_contamination_example_has_event_presence_true_and_valid_masks(network, signature_library) -> None:
    example = _example(network, signature_library, event_type=EventType.CONTAMINATION)
    validate_targets_v2(example.targets)
    assert bool(example.targets["event_presence"]) is True
    assert int(example.targets["event_cause"]) == EVENT_CAUSE_INDEX[EventCause.CONTAMINATION]
    assert bool(example.targets["source_node_mask"]) is True
    assert bool(example.targets["start_time_mask"]) is True
    assert bool(example.targets["duration_mask"]) is True
    assert bool(example.targets["relative_strength_mask"]) is True
    assert bool(example.targets["source_region_mask"]) is True


def test_normal_example_has_event_presence_false_and_masked_targets(network, signature_library) -> None:
    example = _example(network, signature_library, event_type=EventType.NORMAL)
    validate_targets_v2(example.targets)
    assert bool(example.targets["event_presence"]) is False
    assert int(example.targets["event_cause"]) == EVENT_CAUSE_INDEX[EventCause.NORMAL]
    assert bool(example.targets["source_node_mask"]) is False
    assert bool(example.targets["start_time_mask"]) is False
    assert bool(example.targets["duration_mask"]) is False
    assert bool(example.targets["relative_strength_mask"]) is False
    assert bool(example.targets["source_region_mask"]) is False
    assert not bool(example.targets["sensor_fault"].any())


def test_sensor_fault_only_example_has_correct_cause_and_shows_the_fault(network, signature_library) -> None:
    example = _example(network, signature_library, event_type=EventType.SENSOR_FAULT_ONLY)
    validate_targets_v2(example.targets)
    assert bool(example.targets["event_presence"]) is False
    assert int(example.targets["event_cause"]) == EVENT_CAUSE_INDEX[EventCause.SENSOR_FAULT]
    assert bool(example.targets["sensor_fault"].any())  # the forced fault is visible


def test_normal_operation_is_not_labeled_as_contamination(network, signature_library) -> None:
    # Direct plan requirement (Task 2.2 test list).
    example = _example(network, signature_library, event_type=EventType.NORMAL)
    assert int(example.targets["event_cause"]) != EVENT_CAUSE_INDEX[EventCause.CONTAMINATION]


def test_fault_only_scenario_has_correct_event_cause(network, signature_library) -> None:
    # Direct plan requirement (Task 2.2 test list).
    example = _example(network, signature_library, event_type=EventType.SENSOR_FAULT_ONLY)
    assert int(example.targets["event_cause"]) == EVENT_CAUSE_INDEX[EventCause.SENSOR_FAULT]


def test_evidence_sufficiency_agrees_with_clean_high_health_scenario(network, signature_library) -> None:
    # A clean, fully-observed scenario should have every sensor at health=1.0
    # for the whole window, well above the sufficiency rule's threshold.
    example = _example(network, signature_library, event_type=EventType.CONTAMINATION, seed=3000)
    assert bool(example.targets["evidence_sufficiency"]) is True


def test_source_node_index_is_within_masked_out_range_but_harmless_when_masked(network, signature_library) -> None:
    # When event_presence is False, source_node's value is a placeholder;
    # confirm it does not crash downstream consumers and the mask correctly
    # flags it as not to be trained on.
    example = _example(network, signature_library, event_type=EventType.NORMAL)
    assert int(example.targets["source_node"]) == 0
    assert bool(example.targets["source_node_mask"]) is False
