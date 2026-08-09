"""core-issues2.txt Phase 7: full incident trajectory assembly."""

from __future__ import annotations

import pytest

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import fit_signature_library
from hydroswarm.training.full_trajectory import build_incident_trajectory
from hydroswarm.training.ood_categories import OODCategory
from hydroswarm.training.scout_labels import build_signature_artifact_for_network
from hydroswarm.training.targets_v2 import validate_targets_v2

#: Every test in this module runs many real WNTR/EPANET verifications
#: (audited call count >=10 each) -- see pyproject.toml's full_simulation
#: marker docstring.
pytestmark = pytest.mark.full_simulation

_VALIDATED = frozenset({"reference-topology-hash"})


def _artifact(network, cache_dir):
    cache = SignatureCache(cache_dir)
    key = SignatureCacheKey(
        network_hash="test-net", hydraulic_state_hash="test-state", simulator_version="test",
        configuration_hash="full-trajectory-test-v1", sensor_layout_hash="all-junctions",
    )
    return build_signature_artifact_for_network(
        network, cache, key=key,
        sample_times_seconds=(0, 3600, 7200),
        start_time_bins=(0,), duration_bins=(60,), strength_bins=(1.0,),
    )


def _signature_library(network):
    junctions = tuple(sorted(network.junction_name_list))
    generator = WNTRScenarioGenerator()
    scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=1000 + index * 10, network_id="ref", network_family="reference",
                split=DatasetSplit.TRAIN, source_node=source, sensor_count=3,
                start_time_bins_min=(0,), duration_bins_min=(60,), strength_bins=(1.0,),
            ),
        )
        for index, source in enumerate(junctions)
    ]
    return fit_signature_library(scenarios, junctions)


def _scenario(network, *, source_node: str, seed: int):
    generator = WNTRScenarioGenerator()
    return generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
            source_node=source_node, sensor_count=3, start_time_bins_min=(0,),
            duration_bins_min=(60,), strength_bins=(1.0,),
        ),
    )


def test_incident_trajectory_bundles_every_piece(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _artifact(network, tmp_path / "cache")
    library = _signature_library(network)
    scenario = _scenario(network, source_node="J2", seed=10)
    result = build_incident_trajectory(
        scenario, network, library, artifact,
        topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED,
    )

    assert result.example.scenario_id == str(scenario.manifest.scenario_id)
    assert result.ood_category == OODCategory.NONE
    assert len(result.scout.steps) >= 1
    assert len(result.strategist.steps) == 1
    assert any(label.is_no_response_comparator for label in result.strategist.steps[0].labels)


def test_merged_example_targets_include_every_new_governed_target(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _artifact(network, tmp_path / "cache")
    library = _signature_library(network)
    scenario = _scenario(network, source_node="J3", seed=11)
    result = build_incident_trajectory(
        scenario, network, library, artifact,
        topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED,
    )
    for key in (
        "ood_class", "next_step", "evidence_sufficiency",
        "sensor_reconstruction", "sensor_reconstruction_mask",
        "future_concentration", "future_concentration_mask",
        "travel_time", "travel_time_mask",
        # original sentinel-category targets must still be present
        "source_node", "source_node_mask", "event_cause", "sensor_fault",
    ):
        assert key in result.example.targets, key


def test_merged_example_targets_pass_validate_targets_v2(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _artifact(network, tmp_path / "cache")
    library = _signature_library(network)
    scenario = _scenario(network, source_node="J4", seed=12)
    result = build_incident_trajectory(
        scenario, network, library, artifact,
        topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED,
    )
    validate_targets_v2(result.example.targets, topology=result.example.topology)


def test_evidence_sufficiency_target_is_overwritten_not_duplicated(tmp_path) -> None:
    # scenario_to_example's own (partial-rule) evidence_sufficiency value
    # must be replaced by the fuller Phase 5 rule, not left alongside it
    # under a different key.
    network = build_wntr_network()
    artifact = _artifact(network, tmp_path / "cache")
    library = _signature_library(network)
    scenario = _scenario(network, source_node="J2", seed=13)
    result = build_incident_trajectory(
        scenario, network, library, artifact,
        topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED,
    )
    assert sum(1 for key in result.example.targets if key == "evidence_sufficiency") == 1


def test_unseen_topology_hash_flags_the_category_and_next_step_abstains(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _artifact(network, tmp_path / "cache")
    library = _signature_library(network)
    scenario = _scenario(network, source_node="J2", seed=14)
    result = build_incident_trajectory(
        scenario, network, library, artifact,
        topology_hash="some-unvalidated-hash", validated_topology_hashes=_VALIDATED,
    )
    assert result.ood_category == OODCategory.UNSEEN_TOPOLOGY
    from hydroswarm.training.targets_v2 import NextStep

    next_step_index = int(result.example.targets["next_step"])
    assert list(NextStep)[next_step_index] == NextStep.ABSTAIN
