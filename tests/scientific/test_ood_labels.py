"""core-issues2.txt Phase 4: learned OOD category-label generation."""

from __future__ import annotations

import torch

from hydroswarm.data.scenarios import CurriculumStage, DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.ood_categories import OODCategory
from hydroswarm.training.ood_labels import classify_ood_category, ood_class_target
from hydroswarm.training.targets_v2 import validate_targets_v2

_VALIDATED = frozenset({"reference-topology-hash"})


def _generate(network, **overrides):
    generator = WNTRScenarioGenerator()
    base = dict(
        seed=100, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
        source_node="J2", sensor_count=4,
    )
    base.update(overrides)
    return generator.generate(network, ScenarioGenerationConfig(**base))


def test_ordinary_scenario_is_in_distribution(tmp_path) -> None:
    network = build_wntr_network()
    # The corpus generator's own worst in-distribution degradation knobs
    # (scripts/generate_cycle_b_corpus.py's _degradation_probabilities for
    # CurriculumStage.ADVERSARIAL) -- passing stage= alone does not
    # auto-wire these; the real generator computes and passes them
    # explicitly, so this test mirrors that instead of relying on stage=
    # having a side effect it does not have.
    scenario = _generate(
        network, stage=CurriculumStage.ADVERSARIAL, missing_probability=0.08,
        frozen_probability=0.06, communication_outage_probability=0.06, unit_mismatch_probability=0.02,
    )
    category = classify_ood_category(
        scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.NONE


def test_unseen_topology_hash_is_flagged() -> None:
    network = build_wntr_network()
    scenario = _generate(network)
    category = classify_ood_category(
        scenario, topology_hash="some-other-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.UNSEEN_TOPOLOGY


def test_unseen_topology_is_not_flagged_when_a_broader_artifact_covers_it() -> None:
    network = build_wntr_network()
    scenario = _generate(network)
    category = classify_ood_category(
        scenario, topology_hash="some-other-hash", validated_topology_hashes=_VALIDATED,
        broader_validated_artifact_exists=True,
    )
    assert category != OODCategory.UNSEEN_TOPOLOGY


def test_extreme_demand_is_flagged() -> None:
    network = build_wntr_network()
    scenario = _generate(network, demand_regimes=(2.5,))
    category = classify_ood_category(
        scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.EXTREME_DEMAND


def test_tank_state_shift_is_flagged() -> None:
    network = build_wntr_network()
    scenario = _generate(network, tank_level_variation_fraction=0.6)
    category = classify_ood_category(
        scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.TANK_STATE_SHIFT


def test_roughness_mismatch_is_flagged() -> None:
    network = build_wntr_network()
    scenario = _generate(network, roughness_variation_fraction=0.4)
    category = classify_ood_category(
        scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.ROUGHNESS_MISMATCH


def test_severe_missingness_is_flagged() -> None:
    network = build_wntr_network()
    scenario = _generate(network, missing_probability=0.6)
    category = classify_ood_category(
        scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.SEVERE_MISSINGNESS


def test_frozen_drifting_sensor_is_flagged() -> None:
    network = build_wntr_network()
    scenario = _generate(network, frozen_probability=1.0, sensor_count=6)
    category = classify_ood_category(
        scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.FROZEN_DRIFTING_SENSOR


def test_priority_order_prefers_the_most_structural_category() -> None:
    # A scenario that is BOTH on an unseen topology AND has extreme demand
    # must be labeled by the higher-priority (more structural) category.
    network = build_wntr_network()
    scenario = _generate(network, demand_regimes=(2.5,))
    category = classify_ood_category(
        scenario, topology_hash="some-other-hash", validated_topology_hashes=_VALIDATED
    )
    assert category == OODCategory.UNSEEN_TOPOLOGY


def test_ood_class_target_is_governed_and_matches_the_classified_category() -> None:
    for category in OODCategory:
        target = ood_class_target(category)
        validate_targets_v2(target)  # must not raise for any real category, including the highest index
        assert isinstance(target["ood_class"], torch.Tensor)


def test_every_category_maps_to_a_distinct_index() -> None:
    indices = {int(ood_class_target(category)["ood_class"]) for category in OODCategory}
    assert len(indices) == len(list(OODCategory))
