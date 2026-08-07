"""core-issues2.txt Phase 4: learned OOD category-label generation."""

from __future__ import annotations

import torch

from hydroswarm.data.scenarios import CurriculumStage, DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.ood_categories import OOD_CATEGORY_BEHAVIOR, OODCategory
from hydroswarm.training.ood_labels import (
    OOD_TRIGGERING_CONFIG_OVERRIDES,
    SUPPORTED_OOD_CATEGORIES,
    UNSUPPORTED_OOD_CATEGORIES,
    classify_ood_category,
    ood_class_target,
)
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


def test_supported_and_unsupported_ood_categories_partition_the_full_taxonomy() -> None:
    assert SUPPORTED_OOD_CATEGORIES | UNSUPPORTED_OOD_CATEGORIES == frozenset(OODCategory)
    assert SUPPORTED_OOD_CATEGORIES & UNSUPPORTED_OOD_CATEGORIES == frozenset()


def test_classify_ood_category_never_returns_an_unsupported_category(tmp_path) -> None:
    """core-issues3.txt Phase 6.2: the SUPPORTED_OOD_CATEGORIES registry
    must match classify_ood_category's real behavior, not silently drift
    from it -- exercised across every configuration knob combination this
    module's own threshold table names, not just the individually-flagged
    cases each already have their own test."""

    network = build_wntr_network()
    configurations = [
        {},
        {"stage": CurriculumStage.ADVERSARIAL, "missing_probability": 0.08, "frozen_probability": 0.06,
         "communication_outage_probability": 0.06, "unit_mismatch_probability": 0.02},
        *OOD_TRIGGERING_CONFIG_OVERRIDES.values(),
    ]
    for overrides in configurations:
        for topology_hash, validated in (
            ("reference-topology-hash", _VALIDATED),
            ("some-other-hash", _VALIDATED),
        ):
            scenario = _generate(network, **overrides)
            category = classify_ood_category(
                scenario, topology_hash=topology_hash, validated_topology_hashes=validated
            )
            assert category in SUPPORTED_OOD_CATEGORIES


def test_every_recipe_override_reliably_triggers_its_category(tmp_path) -> None:
    """core-issues3.txt Phase 6.3: OOD_TRIGGERING_CONFIG_OVERRIDES is the
    reusable recipe a future balanced-OOD corpus-generation pass would use
    -- verified against classify_ood_category's real behavior across
    several seeds each, not asserted from the threshold table alone."""

    network = build_wntr_network()
    for category, overrides in OOD_TRIGGERING_CONFIG_OVERRIDES.items():
        for seed in (100, 200, 300):
            scenario = _generate(network, seed=seed, **overrides)
            classified = classify_ood_category(
                scenario, topology_hash="reference-topology-hash", validated_topology_hashes=_VALIDATED
            )
            assert classified == category, f"{category} recipe {overrides} gave {classified} at seed {seed}"


def test_every_non_none_category_currently_suppresses_planning_and_invalidates_calibration() -> None:
    """core-issues3.txt Phase 6.6: guards against silently collapsing every
    non-NONE category to OUTSIDE_VALIDATED_RANGE severity in caller code
    (e.g. full_trajectory.py's `category != OODCategory.NONE` check) UNLESS
    the governed OOD_CATEGORY_BEHAVIOR table itself says so for every such
    category. Currently true for the entire taxonomy (no CAUTION-only
    category is defined yet) -- if a future category is added with
    planning_permitted=True (a partial-degradation CAUTION case), any
    `category != NONE` shortcut elsewhere must be revisited alongside it,
    which this test would then catch by failing."""

    for category in OODCategory:
        if category == OODCategory.NONE:
            continue
        behavior = OOD_CATEGORY_BEHAVIOR[category]
        assert behavior.planning_permitted is False
        assert behavior.calibration_valid is False
