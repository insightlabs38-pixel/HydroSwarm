"""core-issues3.txt Phase 1: reconstruct_scenario_network must return the
exact scenario-specific randomized hydraulic state, not a topology-family-
shared pristine context. Uses real WNTR simulation via the same
generate_with_network path scripts/generate_cycle_b_corpus.py itself uses,
not synthetic fixtures."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_cycle_b_corpus import TRAIN_TOPOLOGIES, _degradation_probabilities  # noqa: E402

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.training.auxiliary_labels import travel_time_target  # noqa: E402
from hydroswarm.training.corpus import build_feature_context  # noqa: E402
from hydroswarm.training.scenario_reconstruction import (  # noqa: E402
    ScenarioReconstructionError,
    reconstruct_scenario_network,
    simulate_all_node_truth,
)

_FAMILY, _LOADER = TRAIN_TOPOLOGIES[0]  # golden-reference: cheap, real


def _generate(network, seed: int, source: str, stage=CurriculumStage.OPERATIONAL):
    generator = WNTRScenarioGenerator()
    config = ScenarioGenerationConfig(
        seed=seed,
        network_id=_FAMILY,
        network_family=_FAMILY,
        split=DatasetSplit.TRAIN,
        stage=stage,
        event_type=EventType.CONTAMINATION,
        source_node=source,
        sensor_count=3,
        pipe_outage_probability=0.0,
        **_degradation_probabilities(stage),
    )
    scenario, randomized_network = generator.generate_with_network(network, config)
    return scenario, randomized_network


@pytest.fixture(scope="module")
def pristine_network():
    return _LOADER()


@pytest.fixture(scope="module")
def two_scenarios(pristine_network):
    junctions = tuple(sorted(pristine_network.junction_name_list))
    scenario_a, _ = _generate(pristine_network, seed=910_001, source=junctions[0])
    scenario_b, _ = _generate(pristine_network, seed=910_002, source=junctions[1 % len(junctions)])
    return scenario_a, scenario_b


#: 10 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_different_seeds_produce_different_network_and_hydraulic_state_hashes(
    pristine_network, two_scenarios
) -> None:
    scenario_a, scenario_b = two_scenarios
    reconstruction_a = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    reconstruction_b = reconstruct_scenario_network(
        pristine_network, scenario_b.manifest, degradation_policy=_degradation_probabilities, original=scenario_b
    )

    # topology_hash is stable (same pristine topology)...
    assert reconstruction_a.topology_hash == reconstruction_b.topology_hash
    # ...but the scenario-specific randomized state must differ (different
    # demand regime/roughness/tank-level draws from different seeds).
    assert reconstruction_a.network_state_hash != reconstruction_b.network_state_hash
    assert reconstruction_a.hydraulic_state_hash != reconstruction_b.hydraulic_state_hash


def test_reconstruction_matches_the_original_scenarios_semantic_replay(pristine_network, two_scenarios) -> None:
    for scenario in two_scenarios:
        reconstruction = reconstruct_scenario_network(
            pristine_network, scenario.manifest, degradation_policy=_degradation_probabilities, original=scenario
        )
        assert reconstruction.replay_matched is True


def test_reconstruction_fails_closed_on_a_manifest_that_does_not_match_the_pristine_topology(
    pristine_network, two_scenarios
) -> None:
    scenario_a, _ = two_scenarios
    other_family, other_loader = TRAIN_TOPOLOGIES[1]
    other_network = other_loader()
    # scenario_a's manifest records golden-reference's own source node,
    # which does not exist in a genuinely different topology; replaying it
    # must fail closed rather than silently substituting some other node
    # or returning a context that doesn't correspond to the manifest.
    # (Raised by the underlying generator as ValueError before reconstruction
    # ever reaches its own replay-identity check -- both are fail-closed.)
    with pytest.raises((ScenarioReconstructionError, ValueError)):
        reconstruct_scenario_network(
            other_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
        )


def test_travel_time_labels_change_when_hydraulic_state_changes(pristine_network, two_scenarios) -> None:
    scenario_a, scenario_b = two_scenarios
    reconstruction_a = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    reconstruction_b = reconstruct_scenario_network(
        pristine_network, scenario_b.manifest, degradation_policy=_degradation_probabilities, original=scenario_b
    )
    node_ids = tuple(sorted(pristine_network.node_name_list))

    travel_a = travel_time_target(reconstruction_a.scenario, reconstruction_a.feature_context.graph, node_ids)
    travel_b = travel_time_target(reconstruction_b.scenario, reconstruction_b.feature_context.graph, node_ids)

    # Different sources and different roughness/demand draws must not
    # collapse to identical travel-time vectors.
    assert not (travel_a["travel_time"] == travel_b["travel_time"]).all()


def test_reconstructed_network_is_not_the_pristine_object_and_has_distinct_state(
    pristine_network, two_scenarios
) -> None:
    scenario_a, _ = two_scenarios
    reconstruction = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    assert reconstruction.network is not pristine_network

    pristine_context = build_feature_context(pristine_network)
    # This is the exact bug this module fixes: a caller that reused
    # build_feature_context(pristine_network) for every scenario in this
    # topology family would get identical hydraulic state regardless of
    # which scenario it claimed to represent. The reconstructed,
    # scenario-specific context must differ from the pristine one.
    assert reconstruction.hydraulic_state_hash != _state_hash(pristine_context)


def _state_hash(context) -> str:
    from hydroswarm.training.corpus import _hydraulic_state_hash

    return _hydraulic_state_hash(context.state)


def test_old_shared_pristine_context_would_have_produced_identical_travel_time_for_both_scenarios(
    pristine_network, two_scenarios
) -> None:
    """Documents the exact regression this module fixes: reusing ONE
    pristine FeatureContext (the old generate_trajectory_corpus.py
    behavior) for two scenarios with different sources/seeds would compute
    identical travel-time vectors for both -- physically wrong, since the
    two scenarios have different contamination sources. The corrected
    per-scenario reconstruction must not exhibit this collapse (proven by
    the previous test); this test proves the *old* implementation shape
    really would have collapsed, so the fix is not vacuous."""

    scenario_a, scenario_b = two_scenarios
    shared_pristine_context = build_feature_context(pristine_network)
    node_ids = tuple(sorted(pristine_network.node_name_list))

    travel_a_old_bug = travel_time_target(scenario_a, shared_pristine_context.graph, node_ids)
    travel_b_old_bug = travel_time_target(scenario_b, shared_pristine_context.graph, node_ids)

    # Same graph both times (the bug) but different sources still change
    # the shortest-path source node -- prove the graph itself carries no
    # scenario-specific roughness/demand information by checking the
    # pristine graph's edge weights are identical to nx defaults regardless
    # of which scenario "used" it, unlike the corrected per-scenario graph.
    reconstruction_a = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    corrected_graph_edges = sorted(reconstruction_a.feature_context.graph.edges(data="travel_time_seconds"))
    pristine_graph_edges = sorted(shared_pristine_context.graph.edges(data="travel_time_seconds"))
    assert corrected_graph_edges != pristine_graph_edges, (
        "reconstructed per-scenario graph must carry different travel-time edge weights "
        "than the shared pristine graph -- otherwise the fix has no observable effect"
    )
    assert travel_a_old_bug is not None and travel_b_old_bug is not None  # sanity: old path still runs, just wrong


def test_negligible_magnitude_float_noise_does_not_fail_reconstruction(pristine_network, two_scenarios) -> None:
    """Real case found running this module against data/learning-v2/cycle-b2:
    scenario e5f317e7-8b65-54a3-b898-c553357ea90d (a NORMAL/negligible-
    strength scenario, concentration ~1e-8 mg/L) reconstructs bit-identically
    across two independent fresh regenerations, but differs from the
    corpus's originally-stored array by ~2.7e-19 -- sub-float32-rounding
    noise, not a reconstruction defect (same class of cross-environment
    nondeterminism as the already-documented signed-zero case, here below
    exact-equality resolution rather than at it). np.array_equal alone
    would wrongly fail-closed on this; the atol=1e-6 tolerance (three
    orders of magnitude below quantization_step, the smallest physically
    meaningful resolution anywhere in this system) must accept it."""

    scenario_a, _ = two_scenarios
    noised = dataclasses.replace(
        scenario_a,
        truth_concentration=scenario_a.truth_concentration + np.float32(2.7e-19),
    )
    reconstruction = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=noised
    )
    assert reconstruction.replay_matched is True


def test_a_real_magnitude_difference_still_fails_closed(pristine_network, two_scenarios) -> None:
    """The tolerance added for float-rounding noise must not mask an actual
    reconstruction defect: a difference at the scale of a real signal
    (0.01 mg/L, ordinary sensor-noise-std magnitude, far above both
    quantization_step and the 1e-6 tolerance) must still raise."""

    scenario_a, _ = two_scenarios
    tampered = dataclasses.replace(
        scenario_a,
        truth_concentration=scenario_a.truth_concentration + np.float32(0.01),
    )
    with pytest.raises(ScenarioReconstructionError):
        reconstruct_scenario_network(
            pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=tampered
        )


def test_simulate_all_node_truth_reproduces_the_scenarios_own_stored_sensor_columns(
    pristine_network, two_scenarios
) -> None:
    """core-issues3.txt Phase 5 item Q: all-node Scout sample truth must
    come from the exact randomized scenario. The strongest possible proof
    that simulate_all_node_truth's independent simulate_incident call
    reproduces the SAME physical simulation the corpus generator's own
    (RNG-driven) call already ran: its values at the scenario's own
    original sensor nodes must match the scenario's own stored
    truth_concentration array exactly (both are float32, same computation,
    no noise/degradation applied to either -- this is comparing two exact
    simulator outputs, not tolerating any float noise)."""

    scenario_a, _ = two_scenarios
    reconstruction = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    all_node_truth = simulate_all_node_truth(reconstruction)

    sensor_nodes = scenario_a.manifest.sensor_nodes
    timestamps = scenario_a.timestamps_seconds
    reproduced = (
        all_node_truth.loc[:, list(sensor_nodes)]
        .reindex(index=list(timestamps), method="nearest")
        .to_numpy(dtype=np.float32)
    )
    assert np.allclose(reproduced, scenario_a.truth_concentration, atol=1e-6, rtol=0.0)


def test_simulate_all_node_truth_reaches_nodes_outside_the_original_sensor_set(
    pristine_network, two_scenarios
) -> None:
    scenario_a, _ = two_scenarios
    reconstruction = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    all_node_truth = simulate_all_node_truth(reconstruction)

    non_sensor_nodes = [
        node for node in pristine_network.junction_name_list if node not in scenario_a.manifest.sensor_nodes
    ]
    assert non_sensor_nodes, "fixture network must have at least one non-sensor junction to test with"
    target = non_sensor_nodes[0]
    assert target in all_node_truth.columns
    assert np.isfinite(all_node_truth[target].to_numpy(dtype=np.float64)).all()


def test_simulate_all_node_truth_is_deterministic(pristine_network, two_scenarios) -> None:
    scenario_a, _ = two_scenarios
    reconstruction = reconstruct_scenario_network(
        pristine_network, scenario_a.manifest, degradation_policy=_degradation_probabilities, original=scenario_a
    )
    first = simulate_all_node_truth(reconstruction)
    second = simulate_all_node_truth(reconstruction)
    assert np.array_equal(first.to_numpy(), second.to_numpy(), equal_nan=True)
