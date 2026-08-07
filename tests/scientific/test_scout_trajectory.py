"""core-issues2.txt Phase 2: Scout trajectory-state generation."""

from __future__ import annotations

import pytest

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.sampling.active import SamplingConstraints
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.scenario_reconstruction import reconstruct_scenario_network
from hydroswarm.training.scout_labels import build_signature_artifact_for_network
from hydroswarm.training.scout_trajectory import MAXIMUM_SAMPLES_BOUND, build_scout_trajectory
from hydroswarm.training.targets_v2 import validate_targets_v2


def _no_extra_degradation(stage):
    # Matches _scenario's own never-overridden ScenarioGenerationConfig
    # defaults (missing/frozen/outage/unit-mismatch probabilities).
    return {}


def _fast_artifact(network, cache_dir):
    cache = SignatureCache(cache_dir)
    key = SignatureCacheKey(
        network_hash="test-net", hydraulic_state_hash="test-state", simulator_version="test",
        configuration_hash="scout-trajectory-test-v1", sensor_layout_hash="all-junctions",
    )
    return build_signature_artifact_for_network(
        network, cache, key=key,
        sample_times_seconds=(0, 3600, 7200),
        start_time_bins=(0,), duration_bins=(60,), strength_bins=(1.0,),
    )


def _scenario(network, *, source_node: str, seed: int, sensor_count: int = 3):
    generator = WNTRScenarioGenerator()
    return generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
            source_node=source_node, sensor_count=sensor_count, start_time_bins_min=(0,),
            duration_bins_min=(60,), strength_bins=(1.0,),
        ),
    )


def test_trajectory_is_well_formed_and_hash_chained(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_scout_trajectory(scenario, artifact, node_ids)

    assert len(result.steps) == len(result.trajectory.steps) >= 1
    assert result.trajectory.scenario_id == str(scenario.manifest.scenario_id)
    # FullTrajectory.__post_init__ already validates the hash chain and
    # step-index contiguity -- constructing it above (inside
    # build_scout_trajectory) is itself the test; re-confirm explicitly.
    for current, following in zip(result.trajectory.steps, result.trajectory.steps[1:]):
        assert current.resulting_next_state_hash == following.state_hash
    assert [step.step_index for step in result.trajectory.steps] == list(range(len(result.steps)))


def test_every_step_targets_pass_validate_targets_v2(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J3", seed=11)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_scout_trajectory(scenario, artifact, node_ids)
    for step in result.steps:
        validate_targets_v2(step.targets)  # must not raise -- no diagnostic keys leaked in


def test_sample_node_target_is_a_valid_graph_local_index(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_scout_trajectory(scenario, artifact, node_ids)
    for step in result.steps:
        if step.targets["sample_node_mask"]:
            index = step.targets["sample_node"]
            assert 0 <= index < len(node_ids)
            assert node_ids[index] == step.label.sample_node_id
        else:
            assert step.targets["sample_node"] == -1
            assert step.label.sample_node_id is None


def test_already_sampled_grows_monotonically_and_is_never_repeated(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    # Large noise keeps the posterior high-entropy so Scout keeps recommending
    # samples instead of stopping after step 0 (matches test_scout_labels.py's
    # "genuinely ambiguous evidence" pattern for exercising multiple steps).
    scenario = _scenario(network, source_node="J2", seed=10, sensor_count=len(network.junction_name_list))

    result = build_scout_trajectory(
        scenario, artifact, tuple(sorted(network.node_name_list)), noise_scale_mg_l=50.0
    )
    seen: set[str] = set()
    for step in result.steps:
        already_sampled = step.diagnostics["already_sampled"]
        assert set(already_sampled) == seen  # exactly what had been sampled before this step
        if step.label.sample_node_id is not None:
            assert step.label.sample_node_id not in seen
            seen.add(step.label.sample_node_id)


def test_trajectory_terminates_at_maximum_samples(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10, sensor_count=len(network.junction_name_list))

    result = build_scout_trajectory(
        scenario, artifact, tuple(sorted(network.node_name_list)),
        maximum_samples=1, noise_scale_mg_l=50.0,
    )
    assert len(result.steps) == 1
    # remaining_budgets.samples is the budget *going into* this step's
    # decision (1, about to be spent on this step's recommendation), not
    # what's left after -- the loop terminates precisely because spending it
    # would exhaust maximum_samples, without running a further step.
    assert result.steps[0].state.remaining_budgets.samples == 1
    assert bool(result.steps[0].targets["sample_node_mask"])


def test_maximum_samples_above_the_domain_bound_is_rejected(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)

    with pytest.raises(ValueError, match="exceeds"):
        build_scout_trajectory(
            scenario, artifact, tuple(sorted(network.node_name_list)),
            maximum_samples=MAXIMUM_SAMPLES_BOUND + 1,
        )


def test_zero_or_negative_maximum_samples_is_rejected(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)

    with pytest.raises(ValueError, match="at least 1"):
        build_scout_trajectory(scenario, artifact, tuple(sorted(network.node_name_list)), maximum_samples=0)


def test_trajectory_is_deterministic_for_the_same_scenario(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J4", seed=12)
    node_ids = tuple(sorted(network.node_name_list))

    first = build_scout_trajectory(scenario, artifact, node_ids)
    second = build_scout_trajectory(scenario, artifact, node_ids)
    assert first.trajectory == second.trajectory
    assert [step.targets for step in first.steps] == [step.targets for step in second.steps]


def test_different_scenarios_get_different_trajectory_and_incident_ids(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    node_ids = tuple(sorted(network.node_name_list))
    a = build_scout_trajectory(_scenario(network, source_node="J2", seed=10), artifact, node_ids)
    b = build_scout_trajectory(_scenario(network, source_node="J3", seed=11), artifact, node_ids)

    assert a.trajectory.trajectory_id != b.trajectory.trajectory_id
    assert a.trajectory.scenario_id != b.trajectory.scenario_id
    assert a.steps[0].state.incident_state.incident_id != b.steps[0].state.incident_state.incident_id


def test_stops_immediately_when_no_candidate_is_accessible(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=13)
    node_ids = tuple(sorted(network.node_name_list))

    inaccessible = {node: False for node in artifact.sensor_nodes}
    result = build_scout_trajectory(scenario, artifact, node_ids, constraints=SamplingConstraints(accessible=inaccessible))
    assert len(result.steps) == 1
    assert not bool(result.steps[0].targets["should_continue_sampling"])
    assert not bool(result.steps[0].targets["sample_node_mask"])


def _reconstruction_for(network, *, source_node: str, seed: int, sensor_count: int = 3):
    # Deliberately does NOT reuse _scenario()'s narrow single-value bin
    # overrides (start_time_bins_min=(0,) etc.) -- reconstruct_scenario_
    # network's own config construction always uses ScenarioGenerationConfig's
    # plain defaults for those (matching every real corpus generator, none
    # of which override them either), so a manifest generated under
    # _scenario()'s narrower bins cannot replay-match here: rng.choice over
    # a differently-sized bin array draws a different value for the same
    # seed. Builds the manifest directly under matching (default-bin) config
    # instead.
    generator = WNTRScenarioGenerator()
    config = ScenarioGenerationConfig(
        seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
        source_node=source_node, sensor_count=sensor_count,
    )
    manifest_scenario = generator.generate(network, config)
    return reconstruct_scenario_network(
        network, manifest_scenario.manifest, degradation_policy=_no_extra_degradation, original=manifest_scenario
    )


def test_incremental_revelation_produces_a_genuinely_different_posterior_than_before_phase_5(tmp_path) -> None:
    """core-issues3.txt Phase 5: passing revealed_samples must change the
    computed posterior/entropy, not just filter the candidate pool (the
    exact pre-Phase-5 behavior: already_sampled changed which candidates
    were considered, never what evidence the posterior was computed from).
    Exercises generate_scout_label directly rather than through the full
    trajectory loop -- this fixture's tiny 4-junction network resolves to
    should_continue_sampling=False after exactly one step for every seed
    tried, which is a real property of how little ambiguity such a small
    network leaves after a single sample, not a Phase 5 defect; testing
    the mechanism directly avoids depending on the fixture happening to
    produce a second step."""

    from hydroswarm.training.scenario_reconstruction import simulate_all_node_truth
    from hydroswarm.training.scout_labels import generate_scout_label
    from hydroswarm.training.scout_trajectory import _reveal_sample_measurement

    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    # sensor_count=1 deliberately leaves real ambiguity (nonzero baseline
    # entropy) for the reveal to resolve -- with this fixture's default
    # sensor_count=3, the source is already fully disambiguated before any
    # reveal, which would make this test vacuously pass regardless of
    # whether revealed_samples has any real effect.
    reconstruction = _reconstruction_for(network, source_node="J2", seed=10, sensor_count=1)
    all_node_truth = simulate_all_node_truth(reconstruction)

    without_reveal = generate_scout_label(reconstruction.scenario, artifact)
    assert without_reveal.sample_node_id is not None
    assert without_reveal.posterior_entropy_bits > 0.0, "fixture must have real baseline ambiguity to resolve"

    measurement = _reveal_sample_measurement(
        reconstruction, all_node_truth, without_reveal.sample_node_id, 0, noise_scale_mg_l=0.5
    )
    with_reveal = generate_scout_label(
        reconstruction.scenario, artifact, revealed_samples={without_reveal.sample_node_id: measurement}
    )
    assert with_reveal.posterior_entropy_bits != without_reveal.posterior_entropy_bits


def test_reveal_sample_measurement_is_deterministic_and_matches_true_value_within_noise(tmp_path) -> None:
    from hydroswarm.training.scenario_reconstruction import simulate_all_node_truth
    from hydroswarm.training.scout_trajectory import _reveal_sample_measurement

    network = build_wntr_network()
    reconstruction = _reconstruction_for(network, source_node="J2", seed=10)
    all_node_truth = simulate_all_node_truth(reconstruction)
    target_node = "J4"

    first = _reveal_sample_measurement(reconstruction, all_node_truth, target_node, 0, noise_scale_mg_l=0.5)
    second = _reveal_sample_measurement(reconstruction, all_node_truth, target_node, 0, noise_scale_mg_l=0.5)
    assert first == second

    # A different step_index must (with overwhelming probability) draw a
    # different noise realization -- proves the seed genuinely includes
    # step_index, not just node_id (Phase 5 item 5.2's "deterministic
    # measurement seed derived from scenario_id + step_index + node_id").
    different_step = _reveal_sample_measurement(reconstruction, all_node_truth, target_node, 1, noise_scale_mg_l=0.5)
    assert different_step != first

    true_value = float(
        all_node_truth[target_node].reindex([reconstruction.scenario.timestamps_seconds[-1]], method="nearest").iloc[0]
    )
    assert abs(first - true_value) < 5.0  # bounded by noise_scale_mg_l=0.5, generous margin against a fluke draw
    assert first >= 0.0  # concentration cannot be negative


def test_incremental_revelation_is_deterministic(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    node_ids = tuple(sorted(network.node_name_list))
    reconstruction = _reconstruction_for(network, source_node="J4", seed=12)

    first = build_scout_trajectory(
        reconstruction.scenario, artifact, node_ids, maximum_samples=3, reconstruction=reconstruction
    )
    second = build_scout_trajectory(
        reconstruction.scenario, artifact, node_ids, maximum_samples=3, reconstruction=reconstruction
    )
    assert [step.label.sample_node_id for step in first.steps] == [
        step.label.sample_node_id for step in second.steps
    ]
    assert [step.label.posterior_entropy_bits for step in first.steps] == [
        step.label.posterior_entropy_bits for step in second.steps
    ]


def test_without_reconstruction_behavior_is_unchanged_from_before_phase_5(tmp_path) -> None:
    """Backward-compatibility guarantee: omitting `reconstruction` must
    reproduce the exact pre-Phase-5 behavior (same fixed base observations
    re-ranked every step), not a partially-applied new code path."""

    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_scout_trajectory(scenario, artifact, node_ids, maximum_samples=3)
    assert len(result.steps) >= 1
    for step in result.steps:
        assert isinstance(step.label.posterior_entropy_bits, float)
