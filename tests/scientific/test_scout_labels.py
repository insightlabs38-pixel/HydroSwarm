"""Task 2.3: Scout label generation glue over rank_sample_locations."""

from __future__ import annotations

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.sampling.active import SamplingConstraints
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.scout_labels import build_signature_artifact_for_network, generate_scout_label


def _fast_artifact(network, cache_dir):
    cache = SignatureCache(cache_dir)
    key = SignatureCacheKey(
        network_hash="test-net", hydraulic_state_hash="test-state", simulator_version="test",
        configuration_hash="scout-test-v1", sensor_layout_hash="all-junctions",
    )
    # Reduced bins so the test only runs one WNTR simulation per source
    # candidate instead of the full combinatorial hypothesis space.
    return build_signature_artifact_for_network(
        network, cache, key=key,
        sample_times_seconds=(0, 3600, 7200),
        start_time_bins=(0,), duration_bins=(60,), strength_bins=(1.0,),
    )


def _scenario(network, *, source_node: str, seed: int):
    generator = WNTRScenarioGenerator()
    return generator.generate(
        network,
        ScenarioGenerationConfig(
            seed=seed, network_id="ref", network_family="reference", split=DatasetSplit.TRAIN,
            source_node=source_node, sensor_count=2, start_time_bins_min=(0,), duration_bins_min=(60,),
            strength_bins=(1.0,),
        ),
    )


def test_artifact_builds_and_caches(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    assert not artifact.cache_hit
    reloaded = _fast_artifact(network, tmp_path / "cache")
    assert reloaded.cache_hit
    assert reloaded.artifact_hash == artifact.artifact_hash


def test_scout_label_is_well_formed(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)

    label = generate_scout_label(scenario, artifact)
    # With the reduced single-hypothesis-per-source bins used here, the
    # classical posterior is often already near-certain (this scenario's
    # concentration signature matches its source's signature almost
    # exactly), so Scout may correctly decline to recommend a redundant
    # sample -- that is the *correct* behavior, not a bug. Either outcome
    # must be internally consistent.
    if label.sample_node_id is not None:
        assert label.sample_node_id in artifact.sensor_nodes
        assert label.should_continue_sampling is True
    else:
        assert label.should_continue_sampling is False
    assert label.information_gain_bits >= 0.0
    assert 0.0 <= label.candidate_reduction_fraction <= 1.0


def test_reindex_to_signature_grid_is_usable_directly_with_localize_with_signatures(tmp_path) -> None:
    # core-issues3.txt Phase 12 Stage D: an operational Scout-policy
    # comparison needs per-node marginal source probabilities (a different
    # question from "which node to sample next"), which
    # hydroswarm.classical.signatures.localize_with_signatures already
    # computes from the exact same reindexed observations this module
    # feeds bayesian_source_posterior -- reindex_to_signature_grid is
    # public specifically so a second caller does not have to duplicate
    # the reindexing logic.
    from hydroswarm.classical.signatures import localize_with_signatures
    from hydroswarm.training.scout_labels import reindex_to_signature_grid

    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)

    observed, valid = reindex_to_signature_grid(scenario, artifact.sensor_nodes, artifact.sample_times_seconds)
    result = localize_with_signatures(observed, artifact, observation_mask=valid, noise_scale=0.5)
    assert set(result.source_probabilities) == {hypothesis.source_node for hypothesis in artifact.hypotheses}
    assert abs(sum(result.source_probabilities.values()) - 1.0) < 1e-6


def test_scout_label_recommends_a_sample_when_evidence_is_genuinely_ambiguous(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)

    # A large noise_scale makes every hypothesis look nearly equally likely
    # (residual differences are swamped by "sensor noise"), so the posterior
    # should stay high-entropy and a sample should be worth recommending.
    label = generate_scout_label(scenario, artifact, noise_scale_mg_l=50.0)
    assert label.posterior_entropy_bits > 0.5
    assert label.sample_node_id is not None
    assert label.should_continue_sampling is True


def test_scout_label_excludes_already_sampled_nodes(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=11)

    already_sampled = artifact.sensor_nodes[:-1]  # leave exactly one candidate
    label = generate_scout_label(scenario, artifact, already_sampled=already_sampled)
    assert label.sample_node_id == artifact.sensor_nodes[-1] or label.sample_node_id is None


def test_scout_label_stops_when_all_candidates_are_sampled(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=12)

    label = generate_scout_label(scenario, artifact, already_sampled=artifact.sensor_nodes)
    assert label.sample_node_id is None
    assert label.should_continue_sampling is False
    assert label.stop_reason == "no_accessible_sample"


def test_scout_label_respects_accessibility_constraints(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=13)

    inaccessible = {node: False for node in artifact.sensor_nodes}
    constraints = SamplingConstraints(accessible=inaccessible)
    label = generate_scout_label(scenario, artifact, constraints=constraints)
    assert label.sample_node_id is None
    assert label.should_continue_sampling is False


def test_information_gain_is_nonnegative_within_tolerance(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    # Fixed per-source seeds, not hash(source): Python's built-in hash() on
    # a str is randomized per-process (PYTHONHASHSEED) by default, so this
    # test's own seed silently varied run-to-run -- occasionally landing on
    # a seed whose randomized hydraulics produced a numerically unstable
    # EPANET simulation (SimulationUnstableError), unrelated to the actual
    # thing under test (information-gain nonnegativity). Confirmed flaky by
    # repeated runs before this fix.
    for index, source in enumerate(sorted(network.junction_name_list)):
        scenario = _scenario(network, source_node=source, seed=20 + index)
        label = generate_scout_label(scenario, artifact)
        assert label.information_gain_bits >= -1e-9


def test_scout_labels_are_deterministic_for_the_same_scenario(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J3", seed=14)

    first = generate_scout_label(scenario, artifact)
    second = generate_scout_label(scenario, artifact)
    assert first == second
