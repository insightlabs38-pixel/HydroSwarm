"""core-issues2.txt Phase 2: Scout trajectory-state generation."""

from __future__ import annotations

import pytest

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.sampling.active import SamplingConstraints
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.scout_labels import build_signature_artifact_for_network
from hydroswarm.training.scout_trajectory import MAXIMUM_SAMPLES_BOUND, build_scout_trajectory
from hydroswarm.training.targets_v2 import validate_targets_v2


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
    node_ids = tuple(sorted(network.junction_name_list))

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
    node_ids = tuple(sorted(network.junction_name_list))

    result = build_scout_trajectory(scenario, artifact, node_ids)
    for step in result.steps:
        validate_targets_v2(step.targets)  # must not raise -- no diagnostic keys leaked in


def test_sample_node_target_is_a_valid_graph_local_index(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    node_ids = tuple(sorted(network.junction_name_list))

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
        scenario, artifact, tuple(sorted(network.junction_name_list)), noise_scale_mg_l=50.0
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
        scenario, artifact, tuple(sorted(network.junction_name_list)),
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
            scenario, artifact, tuple(sorted(network.junction_name_list)),
            maximum_samples=MAXIMUM_SAMPLES_BOUND + 1,
        )


def test_zero_or_negative_maximum_samples_is_rejected(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)

    with pytest.raises(ValueError, match="at least 1"):
        build_scout_trajectory(scenario, artifact, tuple(sorted(network.junction_name_list)), maximum_samples=0)


def test_trajectory_is_deterministic_for_the_same_scenario(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J4", seed=12)
    node_ids = tuple(sorted(network.junction_name_list))

    first = build_scout_trajectory(scenario, artifact, node_ids)
    second = build_scout_trajectory(scenario, artifact, node_ids)
    assert first.trajectory == second.trajectory
    assert [step.targets for step in first.steps] == [step.targets for step in second.steps]


def test_different_scenarios_get_different_trajectory_and_incident_ids(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    node_ids = tuple(sorted(network.junction_name_list))
    a = build_scout_trajectory(_scenario(network, source_node="J2", seed=10), artifact, node_ids)
    b = build_scout_trajectory(_scenario(network, source_node="J3", seed=11), artifact, node_ids)

    assert a.trajectory.trajectory_id != b.trajectory.trajectory_id
    assert a.trajectory.scenario_id != b.trajectory.scenario_id
    assert a.steps[0].state.incident_state.incident_id != b.steps[0].state.incident_state.incident_id


def test_stops_immediately_when_no_candidate_is_accessible(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=13)
    node_ids = tuple(sorted(network.junction_name_list))

    inaccessible = {node: False for node in artifact.sensor_nodes}
    result = build_scout_trajectory(scenario, artifact, node_ids, constraints=SamplingConstraints(accessible=inaccessible))
    assert len(result.steps) == 1
    assert not bool(result.steps[0].targets["should_continue_sampling"])
    assert not bool(result.steps[0].targets["sample_node_mask"])
