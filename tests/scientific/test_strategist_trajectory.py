"""core-issues2.txt Phase 3: Strategist trajectory-state generation."""

from __future__ import annotations

from hydroswarm.classical.signatures import SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, ScenarioGenerationConfig, WNTRScenarioGenerator
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import build_feature_context
from hydroswarm.training.scout_labels import build_signature_artifact_for_network
from hydroswarm.training.strategist_trajectory import ACTION_TEMPLATES, build_strategist_trajectory
from hydroswarm.training.targets_v2 import validate_targets_v2


def _fast_artifact(network, cache_dir):
    cache = SignatureCache(cache_dir)
    key = SignatureCacheKey(
        network_hash="test-net", hydraulic_state_hash="test-state", simulator_version="test",
        configuration_hash="strategist-trajectory-test-v1", sensor_layout_hash="all-junctions",
    )
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
            source_node=source_node, sensor_count=3, start_time_bins_min=(0,),
            duration_bins_min=(60,), strength_bins=(1.0,),
        ),
    )


def test_trajectory_is_well_formed_with_a_no_response_comparator(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)

    assert len(result.trajectory.steps) == 1
    step = result.steps[0]
    assert len(step.labels) == len(step.targets) >= 2
    assert any(label.is_no_response_comparator for label in step.labels)


def test_every_label_targets_pass_validate_targets_v2(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J3", seed=11)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    for target in result.steps[0].targets:
        validate_targets_v2(target)  # must not raise -- no diagnostic keys leaked in


def test_action_template_index_matches_the_canonical_vocabulary(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    for label, target in zip(result.steps[0].labels, result.steps[0].targets):
        index = int(target["action_template"])
        assert 0 <= index < len(ACTION_TEMPLATES)
        assert ACTION_TEMPLATES[index] == label.action_template


def test_plan_validity_is_read_from_wntr_not_the_predicted_score(tmp_path) -> None:
    from hydroswarm.domain import PlanDecision
    from hydroswarm.planning.response import PlanGenerationContext, generate_response_plans
    from hydroswarm.simulation.verifier import PlanVerifier
    from hydroswarm.simulation.wrapper import HydraulicSimulator

    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    step = result.steps[0]

    # Independently re-verify each labeled plan through a fresh PlanVerifier
    # built from an unrelated PlanGenerationContext -- confirms
    # plan_validity is sourced from WNTR's own decision on *this exact*
    # plan, not copied from a template's predicted_validity score.
    reference_context = PlanGenerationContext(
        incident_id=step.state.incident_state.incident_id,
        model_version="reference",
        probable_source_nodes=("J2",),
        isolatable_links=tuple(network.pipe_name_list),
        downstream_flush_nodes=node_ids,
        critical_demand_nodes=node_ids[:2],
        monitor_nodes=node_ids,
    )
    proposals_by_template = {
        proposal.template: proposal for proposal in generate_response_plans(reference_context)
    }
    verifier = PlanVerifier(HydraulicSimulator(network))
    for label in step.labels:
        proposal = proposals_by_template.get(label.action_template)
        if proposal is None:
            continue  # a template excluded by prescreening in this trajectory's own context
        verification = verifier.verify(proposal.plan)
        assert (verification.decision == PlanDecision.VERIFIED) == label.plan_validity


def test_target_pointer_masked_off_for_no_response_comparator(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    for label, target in zip(result.steps[0].labels, result.steps[0].targets):
        if label.is_no_response_comparator:
            assert not bool(target["target_pointer_mask"])


def test_trajectory_is_deterministic_for_the_same_scenario(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J4", seed=12)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    first = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    second = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    assert first.trajectory == second.trajectory
    assert [t.action_template for t in first.steps[0].labels] == [
        t.action_template for t in second.steps[0].labels
    ]


def test_different_scenarios_get_different_trajectory_and_incident_ids(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    a = build_strategist_trajectory(_scenario(network, source_node="J2", seed=10), network, context, artifact, node_ids)
    b = build_strategist_trajectory(_scenario(network, source_node="J3", seed=11), network, context, artifact, node_ids)

    assert a.trajectory.trajectory_id != b.trajectory.trajectory_id
    assert a.steps[0].state.incident_state.incident_id != b.steps[0].state.incident_state.incident_id
