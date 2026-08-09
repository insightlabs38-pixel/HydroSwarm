"""core-issues2.txt Phase 3: Strategist trajectory-state generation."""

from __future__ import annotations

import pytest

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


#: 25 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
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


#: 25 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_every_label_targets_pass_validate_targets_v2(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J3", seed=11)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    for target in result.steps[0].targets:
        validate_targets_v2(target)  # must not raise -- no diagnostic keys leaked in


#: 25 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
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


#: 41 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
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


#: 25 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
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


#: 25 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_link_target_pointer_resolves_against_edge_ids_not_node_ids(tmp_path) -> None:
    """core-issues3.txt Phase 3.4 repair: a LINK-target plan's target_pointer
    must resolve against edge_ids (in sorted(network.link_name_list) order),
    never silently reuse the node index space or masked off just because a
    caller forgot to pass edge_ids -- if edge_ids IS supplied, resolution
    must actually work."""

    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))
    link_names = sorted(network.link_name_list)
    edge_ids = tuple(
        (network.get_link(name).start_node_name, network.get_link(name).end_node_name) for name in link_names
    )

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids, edge_ids)
    isolate_label = next(
        (label for label in result.steps[0].labels if label.action_template == "ISOLATE_SOURCE"), None
    )
    assert isolate_label is not None
    assert isolate_label.primary_target_type == "LINK"

    isolate_target = next(
        target
        for label, target in zip(result.steps[0].labels, result.steps[0].targets)
        if label.action_template == "ISOLATE_SOURCE"
    )
    assert bool(isolate_target["target_pointer_mask"])
    resolved_index = int(isolate_target["target_pointer"])
    assert link_names[resolved_index] == isolate_label.primary_target_id
    # And it must NOT coincidentally be interpretable as a valid node index
    # pointing at the same target name (the two spaces are different sizes
    # and orderings on this network -- this assertion would catch a
    # regression back to resolving link targets against node_ids).
    assert node_ids[resolved_index] != isolate_label.primary_target_id or len(node_ids) != len(link_names)


#: 25 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_node_target_pointer_uses_canonical_space_not_junction_only_order(tmp_path) -> None:
    """The old bug: target resolution used sorted(network.junction_name_list)
    (junctions only), which silently disagrees with the canonical node
    space (junctions + reservoirs + tanks, source_node_logits/sensor_fault's
    own index space) on any real network with a reservoir or tank."""

    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    scenario = _scenario(network, source_node="J2", seed=10)
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))
    junction_only_ids = tuple(sorted(network.junction_name_list))
    assert node_ids != junction_only_ids  # sanity: this network really does have non-junction nodes

    result = build_strategist_trajectory(scenario, network, context, artifact, node_ids)
    flush_label, flush_target = next(
        (label, target)
        for label, target in zip(result.steps[0].labels, result.steps[0].targets)
        if label.action_template == "FLUSH_DOWNSTREAM"
    )
    assert bool(flush_target["target_pointer_mask"])
    resolved_index = int(flush_target["target_pointer"])
    # Self-consistency: the resolved index, read back against the same
    # canonical node_ids space, must recover the label's own semantic target.
    assert node_ids[resolved_index] == flush_label.primary_target_id
    assert junction_only_ids  # sanity: fixture still has junctions to compare against


def test_resolve_target_pointer_uses_the_space_the_caller_supplies_not_a_recomputed_one() -> None:
    """Focused unit test of _resolve_target_pointer, independent of any one
    real network's incidental alphabetical sort order (the full-pipeline
    test above can't reliably exercise a case where the canonical and
    junction-only spaces disagree for a specific target, since that
    depends on where a network's reservoir/tank names happen to sort).
    Constructs canonical vs. junction-only spaces that disagree by
    construction and confirms the function resolves against whichever
    space it is actually given."""

    from hydroswarm.training.strategist_labels import StrategistLabel
    from hydroswarm.training.strategist_trajectory import _resolve_target_pointer

    label = StrategistLabel(
        action_template="FLUSH_DOWNSTREAM",
        primary_target_id="J4",
        primary_target_type="NODE",
        plan_validity=True,
        plan_value=1.0,
        regret=0.0,
        exposure_proxy=0.0,
        pressure_risk_proxy=0.0,
        service_loss_proxy=0.0,
        containment_time_proxy=0.0,
        plan_regret_proxy=0.0,
        rejection_codes=(),
        consequence_vector=(0.0, 0.0, 1.0, 0.0),
        is_no_response_comparator=False,
    )
    canonical_node_ids = ("R1", "J1", "J2", "J3", "J4")  # reservoir sorted first: index 4
    junction_only_ids = ("J1", "J2", "J3", "J4")  # index 3 -- deliberately different

    class _StubNetwork:
        link_name_list = ()

    canonical_index, canonical_found = _resolve_target_pointer(label, canonical_node_ids, (), _StubNetwork())
    junction_only_index, junction_only_found = _resolve_target_pointer(label, junction_only_ids, (), _StubNetwork())

    assert canonical_found and junction_only_found
    assert canonical_index == 4
    assert junction_only_index == 3
    assert canonical_index != junction_only_index


#: 43 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
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


#: 45 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_different_scenarios_get_different_trajectory_and_incident_ids(tmp_path) -> None:
    network = build_wntr_network()
    artifact = _fast_artifact(network, tmp_path / "cache")
    context = build_feature_context(network)
    node_ids = tuple(sorted(network.node_name_list))

    a = build_strategist_trajectory(_scenario(network, source_node="J2", seed=10), network, context, artifact, node_ids)
    b = build_strategist_trajectory(_scenario(network, source_node="J3", seed=11), network, context, artifact, node_ids)

    assert a.trajectory.trajectory_id != b.trajectory.trajectory_id
    assert a.steps[0].state.incident_state.incident_id != b.steps[0].state.incident_state.incident_id
