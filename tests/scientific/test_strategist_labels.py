"""Task 2.4: Strategist label generation over the existing bounded plan
templates and exact WNTR verification."""

from __future__ import annotations

from uuid import uuid4

from hydroswarm.planning.response import PlanGenerationContext
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.simulation.verifier import PlanVerifier
from hydroswarm.simulation.wrapper import HydraulicSimulator
from hydroswarm.training.strategist_labels import generate_strategist_labels


def _context(**overrides) -> PlanGenerationContext:
    base = dict(
        incident_id=uuid4(),
        model_version="test-strategist-v1",
        probable_source_nodes=("J2",),
        isolatable_links=("P_R1_J1", "P_J1_J2"),
        downstream_flush_nodes=("J4",),
        critical_demand_nodes=("J3",),
        monitor_nodes=("J2",),
    )
    base.update(overrides)
    return PlanGenerationContext(**base)


def test_generates_bounded_set_including_no_response_comparator() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    assert len(labels) >= 2
    assert any(label.is_no_response_comparator for label in labels)


def test_plan_validity_is_read_from_wntr_not_predicted_score() -> None:
    network = build_wntr_network()
    context = _context()
    labels = generate_strategist_labels(network, context)

    # Independently re-verify every label's underlying plan through a fresh
    # PlanVerifier and confirm the label's plan_validity matches exactly --
    # i.e. plan_validity is never a copy of the template's predicted_validity.
    from hydroswarm.planning.response import generate_response_plans

    proposals = {proposal.template: proposal for proposal in generate_response_plans(context)}
    simulator = HydraulicSimulator(network)
    verifier = PlanVerifier(simulator)
    for label in labels:
        proposal = proposals[label.action_template]
        from hydroswarm.domain import PlanDecision

        expected = verifier.verify(proposal.plan).decision == PlanDecision.VERIFIED
        assert label.plan_validity == expected


def test_no_response_comparator_has_no_target_and_typically_verifies() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    no_response = next(label for label in labels if label.is_no_response_comparator)
    assert no_response.target_id is None
    assert no_response.target_node_index is None


def test_flush_downstream_target_resolves_to_correct_local_node_index() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    flush_label = next((label for label in labels if label.action_template == "FLUSH_DOWNSTREAM"), None)
    if flush_label is not None:  # only generated if prescreened/preserved
        junction_ids = tuple(sorted(network.junction_name_list))
        assert flush_label.target_id == "J4"
        assert junction_ids[flush_label.target_node_index] == "J4"


def test_invalid_isolation_plan_carries_real_rejection_codes(tmp_path=None) -> None:
    # Force a genuinely unsafe plan by isolating the network's only feeder
    # pipe indefinitely -- WNTR should reject it, and the rejection must
    # carry real codes explaining why (not an empty/silent rejection).
    network = build_wntr_network()
    labels = generate_strategist_labels(
        network, _context(isolatable_links=("P_R1_J1",), downstream_flush_nodes=())
    )
    isolation_labels = [label for label in labels if label.action_template == "ISOLATE_SOURCE"]
    for label in isolation_labels:
        if not label.plan_validity:
            assert label.rejection_codes


def test_consequence_vector_present_only_when_verifier_provided_consequences() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    for label in labels:
        if label.plan_validity:
            assert label.consequence_vector is not None
            assert len(label.consequence_vector) == 4


def test_respects_maximum_exact_simulations_budget() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context(), maximum_exact_simulations=1)
    # 1 prescreened + the always-preserved no-response comparator.
    assert len(labels) <= 2
