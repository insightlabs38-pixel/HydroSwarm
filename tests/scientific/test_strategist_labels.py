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

    proposals = {proposal.template: proposal for proposal in generate_response_plans(context, maximum_plans=9)}
    simulator = HydraulicSimulator(network)
    verifier = PlanVerifier(simulator)
    for label in labels:
        proposal = proposals[label.action_template]
        from hydroswarm.domain import PlanDecision

        expected = verifier.verify(proposal.plan).decision == PlanDecision.VERIFIED
        assert label.plan_validity == expected


def test_no_response_comparator_has_no_target() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    no_response = next(label for label in labels if label.is_no_response_comparator)
    assert no_response.primary_target_id is None
    assert no_response.primary_target_type == "NONE"


def test_flush_downstream_target_identity_is_semantic_not_a_premature_index() -> None:
    """core-issues3.txt Phase 3.4: StrategistLabel carries the raw node/link
    name and its type, not a graph-local integer computed at label-
    generation time (which used to silently disagree with the canonical
    node space every other node-indexed target uses -- see
    strategist_trajectory.py's _resolve_target_pointer for where index
    resolution now correctly happens instead)."""

    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    flush_label = next((label for label in labels if label.action_template == "FLUSH_DOWNSTREAM"), None)
    assert flush_label is not None  # the full bounded set is always generated (Phase 3.1)
    assert flush_label.primary_target_id == "J4"
    assert flush_label.primary_target_type == "NODE"


def test_link_target_templates_carry_link_type_not_node() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    isolate_label = next((label for label in labels if label.action_template == "ISOLATE_SOURCE"), None)
    assert isolate_label is not None
    assert isolate_label.primary_target_id == "P_R1_J1"
    assert isolate_label.primary_target_type == "LINK"


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


def test_verifies_the_full_bounded_candidate_set_regardless_of_the_exact_simulation_budget() -> None:
    """core-issues3.txt Phase 3.1 repair: training-label generation must
    verify every candidate generate_response_plans() produces, not a
    heuristically prescreened subset -- using the old prescreener to decide
    which candidates receive labels would bias the corpus toward that same
    heuristic's blind spots. maximum_exact_simulations is accepted for
    call-site compatibility but must have no effect on how many labels are
    produced here (it is a RUNTIME budget concern, not a training-corpus one)."""

    network = build_wntr_network()
    labels_budget_1 = generate_strategist_labels(network, _context(), maximum_exact_simulations=1)
    labels_budget_3 = generate_strategist_labels(network, _context(), maximum_exact_simulations=3)
    assert len(labels_budget_1) == len(labels_budget_3)
    # This scenario's context makes every one of the 9 canonical templates
    # eligible (isolatable_links has 2 entries for ALTERNATE_VALVE_CUT,
    # downstream_flush_nodes/critical_demand_nodes/monitor_nodes all set).
    from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT

    assert len(labels_budget_1) == ACTION_TEMPLATE_COUNT


def test_best_plan_has_zero_regret_and_no_action_is_a_real_comparator() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    valid_with_value = [label for label in labels if label.plan_value is not None]
    assert valid_with_value  # at least NO_ACTION itself must be verifiable
    assert min(label.regret for label in valid_with_value) == 0.0
    assert max(label.plan_value for label in valid_with_value) == 1.0
    # NO_ACTION must be scored by the same policy as everything else, not
    # given an automatic free pass.
    no_response = next(label for label in labels if label.is_no_response_comparator)
    assert no_response.plan_value is not None
    assert no_response.regret is not None


def test_invalid_plans_never_carry_a_plan_value() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(
        network, _context(isolatable_links=("P_R1_J1",), downstream_flush_nodes=())
    )
    for label in labels:
        if not label.plan_validity:
            assert label.plan_value is None
            assert label.regret is None


def test_proxy_targets_are_populated_alongside_plan_value() -> None:
    network = build_wntr_network()
    labels = generate_strategist_labels(network, _context())
    for label in labels:
        has_value = label.plan_value is not None
        assert (label.exposure_proxy is not None) == has_value
        assert (label.pressure_risk_proxy is not None) == has_value
        assert (label.service_loss_proxy is not None) == has_value
        assert (label.containment_time_proxy is not None) == has_value
        assert (label.plan_regret_proxy is not None) == has_value
