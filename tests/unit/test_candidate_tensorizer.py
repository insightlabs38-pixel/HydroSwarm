"""core-issues5.txt Section 6 (P0 blocker): canonical deterministic-
candidate -> HydroCore plan-tensor conversion."""

from __future__ import annotations

from uuid import uuid4

import networkx as nx

from hydroswarm.planning.action_templates import ACTION_TEMPLATE_INDEX
from hydroswarm.planning.candidate_tensorizer import (
    PLAN_FEATURE_NAMES,
    plan_proposals_to_candidate_tensors,
)
from hydroswarm.planning.response import PlanGenerationContext, generate_response_plans

NODE_IDS = ("J1", "J2", "J3", "J4", "R1")


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(NODE_IDS)
    # Deliberately inserted out of (source_index, target_index) order, so a
    # correct implementation must actually SORT (matching align_edges), not
    # merely enumerate insertion order.
    graph.add_edge("J3", "J4", key="P_J3_J4", link_id="P_J3_J4")
    graph.add_edge("R1", "J1", key="P_R1_J1", link_id="P_R1_J1")
    graph.add_edge("J1", "J2", key="P_J1_J2", link_id="P_J1_J2")
    return graph


def _context() -> PlanGenerationContext:
    return PlanGenerationContext(
        incident_id=uuid4(),
        model_version="test",
        probable_source_nodes=("J2",),
        isolatable_links=("P_J1_J2", "P_J3_J4"),
        downstream_flush_nodes=("J3",),
        critical_demand_nodes=("J4",),
        monitor_nodes=("J2", "J3"),
    )


def test_every_candidate_has_a_one_to_one_tensor_representation() -> None:
    proposals = generate_response_plans(_context(), maximum_plans=9)
    tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())

    assert tensors["plan_template_ids"].shape == (1, len(proposals))
    assert tensors["plan_target_type"].shape == (1, len(proposals))
    assert tensors["plan_target_node_index"].shape == (1, len(proposals))
    assert tensors["plan_target_link_index"].shape == (1, len(proposals))
    assert tensors["plan_features"].shape == (1, len(proposals), len(PLAN_FEATURE_NAMES))
    assert tensors["plan_mask"].shape == (1, len(proposals))
    assert bool(tensors["plan_mask"].all())


def test_template_ids_use_the_canonical_vocabulary() -> None:
    proposals = generate_response_plans(_context(), maximum_plans=9)
    tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())

    for position, proposal in enumerate(proposals):
        assert int(tensors["plan_template_ids"][0, position]) == ACTION_TEMPLATE_INDEX[proposal.template]


def test_node_target_mapping_is_correct() -> None:
    proposals = generate_response_plans(_context(), maximum_plans=9)
    tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())

    by_template = {proposal.template: position for position, proposal in enumerate(proposals)}
    position = by_template["FLUSH_DOWNSTREAM"]
    assert int(tensors["plan_target_node_index"][0, position]) == NODE_IDS.index("J3")


def test_link_target_mapping_is_correct_and_matches_the_sorted_edge_order() -> None:
    proposals = generate_response_plans(_context(), maximum_plans=9)
    tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())

    # align_edges sorts by (source_index, target_index, original_row):
    # P_J1_J2 (J1=0, J2=1) -> position 0
    # P_J3_J4 (J3=2, J4=3) -> position 1
    # P_R1_J1 (R1=4, J1=0) -> position 2
    by_template = {proposal.template: position for position, proposal in enumerate(proposals)}
    isolate_position = by_template["ISOLATE_SOURCE"]
    assert int(tensors["plan_target_link_index"][0, isolate_position]) == 0  # P_J1_J2


def test_no_target_templates_have_no_target_index() -> None:
    proposals = generate_response_plans(_context(), maximum_plans=9)
    tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())

    by_template = {proposal.template: position for position, proposal in enumerate(proposals)}
    for template in ("NO_ACTION", "WAIT_OBSERVE"):
        position = by_template[template]
        assert int(tensors["plan_target_node_index"][0, position]) == -1
        assert int(tensors["plan_target_link_index"][0, position]) == -1


def test_candidate_order_does_not_change_which_tensor_row_a_template_owns() -> None:
    """Reordering the input proposals must not change any individual
    candidate's own tensor content -- only its row position -- since every
    downstream consumer re-keys by action_template identity, not row
    index."""

    proposals = generate_response_plans(_context(), maximum_plans=9)
    reversed_proposals = tuple(reversed(proposals))

    forward = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())
    backward = plan_proposals_to_candidate_tensors(reversed_proposals, node_ids=NODE_IDS, graph=_graph())

    forward_by_template = {
        proposal.template: int(forward["plan_template_ids"][0, position])
        for position, proposal in enumerate(proposals)
    }
    backward_by_template = {
        proposal.template: int(backward["plan_template_ids"][0, position])
        for position, proposal in enumerate(reversed_proposals)
    }
    assert forward_by_template == backward_by_template


def test_alternate_valve_cut_is_reachable_with_two_isolatable_links() -> None:
    """The canonical 9th template (omitted from the old, stale 8-template
    tuple this fix removes) must be generatable and tensorizable."""

    context = PlanGenerationContext(
        incident_id=uuid4(),
        model_version="test",
        probable_source_nodes=("J2",),
        isolatable_links=("P_J1_J2", "P_J3_J4"),
        downstream_flush_nodes=("J3",),
        critical_demand_nodes=("J4",),
        monitor_nodes=("J2", "J3"),
    )
    proposals = generate_response_plans(context, maximum_plans=9)
    assert any(proposal.template == "ALTERNATE_VALVE_CUT" for proposal in proposals)
    tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=NODE_IDS, graph=_graph())
    by_template = {proposal.template: position for position, proposal in enumerate(proposals)}
    position = by_template["ALTERNATE_VALVE_CUT"]
    assert int(tensors["plan_target_link_index"][0, position]) == 1  # P_J3_J4
