from uuid import uuid4

from hydroswarm.domain import ActionType
from hydroswarm.planning.response import (
    PlanGenerationContext,
    VerifierFeedback,
    generate_response_plans,
    prescreen_top_plans,
    revise_rejected_plan,
)


def context() -> PlanGenerationContext:
    return PlanGenerationContext(
        incident_id=uuid4(), model_version="hydrocore-L", probable_source_nodes=("J1", "J2"),
        isolatable_links=("P1", "P2"), downstream_flush_nodes=("J3",),
        critical_demand_nodes=("J4",), monitor_nodes=("J1", "J2"),
    )


def test_generates_diverse_multi_action_plans_with_hard_budgets() -> None:
    proposals = generate_response_plans(context())
    assert len(proposals) == 8
    assert len({item.diversity_key for item in proposals}) == 8
    assert any(len(item.plan.actions) > 1 for item in proposals)
    assert any(item.template == "NO_ACTION" for item in proposals)
    assert all(len(item.plan.actions) <= 8 for item in proposals)
    top = prescreen_top_plans(proposals)
    assert len(top) == 3
    assert all(item.template != "NO_ACTION" for item in top)


def test_verifier_feedback_repairs_rejected_action() -> None:
    ctx = context()
    proposal = next(item for item in generate_response_plans(ctx) if item.template == "ISOLATE_SOURCE")
    revised = revise_rejected_plan(
        proposal,
        VerifierFeedback(
            rejection_codes=("PRESSURE_BELOW_MINIMUM",), rejected_targets=frozenset({"P1"}),
            severity="hard", round_index=1,
        ),
        ctx,
    )
    assert revised.plan.actions[0].action_type == ActionType.CLOSE_PIPE
    assert revised.plan.actions[0].target_id == "P2"
    assert revised.prescreen_reasons == ("PRESSURE_BELOW_MINIMUM",)
