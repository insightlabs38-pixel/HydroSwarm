"""Structured response-template generation, revision, and comparison."""

from hydroswarm.planning.pareto import FrontierEntry, FrontierMode, compute_verified_pareto_frontier
from hydroswarm.planning.response import (
    PlanGenerationContext,
    PlanProposal,
    VerifierFeedback,
    generate_response_plans,
    prescreen_top_plans,
    revise_rejected_plan,
)

__all__ = [
    "FrontierEntry",
    "FrontierMode",
    "PlanGenerationContext",
    "PlanProposal",
    "VerifierFeedback",
    "compute_verified_pareto_frontier",
    "generate_response_plans",
    "prescreen_top_plans",
    "revise_rejected_plan",
]

