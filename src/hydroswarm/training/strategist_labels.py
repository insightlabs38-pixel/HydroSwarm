"""Strategist label generation (overnight-plan.txt Task 2.4).

Builds on already-implemented, already-tested planning/verification
infrastructure:

- hydroswarm.planning.response.generate_response_plans() already produces a
  bounded deterministic template set (no action, isolate probable source
  zone, flush downstream, isolate-then-flush, protect critical demand,
  increase monitoring, request another sample, wait and observe, alternate
  valve cut) plus a guaranteed no-response comparator -- Task 2.4's
  "candidate plan templates" list, almost verbatim.
- hydroswarm.planning.response.prescreen_top_plans() already bounds how
  many plans get expensively, exactly simulated.
- hydroswarm.simulation.verifier.PlanVerifier is the exact WNTR/EPANET
  authority; this module never assigns plan_validity from anything but
  verifier.verify()'s own decision.

This is the missing glue: running the bounded template generator against
one scenario's network, exactly verifying the prescreened plans (plus the
no-response comparator, which the plan requires be preserved even when not
separately prescreened), and packaging the result as governed StrategistLabel
records with a target_pointer resolved into a graph-local node index where
the plan's primary target is a node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hydroswarm.domain import PlanDecision
from hydroswarm.planning.response import PlanGenerationContext, generate_response_plans, prescreen_top_plans
from hydroswarm.simulation.verifier import PlanVerifier
from hydroswarm.simulation.wrapper import HydraulicSimulator


@dataclass(frozen=True, slots=True)
class StrategistLabel:
    action_template: str
    target_id: str | None
    target_node_index: int | None
    plan_validity: bool
    plan_value: float
    rejection_codes: tuple[str, ...]
    consequence_vector: tuple[float, float, float, float] | None
    is_no_response_comparator: bool


def generate_strategist_labels(
    network: Any,
    context: PlanGenerationContext,
    *,
    maximum_exact_simulations: int = 3,
    maximum_plans: int = 8,
) -> tuple[StrategistLabel, ...]:
    """Generate, prescreen, and exactly verify a bounded set of response
    plans for one incident context, always preserving a no-response
    comparator. WNTR remains authoritative: plan_validity is read only from
    PlanVerifier's own decision, never from the template's predicted
    validity score."""

    proposals = generate_response_plans(context, maximum_plans=maximum_plans)
    prescreened = list(prescreen_top_plans(proposals, maximum_exact_simulations=maximum_exact_simulations))
    no_response = next((item for item in proposals if item.template == "NO_ACTION"), None)
    if no_response is not None and no_response not in prescreened:
        prescreened.append(no_response)

    junction_ids = tuple(sorted(network.junction_name_list))
    positions = {node_id: index for index, node_id in enumerate(junction_ids)}
    simulator = HydraulicSimulator(network)
    verifier = PlanVerifier(simulator)

    labels = []
    for proposal in prescreened:
        verification = verifier.verify(proposal.plan)
        primary_target = next((action.target_id for action in proposal.plan.actions if action.target_id), None)
        target_index = positions.get(primary_target) if primary_target else None
        consequences = verification.consequences
        vector = (
            (
                consequences.contaminant_mass_consumed_mg,
                consequences.pressure_violation_minutes,
                consequences.service_availability,
                (consequences.containment_time_minutes or 0.0) * 60.0,
            )
            if consequences is not None
            else None
        )
        labels.append(
            StrategistLabel(
                action_template=proposal.template,
                target_id=primary_target,
                target_node_index=target_index,
                plan_validity=verification.decision == PlanDecision.VERIFIED,
                plan_value=proposal.predicted_value * proposal.predicted_validity,
                rejection_codes=verification.rejection_codes,
                consequence_vector=vector,
                is_no_response_comparator=proposal.template == "NO_ACTION",
            )
        )
    return tuple(labels)
