"""Strategist label generation (overnight-plan.txt Task 2.4, repaired by
core-issues3.txt Phase 3).

Builds on already-implemented, already-tested planning/verification
infrastructure:

- hydroswarm.planning.response.generate_response_plans() already produces a
  bounded deterministic template set (no action, isolate probable source
  zone, flush downstream, isolate-then-flush, protect critical demand,
  increase monitoring, request another sample, wait and observe, alternate
  valve cut) plus a guaranteed no-response comparator -- Task 2.4's
  "candidate plan templates" list, almost verbatim.
- hydroswarm.simulation.verifier.PlanVerifier is the exact WNTR/EPANET
  authority; this module never assigns plan_validity from anything but
  verifier.verify()'s own decision.
- hydroswarm.planning.plan_value_policy.evaluate_plan_value() derives
  plan_value/regret/consequence-proxy targets from exact WNTR consequences,
  never from a template's predicted score.

core-issues3.txt Phase 3.1 repair: verifies the FULL bounded candidate set
generate_response_plans() produces (not a heuristically prescreened
subset) -- using the old heuristic prescreener to decide which candidates
receive training labels would bias the corpus toward that same heuristic's
own blind spots, preventing a learned prescreener from ever improving on
it. `maximum_exact_simulations`/`prescreen_top_plans` remain available for
RUNTIME use (where the expensive-simulation budget is a real constraint),
just not for training-label generation.

Phase 3.4 repair: StrategistLabel now carries semantic target identity
(`primary_target_id`, `primary_target_type`) instead of a premature
graph-local integer computed from `sorted(network.junction_name_list)` --
that space silently disagreed with the canonical node space
(TopologyMetadata.node_ids: junctions + reservoirs + tanks) every other
node-indexed target uses, and dropped link targets entirely (`positions`
only ever contained junction names). Index resolution against the
scenario's own canonical node_ids/edge_ids now happens only at
tensor-building time (hydroswarm.training.strategist_trajectory), where
that canonical space is actually available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hydroswarm.domain import ConsequenceMetrics, PlanDecision
from hydroswarm.planning.action_templates import LINK_TARGET_TEMPLATES
from hydroswarm.planning.plan_value_policy import evaluate_plan_value
from hydroswarm.planning.response import PlanGenerationContext, generate_response_plans
from hydroswarm.simulation.verifier import PlanVerifier
from hydroswarm.simulation.wrapper import HydraulicSimulator

TargetType = Literal["NONE", "NODE", "LINK"]


@dataclass(frozen=True, slots=True)
class StrategistLabel:
    action_template: str
    primary_target_id: str | None
    primary_target_type: TargetType
    plan_validity: bool
    #: None whenever this plan was not exactly verified as valid with a
    #: computed consequence vector -- targets_v2's own masking rule for
    #: plan_value ("undefined for invalid plans without a computed
    #: consequence vector"). Never a value derived from a heuristic score.
    plan_value: float | None
    regret: float | None
    exposure_proxy: float | None
    pressure_risk_proxy: float | None
    service_loss_proxy: float | None
    containment_time_proxy: float | None
    plan_regret_proxy: float | None
    rejection_codes: tuple[str, ...]
    consequence_vector: tuple[float, float, float, float] | None
    is_no_response_comparator: bool


def _target_identity(proposal: Any) -> tuple[str | None, TargetType]:
    primary_action = next((action for action in proposal.plan.actions if action.target_id), None)
    if primary_action is None:
        return None, "NONE"
    target_type: TargetType = "LINK" if proposal.template in LINK_TARGET_TEMPLATES else "NODE"
    return primary_action.target_id, target_type


def _consequence_vector(consequences: ConsequenceMetrics | None) -> tuple[float, float, float, float] | None:
    if consequences is None:
        return None
    return (
        consequences.contaminant_mass_consumed_mg,
        consequences.pressure_violation_minutes,
        consequences.service_availability,
        (consequences.containment_time_minutes or 0.0) * 60.0,
    )


def generate_strategist_labels(
    network: Any,
    context: PlanGenerationContext,
    *,
    maximum_exact_simulations: int = 3,
    maximum_plans: int = 9,
) -> tuple[StrategistLabel, ...]:
    """Generate and exactly verify the FULL bounded candidate set for one
    incident context (Phase 3.1), always including the no-response
    comparator. WNTR remains authoritative: plan_validity is read only
    from PlanVerifier's own decision. plan_value/regret/consequence-proxy
    targets are derived only from exact WNTR consequences via the governed
    PlanValuePolicy (Phase 3.2/3.3), never from a template's predicted
    score.

    `maximum_exact_simulations` is accepted for call-site compatibility
    with the runtime-budget path but is NOT used to prescreen which
    candidates get labels here -- every proposal generate_response_plans
    returns is exactly verified. Use response.prescreen_top_plans directly
    at runtime, where the expensive-simulation budget is a real operational
    constraint the training corpus does not need to reproduce.
    """

    del maximum_exact_simulations  # training-label generation verifies every proposal; see docstring
    proposals = generate_response_plans(context, maximum_plans=maximum_plans)

    simulator = HydraulicSimulator(network)
    verifier = PlanVerifier(simulator)
    verifications = {proposal.template: verifier.verify(proposal.plan) for proposal in proposals}

    no_action_verification = verifications.get("NO_ACTION")
    no_action_metrics = (
        no_action_verification.consequences
        if no_action_verification is not None and no_action_verification.decision == PlanDecision.VERIFIED
        else None
    )
    valid_metrics_pool = [
        verification.consequences
        for verification in verifications.values()
        if verification.decision == PlanDecision.VERIFIED and verification.consequences is not None
    ]

    labels = []
    for proposal in proposals:
        verification = verifications[proposal.template]
        target_id, target_type = _target_identity(proposal)
        consequences = verification.consequences
        is_valid = verification.decision == PlanDecision.VERIFIED

        result = None
        if is_valid and consequences is not None and no_action_metrics is not None and valid_metrics_pool:
            result = evaluate_plan_value(
                consequences, no_response=no_action_metrics, valid_candidate_metrics=valid_metrics_pool
            )

        labels.append(
            StrategistLabel(
                action_template=proposal.template,
                primary_target_id=target_id,
                primary_target_type=target_type,
                plan_validity=is_valid,
                plan_value=result.plan_value if result is not None else None,
                regret=result.regret if result is not None else None,
                exposure_proxy=result.exposure_proxy if result is not None else None,
                pressure_risk_proxy=result.pressure_risk_proxy if result is not None else None,
                service_loss_proxy=result.service_loss_proxy if result is not None else None,
                containment_time_proxy=result.containment_time_proxy if result is not None else None,
                plan_regret_proxy=result.plan_regret_proxy if result is not None else None,
                rejection_codes=verification.rejection_codes,
                consequence_vector=_consequence_vector(consequences),
                is_no_response_comparator=proposal.template == "NO_ACTION",
            )
        )
    return tuple(labels)
