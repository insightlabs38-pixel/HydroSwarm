"""Classical response templates mixed with model value/validity scores."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence
from uuid import UUID

from hydroswarm.domain import ActionType, OperationalAction, OperationalPlan


@dataclass(frozen=True, slots=True)
class PlanGenerationContext:
    incident_id: UUID
    model_version: str
    probable_source_nodes: tuple[str, ...]
    isolatable_links: tuple[str, ...]
    downstream_flush_nodes: tuple[str, ...]
    critical_demand_nodes: tuple[str, ...]
    monitor_nodes: tuple[str, ...]
    sampled_nodes: frozenset[str] = frozenset()
    maximum_actions: int = 8


@dataclass(frozen=True, slots=True)
class PlanProposal:
    plan: OperationalPlan
    template: str
    predicted_value: float
    predicted_validity: float
    diversity_key: tuple[tuple[str, str | None], ...]
    prescreen_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifierFeedback:
    rejection_codes: tuple[str, ...]
    rejected_targets: frozenset[str]
    severity: str
    round_index: int


def _signature(actions: Sequence[OperationalAction]) -> tuple[tuple[str, str | None], ...]:
    return tuple((action.action_type.value, action.target_id) for action in actions)


def _proposal(
    context: PlanGenerationContext,
    name: str,
    template: str,
    actions: Sequence[OperationalAction],
    *,
    value: float,
    validity: float,
) -> PlanProposal:
    if not actions or len(actions) > context.maximum_actions:
        raise ValueError("plan actions must respect the configured action budget")
    plan = OperationalPlan(
        incident_id=context.incident_id,
        name=name,
        model_version=context.model_version,
        actions=tuple(actions),
    )
    return PlanProposal(plan, template, value, validity, _signature(actions))


def generate_response_plans(
    context: PlanGenerationContext,
    *,
    neural_value_deltas: Mapping[str, float] | None = None,
    neural_validity_deltas: Mapping[str, float] | None = None,
    maximum_plans: int = 8,
) -> tuple[PlanProposal, ...]:
    """Generate six-to-eight diverse plans including a no-action comparator."""
    if maximum_plans < 3 or maximum_plans > 8:
        raise ValueError("maximum_plans must be between 3 and 8")
    links = context.isolatable_links
    flushes = context.downstream_flush_nodes
    monitors = context.monitor_nodes or context.probable_source_nodes
    templates: list[tuple[str, str, list[OperationalAction], float, float]] = []
    templates.append(("No response", "NO_ACTION", [OperationalAction(action_type=ActionType.END_PLAN)], 0.0, 1.0))
    if links:
        templates.append(("Isolate probable source zone", "ISOLATE_SOURCE", [
            OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id=links[0], duration_minutes=60)
        ], 0.65, 0.65))
    if flushes:
        templates.append(("Flush downstream", "FLUSH_DOWNSTREAM", [
            OperationalAction(action_type=ActionType.FLUSH_NODE, target_id=flushes[0], flow_rate_lps=3.0, duration_minutes=45)
        ], 0.55, 0.85))
    if links and flushes:
        templates.append(("Isolate and flush", "ISOLATE_AND_FLUSH", [
            OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id=links[0], duration_minutes=60),
            OperationalAction(action_type=ActionType.FLUSH_NODE, target_id=flushes[0], start_minute=5, flow_rate_lps=3.0, duration_minutes=45),
        ], 0.8, 0.55))
    if context.critical_demand_nodes:
        templates.append(("Protect critical demand", "PROTECT_CRITICAL", [
            OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=context.critical_demand_nodes[0]),
            OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id=context.critical_demand_nodes[0]),
        ], 0.5, 0.95))
    if monitors:
        templates.append(("Increase monitoring", "INCREASE_MONITORING", [
            OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=monitors[0])
        ], 0.25, 1.0))
    unsampled = [node for node in monitors if node not in context.sampled_nodes]
    if unsampled:
        templates.append(("Request another sample", "REQUEST_SAMPLE", [
            OperationalAction(action_type=ActionType.COLLECT_SAMPLE, target_id=unsampled[0])
        ], 0.3, 1.0))
    templates.append(("Wait and observe", "WAIT_OBSERVE", [
        OperationalAction(action_type=ActionType.WAIT, duration_minutes=15),
        OperationalAction(action_type=ActionType.END_PLAN),
    ], 0.1, 1.0))
    if len(links) > 1:
        templates.append(("Alternate valve cut", "ALTERNATE_VALVE_CUT", [
            OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id=links[1], duration_minutes=45)
        ], 0.55, 0.75))

    proposals: list[PlanProposal] = []
    seen: set[tuple[tuple[str, str | None], ...]] = set()
    for name, template, actions, value, validity in templates:
        signature = _signature(actions)
        if signature in seen:
            continue
        seen.add(signature)
        value += float((neural_value_deltas or {}).get(template, 0.0))
        validity += float((neural_validity_deltas or {}).get(template, 0.0))
        proposals.append(_proposal(
            context, name, template, actions, value=value,
            validity=max(0.0, min(1.0, validity)),
        ))
        if len(proposals) >= maximum_plans:
            break
    return tuple(proposals)


def prescreen_top_plans(
    proposals: Sequence[PlanProposal], *, maximum_exact_simulations: int = 3
) -> tuple[PlanProposal, ...]:
    """Select at most three diverse, plausible plans for authoritative simulation."""
    if not 1 <= maximum_exact_simulations <= 3:
        raise ValueError("exact simulation budget must be between one and three")
    eligible = [item for item in proposals if item.predicted_validity >= 0.5]
    eligible.sort(
        key=lambda item: (item.template == "NO_ACTION", -(item.predicted_value * item.predicted_validity))
    )
    return tuple(eligible[:maximum_exact_simulations])


def revise_rejected_plan(
    proposal: PlanProposal,
    feedback: VerifierFeedback,
    context: PlanGenerationContext,
) -> PlanProposal:
    """Feed hard verifier findings back into a deterministic plan repair."""
    if feedback.round_index < 1:
        raise ValueError("revision round must be positive")
    alternatives = [link for link in context.isolatable_links if link not in feedback.rejected_targets]
    revised: list[OperationalAction] = []
    for action in proposal.plan.actions:
        if action.target_id not in feedback.rejected_targets:
            revised.append(action)
        elif action.action_type in {ActionType.CLOSE_PIPE, ActionType.OPEN_PIPE} and alternatives:
            revised.append(action.model_copy(update={"target_id": alternatives[0]}))
    if not revised or all(action.action_type == ActionType.END_PLAN for action in revised):
        target = context.monitor_nodes[0] if context.monitor_nodes else context.probable_source_nodes[0]
        revised = [OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=target)]
    if any(code in {"PRESSURE_BELOW_MINIMUM", "SERVICE_BELOW_MINIMUM"} for code in feedback.rejection_codes):
        revised = [action for action in revised if action.action_type != ActionType.FLUSH_NODE]
        if not revised:
            target = context.monitor_nodes[0] if context.monitor_nodes else context.probable_source_nodes[0]
            revised = [OperationalAction(action_type=ActionType.MONITOR_NODE, target_id=target)]
    repaired = _proposal(
        context,
        f"{proposal.plan.name} — revision {feedback.round_index}",
        f"{proposal.template}_REVISION",
        revised,
        value=max(0.0, proposal.predicted_value - 0.05),
        validity=min(1.0, proposal.predicted_validity + 0.1),
    )
    return replace(repaired, prescreen_reasons=feedback.rejection_codes)
