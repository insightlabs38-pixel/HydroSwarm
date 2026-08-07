"""Safety prescreen and exact hydraulic plan verification."""

from __future__ import annotations

from hydroswarm.domain import (
    AbstentionReason,
    ActionType,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
)

from .wrapper import HydraulicSimulator, PlanEvaluationContext


class PlanVerifier:
    """Reject malformed targets before running an authoritative WNTR simulation."""

    def __init__(self, simulator: HydraulicSimulator) -> None:
        self.simulator = simulator

    def prescreen(self, plan: OperationalPlan) -> tuple[str, ...]:
        model = self.simulator.network
        codes: list[str] = list(self.simulator.validate())
        link_actions = {ActionType.CLOSE_PIPE, ActionType.OPEN_PIPE}
        node_actions = {ActionType.FLUSH_NODE, ActionType.MONITOR_NODE, ActionType.COLLECT_SAMPLE}
        for action in plan.actions:
            target = action.target_id
            if action.action_type in link_actions:
                if target not in model.link_name_list:
                    codes.append(f"UNKNOWN_TARGET:{target}")
                    continue
                link = model.get_link(target)
                if target not in model.pipe_name_list or not self.simulator._is_link_operable(link):
                    codes.append(f"INOPERABLE_TARGET:{target}")
            elif action.action_type in node_actions:
                if target not in model.node_name_list:
                    codes.append(f"UNKNOWN_TARGET:{target}")
                elif action.action_type == ActionType.FLUSH_NODE and target not in model.junction_name_list:
                    codes.append(f"INOPERABLE_TARGET:{target}")
            elif action.action_type == ActionType.ISOLATE_ZONE:
                codes.append(f"INOPERABLE_TARGET:{target}")
        return tuple(sorted(set(codes)))

    def verify(
        self,
        plan: OperationalPlan,
        evaluation_context: PlanEvaluationContext | None = None,
    ) -> PlanVerification:
        """Exact WNTR/EPANET plan verification.

        `evaluation_context` is the canonical exposure-aware path
        (important-issues.txt requirement 12): when supplied, verification
        runs `HydraulicSimulator.evaluate_plan_consequences` -- one EPANET
        chemical-transport pass per evaluated source hypothesis over the
        plan-modified network -- and the returned `consequences` are real,
        measured contamination-exposure values
        (`ConsequenceMetrics.exposure_evaluated=True`).

        When omitted, verification falls back to the legacy hydraulic-only
        `evaluate_plan` path; the returned `consequences` have
        `exposure_evaluated=False` and must never be presented as measured
        exposure by any caller (requirement 12: "never populate zero/None
        defaults as though they were measured").
        """

        state_hash = self.simulator.state_hash(plan)
        rejection_codes = self.prescreen(plan)
        if rejection_codes:
            return PlanVerification(
                plan_id=plan.plan_id,
                decision=PlanDecision.REJECTED,
                simulator=self.simulator.simulator_name,
                simulator_version=self.simulator.simulator_version,
                state_hash=state_hash,
                rejection_codes=rejection_codes,
            )
        if evaluation_context is not None:
            try:
                exposure_evaluation = self.simulator.evaluate_plan_consequences(plan, evaluation_context)
            except Exception:
                return PlanVerification(
                    plan_id=plan.plan_id,
                    decision=PlanDecision.ABSTAINED,
                    simulator=self.simulator.simulator_name,
                    simulator_version=self.simulator.simulator_version,
                    state_hash=state_hash,
                    abstention_reason=AbstentionReason.SIMULATION_FAILURE,
                )
            decision = (
                PlanDecision.REJECTED if exposure_evaluation.rejection_codes else PlanDecision.VERIFIED
            )
            provenance = {
                "aggregation_policy": evaluation_context.aggregation_policy,
                "contamination_threshold_mg_l": evaluation_context.contamination_threshold_mg_l,
                "population_map_identity": evaluation_context.population_map_identity,
                "consequence_policy_version": evaluation_context.consequence_policy_version,
                "hypotheses": evaluation_context.hypothesis_identity,
                #: requirement 10: real EPANET runs THIS verify() call
                #: consumed -- never silently reported as one simulator call.
                "exact_simulation_count": exposure_evaluation.exact_simulation_count,
                "simulator_version": self.simulator.simulator_version,
                "cache_hit": exposure_evaluation.cache_hit,
            }
            return PlanVerification(
                plan_id=plan.plan_id,
                decision=decision,
                simulator=self.simulator.simulator_name,
                simulator_version=self.simulator.simulator_version,
                state_hash=exposure_evaluation.state_hash,
                consequences=exposure_evaluation.consequences,
                worst_case_consequences=exposure_evaluation.worst_case_consequences,
                evaluation_provenance=provenance,
                rejection_codes=exposure_evaluation.rejection_codes,
            )
        try:
            evaluation = self.simulator.evaluate_plan(plan)
        except Exception:
            return PlanVerification(
                plan_id=plan.plan_id,
                decision=PlanDecision.ABSTAINED,
                simulator=self.simulator.simulator_name,
                simulator_version=self.simulator.simulator_version,
                state_hash=state_hash,
                abstention_reason=AbstentionReason.SIMULATION_FAILURE,
            )
        decision = PlanDecision.REJECTED if evaluation.rejection_codes else PlanDecision.VERIFIED
        return PlanVerification(
            plan_id=plan.plan_id,
            decision=decision,
            simulator=self.simulator.simulator_name,
            simulator_version=self.simulator.simulator_version,
            state_hash=evaluation.state_hash,
            consequences=evaluation.consequences,
            rejection_codes=evaluation.rejection_codes,
        )

    evaluate = verify
