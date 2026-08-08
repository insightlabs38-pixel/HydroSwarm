"""Safety prescreen and exact hydraulic plan verification."""

from __future__ import annotations

from hydroswarm.domain import (
    AbstentionReason,
    ActionType,
    OperationalPlan,
    PlanDecision,
    PlanVerification,
)

from .wrapper import (
    HydraulicSimulator,
    PlanEvaluationContext,
    SimulationBudgetExceeded,
    SimulationIncompleteError,
    SimulationTimeoutError,
    SimulationUnstableError,
)

#: core-issues5.txt Section 9: map hydroswarm.simulation.wrapper's own
#: already-distinct SimulationError subclasses to a specific, auditable
#: AbstentionReason instead of collapsing every failure into one generic
#: SIMULATION_FAILURE bucket. Checked in this order (most specific first)
#: since SimulationBudgetExceeded etc. are all SimulationError subclasses,
#: not siblings of it.
_FAILURE_CATEGORY: tuple[tuple[type[Exception], AbstentionReason], ...] = (
    (SimulationTimeoutError, AbstentionReason.SIMULATION_TIMEOUT),
    (SimulationBudgetExceeded, AbstentionReason.SIMULATION_BUDGET_EXCEEDED),
    (SimulationUnstableError, AbstentionReason.SIMULATION_UNSTABLE),
    (SimulationIncompleteError, AbstentionReason.SIMULATION_INCOMPLETE),
)


def _categorize_failure(error: Exception) -> AbstentionReason:
    for exception_type, reason in _FAILURE_CATEGORY:
        if isinstance(error, exception_type):
            return reason
    return AbstentionReason.SIMULATION_FAILURE


def _hypotheses_per_plan(evaluation_context: PlanEvaluationContext) -> int:
    """The real number of source hypotheses this evaluation covers -- NOT
    `len(evaluation_context.hypothesis_identity)`, which is a fixed-shape
    `{"mode": ..., ...}` dict (always 1-2 keys) and would silently report
    the wrong count regardless of the actual hypothesis set size."""

    return len(evaluation_context.hypotheses) if evaluation_context.hypotheses else 1


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
            # core-issues5.txt Section 9: captured BEFORE the call so the
            # except branch can still report exactly how many new exact
            # EPANET runs were consumed before a mid-loop failure, even
            # though evaluate_plan_consequences itself never returns in
            # that case.
            runs_before = self.simulator.exact_runs
            try:
                exposure_evaluation = self.simulator.evaluate_plan_consequences(plan, evaluation_context)
            except Exception as error:
                return PlanVerification(
                    plan_id=plan.plan_id,
                    decision=PlanDecision.ABSTAINED,
                    simulator=self.simulator.simulator_name,
                    simulator_version=self.simulator.simulator_version,
                    state_hash=state_hash,
                    abstention_reason=_categorize_failure(error),
                    evaluation_provenance={
                        "aggregation_policy": evaluation_context.aggregation_policy,
                        "hypotheses": evaluation_context.hypothesis_identity,
                        "hypotheses_per_plan": _hypotheses_per_plan(evaluation_context),
                        #: requirement 10 / Section 9: the real number of new
                        #: exact EPANET runs consumed before this failure --
                        #: never silently dropped just because the call
                        #: raised instead of returning.
                        "exact_simulation_count": self.simulator.exact_runs - runs_before,
                        "simulator_version": self.simulator.simulator_version,
                        "cache_hit": False,
                        "failure_type": type(error).__name__,
                    },
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
                "hypotheses_per_plan": _hypotheses_per_plan(evaluation_context),
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
        runs_before = self.simulator.exact_runs
        try:
            evaluation = self.simulator.evaluate_plan(plan)
        except Exception as error:
            return PlanVerification(
                plan_id=plan.plan_id,
                decision=PlanDecision.ABSTAINED,
                simulator=self.simulator.simulator_name,
                simulator_version=self.simulator.simulator_version,
                state_hash=state_hash,
                abstention_reason=_categorize_failure(error),
                evaluation_provenance={
                    "exact_simulation_count": self.simulator.exact_runs - runs_before,
                    "simulator_version": self.simulator.simulator_version,
                    "failure_type": type(error).__name__,
                },
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
