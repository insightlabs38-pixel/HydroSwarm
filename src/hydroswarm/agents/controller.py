"""Deterministic finite-state orchestration for the HydroSwarm specialists."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any
from uuid import UUID

from hydroswarm.domain import (
    CandidateSet,
    IncidentState,
    OODLevel,
    PlanDecision,
    PlanVerification,
    SensorObservation,
)

from .schemas import (
    FSMState,
    FallbackMode,
    ReplayResult,
    ScoutAction,
    ScoutOutput,
    SentinelOutput,
    StrategistOutput,
    SwarmEvent,
    SwarmResult,
)
from .scout import HydroScout
from .sentinel import HydroSentinel
from .strategist import HydroStrategist


GENESIS_HASH = "0" * 64


ALLOWED_TRANSITIONS: dict[FSMState, frozenset[FSMState]] = {
    FSMState.IDLE: frozenset({FSMState.NETWORK_LOADED}),
    FSMState.NETWORK_LOADED: frozenset({FSMState.NETWORK_VALIDATED, FSMState.COMPLETE}),
    FSMState.NETWORK_VALIDATED: frozenset({FSMState.INCIDENT_LOADED}),
    FSMState.INCIDENT_LOADED: frozenset({FSMState.DATA_QUALITY_CHECK}),
    FSMState.DATA_QUALITY_CHECK: frozenset({FSMState.HYDRAULIC_STATE_ESTIMATION, FSMState.COMPLETE}),
    FSMState.HYDRAULIC_STATE_ESTIMATION: frozenset({FSMState.SOURCE_LOCALIZATION, FSMState.COMPLETE}),
    FSMState.SOURCE_LOCALIZATION: frozenset({FSMState.EVIDENCE_CHECK, FSMState.COMPLETE}),
    FSMState.EVIDENCE_CHECK: frozenset(
        {FSMState.SAMPLE_SELECTION, FSMState.PLAN_GENERATION, FSMState.COMPLETE}
    ),
    FSMState.SAMPLE_SELECTION: frozenset({FSMState.SOURCE_UPDATE, FSMState.COMPLETE}),
    FSMState.SOURCE_UPDATE: frozenset({FSMState.SOURCE_LOCALIZATION}),
    FSMState.PLAN_GENERATION: frozenset({FSMState.CONSTRAINT_CHECK, FSMState.COMPLETE}),
    FSMState.CONSTRAINT_CHECK: frozenset(
        {FSMState.WNTR_VERIFY, FSMState.REVISE_PLANS, FSMState.COMPLETE}
    ),
    FSMState.WNTR_VERIFY: frozenset(
        {FSMState.REVISE_PLANS, FSMState.PLAN_COMPARISON, FSMState.COMPLETE}
    ),
    FSMState.REVISE_PLANS: frozenset({FSMState.CONSTRAINT_CHECK, FSMState.COMPLETE}),
    FSMState.PLAN_COMPARISON: frozenset({FSMState.HUMAN_APPROVAL, FSMState.COMPLETE}),
    FSMState.HUMAN_APPROVAL: frozenset({FSMState.SIMULATE_RESPONSE, FSMState.COMPLETE}),
    FSMState.SIMULATE_RESPONSE: frozenset({FSMState.COMPLETE}),
    FSMState.COMPLETE: frozenset(),
}

# Cancellation, deadline exhaustion, and safety abstention must fail closed from
# every nonterminal state, including states that normally advance immediately.
for _state in FSMState:
    if _state != FSMState.COMPLETE:
        ALLOWED_TRANSITIONS[_state] = ALLOWED_TRANSITIONS[_state] | frozenset({FSMState.COMPLETE})


@dataclass(frozen=True, slots=True)
class SwarmLimits:
    max_sampling_rounds: int = 5
    max_planning_rounds: int = 3
    max_generated_plans_per_round: int = 8
    max_exact_simulations: int = 3
    max_repeated_invalid_actions: int = 2
    max_inference_calls: int = 16
    max_incident_runtime_seconds: float = 300.0
    inference_timeout_seconds: float = 15.0
    simulation_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(slots=True)
class _Runtime:
    network: Any
    incident: IncidentState
    node_ids: tuple[str, ...]
    hydraulic_state: object | None = None
    hydraulic_graph: object | None = None
    sentinel_output: SentinelOutput | None = None
    sample_recommendation: ScoutOutput | None = None
    sampling_history: list[str] = field(default_factory=list)
    plans: list[Any] = field(default_factory=list)
    valid_plans: list[Any] = field(default_factory=list)
    verifications: list[PlanVerification] = field(default_factory=list)
    verifier_feedback: list[str] = field(default_factory=list)
    selected_plan: Any | None = None
    selected_verification: PlanVerification | None = None
    planning_round: int = 0
    inference_calls: int = 0
    simulation_calls: int = 0
    invalid_signatures: Counter[tuple[tuple[str, str | None], ...]] = field(default_factory=Counter)


class SwarmController:
    """Own the incident trajectory, budgets, permissions, fallbacks and approval gate."""

    def __init__(
        self,
        *,
        sentinel: HydroSentinel,
        scout: HydroScout,
        strategist: HydroStrategist,
        verifier: Any,
        limits: SwarmLimits | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.sentinel = sentinel
        self.scout = scout
        self.strategist = strategist
        self.verifier = verifier
        self.limits = limits or SwarmLimits()
        self.clock = clock
        self.state = FSMState.IDLE
        self.fallback_mode = FallbackMode.FULL_HYBRID
        self.termination_reason: str | None = None
        self.events: list[SwarmEvent] = []
        self._runtime: _Runtime | None = None
        self._started_at: float | None = None
        self._run_key: str | None = None

    @staticmethod
    def _hash(payload: object) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    @staticmethod
    def _network_nodes(network: Any) -> tuple[str, ...]:
        if hasattr(network, "node_name_list"):
            return tuple(sorted(str(node) for node in network.node_name_list))
        if hasattr(network, "nodes"):
            return tuple(sorted(str(node) for node in network.nodes))
        raise TypeError("network must expose node_name_list or nodes")

    def _key(self, network: Any, incident: IncidentState) -> str:
        simulator = getattr(self.verifier, "simulator", None)
        network_hash = None
        if simulator is not None and hasattr(simulator, "state_hash"):
            try:
                network_hash = simulator.state_hash()
            except Exception:
                network_hash = None
        return self._hash(
            {
                "network": network_hash or self._network_nodes(network),
                "incident": incident.model_dump(mode="json"),
                "limits": {name: getattr(self.limits, name) for name in self.limits.__dataclass_fields__},
            }
        )

    def start(self, network: Any, incident: IncidentState) -> SwarmResult:
        run_key = self._key(network, incident)
        if self.state != FSMState.IDLE:
            if run_key != self._run_key:
                raise RuntimeError("controller already owns a different incident")
            return self.result()
        self._run_key = run_key
        self._runtime = _Runtime(network=network, incident=incident, node_ids=self._network_nodes(network))
        self._started_at = self.clock()
        self.transition(FSMState.NETWORK_LOADED, "network_loaded", {"node_count": len(self._runtime.node_ids)})
        return self.result()

    def transition(
        self, next_state: FSMState, event_type: str = "transition", payload: dict[str, object] | None = None
    ) -> None:
        next_state = FSMState(next_state)
        if next_state not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"invalid swarm transition {self.state} -> {next_state}")
        payload = payload or {}
        previous_hash = self.events[-1].event_hash if self.events else GENESIS_HASH
        body = {
            "sequence": len(self.events),
            "from": self.state.value,
            "to": next_state.value,
            "type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        event = SwarmEvent(
            sequence=len(self.events),
            from_state=self.state,
            to_state=next_state,
            event_type=event_type,
            payload=payload,
            previous_hash=previous_hash,
            event_hash=self._hash(body),
        )
        self.events.append(event)
        self.state = next_state

    def _terminate(self, reason: str) -> None:
        self.termination_reason = reason
        if self.state != FSMState.COMPLETE:
            self.transition(FSMState.COMPLETE, "terminated", {"reason": reason})

    def _audit(self, event_type: str, payload: dict[str, object]) -> None:
        """Record a deterministic fact without advancing the FSM."""

        previous_hash = self.events[-1].event_hash if self.events else GENESIS_HASH
        body = {
            "sequence": len(self.events),
            "from": self.state.value,
            "to": self.state.value,
            "type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        self.events.append(
            SwarmEvent(
                sequence=len(self.events),
                from_state=self.state,
                to_state=self.state,
                event_type=event_type,
                payload=payload,
                previous_hash=previous_hash,
                event_hash=self._hash(body),
            )
        )

    def _check_limits(self) -> bool:
        runtime = self._require_runtime()
        if self.clock() - float(self._started_at) > self.limits.max_incident_runtime_seconds:
            self._terminate("MAX_INCIDENT_RUNTIME")
            return False
        if runtime.inference_calls >= self.limits.max_inference_calls and self.state in {
            FSMState.SOURCE_LOCALIZATION,
            FSMState.SAMPLE_SELECTION,
            FSMState.PLAN_GENERATION,
            FSMState.REVISE_PLANS,
        }:
            self._terminate("MAX_INFERENCE_CALLS")
            return False
        return True

    def _require_runtime(self) -> _Runtime:
        if self._runtime is None:
            raise RuntimeError("start() must be called before running the controller")
        return self._runtime

    def _agent_state(self) -> dict[str, object]:
        runtime = self._require_runtime()
        sentinel = runtime.sentinel_output
        probabilities = (
            {item.node_id: item.probability for item in sentinel.top_candidates} if sentinel else {}
        )
        return {
            "incident_id": runtime.incident.incident_id,
            "observations": runtime.incident.observations,
            "hydraulic_state": runtime.hydraulic_state,
            "hydraulic_graph": runtime.hydraulic_graph,
            "node_ids": runtime.node_ids,
            "prior_candidates": (
                runtime.incident.candidates.node_probabilities if runtime.incident.candidates else probabilities
            ),
            "candidate_region": sentinel.candidate_region if sentinel else (),
            "candidate_probabilities": probabilities,
            "sampling_history": tuple(runtime.sampling_history),
            "verifier_feedback": tuple(runtime.verifier_feedback),
            "planning_round": runtime.planning_round,
        }

    def _record_agent_recovery(self, agent: Any) -> None:
        severity = {
            FallbackMode.FULL_HYBRID: 0,
            FallbackMode.DEGRADED_HYBRID: 1,
            FallbackMode.CLASSICAL_SAFE: 2,
            FallbackMode.SIMULATION_ONLY: 3,
        }
        if severity[agent.last_mode] > severity[self.fallback_mode]:
            self.fallback_mode = agent.last_mode
        if agent.last_error:
            self._audit(
                "agent_output_recovered",
                {"agent": type(agent).__name__, "mode": agent.last_mode.value},
            )

    @staticmethod
    def _plan_signature(plan: Any) -> tuple[tuple[str, str | None], ...]:
        return tuple((action.action_type.value, action.target_id) for action in plan.actions)

    def _bounded_verify(self, plan: Any) -> PlanVerification:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydroswarm-verifier")
        future = executor.submit(self.verifier.verify, plan)
        try:
            result = future.result(timeout=self.limits.simulation_timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError("verification exceeded simulation timeout") from exc
        except BaseException:
            executor.shutdown(wait=True)
            raise
        else:
            executor.shutdown(wait=True)
            return PlanVerification.model_validate(result)

    def run(self) -> SwarmResult:
        self._require_runtime()
        while self.state not in {
            FSMState.COMPLETE,
            FSMState.HUMAN_APPROVAL,
            FSMState.SAMPLE_SELECTION,
        }:
            if not self._check_limits():
                break
            self._step()
        # SAMPLE_SELECTION invokes Scout once, then pauses for a real observation.
        if self.state == FSMState.SAMPLE_SELECTION and self._require_runtime().sample_recommendation is None:
            if self._check_limits():
                self._step()
        return self.result()

    def execute(self, network: Any | None = None, incident: IncidentState | None = None) -> SwarmResult:
        if network is not None or incident is not None:
            if network is None or incident is None:
                raise ValueError("network and incident must be supplied together")
            self.start(network, incident)
        return self.run()

    def _step(self) -> None:
        runtime = self._require_runtime()
        if self.state == FSMState.NETWORK_LOADED:
            issues: tuple[str, ...] = ()
            simulator = getattr(self.verifier, "simulator", None)
            if simulator is not None and hasattr(simulator, "validate"):
                issues = tuple(simulator.validate())
            if issues:
                self._terminate(f"INVALID_NETWORK:{','.join(issues)}")
            else:
                self.transition(FSMState.NETWORK_VALIDATED, "network_validated")
        elif self.state == FSMState.NETWORK_VALIDATED:
            self.transition(FSMState.INCIDENT_LOADED, "incident_loaded")
        elif self.state == FSMState.INCIDENT_LOADED:
            self.transition(FSMState.DATA_QUALITY_CHECK, "data_quality_check")
        elif self.state == FSMState.DATA_QUALITY_CHECK:
            usable = [item for item in runtime.incident.observations if not item.missing and item.quality > 0]
            if not usable:
                self._terminate("NO_USABLE_SENSOR_DATA")
            else:
                self.transition(
                    FSMState.HYDRAULIC_STATE_ESTIMATION,
                    "data_quality_accepted",
                    {"usable_observations": len(usable)},
                )
        elif self.state == FSMState.HYDRAULIC_STATE_ESTIMATION:
            simulator = getattr(self.verifier, "simulator", None)
            try:
                if simulator is not None:
                    runtime.hydraulic_state = simulator.calculate_state()
                    runtime.hydraulic_graph = simulator.build_dynamic_graph(runtime.hydraulic_state)
            except Exception:
                self.fallback_mode = FallbackMode.CLASSICAL_SAFE
                runtime.hydraulic_state = None
                runtime.hydraulic_graph = None
            self.transition(FSMState.SOURCE_LOCALIZATION, "hydraulic_state_estimated")
        elif self.state == FSMState.SOURCE_LOCALIZATION:
            runtime.inference_calls += 1
            runtime.sentinel_output = self.sentinel.invoke(
                self._agent_state(), timeout_seconds=self.limits.inference_timeout_seconds
            )
            self._record_agent_recovery(self.sentinel)
            output = runtime.sentinel_output
            probabilities = {node: 0.0 for node in output.candidate_region}
            probabilities.update({item.node_id: item.probability for item in output.top_candidates})
            total = sum(probabilities.values())
            normalized = {node: value / total for node, value in probabilities.items()}
            runtime.incident = runtime.incident.model_copy(
                update={
                    "candidates": CandidateSet(
                        node_probabilities=normalized,
                        node_ids=output.candidate_region,
                        calibrated=False,
                    ),
                    "ood_level": output.ood_level,
                }
            )
            self.transition(
                FSMState.EVIDENCE_CHECK,
                "source_localized",
                {
                    "candidate_region": list(output.candidate_region),
                    "probabilities": normalized,
                    "evidence_sufficient": output.evidence_sufficient,
                    "ood_level": output.ood_level.value,
                },
            )
        elif self.state == FSMState.EVIDENCE_CHECK:
            output = runtime.sentinel_output
            if output is None:
                self._terminate("MISSING_SENTINEL_OUTPUT")
            elif output.ood_level == OODLevel.OUTSIDE_VALIDATED_RANGE:
                self._terminate("OUT_OF_DISTRIBUTION")
            elif output.evidence_sufficient:
                self.transition(FSMState.PLAN_GENERATION, "evidence_sufficient")
            elif runtime.incident.sample_count >= min(5, self.limits.max_sampling_rounds):
                self._terminate("MAX_SAMPLING_ROUNDS")
            else:
                runtime.sample_recommendation = None
                self.transition(FSMState.SAMPLE_SELECTION, "more_evidence_required")
        elif self.state == FSMState.SAMPLE_SELECTION:
            runtime.inference_calls += 1
            recommendation = self.scout.invoke(
                self._agent_state(), timeout_seconds=self.limits.inference_timeout_seconds
            )
            self._record_agent_recovery(self.scout)
            if recommendation.action == ScoutAction.STOP:
                self._terminate("SCOUT_STOPPED_SAMPLING")
            else:
                runtime.sample_recommendation = recommendation
                self._audit(
                    "sample_recommended",
                    {
                        "node_id": recommendation.node_id,
                        "expected_information_gain": recommendation.expected_information_gain,
                        "alternatives": list(recommendation.alternatives),
                    },
                )
        elif self.state == FSMState.SOURCE_UPDATE:
            self.transition(FSMState.SOURCE_LOCALIZATION, "source_update_ready")
        elif self.state == FSMState.PLAN_GENERATION:
            runtime.inference_calls += 1
            output = self.strategist.invoke(
                self._agent_state(), timeout_seconds=self.limits.inference_timeout_seconds
            )
            self._record_agent_recovery(self.strategist)
            runtime.plans = [
                candidate.plan for candidate in output.plans[: self.limits.max_generated_plans_per_round]
            ]
            runtime.valid_plans = []
            self.transition(
                FSMState.CONSTRAINT_CHECK,
                "plans_generated",
                {
                    "count": len(runtime.plans),
                    "plans": [
                        {
                            "plan_id": str(plan.plan_id),
                            "actions": [
                                {"type": action.action_type.value, "target": action.target_id}
                                for action in plan.actions
                            ],
                        }
                        for plan in runtime.plans
                    ],
                },
            )
        elif self.state == FSMState.CONSTRAINT_CHECK:
            runtime.valid_plans = []
            for plan in runtime.plans:
                codes = tuple(self.verifier.prescreen(plan)) if hasattr(self.verifier, "prescreen") else ()
                if codes:
                    runtime.verifier_feedback.extend(codes)
                    runtime.invalid_signatures[self._plan_signature(plan)] += 1
                elif runtime.invalid_signatures[self._plan_signature(plan)] < self.limits.max_repeated_invalid_actions:
                    runtime.valid_plans.append(plan)
            if runtime.valid_plans:
                self.transition(FSMState.WNTR_VERIFY, "constraint_check_passed")
            elif runtime.planning_round + 1 < self.limits.max_planning_rounds:
                self.transition(FSMState.REVISE_PLANS, "all_plans_failed_prescreen")
            else:
                self._terminate("NO_VALID_PLAN")
        elif self.state == FSMState.WNTR_VERIFY:
            for plan in runtime.valid_plans:
                if runtime.simulation_calls >= self.limits.max_exact_simulations:
                    break
                runtime.simulation_calls += 1
                try:
                    verification = self._bounded_verify(plan)
                except TimeoutError:
                    runtime.verifier_feedback.append("SIMULATION_TIMEOUT")
                    self.fallback_mode = FallbackMode.SIMULATION_ONLY
                    continue
                except Exception:
                    runtime.verifier_feedback.append("SIMULATION_FAILURE")
                    self.fallback_mode = FallbackMode.SIMULATION_ONLY
                    continue
                runtime.verifications.append(verification)
                self._audit(
                    "plan_verified",
                    {
                        "plan_id": str(plan.plan_id),
                        "decision": verification.decision.value,
                        "rejection_codes": list(verification.rejection_codes),
                        "state_hash": verification.state_hash,
                    },
                )
                if verification.decision != PlanDecision.VERIFIED:
                    runtime.verifier_feedback.extend(verification.rejection_codes)
                    runtime.invalid_signatures[self._plan_signature(plan)] += 1
            if any(item.decision == PlanDecision.VERIFIED for item in runtime.verifications):
                self.transition(FSMState.PLAN_COMPARISON, "verified_plans_available")
            elif (
                runtime.planning_round + 1 < self.limits.max_planning_rounds
                and runtime.simulation_calls < self.limits.max_exact_simulations
            ):
                self.transition(FSMState.REVISE_PLANS, "verifier_rejected_plans")
            else:
                self._terminate("NO_VALID_PLAN")
        elif self.state == FSMState.REVISE_PLANS:
            runtime.planning_round += 1
            runtime.inference_calls += 1
            output: StrategistOutput = self.strategist.invoke(
                self._agent_state(), timeout_seconds=self.limits.inference_timeout_seconds
            )
            self._record_agent_recovery(self.strategist)
            runtime.plans = [
                candidate.plan for candidate in output.plans[: self.limits.max_generated_plans_per_round]
            ]
            self.transition(
                FSMState.CONSTRAINT_CHECK,
                "plans_revised",
                {"round": runtime.planning_round, "feedback": sorted(set(runtime.verifier_feedback))},
            )
        elif self.state == FSMState.PLAN_COMPARISON:
            verified = [item for item in runtime.verifications if item.decision == PlanDecision.VERIFIED]
            if not verified:
                self._terminate("NO_VALID_PLAN")
                return
            verified.sort(
                key=lambda item: (
                    -item.consequences.service_availability,
                    -item.consequences.minimum_pressure_m,
                    str(item.plan_id),
                )
            )
            runtime.selected_verification = verified[0]
            runtime.selected_plan = next(plan for plan in runtime.plans if plan.plan_id == verified[0].plan_id)
            self.transition(FSMState.HUMAN_APPROVAL, "plan_selected", {"plan_id": str(verified[0].plan_id)})
        elif self.state == FSMState.SIMULATE_RESPONSE:
            # Exact WNTR verification is the response counterfactual; physical actuation is never automatic.
            self.transition(FSMState.COMPLETE, "response_simulated")
        else:
            raise RuntimeError(f"no step handler for {self.state}")

    def add_sample(self, observation: SensorObservation) -> SwarmResult:
        runtime = self._require_runtime()
        if self.state != FSMState.SAMPLE_SELECTION or runtime.sample_recommendation is None:
            raise RuntimeError("controller is not awaiting a sample")
        runtime.sampling_history.append(observation.node_id)
        runtime.incident = runtime.incident.model_copy(
            update={
                "observations": (*runtime.incident.observations, observation),
                "sample_count": runtime.incident.sample_count + 1,
                "status": "SAMPLING",
            }
        )
        runtime.sample_recommendation = None
        self.transition(FSMState.SOURCE_UPDATE, "sample_received", {"node_id": observation.node_id})
        return self.run()

    def approve(self, plan_id: UUID | None = None) -> SwarmResult:
        runtime = self._require_runtime()
        if self.state != FSMState.HUMAN_APPROVAL:
            raise RuntimeError("approval is accepted only at HUMAN_APPROVAL")
        if plan_id is not None and runtime.selected_plan.plan_id != plan_id:
            raise ValueError("only the selected verified plan can be approved")
        self.transition(
            FSMState.SIMULATE_RESPONSE,
            "human_approved",
            {"plan_id": str(runtime.selected_plan.plan_id)},
        )
        return self.run()

    def abort(self, reason: str = "OPERATOR_ABORTED") -> SwarmResult:
        if self.state == FSMState.COMPLETE:
            return self.result()
        self._terminate(reason)
        return self.result()

    def result(self) -> SwarmResult:
        runtime = self._runtime
        return SwarmResult(
            state=self.state,
            fallback_mode=self.fallback_mode,
            awaiting_sample=runtime.sample_recommendation if runtime else None,
            selected_plan=runtime.selected_plan if runtime else None,
            verification=runtime.selected_verification if runtime else None,
            termination_reason=self.termination_reason,
            events=tuple(self.events),
            run_key=self._run_key or GENESIS_HASH,
            plans_exactly_verified=len(runtime.verifications) if runtime else 0,
        )

    @classmethod
    def replay(cls, events: tuple[SwarmEvent, ...] | list[SwarmEvent]) -> ReplayResult:
        state = FSMState.IDLE
        previous_hash = GENESIS_HASH
        for index, supplied in enumerate(events):
            event = SwarmEvent.model_validate(supplied)
            if event.sequence != index or event.previous_hash != previous_hash or event.from_state != state:
                raise ValueError("event sequence, state, or hash chain is invalid")
            if event.to_state != state and event.to_state not in ALLOWED_TRANSITIONS[state]:
                raise ValueError(f"invalid replay transition {state} -> {event.to_state}")
            body = {
                "sequence": event.sequence,
                "from": event.from_state.value,
                "to": event.to_state.value,
                "type": event.event_type,
                "payload": event.payload,
                "previous_hash": event.previous_hash,
            }
            if event.event_hash != cls._hash(body):
                raise ValueError("event hash mismatch")
            state = event.to_state
            previous_hash = event.event_hash
        return ReplayResult(
            state=state,
            event_count=len(events),
            final_hash=previous_hash,
            trajectory=tuple(SwarmEvent.model_validate(event) for event in events),
        )
