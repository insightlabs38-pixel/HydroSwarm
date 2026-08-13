"""End-to-end classical, neural, calibrated and operational hybrid inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import time
from typing import Any
from uuid import UUID, uuid5

import numpy as np
import torch

from hydroswarm.classical import (
    HydraulicStateEstimator,
    OperationalTelemetry,
    SignatureArtifact,
    localize_with_signatures,
)
from hydroswarm.calibration import CalibrationArtifact, SplitConformalCalibrator
from hydroswarm.data.scenarios import network_sha256
from hydroswarm.inference.fusion import (
    DYNAMIC_TRUST_FUSION_CONFIG,
    ControlAction,
    FusionDiagnostics,
    TrustFeatures,
    fuse_source_probabilities,
    uncertainty_control,
)
from hydroswarm.planning import (
    PlanGenerationContext,
    PlanProposal,
    generate_response_plans,
)
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.planning.candidate_tensorizer import plan_proposals_to_candidate_tensors
from hydroswarm.preprocessing import BuiltHydroBatch, HydraulicFeatureBuilder, SensorSeries
from hydroswarm.sampling import ActiveSamplingResult, SamplingConstraints, rank_sample_locations
from hydroswarm.simulation.wrapper import FEATURE_SNAPSHOT_TIME_SECONDS
from hydroswarm.tasks import RUNTIME_TASKS, validate_tasks

# hydroswarm.training's package __init__ imports hydroswarm.training.full_trajectory,
# which imports HybridInferencePipeline from this module -- a module-level
# import here of anything under hydroswarm.training would be a circular
# import. See _model_semantics's own local imports below.

from .ood import OODDetector
from .results import (
    EvidenceChange,
    EvidenceSnapshot,
    HybridRuntimeMode,
    IncidentAnalysisResult,
    PosteriorSnapshot,
    SemanticPredictions,
)


#: Architecture identifier surfaced in plan/provenance records and the
#: incident-view API contract (overnight-plan.txt Task 3.2). Bump this when
#: the hybrid architecture itself changes; checkpoint identity is tracked
#: separately by ``self._model_hash``.
MODEL_VERSION = "hydrocore-hybrid-v1"


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or total <= 0:
        raise ValueError("belief vector must contain finite positive mass")
    return values / total


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("model logits must be finite and non-empty")
    shifted = values - values.max()
    return _normalise(np.exp(shifted))


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log2(positive)).sum())


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float64)


def _fail_closed_control_action(
    proposed: ControlAction,
    suppression_reasons: Sequence[str],
    *,
    sample_budget_remaining: int,
) -> ControlAction:
    """Prevent a generic uncertainty recommendation from outranking authority.

    ``uncertainty_control`` deliberately does not know about calibration and
    other pipeline-level gates.  If it recommends plan generation after one
    of those gates suppressed planning, map the known reason to an existing
    non-planning action.  This leaves the underlying scientific thresholds
    untouched and keeps ``GENERATE_PLANS`` synonymous with actual authority.
    """

    if proposed != ControlAction.GENERATE_PLANS:
        return proposed
    reasons = frozenset(suppression_reasons)
    if (
        "ALL_SENSORS_FROZEN" in reasons
        or "CALIBRATION_INVALID_OR_MISSING" in reasons
        or any(reason.startswith("OOD_") for reason in reasons)
    ):
        return ControlAction.ABSTAIN
    if "HIGH_CLASSICAL_NEURAL_DISAGREEMENT" in reasons:
        return ControlAction.INSPECT_SENSORS
    if (
        "CANDIDATE_REGION_TOO_BROAD" in reasons
        or "MODEL_EVIDENCE_INSUFFICIENT" in reasons
    ):
        return (
            ControlAction.REQUEST_SAMPLE
            if sample_budget_remaining > 0
            else ControlAction.ABSTAIN
        )
    # Future explicit suppressions must fail closed even before a more
    # specific action is designed for them.
    return ControlAction.CONTINUE_ANALYSIS


class HybridInferencePipeline:
    """Compose hydraulic evidence and learned residuals without hiding fallbacks."""

    def __init__(
        self,
        *,
        simulator: Any,
        signature_artifact: SignatureArtifact,
        model: Any | None,
        model_hash: str | None = None,
        calibration_artifact: CalibrationArtifact | None = None,
        state_estimator: HydraulicStateEstimator | None = None,
        feature_builder: HydraulicFeatureBuilder | None = None,
        ood_detector: OODDetector | None = None,
        localizer: Callable[..., Any] = localize_with_signatures,
        sampling_ranker: Callable[..., ActiveSamplingResult] = rank_sample_locations,
        planner: Callable[..., tuple[PlanProposal, ...]] = generate_response_plans,
        maximum_planning_candidates: int = 3,
        disagreement_threshold: float = 0.50,
        evidence_threshold: float = 0.55,
        clock: Callable[[], float] = time.perf_counter,
        trained_tasks: frozenset[str] | None = None,
        fusion_config_hash: str | None = None,
        runtime_enabled_outputs: frozenset[str] | None = None,
    ) -> None:
        if maximum_planning_candidates < 1:
            raise ValueError("maximum_planning_candidates must be positive")
        # core-issues.txt repair item 8: `None` means "no gating" (every
        # existing caller that predates this parameter, and every test that
        # wires a fake model to exercise a head directly, keeps behaving
        # exactly as before). Production wiring (runtime/defaults.py) always
        # passes the checkpoint's own declared `trained_tasks`, which is the
        # only place this actually needs to fail closed.
        self.trained_tasks = RUNTIME_TASKS if trained_tasks is None else frozenset(trained_tasks)
        validate_tasks(self.trained_tasks, label="trained_tasks")
        # core-issues3.txt Phase 15 item 2: granular output-level gating,
        # additive to (not a replacement for) trained_tasks's coarser
        # role-level gating above -- trained_tasks still governs
        # Scout/Strategist/OOD wholesale (none of those roles are promoted
        # today; see reports/results/v4/phase14-promotion-gates.md).
        # `None` (the default) means "no v4 checkpoint identity available",
        # exactly like trained_tasks=None -- every existing caller/test
        # keeps behaving exactly as before. A v4-aware factory
        # (hydroswarm.runtime.v4_defaults) always passes the checkpoint
        # identity's own declared runtime_enabled_outputs.
        self.runtime_enabled_outputs = runtime_enabled_outputs
        # core-issues.txt repair item 10: `None` (the default) skips the
        # fusion_config_hash check entirely -- every existing
        # CalibrationArtifact test fixture and the currently-promoted
        # checkpoint's own calibration.json predate this field and would
        # otherwise be spuriously invalidated. Production wiring
        # (runtime/defaults.py) passes DYNAMIC_TRUST_FUSION_CONFIG
        # explicitly, since that is the one place this actually needs to
        # fail closed on a real, deployed mismatch.
        self.fusion_config_hash = fusion_config_hash
        self.simulator = simulator
        self.signature_artifact = signature_artifact
        self.model = model
        self.calibration_artifact = calibration_artifact
        self.state_estimator = state_estimator or HydraulicStateEstimator()
        self.feature_builder = feature_builder or HydraulicFeatureBuilder()
        self.ood_detector = ood_detector or OODDetector()
        self.localizer = localizer
        self.sampling_ranker = sampling_ranker
        self.planner = planner
        self.maximum_planning_candidates = maximum_planning_candidates
        self.disagreement_threshold = disagreement_threshold
        self.evidence_threshold = evidence_threshold
        self.clock = clock
        self._history: dict[UUID, tuple[PosteriorSnapshot, ...]] = {}
        self._evidence: dict[UUID, tuple[EvidenceSnapshot, ...]] = {}
        self._changes: dict[UUID, tuple[EvidenceChange, ...]] = {}
        self._cache: dict[tuple[UUID, str], IncidentAnalysisResult] = {}
        self._model_hash = model_hash or self._fingerprint_model(model)

    @staticmethod
    def _fingerprint_model(model: Any | None) -> str:
        digest = hashlib.sha256()
        if model is None:
            digest.update(b"classical-safe:no-model")
        elif hasattr(model, "state_dict"):
            digest.update(type(model).__qualname__.encode())
            for name, tensor in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(f"{type(model).__module__}.{type(model).__qualname__}".encode())
        return digest.hexdigest()

    @staticmethod
    def _evidence_payload(
        sensor_series: Sequence[SensorSeries], telemetry: OperationalTelemetry
    ) -> dict[str, object]:
        return {
            "sensors": [
                {
                    "node": item.node_id,
                    "times": item.timestamps_seconds,
                    "concentration": item.concentration_mg_l,
                    "pressure": item.pressure_m,
                    "health": item.health,
                    "missing": item.missing,
                    "drift": item.drift,
                    "delayed": item.delayed,
                    "frozen": item.frozen,
                }
                for item in sorted(sensor_series, key=lambda value: value.node_id)
            ],
            "telemetry": {
                "pressure": dict(telemetry.pressure_m),
                "demand": dict(telemetry.demand_m3s),
                "flow": dict(telemetry.flow_m3s),
                "tank": dict(telemetry.tank_level_m),
                "pump": dict(telemetry.pump_open),
                "valve": dict(telemetry.valve_open),
                "overrides": dict(telemetry.operator_overrides),
            },
        }

    @staticmethod
    def _signature_observations(
        sensor_series: Sequence[SensorSeries], artifact: SignatureArtifact
    ) -> tuple[np.ndarray, np.ndarray]:
        by_node = {item.node_id: item for item in sensor_series}
        observations = np.zeros(
            (len(artifact.sample_times_seconds), len(artifact.sensor_nodes)), dtype=np.float64
        )
        mask = np.zeros_like(observations, dtype=bool)
        for sensor_index, node in enumerate(artifact.sensor_nodes):
            series = by_node.get(node)
            if series is None:
                continue
            timestamps = np.asarray(series.timestamps_seconds, dtype=float)
            for time_index, target in enumerate(artifact.sample_times_seconds):
                source_index = int(np.argmin(np.abs(timestamps - target)))
                value = series.concentration_mg_l[source_index]
                valid = value is not None and not series.missing[source_index] and np.isfinite(value)
                if valid:
                    observations[time_index, sensor_index] = float(value)
                    mask[time_index, sensor_index] = True
        if not mask.any():
            raise ValueError("signature localization requires at least one valid concentration observation")
        return observations, mask

    def _model_semantics(
        self, output: Mapping[str, Any], node_ids: tuple[str, ...]
    ) -> SemanticPredictions:
        # Local imports: see this module's top-of-file circular-import note.
        from hydroswarm.training.checkpoint_identity import OOD_CATEGORY_NAMES
        from hydroswarm.training.control_labels import NEXT_STEP_RUNTIME_ENABLED
        from hydroswarm.training.corpus import EVENT_CAUSE_INDEX, SUPPORTED_EVENT_CAUSES
        from hydroswarm.training.ood_labels import SUPPORTED_OOD_CATEGORIES
        from hydroswarm.training.targets_v2 import EventCause, NextStep

        def scalar(key: str) -> float | None:
            value = output.get(key)
            return float(_array(value).reshape(-1)[0]) if value is not None else None

        # core-issues3.txt Phase 15 item 2: when a v4 checkpoint identity is
        # active (runtime_enabled_outputs is not None), an output not in it
        # must contribute exactly zero -- checked per-field below, distinct
        # from "the head is simply absent from this model" (which already,
        # separately, produces None via output.get(...) returning None).
        enabled = self.runtime_enabled_outputs

        def granular_enabled(canonical_name: str) -> bool:
            return enabled is None or canonical_name in enabled

        sample_values = output.get("expected_information_gain")
        faults = output.get("sensor_fault_logits")
        plan_values = output.get("plan_value")
        plan_validity = output.get("plan_validity_logits")

        event_presence_probability: float | None = None
        event_presence: bool | None = None
        if granular_enabled("event_presence"):
            event_presence_probability = scalar("event_presence_logits")
            if event_presence_probability is not None:
                event_presence_probability = float(1.0 / (1.0 + np.exp(-event_presence_probability)))
                event_presence = event_presence_probability >= 0.5

        event_cause: str | None = None
        cause_logits = output.get("event_cause_logits")
        if granular_enabled("event_cause") and cause_logits is not None:
            predicted_index = int(np.argmax(_array(cause_logits).reshape(-1)[: len(EventCause)]))
            predicted_cause = next(cause for cause, index in EVENT_CAUSE_INDEX.items() if index == predicted_index)
            # Phase 6.5/9.3: never surface a currently-unsupported class
            # (AMBIGUOUS/HYDRAULIC_MISMATCH have zero real training
            # examples -- see reports/results/v4/phase13-metrics-and-baselines.md)
            # as a live prediction.
            if predicted_cause in SUPPORTED_EVENT_CAUSES:
                event_cause = predicted_cause.value

        next_step: str | None = None
        next_step_logits = output.get("next_step_logits")
        if granular_enabled("next_step") and next_step_logits is not None:
            ordered_next_steps = tuple(NextStep)
            predicted_index = int(np.argmax(_array(next_step_logits).reshape(-1)[: len(ordered_next_steps)]))
            predicted_step = ordered_next_steps[predicted_index]
            if predicted_step in NEXT_STEP_RUNTIME_ENABLED:
                next_step = predicted_step.value

        # core-issues5.txt Section 18.1: advisory only -- never read by
        # OODDetector/the deterministic controller. Gated by
        # runtime_enabled_outputs like every other v4 advisory field above;
        # every real checkpoint identity built so far excludes
        # "ood_category" from it (see output_governance.OOD_CONTROL_OUTPUTS),
        # so this resolves to None in production today, exactly like
        # sensor_fault_probability.
        ood_category: str | None = None
        ood_category_logits = output.get("ood_category_logits")
        if granular_enabled("ood_category") and ood_category_logits is not None:
            predicted_index = int(np.argmax(_array(ood_category_logits).reshape(-1)[: len(OOD_CATEGORY_NAMES)]))
            predicted_category_name = OOD_CATEGORY_NAMES[predicted_index]
            if predicted_category_name in {category.value for category in SUPPORTED_OOD_CATEGORIES}:
                ood_category = predicted_category_name

        return SemanticPredictions(
            evidence_sufficiency=scalar("evidence_sufficiency") if granular_enabled("evidence_sufficiency") else None,
            uncertainty=scalar("uncertainty") if granular_enabled("uncertainty") else None,
            expected_information_gain=(
                dict(zip(node_ids, _array(sample_values).reshape(-1)[-len(node_ids):], strict=True))
                if sample_values is not None and granular_enabled("information_gain")
                else None
            ),
            sensor_fault_probability=(
                dict(
                    zip(
                        node_ids,
                        1.0 / (1.0 + np.exp(-_array(faults).reshape(-1)[-len(node_ids):])),
                        strict=True,
                    )
                )
                if faults is not None and granular_enabled("sensor_fault")
                else None
            ),
            # core-issues5.txt Section 11 (P0 governance fix): granular
            # v4 governance must be independently authoritative here too --
            # previously only the coarse trained_tasks role switch
            # (analyze()'s own "strategist" not in self.trained_tasks
            # override) gated these two fields; a v4 identity that granularly
            # excluded plan_value/plan_validity from runtime_enabled_outputs
            # while trained_tasks still permitted "strategist" would not
            # have been enforced at all at this layer. effective_enabled =
            # legacy_role_allows AND runtime_enabled_outputs contains
            # output -- the AND is now real on both sides, not just one.
            plan_values=tuple(float(value) for value in _array(plan_values).reshape(-1))
            if plan_values is not None and granular_enabled("plan_value")
            else (),
            plan_validity=tuple(
                float(_softmax(row)[-1]) for row in _array(plan_validity).reshape(-1, 2)
            )
            if plan_validity is not None and granular_enabled("plan_validity")
            else (),
            event_presence=event_presence,
            event_presence_probability=event_presence_probability,
            event_cause=event_cause,
            next_step=next_step,
            ood_category=ood_category,
        )

    def _run_model(self, built: BuiltHydroBatch) -> Mapping[str, Any]:
        if self.model is None:
            raise RuntimeError("no neural model configured")
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            output = self.model(built.batch)
        if not isinstance(output, Mapping):
            raise TypeError("HydroCore must return a mapping of semantic tensors")
        return output

    def _score_candidate_plans(
        self, proposals: tuple[PlanProposal, ...], built: BuiltHydroBatch, graph: Any
    ) -> tuple[dict[str, float], dict[str, float]]:
        """core-issues5.txt Section 6 (P0 blocker): PASS 2 -- score the real,
        bounded deterministic candidate set already generated by
        `self.planner`, using the actual candidate-conditioned architecture
        that was trained (not an anonymous positional-delta approximation).

        Returns (value_deltas, validity_deltas) keyed by `template` name --
        a stable identity, not tensor position, so
        `generate_response_plans` (which already accepts
        `Mapping[str, float]` deltas keyed this same way) can apply them
        regardless of candidate order. Both dicts are empty whenever
        Strategist scoring is unavailable or not governed as runtime-
        enabled: no model, the coarse `trained_tasks` role switch excludes
        "strategist" (core-issues.txt repair item 8's existing convention),
        the granular v4 `runtime_enabled_outputs` identity excludes
        `plan_value`/`plan_validity` (neither is runtime-enabled for any
        checkpoint built so far -- Phase 14's gate 7, >= 2 finalist seeds,
        is unmet), or the PASS-2 forward itself raises. Every one of these
        is the SAME "no learned signal available" outcome from
        `generate_response_plans`'s perspective: deterministic heuristic
        ordering, never a hard failure.
        """

        empty: tuple[dict[str, float], dict[str, float]] = ({}, {})
        if self.model is None or "strategist" not in self.trained_tasks or not proposals:
            return empty
        enabled = self.runtime_enabled_outputs

        def granular_enabled(name: str) -> bool:
            return enabled is None or name in enabled

        if not granular_enabled("plan_value") and not granular_enabled("plan_validity"):
            return empty
        # The entire PASS-2 forward AND delta extraction is one fail-closed
        # unit: any failure (shape mismatch against a stub/incompatible
        # model, a corrupted checkpoint, an architecture that does not
        # actually support candidate-conditioned scoring) falls back to
        # empty deltas -- deterministic heuristic ordering only, exactly
        # like a PASS-1 neural failure falls back to CLASSICAL_SAFE. Never
        # crash the whole incident analysis over a planning-scoring error.
        try:
            plan_tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=built.node_ids, graph=graph)
            batch = {**built.batch, **plan_tensors}
            if hasattr(self.model, "eval"):
                self.model.eval()
            with torch.no_grad():
                output = self.model(batch)
            if not isinstance(output, Mapping):
                raise TypeError("HydroCore must return a mapping of semantic tensors")

            value_deltas: dict[str, float] = {}
            if granular_enabled("plan_value"):
                plan_values = output.get("plan_value")
                if plan_values is not None:
                    values = _array(plan_values).reshape(-1)
                    for proposal, value in zip(proposals, values, strict=True):
                        value_deltas[proposal.template] = math.tanh(float(value)) * 0.1

            validity_deltas: dict[str, float] = {}
            if granular_enabled("plan_validity"):
                plan_validity = output.get("plan_validity_logits")
                if plan_validity is not None:
                    rows = _array(plan_validity).reshape(len(proposals), -1)
                    for proposal, row in zip(proposals, rows, strict=True):
                        probability = float(_softmax(row)[-1])
                        validity_deltas[proposal.template] = (probability - 0.5) * 0.2

            return value_deltas, validity_deltas
        except Exception:
            return empty

    @staticmethod
    def _credible_nodes(belief: Mapping[str, float], mass_target: float = 0.90) -> tuple[str, ...]:
        selected: list[str] = []
        mass = 0.0
        for node, probability in sorted(belief.items(), key=lambda item: (-item[1], item[0])):
            if probability <= 0:
                continue
            selected.append(node)
            mass += probability
            if mass >= mass_target:
                break
        return tuple(selected)

    @staticmethod
    def _planning_context(
        incident_id: UUID,
        network: Any,
        graph: Any,
        probable_nodes: tuple[str, ...],
        sampled_nodes: frozenset[str],
    ) -> PlanGenerationContext:
        junctions = tuple(str(node) for node in network.junction_name_list)
        monitors = probable_nodes + tuple(node for node in junctions if node not in probable_nodes)
        downstream: list[str] = []
        for source in probable_nodes:
            if source in graph:
                downstream.extend(str(node) for node in graph.successors(source) if str(node) in junctions)
        if not downstream:
            downstream = list(monitors)
        demand_ranked = sorted(
            junctions,
            key=lambda node: -float(graph.nodes[node].get("demand_m3s", 0.0)),
        )
        return PlanGenerationContext(
            incident_id=incident_id,
            model_version=MODEL_VERSION,
            probable_source_nodes=probable_nodes,
            isolatable_links=tuple(str(link) for link in network.pipe_name_list),
            downstream_flush_nodes=tuple(dict.fromkeys(downstream)),
            critical_demand_nodes=tuple(demand_ranked[:2]),
            monitor_nodes=tuple(dict.fromkeys(monitors)),
            sampled_nodes=sampled_nodes,
        )

    @staticmethod
    def _deterministic_plans(
        proposals: tuple[PlanProposal, ...], incident_id: UUID, sensor_series: Sequence[SensorSeries]
    ) -> tuple[PlanProposal, ...]:
        earliest = min(time for series in sensor_series for time in series.timestamps_seconds)
        created_at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=float(earliest))
        deterministic: list[PlanProposal] = []
        for proposal in proposals:
            signature = ";".join(
                f"{action.action_type.value}:{action.target_id}:{action.start_minute}:{action.duration_minutes}"
                for action in proposal.plan.actions
            )
            plan = proposal.plan.model_copy(
                update={
                    "plan_id": uuid5(incident_id, f"{proposal.template}:{signature}"),
                    "created_at": created_at,
                }
            )
            deterministic.append(replace(proposal, plan=plan))
        return tuple(deterministic)

    def analyze(
        self,
        incident_id: UUID,
        network: Any,
        sensor_series: Sequence[SensorSeries],
        *,
        telemetry: OperationalTelemetry | None = None,
        signature_artifact: SignatureArtifact | None = None,
        calibration_artifact: CalibrationArtifact | None = None,
        sampling_constraints: SamplingConstraints | None = None,
        hypothesis_prior: Mapping[str, float] | None = None,
        sample_budget_remaining: int = 5,
        previous_result: IncidentAnalysisResult | None = None,
        noise_scale: float = 0.05,
    ) -> IncidentAnalysisResult:
        if not sensor_series:
            raise ValueError("at least one sensor series is required")
        artifact = signature_artifact or self.signature_artifact
        calibration = calibration_artifact if calibration_artifact is not None else self.calibration_artifact
        telemetry = telemetry or OperationalTelemetry()
        evidence_hash = _hash(self._evidence_payload(sensor_series, telemetry))
        analysis_hash = _hash(
            {
                "evidence": evidence_hash,
                "signature": artifact.artifact_hash,
                "calibration": calibration.artifact_hash if calibration else None,
                "hypothesis_prior": dict(hypothesis_prior or {}),
                "sample_budget": sample_budget_remaining,
                "sampling_constraints": sampling_constraints,
                "noise_scale": noise_scale,
            }
        )
        cache_key = (incident_id, analysis_hash)
        if cache_key in self._cache:
            return self._cache[cache_key]
        latencies: dict[str, float] = {}

        def stage(name: str, function: Callable[[], Any]) -> Any:
            started = self.clock()
            value = function()
            latencies[name] = max(0.0, (self.clock() - started) * 1000.0)
            return value

        network_hash = (
            self.simulator.state_hash()
            if hasattr(self.simulator, "state_hash")
            else _hash(tuple(sorted(str(node) for node in network.node_name_list)))
        )
        # core-issues.txt repair item 10: the STRUCTURAL topology identity
        # (stable per named network family, unlike network_hash above which
        # also captures this incident's randomized/live hydraulic state) --
        # the same function hydroswarm.training.corpus uses to populate
        # TopologyMetadata.topology_hash, so a CalibrationArtifact fit
        # against real corpus examples can be checked against it here.
        topology_hash = network_sha256(network)
        # Phase 3 item 18 discovery: calculate_state() with no argument
        # defaults to the network's LAST simulated timestamp, not the
        # FEATURE_SNAPSHOT_TIME_SECONDS snapshot every feature-building path
        # (hydroswarm.training.corpus.build_feature_context, hydroswarm.cli's
        # fixed-inference verification) actually uses -- for a typical
        # multi-hour simulation this silently fed the model a hydraulic
        # snapshot far outside its training distribution. See
        # FEATURE_SNAPSHOT_TIME_SECONDS's docstring for the full writeup.
        raw_state = stage(
            "hydraulic_simulation",
            lambda: self.simulator.calculate_state(FEATURE_SNAPSHOT_TIME_SECONDS),
        )
        tank_levels = {name: raw_state.pressure_m.get(name, 0.0) for name in network.tank_name_list}
        estimated = stage(
            "state_estimation",
            lambda: self.state_estimator.estimate(
                raw_state, telemetry, model_tank_levels_m=tank_levels
            ),
        )
        graph = stage(
            "dynamic_graph",
            lambda: self.simulator.build_dynamic_graph(estimated.as_hydraulic_state()),
        )
        observations, observation_mask = self._signature_observations(sensor_series, artifact)
        feasible = {
            hypothesis.source_node: bool(
                hypothesis.source_node in graph
                and graph.nodes[hypothesis.source_node].get("reservoir_reachable", True)
            )
            for hypothesis in artifact.hypotheses
        }
        # core-issues5.txt delta item 1: `classical` below is
        # `live_classical_localization` -- the richer Bayesian posterior
        # over the runtime hypothesis-grid artifact, kept for deterministic
        # reasoning/fusion/operator evidence (classical_vector/
        # classical_belief, trust features, explanation). It must NOT be
        # fed to HydroCore as the model input; see
        # `model_input_prior` immediately below.
        classical = stage(
            "classical_localization",
            lambda: self.localizer(
                observations,
                artifact,
                observation_mask=observation_mask,
                prior=hypothesis_prior,
                feasible_sources=feasible,
                noise_scale=noise_scale,
                hydraulic_graph=graph,
            ),
        )

        def _model_input_classical_prior() -> tuple[Mapping[str, float], str, str]:
            # Local import: see this module's top-of-file circular-import
            # note (hydroswarm.training's package __init__ imports
            # full_trajectory, which imports HybridInferencePipeline from
            # this same module).
            from hydroswarm.training.corpus import (
                model_input_classical_prior,
                resolve_model_input_signature_library,
            )

            junctions = tuple(sorted(network.junction_name_list))
            library, reference_timestamps, mode = resolve_model_input_signature_library(
                topology_hash, junctions, network
            )
            prior = model_input_classical_prior(library, junctions, sensor_series, reference_timestamps)
            return prior, mode, library.manifest_hash

        # core-issues5.txt delta item 1 (P0): the MODEL INPUT
        # `classical_prior` feature HydroCore was actually trained on must
        # reproduce the SAME governed algorithm/distribution Stage-F
        # training tensors were built with
        # (hydroswarm.training.corpus.SignatureLibrary.
        # posterior_from_observations) -- NOT `classical.
        # source_probabilities` above, which is a structurally different
        # Bayesian-posterior algorithm over a different (hypothesis-grid)
        # signature representation. Feeding the richer live localizer's
        # posterior here would be exactly the "silently substitute a
        # different algorithm as the neural model's training prior"
        # train/serve skew scripts/run_train_serve_parity_gate.py's own
        # module docstring documents.
        #
        # `model_input_signature_mode` (GOVERNED_KNOWN_NETWORK vs.
        # RUNTIME_GENERATED_IMPORTED_NETWORK -- see
        # hydroswarm.classical.signature_policy) is computed here but not
        # yet threaded into `DecisionProvenance` (Section 13); flagged as a
        # deferred follow-up rather than expanding that schema
        # speculatively in this fix.
        model_input_prior, _model_input_signature_mode, model_input_signature_hash = stage(
            "model_input_classical_prior", _model_input_classical_prior
        )
        built = stage(
            "feature_building",
            lambda: self.feature_builder.build(
                network,
                graph,
                estimated,
                sensor_series,
                classical_prior=model_input_prior,
            ),
        )
        node_ids = built.node_ids
        # live_classical_localization's own belief vector -- used for
        # deterministic reasoning/fusion/trust/explanation, deliberately
        # NOT the tensor HydroCore consumed above.
        classical_vector = _normalise(
            np.asarray([classical.source_probabilities.get(node, 0.0) for node in node_ids])
        )
        classical_belief = dict(zip(node_ids, map(float, classical_vector), strict=True))

        runtime_mode = HybridRuntimeMode.FULL_HYBRID
        neural_failure: str | None = None
        neural_vector: np.ndarray | None = None
        neural_logits: np.ndarray | None = None
        model_output: Mapping[str, Any] = {}
        semantics = SemanticPredictions()
        try:
            model_output = stage("neural_inference", lambda: self._run_model(built))
            # core-issues5.txt delta item 2 (P0 governance fix): the
            # governed decision is that "source_node" is a normal granular
            # output like every other -- gated by runtime_enabled_outputs
            # exactly like event_presence/plan_value/etc. above, not
            # unconditionally authoritative regardless of governance. This
            # was previously a documented, deliberately-unenforced gap
            # (every v4 identity built so far happened to already exclude
            # "source_node", so the gap was latent, not yet exercised).
            # Phase 14's own evidence
            # (reports/results/v4/phase14-promotion-gates.md: "PASS ...
            # already runtime-enabled (v3 path); re-verify under v4
            # metadata in Phase 15") supports treating it as validated when
            # a checkpoint's own governance says so --
            # scripts/build_phase15_v4_checkpoint.py now includes it in
            # VALIDATED_OUTPUTS/RUNTIME_ENABLED_OUTPUTS accordingly. When a
            # v4 identity explicitly excludes it, source localization falls
            # back to classical-only belief (fused_vector below), the same
            # "no learned signal available" degradation Scout/Strategist
            # already use -- deterministic/classical localization and
            # fail-closed behavior are unaffected either way.
            if self.runtime_enabled_outputs is None or "source_node" in self.runtime_enabled_outputs:
                neural_logits = _array(model_output["source_node_logits"]).reshape(-1)[-len(node_ids):]
                neural_vector = _softmax(neural_logits)
            semantics = self._model_semantics(model_output, node_ids)
            # core-issues.txt repair item 8: Scout (sample_node/
            # information_gain heads) and Strategist (plan_value/
            # plan_validity heads) never receive a real training label (see
            # hydroswarm.tasks) -- their raw outputs must not perturb active
            # sampling or plan ranking until a checkpoint actually declares
            # those tasks trained.
            if "scout" not in self.trained_tasks:
                semantics = replace(semantics, expected_information_gain=None)
            if "strategist" not in self.trained_tasks:
                semantics = replace(semantics, plan_values=(), plan_validity=())
        except Exception as exc:
            latencies.setdefault("neural_inference", 0.0)
            runtime_mode = HybridRuntimeMode.CLASSICAL_SAFE
            neural_failure = type(exc).__name__

        healthy_fraction = sum(item.health[-1] for item in sensor_series) / len(sensor_series)
        missing_rate = sum(float(item.missing[-1]) for item in sensor_series) / len(sensor_series)
        residual_values = list(classical.posterior.residual_rmse.values())
        normalized_residual = min(1.0, min(residual_values, default=noise_scale) / (3 * noise_scale))
        latent = None
        if "latent_state" in model_output:
            latent = _array(model_output["latent_state"]).mean(axis=tuple(range(_array(model_output["latent_state"]).ndim - 1)))
        ood_probabilities = None
        # core-issues.txt repair item 8: the ood_head never receives a real
        # ood_class label (see hydroswarm.tasks); until a checkpoint
        # declares "ood" trained, OODDetector.evaluate falls back to its
        # deterministic energy proxy (1 - max(neural_probabilities)) instead
        # of this untrained head's output.
        if "ood" in self.trained_tasks and "ood_logits" in model_output:
            ood_probabilities = _softmax(_array(model_output["ood_logits"]).reshape(-1)[-3:])
        ood_components, ood_level = stage(
            "ood_detection",
            lambda: self.ood_detector.evaluate(
                node_count=len(node_ids),
                network_hash=network_hash,
                state=estimated,
                sensor_series=sensor_series,
                latent=latent,
                neural_probabilities=neural_vector,
                ood_probabilities=ood_probabilities,
            ),
        )
        trust = TrustFeatures(
            healthy_sensor_fraction=float(np.clip(healthy_fraction, 0.0, 1.0)),
            missing_rate=float(np.clip(missing_rate, 0.0, 1.0)),
            normalized_residual=normalized_residual,
            hydraulic_uncertainty=estimated.normalized_uncertainty,
            neural_entropy=(
                _entropy(neural_vector) / max(1.0, math.log2(len(node_ids)))
                if neural_vector is not None
                else 1.0
            ),
            classical_entropy=_entropy(classical_vector) / max(1.0, math.log2(len(node_ids))),
            ood_score=ood_components.combined,
        )
        diagnostics: FusionDiagnostics | None
        if neural_vector is not None and neural_logits is not None:
            physical_mask = classical_vector > 0
            fused_vector, diagnostics = stage(
                "belief_fusion",
                lambda: fuse_source_probabilities(
                    neural_logits, classical_vector, physical_mask, trust
                ),
            )
        else:
            fused_vector, diagnostics = classical_vector.copy(), None
            latencies["belief_fusion"] = 0.0
        neural_belief = (
            dict(zip(node_ids, map(float, neural_vector), strict=True))
            if neural_vector is not None
            else None
        )
        fused_belief = dict(zip(node_ids, map(float, fused_vector), strict=True))
        trust_rationale = (
            f"classical_trust={diagnostics.classical_trust:.3f}; "
            f"healthy={trust.healthy_sensor_fraction:.3f}; missing={trust.missing_rate:.3f}; "
            f"residual={trust.normalized_residual:.3f}; hydraulic_uncertainty={trust.hydraulic_uncertainty:.3f}"
            if diagnostics is not None
            else "neural inference unavailable; fused belief uses the residual-based classical posterior"
        )

        calibrated = False
        if calibration is not None:
            try:
                calibration.validate_runtime(
                    model_hash=self._model_hash,
                    feature_schema_hash=built.feature_schema_hash,
                    normalization_hash=built.normalization_hash,
                    fusion_config_hash=self.fusion_config_hash,
                    topology_hash=topology_hash,
                )
            except ValueError:
                calibrated = False
            else:
                calibrated = True
        if calibrated and calibration is not None:
            indices = SplitConformalCalibrator(calibration).candidate_set(
                fused_vector,
                network_id=str(getattr(network, "name", "unknown")),
                ood_level=ood_level.value,
            )
            conformal_nodes = tuple(node_ids[index] for index in indices)
        else:
            conformal_nodes = self._credible_nodes(fused_belief)
        model_evidence = (
            semantics.evidence_sufficiency is None
            or semantics.evidence_sufficiency >= self.evidence_threshold
        )
        evidence_sufficient = bool(
            calibrated
            and 0 < len(conformal_nodes) <= self.maximum_planning_candidates
            and model_evidence
        )
        suppression: list[str] = []
        if not calibrated:
            suppression.append("CALIBRATION_INVALID_OR_MISSING")
        if not conformal_nodes or len(conformal_nodes) > self.maximum_planning_candidates:
            suppression.append("CANDIDATE_REGION_TOO_BROAD")
        disagreement = diagnostics.disagreement_js if diagnostics else 0.0
        if disagreement >= self.disagreement_threshold:
            suppression.append("HIGH_CLASSICAL_NEURAL_DISAGREEMENT")
        if ood_level.value != "NORMAL":
            suppression.append(f"OOD_{ood_level.value}")
        if not model_evidence:
            suppression.append("MODEL_EVIDENCE_INSUFFICIENT")
        # A frozen reading is carried separately for provenance and this
        # deterministic authority guard, while the existing sensor-health
        # feature remains the sole HydroCore input.  No plan may be enabled
        # when every latest sensor reading is explicitly frozen.
        if sensor_series and all(item.frozen[-1] for item in sensor_series):
            suppression.append("ALL_SENSORS_FROZEN")
        planning_allowed = evidence_sufficient and not suppression
        candidate_nodes = conformal_nodes or self._credible_nodes(fused_belief)
        if planning_allowed:
            control = ControlAction.GENERATE_PLANS
        else:
            control = uncertainty_control(
                candidate_count=len(candidate_nodes),
                disagreement_js=disagreement,
                ood_score=ood_components.combined,
                healthy_sensor_fraction=trust.healthy_sensor_fraction,
                sample_budget_remaining=sample_budget_remaining,
            )
            control = _fail_closed_control_action(
                control,
                suppression,
                sample_budget_remaining=sample_budget_remaining,
            )

        sample_result: ActiveSamplingResult | None = None
        if control == ControlAction.REQUEST_SAMPLE and sample_budget_remaining > 0:
            hypothesis_weights: dict[str, float] = {}
            for explanation in classical.ranked_hypotheses:
                source = explanation.hypothesis.source_node
                classical_source = max(classical.source_probabilities.get(source, 0.0), 1e-12)
                hypothesis_weights[explanation.hypothesis.identifier] = (
                    explanation.probability * fused_belief.get(source, 0.0) / classical_source
                )
            total = sum(hypothesis_weights.values())
            hypothesis_weights = {key: value / total for key, value in hypothesis_weights.items()}
            constraints = sampling_constraints or SamplingConstraints(
                already_sampled=frozenset(item.node_id for item in sensor_series)
            )
            neural_deltas = semantics.expected_information_gain
            sample_result = stage(
                "active_sampling",
                lambda: self.sampling_ranker(
                    artifact,
                    hypothesis_weights,
                    constraints=constraints,
                    neural_residual_deltas=neural_deltas,
                ),
            )
        else:
            latencies["active_sampling"] = 0.0

        proposals: tuple[PlanProposal, ...] = ()
        if planning_allowed:
            context = self._planning_context(
                incident_id,
                network,
                graph,
                conformal_nodes,
                frozenset(item.node_id for item in sensor_series),
            )
            # core-issues5.txt Section 6 (P0 blocker): PASS 2 candidate
            # scoring needs the real bounded candidate set FIRST (to
            # tensorize and score it), so generation happens twice --
            # once to produce the candidates PASS 2 scores, once more with
            # the resulting deltas actually applied. generate_response_plans
            # is a pure deterministic function of `context`, so the second
            # call reproduces the identical candidate set (same identities,
            # same order) the first call did; only the value/validity
            # baked into each proposal differs. maximum_plans is the real
            # canonical vocabulary size (9, hydroswarm.planning.
            # action_templates.ACTION_TEMPLATE_COUNT) -- previously
            # hardcoded to 8 to match a stale, separately-duplicated
            # template tuple that omitted ALTERNATE_VALVE_CUT entirely.
            baseline_proposals = stage(
                "planning",
                lambda: self.planner(context, maximum_plans=ACTION_TEMPLATE_COUNT),
            )
            value_deltas, validity_deltas = stage(
                "strategist_scoring",
                lambda: self._score_candidate_plans(baseline_proposals, built, graph),
            )
            if value_deltas or validity_deltas:
                proposals = self.planner(
                    context,
                    neural_value_deltas=value_deltas,
                    neural_validity_deltas=validity_deltas,
                    maximum_plans=ACTION_TEMPLATE_COUNT,
                )
            else:
                # Strategist scoring unavailable/disabled/not governed as
                # runtime-enabled: deterministic heuristic ordering only,
                # never a hard failure (required behavior, see
                # _score_candidate_plans's own docstring).
                proposals = baseline_proposals
            proposals = self._deterministic_plans(proposals, incident_id, sensor_series)
        else:
            latencies.setdefault("strategist_scoring", 0.0)
            latencies["planning"] = 0.0

        prior_history = (
            previous_result.posterior_history
            if previous_result is not None
            else self._history.get(incident_id, ())
        )
        prior_evidence = (
            previous_result.evidence_history
            if previous_result is not None
            else self._evidence.get(incident_id, ())
        )
        prior_changes = (
            previous_result.comparison_history
            if previous_result is not None
            else self._changes.get(incident_id, ())
        )
        snapshot = PosteriorSnapshot(
            round_index=len(prior_history),
            observation_count=sum(len(item.timestamps_seconds) for item in sensor_series),
            fused_belief=fused_belief,
            candidate_nodes=candidate_nodes,
            entropy_bits=_entropy(fused_vector),
            evidence_hash=evidence_hash,
        )
        before_after: EvidenceChange | None = None
        if prior_history:
            previous = prior_history[-1]
            previous_set, current_set = set(previous.candidate_nodes), set(candidate_nodes)
            before_after = EvidenceChange(
                from_round=previous.round_index,
                to_round=snapshot.round_index,
                previous_candidates=previous.candidate_nodes,
                current_candidates=candidate_nodes,
                removed_candidates=tuple(sorted(previous_set - current_set)),
                added_candidates=tuple(sorted(current_set - previous_set)),
                candidate_contraction=len(previous.candidate_nodes) - len(candidate_nodes),
                top_probability_change=max(fused_vector) - max(previous.fused_belief.values()),
            )
        posterior_history = (*prior_history, snapshot)
        evidence_snapshot = EvidenceSnapshot(
            round_index=snapshot.round_index,
            observation_count=snapshot.observation_count,
            valid_concentration_count=sum(
                value is not None and not missing
                for series in sensor_series
                for value, missing in zip(series.concentration_mg_l, series.missing, strict=True)
            ),
            sensor_nodes=tuple(sorted(item.node_id for item in sensor_series)),
            evidence_hash=evidence_hash,
        )
        evidence_history = (*prior_evidence, evidence_snapshot)
        comparison_history = (*prior_changes, *((before_after,) if before_after else ()))
        self._history[incident_id] = posterior_history
        self._evidence[incident_id] = evidence_history
        self._changes[incident_id] = comparison_history
        provenance = {
            "network": network_hash,
            "topology": topology_hash,
            "signature_artifact": artifact.artifact_hash,
            # model_input_signature_mode (a SignatureMode string, not a
            # hash) deliberately excluded from this hash-only mapping --
            # provenance_hashes' own contract is "every value is either a
            # 64-hex-char hash or the literal 'none'"
            # (test_hybrid_result_aligns_native_beliefs_and_records_
            # provenance). Surfacing the mode itself belongs to the
            # Decision Authority / Applicability Certificate contract
            # (core-issues5.txt Section 13), not this dict.
            "model_input_signature_hash": model_input_signature_hash,
            "feature_schema": built.feature_schema_hash,
            "normalization": built.normalization_hash,
            "fusion_config": _hash(DYNAMIC_TRUST_FUSION_CONFIG),
            "model": self._model_hash,
            "calibration": calibration.artifact_hash if calibration else "none",
            "evidence": evidence_hash,
        }
        result = IncidentAnalysisResult(
            incident_id=incident_id,
            node_alignment=node_ids,
            classical_belief=classical_belief,
            neural_belief=neural_belief,
            fused_belief=fused_belief,
            classical_localization=classical,
            estimated_hydraulic_state=estimated,
            trust_features=trust,
            fusion_diagnostics=diagnostics,
            trust_rationale=trust_rationale,
            conformal_candidate_nodes=conformal_nodes,
            calibrated=calibrated,
            calibration_alpha=calibration.alpha if calibrated and calibration else None,
            ood_components=ood_components,
            ood_level=ood_level,
            evidence_sufficient=evidence_sufficient,
            planning_allowed=planning_allowed,
            planning_suppression_reasons=tuple(suppression),
            control_action=control,
            sample_result=sample_result,
            plan_proposals=proposals,
            semantic_predictions=semantics,
            posterior_history=posterior_history,
            evidence_history=evidence_history,
            comparison_history=comparison_history,
            before_after=before_after,
            runtime_mode=runtime_mode,
            neural_failure=neural_failure,
            latencies_ms=latencies,
            provenance_hashes=provenance,
            evidence_hash=evidence_hash,
        )
        self._cache[cache_key] = result
        return result

    def reanalyze_after_sample(
        self,
        previous_result: IncidentAnalysisResult,
        network: Any,
        sensor_series: Sequence[SensorSeries],
        **kwargs: Any,
    ) -> IncidentAnalysisResult:
        """Re-run the complete pipeline and attach an explicit before/after belief comparison."""

        return self.analyze(
            previous_result.incident_id,
            network,
            sensor_series,
            previous_result=previous_result,
            **kwargs,
        )


# Concise public alias for application wiring.
HybridPipeline = HybridInferencePipeline
