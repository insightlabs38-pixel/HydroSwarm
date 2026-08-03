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
from hydroswarm.inference.fusion import (
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
from hydroswarm.preprocessing import BuiltHydroBatch, HydraulicFeatureBuilder, SensorSeries
from hydroswarm.sampling import ActiveSamplingResult, SamplingConstraints, rank_sample_locations

from .ood import OODDetector
from .results import (
    EvidenceChange,
    EvidenceSnapshot,
    HybridRuntimeMode,
    IncidentAnalysisResult,
    PosteriorSnapshot,
    SemanticPredictions,
)


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
    ) -> None:
        if maximum_planning_candidates < 1:
            raise ValueError("maximum_planning_candidates must be positive")
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

    @staticmethod
    def _model_semantics(
        output: Mapping[str, Any], node_ids: tuple[str, ...]
    ) -> SemanticPredictions:
        def scalar(key: str) -> float | None:
            value = output.get(key)
            return float(_array(value).reshape(-1)[0]) if value is not None else None

        sample_values = output.get("expected_information_gain")
        faults = output.get("sensor_fault_logits")
        plan_values = output.get("plan_value")
        plan_validity = output.get("plan_validity_logits")
        return SemanticPredictions(
            evidence_sufficiency=scalar("evidence_sufficiency"),
            uncertainty=scalar("uncertainty"),
            expected_information_gain=(
                dict(zip(node_ids, _array(sample_values).reshape(-1)[-len(node_ids):], strict=True))
                if sample_values is not None
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
                if faults is not None
                else None
            ),
            plan_values=tuple(float(value) for value in _array(plan_values).reshape(-1))
            if plan_values is not None
            else (),
            plan_validity=tuple(
                float(_softmax(row)[-1]) for row in _array(plan_validity).reshape(-1, 2)
            )
            if plan_validity is not None
            else (),
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
            model_version="hydrocore-hybrid-v1",
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
        raw_state = stage("hydraulic_simulation", self.simulator.calculate_state)
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
        built = stage(
            "feature_building",
            lambda: self.feature_builder.build(
                network,
                graph,
                estimated,
                sensor_series,
                classical_prior=classical.source_probabilities,
            ),
        )
        node_ids = built.node_ids
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
            neural_logits = _array(model_output["source_node_logits"]).reshape(-1)[-len(node_ids):]
            neural_vector = _softmax(neural_logits)
            semantics = self._model_semantics(model_output, node_ids)
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
        if "ood_logits" in model_output:
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
            templates = (
                "NO_ACTION", "ISOLATE_SOURCE", "FLUSH_DOWNSTREAM", "ISOLATE_AND_FLUSH",
                "PROTECT_CRITICAL", "INCREASE_MONITORING", "REQUEST_SAMPLE", "WAIT_OBSERVE",
            )
            value_deltas = {
                template: math.tanh(value) * 0.1
                for template, value in zip(templates, semantics.plan_values, strict=False)
            }
            validity_deltas = {
                template: (value - 0.5) * 0.2
                for template, value in zip(templates, semantics.plan_validity, strict=False)
            }
            proposals = stage(
                "planning",
                lambda: self.planner(
                    context,
                    neural_value_deltas=value_deltas,
                    neural_validity_deltas=validity_deltas,
                    maximum_plans=8,
                ),
            )
            proposals = self._deterministic_plans(proposals, incident_id, sensor_series)
        else:
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
            "signature_artifact": artifact.artifact_hash,
            "feature_schema": built.feature_schema_hash,
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
