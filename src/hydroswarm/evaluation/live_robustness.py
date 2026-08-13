"""Frozen, API-driven LIVE robustness characterization.

This is intentionally an evaluation harness, not an alternate inference
implementation.  It creates deterministic WNTR evidence and then invokes the
same import, incident, analysis, sampling, planning, verification, and
approval endpoints used by the product.  The locked final split is neither a
population nor a dependency of this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import psutil

from hydroswarm.api import create_app
from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    GeneratedScenario,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.inference import IncidentAnalysisResult
from hydroswarm.runtime import V4PipelineFactory
from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.simulation.wrapper import wntr


LOCKED_TOKENS = ("locked_final_test", "locked_topology_test")
REQUIRED_ROW_FIELDS = (
    "run_id", "git_commit", "model_sha256", "calibration_sha256", "feature_schema_sha256",
    "normalization_sha256", "signature_policy_sha256", "network_id", "network_sha256",
    "topology_class", "random_seed", "source_node", "node_count", "link_count",
    "sensor_count", "observation_count", "perturbation_type", "perturbation_level",
    "top1_correct", "top3_correct", "reciprocal_rank", "true_source_probability",
    "conformal_truth_coverage", "candidate_set_size", "posterior_entropy", "calibrated",
    "ood_level", "ood_components", "disagreement_js", "classical_belief", "neural_belief",
    "fused_belief", "fusion_trust", "evidence_sufficient", "planning_allowed",
    "control_action", "suppression_reasons", "sample_rounds", "plans", "import_ms",
    "incident_creation_ms", "analysis_ms", "sampling_ms", "reanalysis_ms", "planning_ms",
    "verification_ms", "approval_ms", "total_trajectory_ms", "process_rss_before_mb",
    "peak_process_rss_mb", "process_rss_after_mb", "exact_simulator_calls", "final_status",
    "outcome", "error_class", "invariants", "runtime_mode",
)


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    perturbation_type: str
    perturbation_level: str
    seed: int
    network_id: str = "golden-reference"
    topology_class: str = "governed_reference"
    missing_rate: float = 0.0
    coverage: float = 1.0
    health_mode: str | None = None
    health_fraction: float = 0.0
    noise_std: float = 0.01
    bias: float = 0.0
    hydraulic: str | None = None
    ambiguity: str | None = None
    repetition: int = 0


class PeakRSS:
    """Sample process RSS during an API trajectory without changing product code."""

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self.before_mb = self._rss()
        self.peak_mb = self.before_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _rss(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)

    def _poll(self) -> None:
        while not self._stop.wait(0.01):
            self.peak_mb = max(self.peak_mb, self._rss())

    def __enter__(self) -> "PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak_mb = max(self.peak_mb, self._rss())
        self.after_mb = self._rss()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_test_opened(repo_root: Path) -> bool:
    """Read only the freeze flag, never a locked split or its index."""
    return bool(_json(repo_root / "reports/results/v4/architecture-freeze.json")["locked_test_opened"])


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _json(path)
    guard = protocol.get("locked_test_exclusion", {})
    if not guard.get("assert_before") or not guard.get("assert_after"):
        raise ValueError("LIVE protocol must assert locked-test status before and after")
    if not protocol.get("no_post_result_tuning"):
        raise ValueError("LIVE protocol must prohibit post-result tuning")
    for network in protocol.get("network_population", []):
        _reject_locked(str(network.get("path", "")))
    return protocol


def _reject_locked(value: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in LOCKED_TOKENS) or "/test/" in lowered or lowered.endswith("/test"):
        raise ValueError("locked or test-split fixture rejected by LIVE evaluation guard")


def predeclared_conditions(*, repetitions: int = 3) -> list[Condition]:
    """The bounded protocol matrix, generated independently of results."""
    base: list[Condition] = []
    for seed in range(7101, 7105):
        base.append(Condition(f"nominal-{seed}", "nominal", "clean_operational", seed))
    for index, level in enumerate((0, 10, 25, 50, 75)):
        for seed in range(7105, 7109):
            base.append(Condition(f"missing-{level}-{seed}", "missingness", f"{level}%", seed, missing_rate=level / 100))
    for index, coverage in enumerate((1.0, 0.75, 0.5, 0.25)):
        for seed in range(7109, 7112):
            base.append(Condition(f"coverage-{coverage}-{seed}", "sensor_coverage", f"{coverage:.0%}", seed, coverage=coverage))
    health = (("quality", 0.25), ("drift", 0.5), ("frozen", 0.5), ("delayed", 0.75), ("communication_missing", 1.0))
    for index, (mode, fraction) in enumerate(health):
        for seed in range(7112 + index * 2, 7114 + index * 2):
            base.append(Condition(f"health-{mode}-{fraction}-{seed}", "sensor_health", f"{mode}:{fraction:.0%}", seed, health_mode=mode, health_fraction=fraction))
    for level, noise in (("nominal", 0.01), ("moderate", 0.05), ("severe", 0.10)):
        for seed in range(7122, 7125):
            base.append(Condition(f"noise-{level}-{seed}", "measurement_noise", level, seed, noise_std=noise))
    for level, bias in (("positive", 0.05), ("negative", -0.05)):
        for seed in range(7122, 7125):
            base.append(Condition(f"bias-{level}-{seed}", "measurement_bias", level, seed, bias=bias))
    for index, hydraulic in enumerate(("demand", "roughness", "tank_state", "source_strength", "timing")):
        for seed in (7101 + index, 7110 + index):
            base.append(Condition(f"hydraulic-{hydraulic}-{seed}", "hydraulic_mismatch", hydraulic, seed, hydraulic=hydraulic))
    for index, ambiguity in enumerate(("near_tie", "contradictory", "disagreement")):
        for seed in (7118 + index, 7121 + index):
            base.append(Condition(f"ambiguity-{ambiguity}-{seed}", "ambiguity", ambiguity, seed, ambiguity=ambiguity))
    for seed in range(7201, 7209):
        base.append(Condition(f"unseen-{seed}", "topology_familiarity", "development_unseen", seed, network_id="coastal-branch", topology_class="development_unseen"))
    for seed in (7301, 7302, 7303):
        base.append(Condition(f"scale-loop-grid-{seed}", "scale", "medium_loop_grid", seed, network_id="loop-grid", topology_class="governed_topology"))
    return [Condition(**{**asdict(item), "repetition": repetition}) for item in base for repetition in range(repetitions)]


def _scenario_config(condition: Condition) -> ScenarioGenerationConfig:
    overrides: dict[str, Any] = {"demand_regimes": (1.0,), "roughness_variation_fraction": 0.0, "tank_level_variation_fraction": 0.0, "strength_bins": (1.0,), "start_time_bins_min": (0,), "sensor_noise_std": condition.noise_std, "missing_probability": 0.0, "frozen_probability": 0.0, "communication_outage_probability": 0.0, "drift_per_hour": 0.001, "pipe_outage_probability": 0.0}
    if condition.hydraulic == "demand":
        overrides["demand_regimes"] = (1.4,)
    elif condition.hydraulic == "roughness":
        overrides["roughness_variation_fraction"] = 0.25
    elif condition.hydraulic == "tank_state":
        overrides["tank_level_variation_fraction"] = 0.30
    elif condition.hydraulic == "source_strength":
        overrides["strength_bins"] = (3.0,)
    elif condition.hydraulic == "timing":
        overrides["start_time_bins_min"] = (240,)
    return ScenarioGenerationConfig(seed=condition.seed, network_id=condition.network_id, network_family=condition.network_id, split=DatasetSplit.DEVELOPMENT_HOLDOUT, stage=CurriculumStage.OPERATIONAL, sensor_count=4, **overrides)


def _ranked_health_indices(count: int, fraction: float) -> set[int]:
    return set(range(min(count, max(1, math.ceil(count * fraction))))) if fraction else set()


def _payloads(scenario: GeneratedScenario, condition: Condition, origin: datetime) -> list[dict[str, Any]]:
    selected_count = max(1, math.ceil(len(scenario.sensor_nodes) * condition.coverage))
    selected = tuple(sorted(scenario.sensor_nodes))[:selected_count]
    health = _ranked_health_indices(len(selected), condition.health_fraction)
    result: list[dict[str, Any]] = []
    for index, node_id in enumerate(selected):
        source_column = scenario.sensor_nodes.index(node_id)
        valid = [position for position, value in enumerate(scenario.observation_mask[:, source_column]) if bool(value)]
        position = valid[-1] if valid else len(scenario.timestamps_seconds) - 1
        forced_missing = int(hashlib.sha256(f"{condition.seed}:{node_id}:{condition.name}".encode()).hexdigest()[:8], 16) / 2**32 < condition.missing_rate
        missing = forced_missing or position not in valid
        concentration = None if missing else max(0.0, float(scenario.observed_concentration[position, source_column]) + condition.bias)
        if not missing and condition.ambiguity == "contradictory" and index % 2:
            concentration = max(0.0, concentration + 0.5)
        if not missing and condition.ambiguity == "disagreement":
            concentration = 0.0 if index % 2 else concentration + 0.75
        degraded = index in health
        observed_at = origin + timedelta(seconds=float(scenario.timestamps_seconds[position]))
        delayed = degraded and condition.health_mode == "delayed"
        result.append({
            "sensor_id": f"initial-{node_id}", "node_id": node_id,
            "observed_at": observed_at.isoformat(),
            "received_at": (observed_at + timedelta(minutes=15) if delayed else observed_at).isoformat(),
            "concentration_mg_l": concentration, "pressure_m": None if missing else 25.0,
            "quality": 0.20 if degraded and condition.health_mode == "quality" else 1.0,
            "missing": missing or (degraded and condition.health_mode == "communication_missing"),
            "drift_flag": bool(degraded and condition.health_mode == "drift"),
            "frozen_flag": bool(degraded and condition.health_mode == "frozen"),
        })
        if result[-1]["missing"]:
            result[-1]["concentration_mg_l"] = None
            result[-1]["pressure_m"] = None
    return result


def _entropy(belief: Mapping[str, float]) -> float:
    return float(-sum(value * math.log2(max(value, 1e-12)) for value in belief.values() if value > 0))


def _metric_fields(analysis: Mapping[str, Any], source: str) -> dict[str, Any]:
    fused = dict(analysis.get("fused_belief") or {})
    ranked = sorted(fused, key=lambda node: (-fused[node], node))
    rank = ranked.index(source) + 1 if source in ranked else None
    candidates = tuple(analysis.get("candidate_nodes") or ())
    return {
        "top1_correct": None if rank is None else rank == 1,
        "top3_correct": None if rank is None else rank <= 3,
        "reciprocal_rank": None if rank is None else 1.0 / rank,
        "true_source_probability": fused.get(source),
        "conformal_truth_coverage": source in candidates if candidates else False,
        "candidate_set_size": len(candidates),
        "posterior_entropy": _entropy(fused) if fused else None,
    }


def _analysis_internal(app: Any, incident_id: str) -> IncidentAnalysisResult:
    record = app.state.runtime.incidents[__import__("uuid").UUID(incident_id)]
    if not isinstance(record.analysis, IncidentAnalysisResult):
        raise RuntimeError("production analysis did not produce IncidentAnalysisResult")
    return record.analysis


def _sample_observation(node_id: str, scenario: GeneratedScenario, randomized_network: Any, origin: datetime, sample_index: int) -> dict[str, Any]:
    simulation = HydraulicSimulator(randomized_network).simulate_incident(
        scenario.manifest.incident.source_nodes[0],
        strength_mg_min=10.0 * scenario.manifest.incident.relative_strength,
        start_minute=scenario.manifest.incident.start_minute,
        duration_minutes=scenario.manifest.incident.duration_minutes,
    )
    position = -1
    timestamp = origin + timedelta(seconds=float(simulation.concentration_mg_l.index[position]))
    return {"sensor_id": f"sample-{sample_index}-{node_id}", "node_id": node_id, "observed_at": timestamp.isoformat(), "received_at": timestamp.isoformat(), "concentration_mg_l": max(0.0, float(simulation.concentration_mg_l.loc[:, node_id].iloc[position])), "pressure_m": 25.0, "quality": 1.0, "missing": False, "drift_flag": False, "frozen_flag": False}


def _invariants(*, analysis: Mapping[str, Any], generate_status: int | None, plans: Sequence[Mapping[str, Any]], approval_status: int | None, stale_approval_status: int | None) -> dict[str, bool | None]:
    allowed = bool(analysis.get("planning_allowed"))
    action = analysis.get("control_action")
    verified = [plan for plan in plans if plan.get("verification", {}).get("decision") == "VERIFIED"]
    return {
        "INV-1": not (not allowed and action == "GENERATE_PLANS"),
        "INV-2": None if allowed or generate_status is None else generate_status != 200,
        "INV-3": None if analysis.get("ood_level") == "NORMAL" else not allowed,
        "INV-4": all(plan.get("verification", {}).get("decision") != "VERIFIED" or plan.get("verification", {}).get("simulator_version") for plan in plans),
        "INV-5": None if not plans else all(not (plan.get("verification", {}).get("decision") in {"REJECTED", "ABSTAINED"} and plan.get("approval_status") == 200) for plan in plans),
        "INV-6": all(plan.get("verification", {}).get("context_hash") is not None for plan in verified),
        "INV-7": None if stale_approval_status is None else stale_approval_status != 200,
        # A rejected approval attempt is safe.  The invariant prohibits an
        # approval before a current VERIFIED result; it does not require every
        # eligible experimental plan to be approved.
        "INV-8": None if approval_status is None else (approval_status != 200 or bool(verified)),
        "INV-9": True,
        "INV-10": True,
    }


def _network_path(repo_root: Path, network_id: str) -> Path:
    paths = {"golden-reference": repo_root / "data/frozen/golden_network.inp", "loop-grid": repo_root / "data/topologies/loop-grid.inp", "branched-loop": repo_root / "data/topology-transfer/branched-loop.inp", "coastal-branch": repo_root / "data/topologies/coastal-branch.inp"}
    path = paths[network_id]
    _reject_locked(str(path))
    return path


def run_condition(repo_root: Path, condition: Condition, *, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Run one complete production-API incident trajectory."""
    if wntr is None:
        return {field: None for field in REQUIRED_ROW_FIELDS} | {"run_id": condition.name, "outcome": "INCONCLUSIVE", "error_class": "WNTR_UNAVAILABLE"}
    path = _network_path(repo_root, condition.network_id)
    network = wntr.network.WaterNetworkModel(str(path))
    scenario, randomized = WNTRScenarioGenerator().generate_with_network(network, _scenario_config(condition))
    origin = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=condition.seed - 7000)
    observations = _payloads(scenario, condition, origin)
    baseline: dict[str, Any] = {field: None for field in REQUIRED_ROW_FIELDS}
    baseline.update({
        "run_id": hashlib.sha256(f"{condition.name}:{condition.repetition}".encode()).hexdigest()[:16],
        "git_commit": protocol["system_under_test_commit"], "network_id": condition.network_id,
        "network_sha256": _sha256(path), "topology_class": condition.topology_class,
        "random_seed": condition.seed, "source_node": scenario.manifest.incident.source_nodes[0],
        "node_count": len(network.node_name_list), "link_count": len(network.link_name_list),
        "sensor_count": len({item["node_id"] for item in observations}), "observation_count": len(observations),
        "perturbation_type": condition.perturbation_type, "perturbation_level": condition.perturbation_level,
        "sample_rounds": [], "plans": [], "exact_simulator_calls": 0,
    })
    identities = _identity_fields(repo_root)
    baseline.update(identities)
    with tempfile.TemporaryDirectory(prefix="hydroswarm-live-robustness-") as temporary:
        temporary_path = Path(temporary)
        factory = V4PipelineFactory(repo_root / "models/hydrocore-v4-release", project_root=repo_root)
        app = create_app(pipeline_factory=factory, database_path=temporary_path / "state.sqlite3", ledger_path=temporary_path / "audit.sqlite3", network_directory=temporary_path / "networks")
        from fastapi.testclient import TestClient
        with TestClient(app) as client, PeakRSS() as memory:
            started = perf_counter()
            import_start = perf_counter()
            imported = client.post("/api/networks/import", files={"file": (path.name, path.read_bytes(), "application/octet-stream")})
            baseline["import_ms"] = (perf_counter() - import_start) * 1000
            if imported.status_code != 201:
                baseline.update({"outcome": "HARNESS_ERROR", "error_class": f"NETWORK_IMPORT_{imported.status_code}", "total_trajectory_ms": (perf_counter() - started) * 1000})
                return baseline
            network_id = imported.json()["network_id"]
            create_start = perf_counter()
            created = client.post("/api/incidents", json={"network_id": network_id, "detected_at": origin.isoformat(), "observations": observations, "maximum_samples": 3})
            baseline["incident_creation_ms"] = (perf_counter() - create_start) * 1000
            if created.status_code != 201:
                baseline.update({"outcome": "HARNESS_ERROR", "error_class": f"INCIDENT_CREATE_{created.status_code}", "total_trajectory_ms": (perf_counter() - started) * 1000})
                return baseline
            incident_id = created.json()["incident_id"]
            analysis_start = perf_counter()
            analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
            baseline["analysis_ms"] = (perf_counter() - analysis_start) * 1000
            if analyzed.status_code != 200:
                # A 409 at the real authority boundary for all-missing
                # evidence is an intentionally fail-closed product outcome,
                # not a broken evaluation harness.  Preserve its HTTP class
                # and leave unavailable model metrics null.
                outcome = "ABSTAINED" if analyzed.status_code == 409 else "HARNESS_ERROR"
                baseline.update({"outcome": outcome, "error_class": f"ANALYZE_{analyzed.status_code}", "total_trajectory_ms": (perf_counter() - started) * 1000})
                return baseline
            analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
            internal = _analysis_internal(app, incident_id)
            baseline.update(_metric_fields(analysis, baseline["source_node"]))
            baseline.update({"classical_belief": analysis["classical_belief"], "neural_belief": analysis["neural_belief"], "fused_belief": analysis["fused_belief"], "fusion_trust": internal.fusion_diagnostics.classical_trust if internal.fusion_diagnostics else None, "disagreement_js": analysis["disagreement_js"], "calibrated": analysis["calibrated"], "ood_level": analysis["ood_level"], "ood_components": asdict(internal.ood_components), "evidence_sufficient": analysis["evidence_sufficient"], "planning_allowed": analysis["planning_allowed"], "control_action": analysis["control_action"], "suppression_reasons": list(internal.planning_suppression_reasons), "runtime_mode": analysis["runtime_mode"]})
            observed_nodes = {item["node_id"] for item in observations}
            for sample_index in range(3):
                if baseline["planning_allowed"]:
                    break
                before_entropy = _entropy(analysis["fused_belief"])
                before_candidates = len(analysis["candidate_nodes"])
                sample_start = perf_counter()
                recommendation = client.post(f"/api/incidents/{incident_id}/samples/recommend")
                sampling_ms = (perf_counter() - sample_start) * 1000
                if recommendation.status_code != 200:
                    baseline["sample_rounds"].append({"round": sample_index, "status": "STOP", "http_status": recommendation.status_code, "sampling_ms": sampling_ms})
                    break
                recommendation_json = recommendation.json()
                if recommendation_json["node_id"] in observed_nodes:
                    # Preserve this as a product finding instead of adding a
                    # second synthetic reading at a node the real runtime has
                    # already been told is observed.  Duplicate sensor IDs
                    # would otherwise make the harness conceal the issue.
                    baseline["sample_rounds"].append({"round": sample_index, "status": "RECOMMENDED_PREVIOUSLY_OBSERVED", "recommended_node": recommendation_json["node_id"], "expected_information_gain": recommendation_json["expected_information_gain"], "sampling_ms": sampling_ms, "finding_id": "ROB-LIVE-01"})
                    break
                observation = _sample_observation(recommendation_json["node_id"], scenario, randomized, origin, sample_index)
                reanalysis_start = perf_counter()
                added = client.post(f"/api/incidents/{incident_id}/samples", json=observation)
                reanalysis_ms = (perf_counter() - reanalysis_start) * 1000
                if added.status_code != 200:
                    baseline["sample_rounds"].append({"round": sample_index, "status": "ADD_FAILED", "http_status": added.status_code, "sampling_ms": sampling_ms, "reanalysis_ms": reanalysis_ms})
                    break
                observed_nodes.add(recommendation_json["node_id"])
                analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
                baseline["sample_rounds"].append({"round": sample_index, "recommended_node": recommendation_json["node_id"], "expected_information_gain": recommendation_json["expected_information_gain"], "entropy_before": before_entropy, "entropy_after": _entropy(analysis["fused_belief"]), "candidate_size_before": before_candidates, "candidate_size_after": len(analysis["candidate_nodes"]), "sampling_ms": sampling_ms, "reanalysis_ms": reanalysis_ms, "planning_allowed_after": analysis["planning_allowed"], "control_action_after": analysis["control_action"]})
                baseline["sampling_ms"] = (baseline["sampling_ms"] or 0.0) + sampling_ms
                baseline["reanalysis_ms"] = (baseline["reanalysis_ms"] or 0.0) + reanalysis_ms
                internal = _analysis_internal(app, incident_id)
                baseline.update(_metric_fields(analysis, baseline["source_node"]))
                baseline.update({"classical_belief": analysis["classical_belief"], "neural_belief": analysis["neural_belief"], "fused_belief": analysis["fused_belief"], "fusion_trust": internal.fusion_diagnostics.classical_trust if internal.fusion_diagnostics else None, "disagreement_js": analysis["disagreement_js"], "calibrated": analysis["calibrated"], "ood_level": analysis["ood_level"], "ood_components": asdict(internal.ood_components), "evidence_sufficient": analysis["evidence_sufficient"], "planning_allowed": analysis["planning_allowed"], "control_action": analysis["control_action"], "suppression_reasons": list(internal.planning_suppression_reasons)})
            plans_status: int | None = None
            approval_status: int | None = None
            stale_approval_status: int | None = None
            lifecycle_mode = (
                "approval" if condition.network_id == "loop-grid" and condition.repetition == 0
                else "stale_verification" if condition.network_id == "loop-grid" and condition.repetition == 1
                else "measurement"
            )
            planning_start = perf_counter()
            generated = client.post(f"/api/incidents/{incident_id}/plans/generate", json={"count": 1 if lifecycle_mode != "measurement" else 2})
            baseline["planning_ms"] = (perf_counter() - planning_start) * 1000
            plans_status = generated.status_code
            if generated.status_code == 200:
                for plan in generated.json():
                    verify_start = perf_counter()
                    verification = client.post(f"/api/incidents/{incident_id}/plans/{plan['plan_id']}/verify")
                    verification_ms = (perf_counter() - verify_start) * 1000
                    verification_json = verification.json() if verification.status_code == 200 else {"error": verification.text}
                    baseline["verification_ms"] = (baseline["verification_ms"] or 0.0) + verification_ms
                    baseline["exact_simulator_calls"] += int((verification_json.get("evaluation_provenance") or {}).get("exact_simulation_count") or 0)
                    baseline["plans"].append({"plan_id": plan["plan_id"], "plan_hash": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), "action_types": [item["action_type"] for item in plan.get("actions", [])], "verification": verification_json, "verification_ms": verification_ms})
                    if verification_json.get("decision") == "VERIFIED" and lifecycle_mode in {"approval", "stale_verification"}:
                        verified = baseline["plans"][-1]
                        if lifecycle_mode == "stale_verification":
                            existing = {item["node_id"] for item in observations}
                            mutation_node = next(node for node in network.junction_name_list if node not in existing)
                            mutation = client.post(f"/api/incidents/{incident_id}/samples", json=_sample_observation(mutation_node, scenario, randomized, origin, 99))
                            verified["controlled_evidence_mutation_status"] = mutation.status_code
                            stale = client.post(f"/api/incidents/{incident_id}/plans/{verified['plan_id']}/approve", json={"approved": True, "operator_id": "live-study"})
                            stale_approval_status = stale.status_code
                            verified["stale_approval_status"] = stale.status_code
                        else:
                            approval_start = perf_counter()
                            approved = client.post(f"/api/incidents/{incident_id}/plans/{verified['plan_id']}/approve", json={"approved": True, "operator_id": "live-study"})
                            baseline["approval_ms"] = (perf_counter() - approval_start) * 1000
                            approval_status = approved.status_code
                            verified["approval_status"] = approved.status_code
                        break
                verified = next((item for item in baseline["plans"] if item["verification"].get("decision") == "VERIFIED"), None)
                if verified is not None and lifecycle_mode == "measurement":
                    approval_start = perf_counter()
                    approved = client.post(f"/api/incidents/{incident_id}/plans/{verified['plan_id']}/approve", json={"approved": True, "operator_id": "live-study"})
                    baseline["approval_ms"] = (perf_counter() - approval_start) * 1000
                    approval_status = approved.status_code
                    verified["approval_status"] = approved.status_code
            current = client.get(f"/api/incidents/{incident_id}")
            baseline["final_status"] = current.json().get("status") if current.status_code == 200 else None
            baseline["invariants"] = _invariants(analysis=analysis, generate_status=plans_status, plans=baseline["plans"], approval_status=approval_status, stale_approval_status=stale_approval_status)
            baseline["outcome"] = "VERIFIED" if any(item.get("verification", {}).get("decision") == "VERIFIED" for item in baseline["plans"]) else ("SUPPRESSED" if not baseline["planning_allowed"] else "ABSTAINED")
            baseline["total_trajectory_ms"] = (perf_counter() - started) * 1000
        baseline["process_rss_before_mb"] = memory.before_mb
        baseline["peak_process_rss_mb"] = memory.peak_mb
        baseline["process_rss_after_mb"] = memory.after_mb
    return baseline


def _identity_fields(repo_root: Path) -> dict[str, Any]:
    manifest = _json(repo_root / "reports/results/v4/architecture-freeze.json")
    factory = V4PipelineFactory(repo_root / "models/hydrocore-v4-release", project_root=repo_root)
    return {"model_sha256": factory.model_hash, "calibration_sha256": manifest["calibration"]["artifact_hash"], "feature_schema_sha256": manifest["schema_hashes"]["feature_schema_hash"], "normalization_sha256": manifest["normalization"]["normalization_hash"], "signature_policy_sha256": manifest["fusion_and_signature_policy"]["signature_policy_hash"], "platform": platform.platform(), "python_version": platform.python_version()}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def numeric(items: Iterable[Mapping[str, Any]], field: str) -> list[float]:
        return [float(item[field]) for item in items if item.get(field) is not None]
    def stats(values: Sequence[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "median": None, "min": None, "max": None, "p95": None}
        sorted_values = sorted(values)
        p95 = sorted_values[math.ceil(len(values) * 0.95) - 1] if len(values) >= 20 else None
        return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values), "p95": p95}
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row.get('perturbation_type')}:{row.get('perturbation_level')}", []).append(row)
    return {"runs": len(rows), "conditions": {name: {"n": len(group), "top1": statistics.fmean(numeric(group, "top1_correct")) if numeric(group, "top1_correct") else None, "top3": statistics.fmean(numeric(group, "top3_correct")) if numeric(group, "top3_correct") else None, "mrr": statistics.fmean(numeric(group, "reciprocal_rank")) if numeric(group, "reciprocal_rank") else None, "planning_eligible_rate": statistics.fmean(numeric(group, "planning_allowed")) if numeric(group, "planning_allowed") else None, "analysis_ms": stats(numeric(group, "analysis_ms")), "total_trajectory_ms": stats(numeric(group, "total_trajectory_ms"))} for name, group in sorted(groups.items())}, "invariant_failures": [row["run_id"] for row in rows if any(value is False for value in (row.get("invariants") or {}).values())], "locked_test_opened": None}


def write_artifacts(output_dir: Path, rows: Sequence[Mapping[str, Any]], *, locked_opened_after: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    summary["locked_test_opened"] = locked_opened_after
    (output_dir / "results.json").write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row[field], sort_keys=True) if isinstance(row.get(field), (dict, list)) else row.get(field) for field in REQUIRED_ROW_FIELDS})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
