"""Deterministic, WNTR-backed golden incident workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import numpy as np

from hydroswarm.agents import HydroScout, HydroSentinel, HydroStrategist, SwarmController, SwarmLimits
from hydroswarm.domain import ActionType, IncidentState, OperationalAction, OperationalPlan, SensorObservation
from hydroswarm.explanation import EvidenceBundle, ExplanationIntent, deterministic_operational_summary, explain
from hydroswarm.simulation import HydraulicSimulator, IncidentSourceProfile, PlanVerifier, build_wntr_network
from hydroswarm.simulation.consequences import calculate_exposure_consequences
from hydroswarm.storage import SimulationResultCache


SCHEMA_VERSION = "hydroswarm-golden-v1"
INCIDENT_ID = UUID("5bf2cfe9-66cc-5364-b779-976e612a02aa")
TIMESTAMP = datetime(2026, 8, 3, tzinfo=UTC)
CANDIDATES = ("J1", "J2", "J3", "J4")
TRUE_SOURCE = "J2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def freeze_golden_inputs(root: str | Path) -> dict[str, Any]:
    """Write the small authoritative input fixture and checksummed manifest."""

    root = Path(root).resolve()
    frozen = root / "data" / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    network_path = frozen / "golden_network.inp"
    scenario_path = frozen / "golden_scenario.json"

    import wntr

    network = build_wntr_network()
    network.options.time.duration = 4 * 3600
    wntr.network.write_inpfile(network, network_path)
    inp_text = network_path.read_text(encoding="utf-8")
    inp_text = re.sub(
        r"(?m)^; Created: .+$",
        "; Created: 2026-08-03 00:00:00 UTC (frozen fixture)",
        inp_text,
    )
    network_path.write_text(inp_text, encoding="utf-8", newline="\n")
    scenario = {
        "schema_version": SCHEMA_VERSION,
        "incident_id": str(INCIDENT_ID),
        "network_id": "golden-network-v1",
        "true_source": TRUE_SOURCE,
        "candidate_sources": list(CANDIDATES),
        "source_profile": {
            "strength_mg_min": 10.0,
            "start_minute": 0,
            "duration_minutes": 60,
        },
        "initial_sensor": "R1",
        "sample_time_seconds": 3600,
        "contamination_threshold_mg_l": 0.001,
        "energy_price_per_kwh": 0.12,
    }
    scenario_path.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "hydroswarm.evaluation.golden.freeze_golden_inputs",
        "artifacts": {
            "golden_network.inp": {"sha256": _sha256(network_path), "bytes": network_path.stat().st_size},
            "golden_scenario.json": {"sha256": _sha256(scenario_path), "bytes": scenario_path.stat().st_size},
        },
    }
    (frozen / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities.values() if value > 0)


def _credible(probabilities: dict[str, float], target: float = 0.90) -> list[str]:
    selected: list[str] = []
    mass = 0.0
    for node, probability in sorted(probabilities.items(), key=lambda item: (-item[1], item[0])):
        selected.append(node)
        mass += probability
        if mass >= target:
            break
    return selected


def _posterior(predictions: dict[str, float], observed: float, noise_scale: float) -> dict[str, float]:
    likelihoods = {
        node: math.exp(-0.5 * ((observed - predicted) / noise_scale) ** 2)
        for node, predicted in predictions.items()
    }
    total = sum(likelihoods.values())
    return {node: value / total for node, value in likelihoods.items()}


def _active_sample(signatures: dict[str, dict[str, float]], prior: dict[str, float]) -> tuple[str, dict[str, float]]:
    prior_entropy = _entropy(prior)
    scores: dict[str, float] = {}
    for sensor in CANDIDATES:
        positive = {source for source in CANDIDATES if signatures[source][sensor] > 1e-9}
        positive_mass = sum(prior[source] for source in positive)
        negative_mass = 1.0 - positive_mass
        conditional = 0.0
        for group, mass in ((positive, positive_mass), (set(CANDIDATES) - positive, negative_mass)):
            if mass > 0:
                conditional += mass * _entropy({source: prior[source] / mass for source in group})
        scores[sensor] = prior_entropy - conditional
    # Equal information gain is resolved by base demand, then stable node ID.
    demand = {"J1": 0.008, "J2": 0.010, "J3": 0.0065, "J4": 0.009}
    selected = max(CANDIDATES, key=lambda node: (scores[node], demand[node], node))
    return selected, scores


def _plan(name: str, actions: tuple[OperationalAction, ...]) -> OperationalPlan:
    signature = ";".join(f"{item.action_type.value}:{item.target_id}" for item in actions)
    return OperationalPlan(
        plan_id=uuid5(INCIDENT_ID, signature),
        incident_id=INCIDENT_ID,
        name=name,
        actions=actions,
        created_at=TIMESTAMP,
        model_version="golden-deterministic-v1",
    )


class GoldenScenarioRunner:
    def __init__(
        self,
        root: str | Path,
        *,
        seed: int = 2026,
        cache_enabled: bool = True,
        cache_directory: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.seed = seed
        self.cache_enabled = cache_enabled
        self.cache_directory = Path(cache_directory).resolve() if cache_directory else None

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        manifest = freeze_golden_inputs(self.root)
        scenario = json.loads((self.root / "data" / "frozen" / "golden_scenario.json").read_text())
        network_path = self.root / "data" / "frozen" / "golden_network.inp"
        cache = (
            SimulationResultCache(
                self.cache_directory or self.root / "data" / "generated" / "golden-cache"
            )
            if self.cache_enabled
            else None
        )
        network = HydraulicSimulator(network_path).network
        simulator = HydraulicSimulator(network, cache=cache, exact_simulation_budget=32)
        profile_kwargs = scenario["source_profile"]

        signatures: dict[str, dict[str, float]] = {}
        simulations = {}
        for source in CANDIDATES:
            simulation = simulator.simulate_hypothesis(
                IncidentSourceProfile(source_node_id=source, **profile_kwargs),
                include_diagnostics=False,
            )
            simulations[source] = simulation
            signatures[source] = {
                node: float(simulation.concentration_mg_l.loc[scenario["sample_time_seconds"], node])
                for node in CANDIDATES
            }

        initial = {node: 1.0 / len(CANDIDATES) for node in CANDIDATES}
        selected_sample, information_gain = _active_sample(signatures, initial)
        rng = np.random.default_rng(self.seed)
        observed = signatures[TRUE_SOURCE][selected_sample] + float(rng.normal(0.0, 0.25))
        contracted = _posterior(
            {source: signatures[source][selected_sample] for source in CANDIDATES},
            observed,
            noise_scale=20.0,
        )
        before_region = _credible(initial)
        after_region = _credible(contracted)

        unsafe_plan = _plan(
            "Close sole reservoir feeder",
            (OperationalAction(action_type=ActionType.CLOSE_PIPE, target_id="P_R1_J1", duration_minutes=60),),
        )
        safe_plan = _plan(
            "Flush downstream J4",
            (
                OperationalAction(
                    action_type=ActionType.FLUSH_NODE,
                    target_id="J4",
                    flow_rate_lps=3.0,
                    duration_minutes=45,
                ),
            ),
        )
        no_response_plan = _plan("No response", (OperationalAction(action_type=ActionType.END_PLAN),))
        verifier = PlanVerifier(simulator)
        verifications = {
            "unsafe": verifier.verify(unsafe_plan).model_copy(update={"verified_at": TIMESTAMP}),
            "safe": verifier.verify(safe_plan).model_copy(update={"verified_at": TIMESTAMP}),
            "no_response": verifier.verify(no_response_plan).model_copy(update={"verified_at": TIMESTAMP}),
        }

        no_response = simulations[TRUE_SOURCE]
        safe_network = copy.deepcopy(network)
        safe_simulator = HydraulicSimulator(safe_network, cache=cache)
        safe_simulator._apply_actions(safe_simulator.network, safe_plan)
        safe_response = safe_simulator.simulate_hypothesis(
            IncidentSourceProfile(source_node_id=TRUE_SOURCE, **profile_kwargs),
            include_diagnostics=False,
        )
        junctions = list(network.junction_name_list)

        def exposure(simulation, operations: int):
            endpoints = {
                name: (
                    network.get_link(name).start_node_name,
                    network.get_link(name).end_node_name,
                    float(network.get_link(name).length),
                )
                for name in network.pipe_name_list
            }
            return calculate_exposure_consequences(
                simulation.concentration_mg_l[junctions],
                simulation.demand_m3s[junctions],
                threshold_mg_l=scenario["contamination_threshold_mg_l"],
                pipe_endpoints=endpoints,
                pressure_m=simulation.pressure_m[junctions],
                operation_count=operations,
            )

        no_response_metrics = exposure(no_response, 0)
        safe_metrics = exposure(safe_response, 1)
        exposure_reduction = (
            no_response_metrics.contaminant_mass_consumed_mg
            - safe_metrics.contaminant_mass_consumed_mg
        )

        # Run the actual bounded FSM through sample pause, exact verification,
        # approval pause and deterministic replay using the measured beliefs.
        sentinel = HydroSentinel(
            inference=lambda state: {
                "top_candidates": [
                    {"node_id": node, "probability": probability}
                    for node, probability in (
                        contracted.items() if len(state.get("observations", ())) > 1 else initial.items()
                    )
                ],
                "candidate_region": after_region if len(state.get("observations", ())) > 1 else before_region,
                "evidence_sufficient": len(state.get("observations", ())) > 1,
                "uncertainty": min(1.0, _entropy(contracted if len(state.get("observations", ())) > 1 else initial) / 2.0),
            }
        )
        scout = HydroScout(
            inference=lambda _state: {
                "action": "SAMPLE",
                "node_id": selected_sample,
                "expected_information_gain": information_gain[selected_sample],
                "expected_candidate_reduction": (len(before_region) - len(after_region)) / len(before_region),
                "alternatives": [node for node in CANDIDATES if node != selected_sample][:2],
                "reason": "largest measured signature split; demand-centrality tie-break",
            }
        )
        strategist = HydroStrategist(
            inference=lambda _state: {
                "plans": [
                    {"plan": unsafe_plan, "estimated_value": 0.7, "template": "unsafe-feeder-close"},
                    {"plan": safe_plan, "estimated_value": 0.6, "template": "verified-flush"},
                ],
                "revision_round": 0,
            }
        )
        controller = SwarmController(
            sentinel=sentinel,
            scout=scout,
            strategist=strategist,
            verifier=PlanVerifier(HydraulicSimulator(network, cache=cache)),
            limits=SwarmLimits(max_exact_simulations=3, max_incident_runtime_seconds=120),
        )
        initial_observation = SensorObservation(
            sensor_id="S-R1",
            node_id="R1",
            observed_at=TIMESTAMP,
            received_at=TIMESTAMP,
            concentration_mg_l=0.0,
            pressure_m=132.0,
        )
        incident = IncidentState(
            incident_id=INCIDENT_ID,
            network_id="golden-network-v1",
            status="DETECTED",
            observations=(initial_observation,),
        )
        controller.start(network, incident)
        sampling_pause = controller.run()
        sample_observation = SensorObservation(
            sensor_id=f"S-{selected_sample}",
            node_id=selected_sample,
            observed_at=TIMESTAMP,
            received_at=TIMESTAMP,
            concentration_mg_l=max(0.0, observed),
            pressure_m=25.0,
        )
        approval_pause = controller.add_sample(sample_observation)
        replay_at_pause = SwarmController.replay(approval_pause.events)
        completed = controller.approve(approval_pause.selected_plan.plan_id)
        completed_replay = SwarmController.replay(completed.events)

        evidence = EvidenceBundle(
            source_node=max(contracted, key=contracted.get),
            source_probability=max(contracted.values()),
            candidate_region=tuple(after_region),
            candidate_coverage=sum(contracted[node] for node in after_region),
            recommended_sample=selected_sample,
            information_gain_bits=information_gain[selected_sample],
            candidates_before=len(before_region),
            candidates_after=len(after_region),
            selected_plan=safe_plan.name,
            rejected_plan=unsafe_plan.name,
            rejection_codes=verifications["unsafe"].rejection_codes,
            exposure_reduction_mg=exposure_reduction,
            pressure_violation_minutes=verifications["safe"].consequences.pressure_violation_minutes,
            service_availability=verifications["safe"].consequences.service_availability,
            disagreement_js=0.0,
            ood_level="NORMAL",
            approval_pending=True,
            abstention_reason=None,
            supporting_sensors=("S-R1", f"S-{selected_sample}"),
            removed_candidates={node: "inconsistent with active sample" for node in before_region if node not in after_region},
        )
        explanations = {
            intent.value: explain(intent, evidence).text
            for intent in (
                ExplanationIntent.WHY_SAMPLE,
                ExplanationIntent.WHAT_CHANGED,
                ExplanationIntent.WHY_PLAN_REJECTED,
                ExplanationIntent.COMPARE_PLANS,
            )
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "fixture_manifest": manifest,
            "seed": self.seed,
            "network": {"nodes": len(network.node_name_list), "links": len(network.link_name_list)},
            "source": {"true": TRUE_SOURCE, "profile": profile_kwargs},
            "localization": {
                "initial_probabilities": initial,
                "initial_credible_region": before_region,
                "initial_entropy_bits": _entropy(initial),
                "sample_node": selected_sample,
                "sample_observed_mg_l": observed,
                "information_gain_by_node_bits": information_gain,
                "posterior_probabilities": contracted,
                "posterior_credible_region": after_region,
                "posterior_entropy_bits": _entropy(contracted),
                "candidate_contraction": len(before_region) - len(after_region),
            },
            "plans": {
                "generated_count": 3,
                "unsafe": {
                    "plan": unsafe_plan.model_dump(mode="json"),
                    "verification": verifications["unsafe"].model_dump(mode="json"),
                },
                "safe": {
                    "plan": safe_plan.model_dump(mode="json"),
                    "verification": verifications["safe"].model_dump(mode="json"),
                },
                "no_response": {
                    "plan": no_response_plan.model_dump(mode="json"),
                    "verification": verifications["no_response"].model_dump(mode="json"),
                },
            },
            "consequences": {
                "no_response": no_response_metrics.model_dump(mode="json"),
                "safe_alternative": safe_metrics.model_dump(mode="json"),
                "exposure_reduction_mg": exposure_reduction,
            },
            "workflow": {
                "sampling_pause_state": sampling_pause.state.value,
                "approval_pause_state": approval_pause.state.value,
                "selected_plan_id": str(approval_pause.selected_plan.plan_id),
                "pause_replay_state": replay_at_pause.state.value,
                "completed_state": completed.state.value,
                "completed_replay_state": completed_replay.state.value,
                "event_count": len(completed.events),
                "final_event_hash": completed_replay.final_hash,
                "events": [event.model_dump(mode="json") for event in completed.events],
            },
            "explanations": explanations,
            "operational_summary": deterministic_operational_summary(evidence),
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "exact_simulation_runs": simulator.exact_runs,
                "cache_enabled": self.cache_enabled,
            },
        }
        reproducible_payload = {key: value for key, value in result.items() if key != "runtime"}
        result["result_sha256"] = hashlib.sha256(_canonical(reproducible_payload).encode()).hexdigest()
        return result
