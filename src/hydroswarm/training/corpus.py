"""Leakage-controlled conversion from WNTR scenarios to HydroCore tensors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from hydroswarm.classical import HydraulicStateEstimator, OperationalTelemetry
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries
from hydroswarm.simulation.wrapper import HydraulicSimulator

from .data import CurriculumStage, ScenarioExample


STAGE_MAP = {
    "clean": CurriculumStage.CLEAN,
    "operational": CurriculumStage.OPERATIONAL,
    "degraded": CurriculumStage.DEGRADED,
    "distribution_shift": CurriculumStage.SHIFT,
    "adversarial": CurriculumStage.ADVERSARIAL,
}


@dataclass(frozen=True, slots=True)
class SignatureLibrary:
    """Training-only concentration templates used by the classical baseline."""

    node_ids: tuple[str, ...]
    signatures: Mapping[str, np.ndarray]
    manifest_hash: str

    def posterior(self, scenario: GeneratedScenario) -> np.ndarray:
        observed, valid = aligned_observations(scenario, self.node_ids)
        transformed = np.log1p(np.nan_to_num(observed, nan=0.0))
        residuals = []
        for node_id in self.node_ids:
            signature = self.signatures[node_id]
            comparable = valid & np.isfinite(signature)
            if not comparable.any():
                residuals.append(1e6)
            else:
                residuals.append(float(np.mean((transformed[comparable] - signature[comparable]) ** 2)))
        values = -np.asarray(residuals, dtype=np.float64)
        values -= values.max()
        probability = np.exp(values / max(float(np.std(values)), 0.05))
        return (probability / probability.sum()).astype(np.float32)


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Reusable canonical hydraulic state and graph for a network family."""

    state: Any
    graph: Any


def build_feature_context(network: Any) -> FeatureContext:
    simulator = HydraulicSimulator(network)
    simulated = simulator.calculate_state(3_600)
    state = HydraulicStateEstimator().estimate(simulated, OperationalTelemetry())
    return FeatureContext(state=state, graph=simulator.build_dynamic_graph(simulated))


def aligned_observations(
    scenario: GeneratedScenario, node_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    steps = len(scenario.timestamps_seconds)
    values = np.full((steps, len(node_ids)), np.nan, dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for source_column, node_id in enumerate(scenario.sensor_nodes):
        target_column = positions[node_id]
        values[:, target_column] = scenario.observed_concentration[:, source_column]
        valid[:, target_column] = scenario.observation_mask[:, source_column]
    return values, valid


def fit_signature_library(
    scenarios: Sequence[GeneratedScenario], node_ids: Sequence[str]
) -> SignatureLibrary:
    """Fit source signatures from training scenarios only."""

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for scenario in scenarios:
        if scenario.manifest.split.value != "train":
            raise ValueError("signature fitting accepts training scenarios only")
        source = scenario.manifest.incident.source_nodes[0]
        observed, _ = aligned_observations(scenario, node_ids)
        grouped[source].append(np.log1p(observed))
    missing = set(node_ids) - set(grouped)
    if missing:
        raise ValueError(f"training corpus has no signatures for sources: {sorted(missing)}")
    signatures = {}
    for node_id in node_ids:
        values = np.stack(grouped[node_id])
        counts = np.isfinite(values).sum(axis=0)
        sums = np.nansum(values, axis=0)
        mean = np.full(sums.shape, np.nan, dtype=np.float64)
        np.divide(sums, counts, out=mean, where=counts > 0)
        signatures[node_id] = mean.astype(np.float32)
    payload = {
        node_id: np.nan_to_num(value, nan=-1.0).round(7).tolist()
        for node_id, value in signatures.items()
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SignatureLibrary(tuple(node_ids), signatures, digest)


def scenario_to_example(
    scenario: GeneratedScenario,
    network: Any,
    signature_library: SignatureLibrary,
    *,
    feature_context: FeatureContext | None = None,
) -> ScenarioExample:
    junction_ids = tuple(sorted(network.junction_name_list))
    if junction_ids != signature_library.node_ids:
        raise ValueError("scenario network nodes do not match the signature library")
    context = feature_context or build_feature_context(network)
    series: list[SensorSeries] = []
    for source_column, node_id in enumerate(scenario.sensor_nodes):
        valid = scenario.observation_mask[:, source_column]
        frozen = scenario.frozen_mask[:, source_column]
        outage = scenario.communication_outage_mask[:, source_column]
        concentration = scenario.observed_concentration[:, source_column]
        pressure = context.state.pressure_m[node_id].estimate
        series.append(
            SensorSeries(
                node_id=node_id,
                timestamps_seconds=tuple(map(float, scenario.timestamps_seconds)),
                concentration_mg_l=tuple(
                    float(value) if is_valid else None
                    for value, is_valid in zip(concentration, valid, strict=True)
                ),
                pressure_m=tuple(pressure if is_valid else None for is_valid in valid),
                health=tuple(
                    0.0 if not is_valid else 0.25 if is_frozen or is_outage else 1.0
                    for is_valid, is_frozen, is_outage in zip(valid, frozen, outage, strict=True)
                ),
                missing=tuple(map(bool, ~valid)),
                drift=tuple(map(bool, frozen)),
                delayed=tuple(map(bool, outage)),
            )
        )
    prior_values = signature_library.posterior(scenario)
    prior = dict(zip(junction_ids, map(float, prior_values), strict=True))
    built = HydraulicFeatureBuilder().build(
        network,
        context.graph,
        context.state,
        series,
        classical_prior=prior,
        window_steps=len(scenario.timestamps_seconds),
    )
    node_ids = built.node_ids
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    source = positions[scenario.manifest.incident.source_nodes[0]]
    start_bins = (0, 60, 120, 240)
    duration_bins = (30, 60, 120)
    strength_bins = (0.5, 1.0, 2.0)
    split = scenario.manifest.split.value
    return ScenarioExample(
        scenario_id=str(scenario.manifest.scenario_id),
        network_id=scenario.manifest.network_id,
        split=split,
        seed=scenario.manifest.seed,
        seed_family=f"{scenario.manifest.network_family}:{scenario.manifest.seed_family}",
        stage=STAGE_MAP[scenario.manifest.stage.value],
        inputs={key: value.squeeze(0) for key, value in built.batch.items()},
        targets={
            "source_node": torch.tensor(source),
            "start_time": torch.tensor(start_bins.index(scenario.manifest.incident.start_minute)),
            "duration": torch.tensor(duration_bins.index(scenario.manifest.incident.duration_minutes)),
            "relative_strength": torch.tensor(strength_bins.index(scenario.manifest.incident.relative_strength)),
            "sensor_fault": torch.tensor([
                float(
                    node_id in scenario.sensor_nodes
                    and (
                        scenario.frozen_mask[:, scenario.sensor_nodes.index(node_id)].any()
                        or scenario.communication_outage_mask[
                            :, scenario.sensor_nodes.index(node_id)
                        ].any()
                    )
                )
                for node_id in node_ids
            ]),
        },
    )


def write_tensor_manifest(
    path: str | Path,
    examples: Sequence[ScenarioExample],
    *,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            record = {
                "scenario_id": example.scenario_id,
                "network_id": example.network_id,
                "split": example.split,
                "seed": example.seed,
                "seed_family": example.seed_family,
                "stage": example.stage.name,
                "inputs": {key: value.tolist() for key, value in example.inputs.items()},
                "targets": {key: value.tolist() for key, value in example.targets.items()},
                "metadata": dict((metadata or {}).get(example.scenario_id, {})),
            }
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            hasher.update(line.encode())
    return hasher.hexdigest()


def signature_metadata(library: SignatureLibrary) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "node_ids": list(library.node_ids),
        "sha256": library.manifest_hash,
        "signatures": {
            node_id: np.nan_to_num(value, nan=-1.0).tolist()
            for node_id, value in library.signatures.items()
        },
    }
