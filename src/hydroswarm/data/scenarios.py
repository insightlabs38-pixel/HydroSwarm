"""Simulator-driven scenario generation, governance, and split integrity."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
import pandas as pd

from hydroswarm.simulation.wrapper import HydraulicSimulator


GENERATOR_VERSION = "hydroswarm-generator-v2"


class CurriculumStage(StrEnum):
    CLEAN = "clean"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    SHIFT = "distribution_shift"
    ADVERSARIAL = "adversarial"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ScenarioGenerationConfig:
    seed: int
    network_id: str
    network_family: str
    split: DatasetSplit | None = None
    stage: CurriculumStage = CurriculumStage.OPERATIONAL
    source_node: str | None = None
    start_time_bins_min: tuple[int, ...] = (0, 60, 120, 240)
    duration_bins_min: tuple[int, ...] = (30, 60, 120)
    strength_bins: tuple[float, ...] = (0.5, 1.0, 2.0)
    demand_regimes: tuple[float, ...] = (0.8, 1.0, 1.2)
    sensor_count: int = 4
    sensor_noise_std: float = 0.01
    missing_probability: float = 0.05
    drift_per_hour: float = 0.001
    frozen_probability: float = 0.02
    communication_outage_probability: float = 0.02
    quantization_step: float = 0.001
    unit_mismatch_probability: float = 0.01
    roughness_variation_fraction: float = 0.05
    tank_level_variation_fraction: float = 0.1
    pipe_outage_probability: float = 0.01
    base_strength_mg_min: float = 10.0


@dataclass(frozen=True, slots=True)
class IncidentTruth:
    source_nodes: tuple[str, ...]
    start_minute: int
    duration_minutes: int
    relative_strength: float
    profile: str
    demand_regime: float


@dataclass(frozen=True, slots=True)
class ScenarioManifest:
    scenario_id: UUID
    split: DatasetSplit
    stage: CurriculumStage
    network_id: str
    network_family: str
    network_sha256: str
    seed: int
    seed_family: int
    simulator_version: str
    generator_version: str
    incident: IncidentTruth
    sensor_nodes: tuple[str, ...]
    model_mismatch: Mapping[str, float | str | bool]
    artifact_sha256: str
    replay_sha256: str
    created_with_python: str


@dataclass(frozen=True, slots=True)
class GeneratedScenario:
    manifest: ScenarioManifest
    timestamps_seconds: np.ndarray
    truth_concentration: np.ndarray
    observed_concentration: np.ndarray
    observation_mask: np.ndarray
    frozen_mask: np.ndarray
    communication_outage_mask: np.ndarray
    timestamp_jitter_seconds: np.ndarray
    sensor_nodes: tuple[str, ...]
    flow_reversal_mask: np.ndarray


class SplitPlanner:
    """Assign ownership before simulation and reserve whole network families for test."""

    def __init__(self, *, held_out_network_families: Sequence[str] = ("C-Town",)) -> None:
        self.held_out = frozenset(held_out_network_families)

    def assign(self, network_family: str, seed: int) -> DatasetSplit:
        if network_family in self.held_out:
            return DatasetSplit.TEST
        seed_family = seed // 100
        bucket = int(hashlib.sha256(f"{network_family}:{seed_family}".encode()).hexdigest()[:8], 16) % 10
        return DatasetSplit.VALIDATION if bucket == 0 else DatasetSplit.TRAIN


def network_sha256(network: Any) -> str:
    payload = {
        "nodes": sorted(network.node_name_list),
        "links": [
            (
                name, network.get_link(name).start_node_name, network.get_link(name).end_node_name,
                float(getattr(network.get_link(name), "length", 0.0)),
                float(getattr(network.get_link(name), "diameter", 0.0)),
                float(getattr(network.get_link(name), "roughness", 0.0)),
            ) for name in sorted(network.link_name_list)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class WNTRScenarioGenerator:
    def __init__(self, split_planner: SplitPlanner | None = None) -> None:
        self.split_planner = split_planner or SplitPlanner()

    def generate(self, network: Any, config: ScenarioGenerationConfig) -> GeneratedScenario:
        rng = np.random.default_rng(config.seed)
        split = config.split or self.split_planner.assign(config.network_family, config.seed)
        if config.network_family in self.split_planner.held_out and split != DatasetSplit.TEST:
            raise ValueError("a held-out network family may only belong to the test split")
        model = copy.deepcopy(network)
        junctions = tuple(sorted(model.junction_name_list))
        if not junctions:
            raise ValueError("scenario network has no junction source candidates")
        source = config.source_node or str(rng.choice(junctions))
        if source not in junctions:
            raise ValueError("configured source must be a junction")
        start = int(rng.choice(config.start_time_bins_min))
        duration = int(rng.choice(config.duration_bins_min))
        strength = float(rng.choice(config.strength_bins))
        demand_regime = float(rng.choice(config.demand_regimes))
        self._randomize_hydraulics(model, config, rng, demand_regime)
        sensor_count = min(max(1, config.sensor_count), len(junctions))
        sensors = tuple(sorted(map(str, rng.choice(junctions, size=sensor_count, replace=False))))
        simulator = HydraulicSimulator(model)
        simulation = simulator.simulate_incident(
            source, strength_mg_min=config.base_strength_mg_min * strength,
            start_minute=start, duration_minutes=duration,
        )
        frame = simulation.concentration_mg_l.loc[:, list(sensors)]
        truth = frame.to_numpy(dtype=np.float32)
        observed, observation_mask, frozen_mask, outage_mask, jitter = self._degrade(
            truth, np.asarray(frame.index, dtype=float), config, rng
        )
        reversals = self._flow_reversals(simulator, np.asarray(frame.index, dtype=int), model.link_name_list)
        replay_payload = {
            "seed": config.seed, "network": network_sha256(network), "source": source,
            "start": start, "duration": duration, "strength": strength,
            "demand": demand_regime, "sensors": sensors, "stage": config.stage.value,
            "split": split.value, "network_family": config.network_family,
        }
        replay_hash = hashlib.sha256(json.dumps(replay_payload, sort_keys=True).encode()).hexdigest()
        scenario_id = uuid5(
            NAMESPACE_URL,
            f"https://hydroswarm.local/scenarios/{replay_hash}",
        )
        array_hash = hashlib.sha256(
            truth.tobytes() + observed.tobytes() + observation_mask.tobytes()
        ).hexdigest()
        mismatch = {
            "roughness_fraction": config.roughness_variation_fraction,
            "demand_multiplier": demand_regime,
            "tank_level_fraction": config.tank_level_variation_fraction,
            "valve_telemetry_incorrect": config.stage in {CurriculumStage.SHIFT, CurriculumStage.ADVERSARIAL},
            "missing_topology_element": config.stage == CurriculumStage.ADVERSARIAL,
        }
        manifest = ScenarioManifest(
            scenario_id=scenario_id, split=split, stage=config.stage, network_id=config.network_id,
            network_family=config.network_family, network_sha256=network_sha256(network), seed=config.seed,
            seed_family=config.seed // 100, simulator_version=simulator.simulator_version,
            generator_version=GENERATOR_VERSION,
            incident=IncidentTruth((source,), start, duration, strength, "pulse", demand_regime),
            sensor_nodes=sensors, model_mismatch=mismatch, artifact_sha256=array_hash,
            replay_sha256=replay_hash, created_with_python=platform.python_version(),
        )
        result = GeneratedScenario(
            manifest, np.asarray(frame.index, dtype=np.int64), truth, observed, observation_mask,
            frozen_mask, outage_mask, jitter, sensors, reversals,
        )
        ScenarioValidator().validate(result)
        return result

    @staticmethod
    def _randomize_hydraulics(model: Any, config: ScenarioGenerationConfig, rng: np.random.Generator, demand_regime: float) -> None:
        for name in model.junction_name_list:
            node = model.get_node(name)
            for demand in node.demand_timeseries_list:
                demand.base_value *= demand_regime
        for name in model.pipe_name_list:
            pipe = model.get_link(name)
            pipe.roughness *= float(rng.uniform(
                1 - config.roughness_variation_fraction, 1 + config.roughness_variation_fraction
            ))
            if rng.random() < config.pipe_outage_probability:
                status_type = type(pipe.initial_status)
                pipe.initial_status = status_type.Closed
        for name in model.tank_name_list:
            tank = model.get_node(name)
            variation = float(rng.uniform(
                1 - config.tank_level_variation_fraction, 1 + config.tank_level_variation_fraction
            ))
            tank.init_level = min(tank.max_level, max(tank.min_level, tank.init_level * variation))

    @staticmethod
    def _degrade(
        truth: np.ndarray, timestamps: np.ndarray, config: ScenarioGenerationConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observed = truth.astype(np.float64, copy=True)
        noise_scale = 0.0 if config.stage == CurriculumStage.CLEAN else config.sensor_noise_std
        observed += rng.normal(0.0, noise_scale, observed.shape)
        hours = (timestamps - timestamps[0]) / 3600.0
        directions = rng.choice([-1.0, 1.0], size=observed.shape[1])
        observed += hours[:, None] * config.drift_per_hour * directions[None, :]
        if config.quantization_step > 0:
            observed = np.round(observed / config.quantization_step) * config.quantization_step
        missing_probability = 0.0 if config.stage == CurriculumStage.CLEAN else config.missing_probability
        mask = rng.random(observed.shape) >= missing_probability
        frozen = np.zeros(observed.shape, dtype=bool)
        for column in range(observed.shape[1]):
            if rng.random() < config.frozen_probability and observed.shape[0] > 2:
                start = int(rng.integers(1, observed.shape[0] - 1))
                observed[start:, column] = observed[start - 1, column]
                frozen[start:, column] = True
        outage = np.zeros(observed.shape, dtype=bool)
        for column in range(observed.shape[1]):
            if rng.random() < config.communication_outage_probability and observed.shape[0] > 2:
                start = int(rng.integers(1, observed.shape[0] - 1))
                stop = min(observed.shape[0], start + max(1, observed.shape[0] // 8))
                mask[start:stop, column] = False
                outage[start:stop, column] = True
        for column in range(observed.shape[1]):
            if rng.random() < config.unit_mismatch_probability:
                observed[:, column] *= 1000.0
        jitter = rng.normal(0.0, 5.0 if config.stage != CurriculumStage.CLEAN else 0.0, observed.shape)
        observed = np.maximum(0.0, observed)
        observed[~mask] = np.nan
        return observed.astype(np.float32), mask, frozen, outage, jitter.astype(np.float32)

    @staticmethod
    def _flow_reversals(simulator: HydraulicSimulator, timestamps: np.ndarray, links: Sequence[str]) -> np.ndarray:
        results = simulator._run_hydraulics(simulator._prepared_network())
        frame = results.link["flowrate"].loc[:, list(links)].reindex(
            index=timestamps.tolist(), method="nearest"
        )
        signs = np.sign(frame.to_numpy(dtype=float))
        reversals = np.zeros_like(signs, dtype=bool)
        reversals[1:] = signs[1:] != signs[:-1]
        return reversals


class ScenarioValidator:
    def validate(self, scenario: GeneratedScenario) -> None:
        shapes = {
            scenario.truth_concentration.shape, scenario.observed_concentration.shape,
            scenario.observation_mask.shape, scenario.frozen_mask.shape,
            scenario.communication_outage_mask.shape,
        }
        if len(shapes) != 1:
            raise ValueError("scenario arrays are not aligned")
        if scenario.truth_concentration.shape[1] != len(scenario.sensor_nodes):
            raise ValueError("sensor layout does not match scenario arrays")
        if np.any(np.diff(scenario.timestamps_seconds) <= 0):
            raise ValueError("scenario time must be strictly increasing")
        if not np.all(np.isfinite(scenario.truth_concentration)) or np.any(scenario.truth_concentration < 0):
            raise ValueError("truth concentration is invalid")
        if not np.all(np.isfinite(scenario.observed_concentration[scenario.observation_mask])):
            raise ValueError("observed finite mask is inconsistent")
        if np.any(np.isfinite(scenario.observed_concentration[~scenario.observation_mask])):
            raise ValueError("missing observations must be NaN")


class ScenarioDatasetWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, scenario: GeneratedScenario) -> Path:
        split_root = self.root / scenario.manifest.split.value
        split_root.mkdir(parents=True, exist_ok=True)
        artifact = split_root / f"{scenario.manifest.scenario_id}.npz"
        np.savez_compressed(
            artifact, timestamps_seconds=scenario.timestamps_seconds,
            truth_concentration=scenario.truth_concentration,
            observed_concentration=scenario.observed_concentration,
            observation_mask=scenario.observation_mask, frozen_mask=scenario.frozen_mask,
            communication_outage_mask=scenario.communication_outage_mask,
            timestamp_jitter_seconds=scenario.timestamp_jitter_seconds,
            flow_reversal_mask=scenario.flow_reversal_mask,
        )
        manifest_dir = self.root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        record = asdict(scenario.manifest)
        record["scenario_id"] = str(scenario.manifest.scenario_id)
        record["split"] = scenario.manifest.split.value
        record["stage"] = scenario.manifest.stage.value
        record["incident"] = asdict(scenario.manifest.incident)
        with (manifest_dir / f"{scenario.manifest.split.value}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        pd.DataFrame({
            "scenario_id": [str(scenario.manifest.scenario_id)], "network_id": [scenario.manifest.network_id],
            "network_family": [scenario.manifest.network_family], "seed": [scenario.manifest.seed],
            "stage": [scenario.manifest.stage.value], "artifact": [artifact.name],
        }).to_parquet(split_root / f"{scenario.manifest.scenario_id}.parquet", index=False)
        return artifact


def load_generated_scenarios(
    root: str | Path, split: DatasetSplit
) -> tuple[GeneratedScenario, ...]:
    """Load governed scenario artifacts with manifest and array validation."""

    dataset_root = Path(root)
    manifest_path = dataset_root / "manifests" / f"{split.value}.jsonl"
    scenarios: list[GeneratedScenario] = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                incident_record = record.pop("incident")
                incident_record["source_nodes"] = tuple(incident_record["source_nodes"])
                incident = IncidentTruth(**incident_record)
                record["scenario_id"] = UUID(record["scenario_id"])
                record["split"] = DatasetSplit(record["split"])
                record["stage"] = CurriculumStage(record["stage"])
                record["sensor_nodes"] = tuple(record["sensor_nodes"])
                record["incident"] = incident
                manifest = ScenarioManifest(**record)
                artifact = dataset_root / split.value / f"{manifest.scenario_id}.npz"
                with np.load(artifact, allow_pickle=False) as arrays:
                    scenario = GeneratedScenario(
                        manifest=manifest,
                        timestamps_seconds=arrays["timestamps_seconds"],
                        truth_concentration=arrays["truth_concentration"],
                        observed_concentration=arrays["observed_concentration"],
                        observation_mask=arrays["observation_mask"],
                        frozen_mask=arrays["frozen_mask"],
                        communication_outage_mask=arrays["communication_outage_mask"],
                        timestamp_jitter_seconds=arrays["timestamp_jitter_seconds"],
                        sensor_nodes=manifest.sensor_nodes,
                        flow_reversal_mask=arrays["flow_reversal_mask"],
                    )
                ScenarioValidator().validate(scenario)
                if hashlib.sha256(
                    scenario.truth_concentration.tobytes()
                    + scenario.observed_concentration.tobytes()
                    + scenario.observation_mask.tobytes()
                ).hexdigest() != manifest.artifact_sha256:
                    raise ValueError("scenario array checksum mismatch")
                scenarios.append(scenario)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid {split.value} scenario manifest line {line_number}: {error}"
                ) from error
    return tuple(scenarios)


def validate_split_integrity(manifests: Sequence[ScenarioManifest]) -> None:
    ownership: dict[tuple[str, int], DatasetSplit] = {}
    source_in_name_leaks = []
    for item in manifests:
        key = (item.network_family, item.seed_family)
        existing = ownership.setdefault(key, item.split)
        if existing != item.split:
            raise ValueError(f"seed-family leakage across splits: {key}")
        if item.incident.source_nodes[0].lower() in str(item.scenario_id).lower():
            source_in_name_leaks.append(str(item.scenario_id))
    if source_in_name_leaks:
        raise ValueError("source identity leaked into scenario filename")


class HardNegativePlanStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self, *, state_hash: str, rejected_plan: Mapping[str, Any], rejection_codes: Sequence[str],
        severity: str, corrected_plan: Mapping[str, Any] | None,
    ) -> None:
        record = {
            "state_hash": state_hash, "rejected_plan": rejected_plan,
            "rejection_codes": list(rejection_codes), "severity": severity,
            "corrected_plan": corrected_plan,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
