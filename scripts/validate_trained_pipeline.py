"""Run the promoted checkpoint through the authoritative hybrid pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.preprocessing import SensorSeries
from hydroswarm.runtime import DefaultPipelineFactory
from hydroswarm.simulation.wrapper import wntr


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    network_path = root / "data" / "frozen" / "golden_network.inp"
    if wntr is None:
        raise SystemExit("WNTR is unavailable")
    network = wntr.network.WaterNetworkModel(str(network_path))
    scenario = WNTRScenarioGenerator().generate(
        network,
        ScenarioGenerationConfig(
            seed=880_100,
            network_id="runtime-validation",
            network_family="runtime-validation",
            split=DatasetSplit.TEST,
            stage=CurriculumStage.OPERATIONAL,
            source_node="J2",
            start_time_bins_min=(60,),
            duration_bins_min=(60,),
            strength_bins=(1.0,),
            demand_regimes=(1.0,),
            sensor_count=4,
            sensor_noise_std=0.0,
            missing_probability=0.0,
            drift_per_hour=0.0,
            pipe_outage_probability=0.0,
            frozen_probability=0.0,
            communication_outage_probability=0.0,
            unit_mismatch_probability=0.0,
        ),
    )
    pressure = 30.0
    series = tuple(
        SensorSeries(
            node_id=node_id,
            timestamps_seconds=tuple(map(float, scenario.timestamps_seconds)),
            concentration_mg_l=tuple(
                float(value) if valid else None
                for value, valid in zip(
                    scenario.observed_concentration[:, column],
                    scenario.observation_mask[:, column],
                    strict=True,
                )
            ),
            pressure_m=tuple(
                pressure if valid else None
                for valid in scenario.observation_mask[:, column]
            ),
            health=tuple(
                1.0 if valid else 0.0
                for valid in scenario.observation_mask[:, column]
            ),
            missing=tuple(map(bool, ~scenario.observation_mask[:, column])),
            drift=tuple(False for _ in scenario.timestamps_seconds),
            delayed=tuple(False for _ in scenario.timestamps_seconds),
        )
        for column, node_id in enumerate(scenario.sensor_nodes)
    )
    factory = DefaultPipelineFactory(root)
    pipeline = factory(None, network_path)
    result = pipeline.analyze(
        UUID("018f4f4e-8d77-7aa1-8000-000000000001"), network, series
    )
    fused_top = max(result.fused_belief, key=result.fused_belief.__getitem__)
    report = {
        "schema_version": 1,
        "expected_source": "J2",
        "fused_top_source": fused_top,
        "source_correct": fused_top == "J2",
        "runtime_mode": result.runtime_mode.value,
        "neural_failure": result.neural_failure,
        "calibrated": result.calibrated,
        "candidate_nodes": list(result.conformal_candidate_nodes),
        "classical_belief": dict(result.classical_belief),
        "neural_belief": dict(result.neural_belief or {}),
        "fused_belief": dict(result.fused_belief),
        "ood_level": result.ood_level.value,
        "control_action": result.control_action.value,
        "latencies_ms": dict(result.latencies_ms),
        "provenance_hashes": dict(result.provenance_hashes),
        "factory_trained_assets_ready": factory.trained_assets_ready,
        "factory_fallback_reason": factory.fallback_reason,
    }
    if result.runtime_mode.value != "FULL_HYBRID" or result.neural_belief is None:
        raise RuntimeError("promoted checkpoint did not execute in the hybrid pipeline")
    output = root / "reports" / "results" / "trained-pipeline-validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
