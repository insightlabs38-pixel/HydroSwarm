"""Causal-prefix capability evaluation for the remediation branch.

This evaluates the production ``HybridInferencePipeline`` directly with the
same ``SensorSeries`` contract that the API constructs.  Each checkpoint only
contains reports observed by that report step; it never supplies the later
incident trajectory.  ``latest_only`` is retained solely as the documented
negative baseline.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import wntr  # noqa: E402

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.builder import SensorSeries  # noqa: E402
from hydroswarm.runtime import V4PipelineFactory  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402


CHECKPOINTS = (1, 2, 3, 6, 12, 25)
SEEDS = tuple(range(2026081401, 2026081421))
OUT = ROOT / "reports/evaluation/capability-remediation/temporal-capability.json"


def _prefix(series: SensorSeries, count: int, *, latest_only: bool = False) -> SensorSeries:
    if not series.timestamps_seconds:
        return series
    stop = min(count, len(series.timestamps_seconds))
    selection = slice(stop - 1, stop) if latest_only else slice(0, stop)
    return SensorSeries(
        node_id=series.node_id,
        timestamps_seconds=series.timestamps_seconds[selection],
        concentration_mg_l=series.concentration_mg_l[selection],
        pressure_m=series.pressure_m[selection],
        health=series.health[selection],
        missing=series.missing[selection],
        drift=series.drift[selection],
        delayed=series.delayed[selection],
        frozen=series.frozen[selection],
    )


def _metrics(result: Any, truth: str) -> dict[str, Any]:
    ranked = sorted(result.fused_belief, key=lambda node: (-result.fused_belief[node], node))
    rank = ranked.index(truth) + 1
    return {
        "top1": rank == 1,
        "top3": rank <= 3,
        "mrr": 1.0 / rank,
        "candidate_size": len(result.conformal_candidate_nodes),
        "entropy": result.posterior_history[-1].entropy_bits,
        "evidence_sufficient": result.evidence_sufficient,
        "planning_eligible": result.planning_allowed,
        "calibrated": result.calibrated,
        "ood_level": result.ood_level.value,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "top1": fmean(float(record["top1"]) for record in records),
        "top3": fmean(float(record["top3"]) for record in records),
        "mrr": fmean(float(record["mrr"]) for record in records),
        "candidate_size": fmean(float(record["candidate_size"]) for record in records),
        "entropy": fmean(float(record["entropy"]) for record in records),
        "evidence_sufficient": fmean(float(record["evidence_sufficient"]) for record in records),
        "planning_eligible": fmean(float(record["planning_eligible"]) for record in records),
        "calibrated": fmean(float(record["calibrated"]) for record in records),
        "ood_normal": fmean(float(record["ood_level"] == "NORMAL") for record in records),
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"
    network_path = ROOT / "data/frozen/golden_network.inp"
    network = wntr.network.WaterNetworkModel(str(network_path))
    pipeline = V4PipelineFactory(ROOT / "models/hydrocore-v4-release", project_root=ROOT)(
        None, network_path
    )
    generator = WNTRScenarioGenerator()
    records: dict[str, list[dict[str, Any]]] = {str(step): [] for step in CHECKPOINTS}
    records["latest_only"] = []
    for seed in SEEDS:
        scenario = generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=seed,
                network_id="golden-reference",
                network_family="golden-reference",
                split=DatasetSplit.DEVELOPMENT_HOLDOUT,
                stage=CurriculumStage.OPERATIONAL,
                sensor_count=4,
            ),
        )
        series = build_sensor_series(scenario, build_feature_context(network))
        truth = scenario.manifest.incident.source_nodes[0]
        for step in CHECKPOINTS:
            result = pipeline.analyze(uuid.uuid4(), network, [_prefix(item, step) for item in series])
            records[str(step)].append({"seed": seed, **_metrics(result, truth)})
        result = pipeline.analyze(
            uuid.uuid4(), network, [_prefix(item, 25, latest_only=True) for item in series]
        )
        records["latest_only"].append({"seed": seed, **_metrics(result, truth)})
    payload = {
        "schema_version": 1,
        "purpose": "post-remediation causal temporal capability; no future observations",
        "network": "golden-reference",
        "population": "development only",
        "locked_test_opened_before": False,
        "locked_test_opened_after": locked_test_opened(ROOT),
        "maximum_causal_window_steps": 25,
        "negative_baseline": "latest_only uses one final report per sensor and is not product behavior",
        "checkpoints": {key: _summarize(value) for key, value in records.items()},
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
