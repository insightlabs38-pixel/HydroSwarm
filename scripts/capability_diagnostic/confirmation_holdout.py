"""Capability diagnostic Section 39: small development holdout for
diagnostic confirmation.

Generates a NEW deterministic development-only scenario set (seed family
20260899, visibly distinct from every other seed family used elsewhere in
this diagnostic -- 2026081[3-8]xx) and checks whether the two primary
findings from the main diagnostic set replicate:

1. Evidence sparsity: LATEST-1 snapshot evidence performs far worse than
   full-trajectory evidence on the SAME real frozen pipeline
   (temporal-ablation.json's headline finding).
2. Network-identity defect: golden-reference incidents served through the
   REAL production network-construction path (file-loaded .inp, what
   V4PipelineFactory actually uses) show calibrated=False and a lower top1
   than the programmatically-built (scenario-generation-identical) network
   path (network-parity.json / calibration-analysis.json's CAP-DATA-01 /
   CAP-CAL-01 finding).

Uses no train, no calibration, no locked examples -- fresh scenarios only.
Not used for optimization; confirmation only.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.runtime.paths import resolve_v4_bundle_dir  # noqa: E402
from hydroswarm.runtime.v4_defaults import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402

CONFIRMATION_SEEDS = [20260899_00 + i for i in range(40)]
LATEST_1_SEEDS = CONFIRMATION_SEEDS[:20]
NETWORK_IDENTITY_SEEDS = CONFIRMATION_SEEDS[20:40]


def _rank(belief: dict[str, float], truth: str) -> dict[str, Any]:
    if not belief or sum(belief.values()) <= 0:
        return {"top1": None, "top3": None, "reciprocal_rank": None}
    return {
        "top1": localization_top_k(belief, truth, k=1),
        "top3": localization_top_k(belief, truth, k=3),
        "reciprocal_rank": mean_reciprocal_rank([belief], [truth]),
    }


def _confirm_evidence_sparsity(factory: V4PipelineFactory, network: Any, context: Any) -> dict[str, Any]:
    generator = WNTRScenarioGenerator()
    pipeline = factory(None, ROOT / "data" / "frozen" / "golden_network.inp")
    latest1_records, full_records = [], []
    for seed in LATEST_1_SEEDS:
        config = ScenarioGenerationConfig(
            seed=seed, network_id="golden-reference", network_family="golden-reference",
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
        )
        scenario = generator.generate(network, config)
        truth = scenario.manifest.incident.source_nodes[0]
        full_series = build_sensor_series(scenario, context)

        latest1 = []
        for series in full_series:
            n = len(series.timestamps_seconds)
            sl = slice(n - 1, n)
            from hydroswarm.preprocessing.builder import SensorSeries

            latest1.append(SensorSeries(
                node_id=series.node_id, timestamps_seconds=series.timestamps_seconds[sl],
                concentration_mg_l=series.concentration_mg_l[sl], pressure_m=series.pressure_m[sl],
                health=series.health[sl], missing=series.missing[sl], drift=series.drift[sl],
                delayed=series.delayed[sl], frozen=series.frozen[sl] if series.frozen else (),
            ))
        try:
            result_latest1 = pipeline.analyze(uuid.uuid4(), network, latest1)
            latest1_records.append({"seed": seed, **_rank(dict(result_latest1.fused_belief), truth)})
        except Exception as exc:  # noqa: BLE001
            latest1_records.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
        try:
            result_full = pipeline.analyze(uuid.uuid4(), network, list(full_series))
            full_records.append({"seed": seed, **_rank(dict(result_full.fused_belief), truth)})
        except Exception as exc:  # noqa: BLE001
            full_records.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})

    def _agg(records: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in records if "error" not in r and r.get("top1") is not None]
        return {"n": len(records), "n_ok": len(ok), "top1": (sum(r["top1"] for r in ok) / len(ok)) if ok else None}

    latest1_agg, full_agg = _agg(latest1_records), _agg(full_records)
    return {
        "n_scenarios": len(LATEST_1_SEEDS),
        "latest1_summary": latest1_agg,
        "full_trajectory_summary": full_agg,
        "gap": (full_agg["top1"] or 0) - (latest1_agg["top1"] or 0),
        "replicates_main_finding": ((full_agg["top1"] or 0) - (latest1_agg["top1"] or 0)) > 0.3,
        "per_scenario": {"latest1": latest1_records, "full": full_records},
    }


def _confirm_network_identity(factory: V4PipelineFactory, context: Any) -> dict[str, Any]:
    import wntr

    programmatic_network = build_wntr_network()
    production_network = wntr.network.WaterNetworkModel(str(ROOT / "data" / "frozen" / "golden_network.inp"))
    pipeline = factory(None, ROOT / "data" / "frozen" / "golden_network.inp")

    generator = WNTRScenarioGenerator()
    scenario_arm, production_arm = [], []
    for seed in NETWORK_IDENTITY_SEEDS:
        config = ScenarioGenerationConfig(
            seed=seed, network_id="golden-reference", network_family="golden-reference",
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
        )
        scenario = generator.generate(programmatic_network, config)
        truth = scenario.manifest.incident.source_nodes[0]
        series_scenario_net = build_sensor_series(scenario, build_feature_context(programmatic_network))
        series_production_net = build_sensor_series(scenario, context)  # context built from production_network below

        try:
            result_scenario = pipeline.analyze(uuid.uuid4(), programmatic_network, series_scenario_net)
            scenario_arm.append({
                "seed": seed, **_rank(dict(result_scenario.fused_belief), truth),
                "calibrated": result_scenario.calibrated,
            })
        except Exception as exc:  # noqa: BLE001
            scenario_arm.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
        try:
            result_production = pipeline.analyze(uuid.uuid4(), production_network, series_production_net)
            production_arm.append({
                "seed": seed, **_rank(dict(result_production.fused_belief), truth),
                "calibrated": result_production.calibrated,
            })
        except Exception as exc:  # noqa: BLE001
            production_arm.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})

    def _agg(records: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in records if "error" not in r and r.get("top1") is not None]
        return {
            "n": len(records), "n_ok": len(ok),
            "top1": (sum(r["top1"] for r in ok) / len(ok)) if ok else None,
            "calibrated_rate": (sum(1 for r in ok if r.get("calibrated")) / len(ok)) if ok else None,
        }

    scenario_agg, production_agg = _agg(scenario_arm), _agg(production_arm)
    return {
        "n_scenarios": len(NETWORK_IDENTITY_SEEDS),
        "scenario_hash_arm_summary": scenario_agg,
        "production_hash_arm_summary": production_agg,
        "scenario_network_sha256": network_sha256(programmatic_network),
        "production_network_sha256": network_sha256(production_network),
        "hashes_match": network_sha256(programmatic_network) == network_sha256(production_network),
        "replicates_cap_data_01": (
            network_sha256(programmatic_network) != network_sha256(production_network)
            and production_agg["calibrated_rate"] == 0.0
        ),
        "per_scenario": {"scenario_hash_arm": scenario_arm, "production_hash_arm": production_arm},
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"

    factory = V4PipelineFactory(resolve_v4_bundle_dir())
    network = build_wntr_network()
    context = build_feature_context(network)

    import wntr

    production_network = wntr.network.WaterNetworkModel(str(ROOT / "data" / "frozen" / "golden_network.inp"))
    production_context = build_feature_context(production_network)

    sparsity_confirmation = _confirm_evidence_sparsity(factory, network, context)
    identity_confirmation = _confirm_network_identity(factory, production_context)

    report = {
        "schema_version": 1,
        "section": "39_confirmation_holdout",
        "seed_family": "20260899xx -- distinct from every other seed family used elsewhere in this diagnostic (2026081[3-8]xx)",
        "excludes_train_calibration_locked": True,
        "used_for_optimization": False,
        "evidence_sparsity_confirmation": sparsity_confirmation,
        "network_identity_defect_confirmation": identity_confirmation,
        "overall_verdict": {
            "evidence_sparsity_finding_replicates": sparsity_confirmation["replicates_main_finding"],
            "network_identity_defect_replicates": identity_confirmation["replicates_cap_data_01"],
        },
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "confirmation-holdout.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["overall_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
