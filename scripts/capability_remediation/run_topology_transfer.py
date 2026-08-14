"""Development-only post-remediation topology and fusion decomposition."""

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
    network_sha256,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.runtime import V4PipelineFactory  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402


NETWORKS = {
    "known_golden_reference": ("golden-reference", ROOT / "data/frozen/golden_network.inp", False),
    "known_branched_loop": ("branched-loop", ROOT / "data/topology-transfer/branched-loop.inp", False),
    "known_loop_grid": ("loop-grid", ROOT / "data/topologies/loop-grid.inp", False),
    "same_family_randomized": ("golden-reference", ROOT / "data/frozen/golden_network.inp", True),
    "unseen_development_coastal": ("coastal-branch", ROOT / "data/topologies/coastal-branch.inp", False),
}
SEEDS = tuple(range(2026081501, 2026081509))
OUT = ROOT / "reports/evaluation/capability-remediation"


def _rank(belief: dict[str, float] | None, truth: str) -> tuple[bool | None, bool | None, float | None]:
    if not belief:
        return None, None, None
    nodes = sorted(belief, key=lambda node: (-belief[node], node))
    position = nodes.index(truth) + 1
    return position == 1, position <= 3, 1.0 / position


def _mean(records: list[dict[str, Any]], name: str) -> float | None:
    values = [float(item[name]) for item in records if item.get(name) is not None]
    return fmean(values) if values else None


def _run() -> list[dict[str, Any]]:
    generator = WNTRScenarioGenerator()
    output: list[dict[str, Any]] = []
    for condition, (family, path, randomized) in NETWORKS.items():
        base = wntr.network.WaterNetworkModel(str(path))
        pipeline = V4PipelineFactory(ROOT / "models/hydrocore-v4-release", project_root=ROOT)(None, path)
        for seed in SEEDS:
            config = ScenarioGenerationConfig(
                seed=seed,
                network_id=family,
                network_family=family,
                split=DatasetSplit.DEVELOPMENT_HOLDOUT,
                stage=CurriculumStage.OPERATIONAL,
                sensor_count=min(4, len(base.junction_name_list)),
                roughness_variation_fraction=0.30 if randomized else 0.0,
                tank_level_variation_fraction=0.30 if randomized else 0.0,
                demand_regimes=(0.7, 1.0, 1.3) if randomized else (1.0,),
            )
            scenario, network = generator.generate_with_network(base, config)
            result = pipeline.analyze(
                uuid.uuid4(), network,
                build_sensor_series(scenario, build_feature_context(network)),
            )
            truth = scenario.manifest.incident.source_nodes[0]
            neural_top1, neural_top3, neural_mrr = _rank(
                dict(result.neural_belief) if result.neural_belief else None, truth
            )
            classical_top1, classical_top3, classical_mrr = _rank(dict(result.classical_belief), truth)
            hybrid_top1, hybrid_top3, hybrid_mrr = _rank(dict(result.fused_belief), truth)
            output.append({
                "condition": condition, "family": family, "seed": seed,
                "topology_hash": network_sha256(network), "truth": truth,
                "neural_top1": neural_top1, "neural_top3": neural_top3, "neural_mrr": neural_mrr,
                "classical_top1": classical_top1, "classical_top3": classical_top3, "classical_mrr": classical_mrr,
                "hybrid_top1": hybrid_top1, "hybrid_top3": hybrid_top3, "hybrid_mrr": hybrid_mrr,
                "calibrated": result.calibrated,
                "calibration_source": result.calibration_source,
                "ood_level": result.ood_level.value,
                "planning_allowed": result.planning_allowed,
            })
    return output


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"
    rows = _run()
    grouped = {name: [row for row in rows if row["condition"] == name] for name in NETWORKS}
    summary = {
        name: {
            "n": len(group),
            **{metric: _mean(group, metric) for metric in (
                "neural_top1", "neural_top3", "neural_mrr", "classical_top1", "classical_top3",
                "classical_mrr", "hybrid_top1", "hybrid_top3", "hybrid_mrr", "calibrated", "planning_allowed",
            )},
            "ood_normal": _mean([{"ood": row["ood_level"] == "NORMAL"} for row in group], "ood"),
        }
        for name, group in grouped.items()
    }
    known = [row for row in rows if row["condition"].startswith("known_")]
    fusion_harms = [
        row for row in known
        if not row["hybrid_top1"] and (row["neural_top1"] or row["classical_top1"])
    ]
    component = {
        "schema_version": 1,
        "population": "development-only causal full-window incidents",
        "locked_test_opened_before": False,
        "locked_test_opened_after": locked_test_opened(ROOT),
        "rows": rows,
        "known_network_component_summary": {
            "n": len(known),
            "neural_top1": _mean(known, "neural_top1"),
            "classical_top1": _mean(known, "classical_top1"),
            "hybrid_top1": _mean(known, "hybrid_top1"),
            "fusion_harms_count": len(fusion_harms),
            "fusion_harms_rate": len(fusion_harms) / len(known) if known else None,
            "fusion_harms": fusion_harms,
        },
    }
    topology = {
        "schema_version": 1,
        "population": "development-only; no locked topology test",
        "locked_test_opened_before": False,
        "locked_test_opened_after": locked_test_opened(ROOT),
        "conditions": summary,
        "interpretation": {
            "known": "Known governed networks should be compatible and calibrated.",
            "same_family_randomized": "A changed static configuration is physically compatible but intentionally loses calibration applicability until commissioned.",
            "unseen_development": "Unseen topology is a zero-shot localization measurement only; planning suppression is expected until commissioning.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "component-decomposition.json").write_text(json.dumps(component, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "topology-transfer.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
