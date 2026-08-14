"""Capability diagnostic Section 11: pressure-representation parity.

The real LIVE robustness harness (`hydroswarm.evaluation.live_robustness._payloads`,
lines ~204-238, verified this session and already cited in
`temporal-ablation.json`/`train-serve-parity.json`) sends `pressure_m` fixed
at exactly `25.0` for every valid observation -- never the real WNTR/
hydraulic-state pressure training uses (`context.state.pressure_m[node_id]
.estimate`, via `hydroswarm.training.corpus.build_sensor_series`). This
script measures, for real, how much that fixture distorts localization
through the REAL, unmodified, frozen `HybridInferencePipeline.analyze()`.

Four SensorSeries construction conditions, same evidence pattern (which
timesteps are valid/frozen/outage), same everything else -- only the
pressure_m values vary:
  A. true_wntr    -- context.state.pressure_m[node].estimate (training-real)
  B. fixed_25m    -- exactly 25.0 for every valid reading, None otherwise
                     (byte-for-byte matches the real LIVE harness fixture)
  C. omitted      -- None for every reading, valid or not
  D. noisy_governed -- true value + N(0, 2.0m), sigma governed/documented
                     here, clipped at 0 (pressure can't be physically
                     negative in this feature's units); seeded per-scenario
                     for reproducibility

Note (confirmed by reading src/hydroswarm/preprocessing/builder.py lines
~155-172 this session): SensorSeries.pressure_m ONLY feeds
`temporal_features` channel index 1 (`pressure_m/100.0` or NaN when None).
The node_features "pressure_value" column (index 4) is built from
`state.pressure_m[node_id].estimate` directly -- the SAME hydraulic-state
estimate regardless of what SensorSeries.pressure_m carries. So this
fixture, however wrong, can only corrupt the TEMPORAL channel, never the
per-node snapshot pressure feature; that scoping is verified structurally
here, not assumed.

No locked-test access: only fresh WNTRScenarioGenerator-generated
golden-reference scenarios, seeds distinct from every seed already used
elsewhere in this diagnostic (20260813_00..19 used by Sections 6-10).
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.builder import SensorSeries  # noqa: E402
from hydroswarm.runtime.paths import resolve_v4_bundle_dir  # noqa: E402
from hydroswarm.runtime.v4_defaults import V4PipelineFactory  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, build_sensor_series  # noqa: E402

SEEDS = [20260814_00 + i for i in range(15)]
NETWORK_PATH = ROOT / "data" / "frozen" / "golden_network.inp"
NOISE_SIGMA_M = 2.0
CONDITIONS = ("true_wntr", "fixed_25m", "omitted", "noisy_governed")


def _make_condition(series: SensorSeries, condition: str, rng: np.random.Generator) -> SensorSeries:
    if condition == "true_wntr":
        return series
    if condition == "fixed_25m":
        new_pressure = tuple(None if missing else 25.0 for missing in series.missing)
    elif condition == "omitted":
        new_pressure = tuple(None for _ in series.missing)
    elif condition == "noisy_governed":
        new_pressure = tuple(
            None if (missing or true_p is None) else max(0.0, float(true_p + rng.normal(0.0, NOISE_SIGMA_M)))
            for missing, true_p in zip(series.missing, series.pressure_m, strict=True)
        )
    else:
        raise ValueError(condition)
    return replace(series, pressure_m=new_pressure)


def _rank_metrics(belief: dict[str, float], truth: str) -> dict[str, Any]:
    if not belief or not np.isfinite(sum(belief.values())) or sum(belief.values()) <= 0:
        return {"top1": None, "top3": None, "reciprocal_rank": None, "true_source_probability": None, "belief_has_nan": any(not np.isfinite(v) for v in belief.values()) if belief else False}
    top1 = localization_top_k(belief, truth, k=1)
    top3 = localization_top_k(belief, truth, k=3)
    mrr = mean_reciprocal_rank([belief], [truth])
    return {
        "top1": top1,
        "top3": top3,
        "reciprocal_rank": mrr,
        "true_source_probability": belief.get(truth, 0.0),
        "belief_has_nan": False,
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    factory = V4PipelineFactory(resolve_v4_bundle_dir())
    network = build_wntr_network()
    pipeline = factory(None, NETWORK_PATH)
    context = build_feature_context(network)
    generator = WNTRScenarioGenerator()

    per_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}

    for seed in SEEDS:
        config = ScenarioGenerationConfig(
            seed=seed, network_id="golden-reference", network_family="golden-reference",
            split=DatasetSplit.VALIDATION, stage=CurriculumStage.OPERATIONAL,
            event_type=EventType.CONTAMINATION, pipe_outage_probability=0.0,
        )
        scenario = generator.generate(network, config)
        truth = scenario.manifest.incident.source_nodes[0]
        true_series = build_sensor_series(scenario, context)
        rng = np.random.default_rng(seed)

        for condition in CONDITIONS:
            varied = [_make_condition(s, condition, rng) for s in true_series]
            try:
                result = pipeline.analyze(uuid.uuid4(), network, varied)
                fused_metrics = _rank_metrics(dict(result.fused_belief), truth)
                neural_metrics = _rank_metrics(dict(result.neural_belief), truth) if result.neural_belief else None
                record = {
                    "seed": seed,
                    "fused": fused_metrics,
                    "neural": neural_metrics,
                    "ood_level": result.ood_level.value,
                    "disagreement_js": result.fusion_diagnostics.disagreement_js if result.fusion_diagnostics else None,
                    "evidence_sufficient": result.evidence_sufficient,
                    "candidate_set_size": len(result.conformal_candidate_nodes) if result.conformal_candidate_nodes else None,
                    "planning_allowed": result.planning_allowed,
                    "calibrated": result.calibrated,
                }
            except Exception as exc:  # noqa: BLE001
                record = {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            per_condition[condition].append(record)

    def _agg(records: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in records if "error" not in r and r["fused"]["top1"] is not None]
        errored = [r for r in records if "error" in r]
        nan_beliefs = [r for r in records if "error" not in r and r["fused"].get("belief_has_nan")]
        return {
            "n": len(records),
            "n_ok": len(ok),
            "n_errors": len(errored),
            "n_nan_belief": len(nan_beliefs),
            "errors": errored,
            "fused_top1_mean": (sum(r["fused"]["top1"] for r in ok) / len(ok)) if ok else None,
            "fused_top3_mean": (sum(r["fused"]["top3"] for r in ok) / len(ok)) if ok else None,
            "fused_mrr_mean": (sum(r["fused"]["reciprocal_rank"] for r in ok) / len(ok)) if ok else None,
            "neural_top1_mean": (
                sum(r["neural"]["top1"] for r in ok if r["neural"] and r["neural"]["top1"] is not None)
                / max(1, sum(1 for r in ok if r["neural"] and r["neural"]["top1"] is not None))
            ) if any(r["neural"] and r["neural"]["top1"] is not None for r in ok) else None,
            "mean_disagreement_js": (
                sum(r["disagreement_js"] for r in ok if r["disagreement_js"] is not None)
                / max(1, sum(1 for r in ok if r["disagreement_js"] is not None))
            ) if any(r["disagreement_js"] is not None for r in ok) else None,
            "mean_candidate_set_size": (
                sum(r["candidate_set_size"] for r in ok if r["candidate_set_size"] is not None)
                / max(1, sum(1 for r in ok if r["candidate_set_size"] is not None))
            ) if any(r["candidate_set_size"] is not None for r in ok) else None,
            "evidence_sufficient_rate": (sum(1 for r in ok if r["evidence_sufficient"]) / len(ok)) if ok else None,
            "ood_level_counts": {
                level: sum(1 for r in ok if r["ood_level"] == level)
                for level in sorted({r["ood_level"] for r in ok})
            },
        }

    summary = {c: _agg(per_condition[c]) for c in CONDITIONS}
    baseline_top1 = summary["true_wntr"]["fused_top1_mean"]
    deltas = {
        c: (
            None if baseline_top1 is None or summary[c]["fused_top1_mean"] is None
            else summary[c]["fused_top1_mean"] - baseline_top1
        )
        for c in CONDITIONS
    }

    fixed25_delta = deltas["fixed_25m"]
    fixed25_distorts = fixed25_delta is not None and abs(fixed25_delta) > 0.05
    classification = (
        "HARNESS_INPUT_ISSUE (fixed-25m pressure fixture, matching the real LIVE harness, measurably distorts "
        "results relative to true pressure -- an evidence-construction defect, not neural capacity)"
        if fixed25_distorts
        else "fixed-25m pressure fixture does NOT measurably distort top1 in this sample (|delta| <= 0.05); "
        "pressure representation is not a primary driver of the LIVE gap by itself"
    )

    report = {
        "schema_version": 1,
        "section": "11_pressure_representation_parity",
        "locked_test_opened_before": locked_before,
        "network": "golden-reference",
        "n_scenarios": len(SEEDS),
        "seeds": SEEDS,
        "noise_sigma_m": NOISE_SIGMA_M,
        "conditions": list(CONDITIONS),
        "structural_finding": (
            "Confirmed by reading src/hydroswarm/preprocessing/builder.py: SensorSeries.pressure_m feeds ONLY "
            "temporal_features[:, :, 1] (pressure_m/100.0, or NaN when None) at builder.py line ~245. The "
            "node_features 'pressure_value' column (index 4, builder.py line ~166) is built independently from "
            "state.pressure_m[node_id].estimate (the real hydraulic-state estimate), never from SensorSeries. "
            "So any pressure-representation error here can only corrupt the temporal channel, not the per-node "
            "snapshot pressure feature."
        ),
        "per_condition_aggregate": summary,
        "delta_fused_top1_vs_true_wntr": deltas,
        "verdict": {
            "fixed_25m_measurably_distorts_top1": fixed25_distorts,
            "fixed_25m_delta_top1": fixed25_delta,
            "classification": classification,
            "caveat": "N=15 scenarios, single network/topology (golden-reference), each scenario a binary "
            "correct/incorrect outcome (~6.7-point swing per scenario) -- small |delta| here is consistent with "
            "'no large effect' but does not rule out a modest one; see per-scenario records for raw values.",
        },
        "per_scenario_by_condition": per_condition,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "pressure-parity.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2, default=str))
    print(json.dumps({c: summary[c]["fused_top1_mean"] for c in CONDITIONS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
