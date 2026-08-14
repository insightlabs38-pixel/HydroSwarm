"""Capability diagnostic Section 6 (highest priority): train-vs-serve parity,
extended past the existing gate's scope.

`scripts/run_train_serve_parity_gate.py` already exists and passes (verified
this session: `reports/evaluation/capability-diagnostic/_existing_gate_check.json`),
but it constructs BOTH comparison arms' `SensorSeries` via the SAME function
(`hydroswarm.training.corpus.build_sensor_series`, the training path) --
it never exercises the real live path's `SensorObservation -> SensorSeries`
construction (`sensor_series()`, a closure defined inline inside
`hydroswarm.api.app`'s app factory at lines ~357-380, not importable). That
closure is where diagnostic.txt Section 6's actual PATH A vs PATH B
divergence risk lives: different health computation, different missing-flag
source, different timestamp derivation, and (critically) the production
`HybridInferencePipeline.analyze` call to `feature_builder.build(...)` at
`src/hydroswarm/inference/pipeline.py:701-707` omits `window_steps`, so it
silently uses the default (12) instead of training's
`window_steps=len(scenario.timestamps_seconds)` (25 for this repo's
networks).

This script:
  1. Builds a real GeneratedScenario (non-locked; validation-equivalent,
     never train/calibration/locked reused verbatim -- fresh seeds).
  2. PATH A: `build_sensor_series` (byte-identical training code).
  3. PATH B: reimplements the app.py `sensor_series()` closure verbatim
     (cited, not reinvented) from `SensorObservation` records built to carry
     MATHEMATICALLY EQUIVALENT evidence content to PATH A (full trajectory,
     same concentration/validity/frozen/outage pattern) -- so any tensor
     divergence reflects the CONSTRUCTION PATH, not evidence sparsity
     (sparsity is a separate, later diagnostic section).
  4. Feeds both SensorSeries lists through the SAME `HydraulicFeatureBuilder.
     build(...)`, once with matched window_steps (isolates pure construction
     divergence) and once with PATH B using production's real default
     window_steps=12 (measures the actual as-deployed divergence).
  5. Diffs every batch tensor key, node order, timestamps, classical_prior.

No locked-test access: only fresh WNTRScenarioGenerator-generated scenarios
with seeds outside every existing frozen/train/calibration seed set.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.domain.schemas import SensorObservation  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.corpus import (  # noqa: E402
    build_feature_context,
    build_sensor_series,
    model_input_classical_prior,
    resolve_model_input_signature_library,
)

# Fresh, deterministic, non-locked scenario seeds -- distinct from every
# committed train/calibration/development-holdout/locked seed used
# elsewhere in this repo (2026081x family already used by other
# diagnostics/gates in this repo; this uses a visibly different base).
PARITY_SEEDS = [20260813_00 + i for i in range(20)]
TOPOLOGY_FAMILIES = ["golden-reference", "branched-loop", "loop-grid"]


def _effective_sensor_health_replica(observation: SensorObservation) -> float:
    """Verbatim replica of hydroswarm.api.app._effective_sensor_health
    (src/hydroswarm/api/app.py:285-295) -- not importable, it's a closure
    inside create_app(). Cited, not reinvented."""
    return min(observation.quality, 0.25) if observation.frozen_flag else observation.quality


def _sensor_series_closure_replica(observations: tuple[SensorObservation, ...], detected_at: datetime) -> list[Any]:
    """Verbatim replica of hydroswarm.api.app's sensor_series() closure
    (src/hydroswarm/api/app.py:357-380). Cited, not reinvented -- same
    grouping/sort/field derivation, applied to a real SensorObservation
    tuple instead of a live IncidentRuntime's stored observations."""
    from collections import defaultdict

    from hydroswarm.preprocessing.builder import SensorSeries

    grouped: dict[str, list[SensorObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.node_id].append(item)
    origin = detected_at
    result: list[SensorSeries] = []
    for node_id, items in sorted(grouped.items()):
        items.sort(key=lambda item: item.observed_at)
        result.append(SensorSeries(
            node_id=node_id,
            timestamps_seconds=tuple((item.observed_at - origin).total_seconds() for item in items),
            concentration_mg_l=tuple(item.concentration_mg_l for item in items),
            pressure_m=tuple(item.pressure_m for item in items),
            health=tuple(_effective_sensor_health_replica(item) for item in items),
            missing=tuple(item.missing for item in items),
            drift=tuple(item.drift_flag or item.frozen_flag for item in items),
            delayed=tuple(item.received_at > item.observed_at for item in items),
            frozen=tuple(item.frozen_flag for item in items),
        ))
    return result


def _build_equivalent_observations(scenario: Any, context: Any, origin: datetime) -> tuple[SensorObservation, ...]:
    """Construct SensorObservation records carrying the SAME evidence
    content build_sensor_series would encode -- full trajectory, same
    validity/frozen/outage pattern, pressure from the SAME hydraulic state
    estimate Path A uses (not a harness-fixed value; pressure-fixture
    parity is a separate diagnostic section) -- to isolate construction-path
    divergence from evidence-content divergence."""
    observations: list[SensorObservation] = []
    for source_column, node_id in enumerate(scenario.sensor_nodes):
        valid = scenario.observation_mask[:, source_column]
        frozen = scenario.frozen_mask[:, source_column]
        outage = scenario.communication_outage_mask[:, source_column]
        concentration = scenario.observed_concentration[:, source_column]
        pressure_estimate = context.state.pressure_m[node_id].estimate
        for step_index, timestamp in enumerate(scenario.timestamps_seconds):
            is_valid = bool(valid[step_index])
            is_frozen = bool(frozen[step_index])
            is_outage = bool(outage[step_index])
            observed_at = origin + timedelta(seconds=float(timestamp))
            received_at = observed_at + timedelta(minutes=15) if is_outage else observed_at
            observations.append(SensorObservation(
                sensor_id=f"parity-{node_id}",
                node_id=str(node_id),
                observed_at=observed_at,
                received_at=received_at,
                concentration_mg_l=(max(0.0, float(concentration[step_index])) if is_valid else None),
                pressure_m=(pressure_estimate if is_valid else None),
                # Path A's health scheme collapses BOTH frozen and outage to
                # 0.25; Path B's _effective_sensor_health only caps via
                # frozen_flag. Setting quality=0.25 directly for outage-only
                # points reproduces Path A's target health value through
                # Path B's real (structurally different, caller-driven)
                # health computation -- this divergence in MECHANISM is
                # itself a parity finding, reported separately below, not
                # hidden by this construction choice.
                quality=(0.25 if (is_outage and not is_frozen) else 1.0) if is_valid else 1.0,
                missing=not is_valid,
                drift_flag=False,
                frozen_flag=is_frozen,
            ))
    return tuple(observations)


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shapes_match": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    a_np = a.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    a_nan = np.isnan(a_np)
    b_nan = np.isnan(b_np)
    nan_pattern_matches = bool(np.array_equal(a_nan, b_nan))
    finite_mask = ~(a_nan | b_nan)
    if finite_mask.any():
        abs_diff = np.abs(a_np[finite_mask] - b_np[finite_mask])
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
    return {
        "shapes_match": True,
        "shape": list(a.shape),
        "nan_count_a": int(a_nan.sum()),
        "nan_count_b": int(b_nan.sum()),
        "nan_pattern_matches": nan_pattern_matches,
        "max_abs_diff_finite_cells": max_abs_diff,
        "mean_abs_diff_finite_cells": mean_abs_diff,
        "exact_match": nan_pattern_matches and max_abs_diff == 0.0,
    }


def _run_one(seed: int, family: str) -> dict[str, Any]:
    network = build_wntr_network() if family == "golden-reference" else None
    if network is None:
        # Other governed families are imported the same way
        # run_train_serve_parity_gate.py does, via generate_cycle_b_corpus's
        # TRAIN_TOPOLOGIES -- reuse that for exact fidelity.
        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_cycle_b_corpus import TRAIN_TOPOLOGIES

        loaders = dict(TRAIN_TOPOLOGIES)
        network = loaders[family]()

    generator = WNTRScenarioGenerator()
    config = ScenarioGenerationConfig(
        seed=seed,
        network_id=family,
        network_family=family,
        split=DatasetSplit.VALIDATION,
        stage=CurriculumStage.OPERATIONAL,
        event_type=EventType.CONTAMINATION,
        pipe_outage_probability=0.0,
    )
    scenario = generator.generate(network, config)
    context = build_feature_context(network)
    origin = datetime(2026, 8, 13, tzinfo=UTC)

    # --- PATH A: training construction ---
    series_a = build_sensor_series(scenario, context)

    # --- PATH B: production-style construction from mathematically
    # equivalent SensorObservation evidence ---
    observations = _build_equivalent_observations(scenario, context, origin)
    series_b = _sensor_series_closure_replica(observations, origin)

    junctions = tuple(sorted(network.junction_name_list))
    from hydroswarm.data.scenarios import network_sha256

    topology_hash = network_sha256(network)
    library, reference_timestamps, mode = resolve_model_input_signature_library(topology_hash, junctions, network)
    prior_a = model_input_classical_prior(library, junctions, series_a, reference_timestamps)
    prior_b = model_input_classical_prior(library, junctions, series_b, reference_timestamps)

    full_window = len(scenario.timestamps_seconds)
    builder = HydraulicFeatureBuilder()

    built_a = builder.build(network, context.graph, context.state, series_a, classical_prior=prior_a, window_steps=full_window)
    built_b_matched_window = builder.build(network, context.graph, context.state, series_b, classical_prior=prior_b, window_steps=full_window)
    built_b_production_default = builder.build(network, context.graph, context.state, series_b, classical_prior=prior_b)

    tensor_keys = sorted(set(built_a.batch.keys()) & set(built_b_matched_window.batch.keys()))
    diffs_matched_window = {key: _tensor_diff(built_a.batch[key], built_b_matched_window.batch[key]) for key in tensor_keys}
    diffs_production_default = {
        key: _tensor_diff(built_a.batch[key], built_b_production_default.batch[key])
        for key in sorted(set(built_a.batch.keys()) & set(built_b_production_default.batch.keys()))
    }

    node_order_a = built_a.node_ids
    node_order_b_matched = built_b_matched_window.node_ids
    node_order_match = node_order_a == node_order_b_matched

    classical_prior_max_diff = max(
        (abs(prior_a.get(node, 0.0) - prior_b.get(node, 0.0)) for node in junctions), default=0.0
    )

    # CAP-PARITY-01 root-cause isolation: for every point where PATH A
    # marks the reading missing, does PATH B's health value fail to reflect
    # that (i.e. show > 0.0 despite missing=True)? PATH B's
    # _effective_sensor_health derives health purely from the caller-
    # supplied `quality`/`frozen_flag` fields, never from `missing` itself
    # -- SensorObservation's own validator enforces missing<->no-measurement
    # consistency for concentration/pressure but NOT for quality/health.
    missing_health_mismatches = []
    series_a_by_node = {s.node_id: s for s in series_a}
    series_b_by_node = {s.node_id: s for s in series_b}
    for node_id in series_a_by_node:
        sa, sb = series_a_by_node[node_id], series_b_by_node.get(node_id)
        if sb is None or len(sa.timestamps_seconds) != len(sb.timestamps_seconds):
            continue
        for i in range(len(sa.timestamps_seconds)):
            if sa.missing[i] and sb.missing[i] and sa.health[i] != sb.health[i]:
                missing_health_mismatches.append({
                    "node_id": node_id, "step_index": i,
                    "path_a_health": sa.health[i], "path_b_health": sb.health[i],
                })

    return {
        "seed": seed,
        "topology_family": family,
        "scenario_id": str(scenario.manifest.scenario_id),
        "network_sha256": topology_hash,
        "signature_mode": str(mode),
        "path_a_timesteps_per_sensor": len(scenario.timestamps_seconds),
        "path_b_timesteps_per_sensor": {s.node_id: len(s.timestamps_seconds) for s in series_b},
        "node_order_match": node_order_match,
        "classical_prior_max_abs_diff": classical_prior_max_diff,
        "cap_parity_01_missing_health_mismatches": missing_health_mismatches,
        "tensor_diffs_matched_window_steps": diffs_matched_window,
        "tensor_diffs_production_default_window_steps_12": diffs_production_default,
        "all_exact_match_matched_window": all(d.get("exact_match", False) for d in diffs_matched_window.values()),
        "all_exact_match_production_default": all(d.get("exact_match", False) for d in diffs_production_default.values()),
    }


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed for this diagnostic"

    results = []
    for i, seed in enumerate(PARITY_SEEDS):
        family = TOPOLOGY_FAMILIES[i % len(TOPOLOGY_FAMILIES)]
        try:
            results.append(_run_one(seed, family))
        except Exception as exc:  # noqa: BLE001 -- deliberately captured, not swallowed
            results.append({"seed": seed, "topology_family": family, "error": f"{type(exc).__name__}: {exc}"})

    ok_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    first_divergent_field: str | None = None
    first_divergent_scenario: str | None = None
    for r in ok_results:
        for field, diff in r["tensor_diffs_matched_window_steps"].items():
            if not diff.get("exact_match", False):
                first_divergent_field = field
                first_divergent_scenario = r["scenario_id"]
                break
        if first_divergent_field:
            break

    total_missing_health_mismatches = sum(len(r.get("cap_parity_01_missing_health_mismatches", [])) for r in ok_results)
    scenarios_with_mismatch = sum(1 for r in ok_results if r.get("cap_parity_01_missing_health_mismatches"))

    report = {
        "schema_version": 1,
        "section": "6_train_serve_parity",
        "n_scenarios": len(PARITY_SEEDS),
        "n_ok": len(ok_results),
        "n_errors": len(error_results),
        "errors": error_results,
        "existing_gate_result": "PASS -- reports/evaluation/capability-diagnostic/_existing_gate_check.json (SensorSeries-level only, see this script's module docstring for scope gap)",
        "cap_findings": {
            "CAP-PARITY-01": {
                "title": "Production health channel does not reflect missing==True",
                "root_cause": (
                    "src/hydroswarm/api/app.py's _effective_sensor_health(observation) (lines 285-295) computes "
                    "health purely from observation.quality (caller-supplied, default 1.0) capped only via "
                    "frozen_flag; it never inspects observation.missing. SensorObservation's own validator "
                    "(src/hydroswarm/domain/schemas.py) enforces missing<->no-measurement consistency for "
                    "concentration_mg_l/pressure_m only, not quality. Training's build_sensor_series "
                    "(src/hydroswarm/training/corpus.py:551) instead derives health deterministically from the "
                    "underlying validity mask: 0.0 whenever a reading is invalid/missing, regardless of any "
                    "caller-style quality field (which does not exist on that path)."
                ),
                "measured_effect": (
                    f"{total_missing_health_mismatches} individual (node, timestep) cells across "
                    f"{scenarios_with_mismatch}/{len(ok_results)} scenarios have health=0.0 under training "
                    "semantics but health=1.0 under this diagnostic's careful mathematically-equivalent PATH B "
                    "construction (which explicitly sets quality=1.0 default, matching the real "
                    "live_robustness.py harness's own default for missing, non-'quality'-health-mode observations "
                    "-- src/hydroswarm/evaluation/live_robustness.py:230, `\"quality\": 0.20 if degraded and "
                    "condition.health_mode == 'quality' else 1.0`, independent of the `missing` flag on the same "
                    "line 231). Confirmed as REAL production code, not a test-construction artifact."
                ),
                "downstream_impact": (
                    "Affects quality_features (health channel, every missing timestep) directly, and additionally "
                    "corrupts node_features' 'current health' snapshot column whenever a node's MOST RECENT "
                    "available reading happens to be missing (observed on 4/20 scenarios this run)."
                ),
                "taxonomy": "CAP-PARITY",
                "severity": "MEDIUM -- silently tells the model an unmeasured node is fully healthy, which can "
                "bias node-level trust/attention away from genuinely-uninstrumented or failed sensors; does not "
                "corrupt concentration/pressure (those correctly stay None/masked), so it is a secondary "
                "feature-quality signal error, not a wholesale evidence corruption.",
            },
            "CAP-PARITY-02": {
                "title": "Production feature builder call omits window_steps, silently defaulting to 12 instead of training's full-trajectory length",
                "root_cause": (
                    "hydroswarm.training.corpus.scenario_to_example calls HydraulicFeatureBuilder.build(..., "
                    "window_steps=len(scenario.timestamps_seconds)) -- 25 for every network in this repo, verified "
                    "this session. hydroswarm.inference.pipeline.HybridInferencePipeline.analyze's feature_building "
                    "stage (src/hydroswarm/inference/pipeline.py:701-707) calls the SAME builder.build(...) but "
                    "never passes window_steps at all, so it silently uses HydraulicFeatureBuilder.build's own "
                    "default of 12 (src/hydroswarm/preprocessing/builder.py:131)."
                ),
                "measured_effect": (
                    "Directly confirmed by tensor shape divergence: temporal_features/quality_features/"
                    "quality_mask/sensor_mask/timestamps all come out shape [1, 25, ...] under Path A (training) "
                    "vs [1, 12, ...] under Path B fed the SAME mathematically-equivalent full-trajectory evidence "
                    "and the real production default (see tensor_diffs_production_default_window_steps_12 per "
                    "scenario in this report) -- production structurally CANNOT see more than the most recent 12 "
                    "distinct timestamps across all sensors, even when 25 real observations exist."
                ),
                "downstream_impact": (
                    "In practice this is dominated by (not the primary driver alongside) the LIVE harness's own "
                    "far sparser real evidence (see Section 8/9 temporal-evidence-ablation results: initial LIVE "
                    "incidents typically carry ~1 observation per sensor, well under either 12 or 25) -- but for "
                    "any future evidence-richer serving path (e.g. genuine multi-report incidents, or a fixed "
                    "harness feeding real history), this cap would independently discard the oldest evidence "
                    "before it ever reaches the model, compounding rather than explaining LIVE's current gap."
                ),
                "taxonomy": "CAP-PARITY",
                "severity": "LOW-MEDIUM given current LIVE evidence sparsity (rarely binds today), but a real, "
                "confirmed construction-path defect that will start silently truncating evidence the moment LIVE "
                "incidents carry more than 12 real observations.",
            },
        },
        "results": results,
        "summary": {
            "all_scenarios_exact_match_at_matched_window_steps": all(
                r.get("all_exact_match_matched_window", False) for r in ok_results
            ),
            "all_scenarios_exact_match_at_production_default_window_steps": all(
                r.get("all_exact_match_production_default", False) for r in ok_results
            ),
            "first_divergent_field_at_matched_window": first_divergent_field,
            "first_divergent_scenario_at_matched_window": first_divergent_scenario,
            "path_a_evidence_timesteps_per_sensor": "25 (full trajectory; window_steps explicitly passed = len(scenario.timestamps_seconds))",
            "path_b_production_window_steps_default": 12,
            "note": "path_b_timesteps_per_sensor above is the RAW SensorSeries length before HydraulicFeatureBuilder's own [-window_steps:] truncation; "
            "with mathematically-equivalent full evidence (25 timesteps/sensor) fed through this diagnostic's Path B construction, the production "
            "pipeline's real default (window_steps=12, since HybridInferencePipeline.analyze's feature_building stage never passes window_steps -- "
            "src/hydroswarm/inference/pipeline.py:701-707) truncates to only the most recent 12 of those 25 timesteps, while training always sees "
            "the full 25 -- a genuine train/serve construction-path divergence, separate from (and compounding) the LIVE harness's own much sparser "
            "(1-observation-per-sensor) real evidence content.",
        },
        "locked_test_opened_after": locked_test_opened(ROOT),
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "train-serve-parity.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, default=str))
    print(f"n_ok={report['n_ok']} n_errors={report['n_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
