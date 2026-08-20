"""Milestone 8.7 Sections 6-14: evaluate every trained CURRENT_CONTROL/
AGE_FIX_ONLY/AGE_FIX_PLUS_RELATIVE_TIME arm/seed checkpoint
(`reports/evaluation/hydrocore-v5/m8-7-runs/*.json`, written by
`run_m8_7_arm.py`) against development_holdout, and apply the frozen
guardrails/promotion logic (`docs/evaluation/HYDROCORE_V5_M8_7_PROTOCOL.md`).

Reuses, unmodified in spirit (only made keyword-arg-forwarding via a new
optional `feature_kwargs` parameter that defaults to None everywhere, so
Milestone 6's own already-committed results stay byte-for-byte
reproducible):
  - `evaluate_m1_depths.py`'s per-depth evaluation pattern (Section 6),
    reimplemented here (not imported -- that script's own
    `_evaluate_checkpoint_at_depth` is private and Milestone-1-scoped) with
    `feature_kwargs` threading added.
  - `run_m6_temporal.py`'s `run_cadence_experiment` (Sections 7/8:
    post-onset N=2/N=3 and matched-physical-time), `run_irregular_
    telemetry_experiment` (Section 11), `_fit_frozen_calibrator`
    (Section 12), and `_generate_cadence_incident_pool`/
    `build_high_resolution_network` -- imported directly, called once per
    arm with that arm's `feature_kwargs`.
  - `run_m8_6_representation_audit.py`'s representation-counterfactual/
    origin-translation methodology (Sections 9/10), reimplemented here at
    M8.7's own (stricter, for B/C) predeclared tolerance.
  - `run_m8_scaling.py`'s `build_grid_network` (Section 14, large-network
    check).

Compute-scoping (recorded explicitly, not hidden): standard localization
(Section 6) and correctness/counterfactual checks (Sections 9/10/14) run
for EVERY available seed of every arm; the more expensive M6-style
diagnostics (Sections 7/8/11/12) run only for each arm's SCREENING-
representative seed (31874, the same seed `_freeze_predictor()` already
treats as this codebase's primary/reference seed everywhere else) to keep
this milestone's already-large evaluation matrix tractable, per the
protocol document's own "seeds... for screening" scoping.

Writes:
  reports/evaluation/hydrocore-v5/m8-7-results.json
  reports/evaluation/hydrocore-v5/m8-7-temporal-diagnostics.json
  reports/evaluation/hydrocore-v5/m8-7-calibration.json
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import classify_runtime_condition  # noqa: E402
from hydroswarm.classical.metrics import localization_top_k, mean_reciprocal_rank  # noqa: E402
from hydroswarm.classical.state_estimation import HydraulicStateEstimator, OperationalTelemetry  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing import HydraulicFeatureBuilder, SensorSeries  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.simulation.wrapper import HydraulicSimulator  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402
from hydroswarm.training.data import collate_scenarios  # noqa: E402
from run_m6_temporal import (  # noqa: E402
    _fit_frozen_calibrator,
    _generate_cadence_incident_pool,
    run_cadence_experiment,
    run_irregular_telemetry_experiment,
)
from run_m8_7_arm import ARM_DEFINITIONS, SHARED_MODEL_CONFIG  # noqa: E402
from run_m8_scaling import build_grid_network  # noqa: E402

RUNS_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-runs"
OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-results.json"
OUTPUT_TEMPORAL = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-temporal-diagnostics.json"
OUTPUT_CALIBRATION = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m8-7-calibration.json"

EPS = 1e-9
ALPHA = 0.1
SCREENING_REPRESENTATIVE_SEED = 31874
EARLY_DEPTHS = (1, 2, 3)
MID_DEPTHS = (4, 6)
MATURE_DEPTHS = (12, 25)

#: Frozen protocol Section 8 tolerances.
STRUCTURAL_TOLERANCE = 1e-4
ORIGIN_TOLERANCE_ARMS_BC = 1e-4
TIMESTAMP_OFFSETS_SECONDS = {"+1h": 3_600.0, "+24h": 86_400.0, "+7d": 604_800.0}

#: Frozen protocol / milestone Section 16 guardrails.
GUARDRAIL_MAX_EARLY_TOP1_REGRESSION_PP = 5.0
GUARDRAIL_MAX_MATURE_TOP1_REGRESSION_PP = 3.0


def _load_run_records() -> dict[str, dict[int, dict[str, Any]]]:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARM_DEFINITIONS}
    for path in sorted(RUNS_ROOT.glob("*.json")):
        record = json.loads(path.read_text())
        by_arm.setdefault(record["arm"], {})[record["seed"]] = record
    return by_arm


def _load_checkpoint(record: dict[str, Any]) -> HydroCore:
    model_kwargs = record["model_kwargs"]
    model = HydroCore.from_variant("small", use_adapters=False, **model_kwargs, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(record["export_path"], device="cpu"), strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Section 6: standard localization by depth.
# ---------------------------------------------------------------------------


def _evaluate_at_depth(model: HydroCore, dev_records, library, depth: int, feature_kwargs: dict[str, Any], *, batch_size: int = 4) -> dict[str, Any]:
    examples = [
        scenario_to_prefix_example(record.scenario, record.network, library, depth, feature_context=record.feature_context, **feature_kwargs)
        for record in dev_records
    ]
    top1s, top3s, mrrs, nlls, entropies = [], [], [], [], []
    ood_finite = True
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs, targets = collate_scenarios(batch)
            output = model(inputs)
            probs = torch.softmax(output["source_node_logits"], dim=-1)
            if "ood_logits" in output and not torch.isfinite(output["ood_logits"]).all():
                ood_finite = False
            for row in range(probs.shape[0]):
                truth = int(targets["source_node"][row].item())
                row_probs_tensor = probs[row]
                row_probs = {position: float(value) for position, value in enumerate(row_probs_tensor.tolist())}
                top1s.append(localization_top_k(row_probs, truth, k=1))
                top3s.append(localization_top_k(row_probs, truth, k=3))
                mrrs.append(mean_reciprocal_rank([row_probs], [truth]))
                p_truth = row_probs.get(truth, 0.0)
                nlls.append(-math.log(p_truth + EPS))
                entropies.append(float(-torch.sum(row_probs_tensor * torch.log(row_probs_tensor + EPS))))
    return {
        "n": len(examples), "top1": statistics.fmean(top1s), "top3": statistics.fmean(top3s),
        "mrr": statistics.fmean(mrrs), "nll": statistics.fmean(nlls), "posterior_entropy": statistics.fmean(entropies),
        "all_finite": bool(np.isfinite(top1s).all() and np.isfinite(mrrs).all()), "ood_logits_finite": ood_finite,
    }


def run_standard_localization(model, dev_records, library, feature_kwargs: dict[str, Any]) -> dict[str, Any]:
    by_depth = {str(depth): _evaluate_at_depth(model, dev_records, library, depth, feature_kwargs) for depth in CAUSAL_PREFIX_DEPTHS}

    def _agg(depths: tuple[int, ...], metric: str) -> float:
        return statistics.fmean(by_depth[str(d)][metric] for d in depths if str(d) in by_depth)

    return {
        "by_depth": by_depth,
        "early_top1": _agg(EARLY_DEPTHS, "top1"), "mid_top1": _agg(MID_DEPTHS, "top1"), "mature_top1": _agg(MATURE_DEPTHS, "top1"),
        "early_mrr": _agg(EARLY_DEPTHS, "mrr"), "mid_mrr": _agg(MID_DEPTHS, "mrr"), "mature_mrr": _agg(MATURE_DEPTHS, "mrr"),
        "overall_mrr": statistics.fmean(row["mrr"] for row in by_depth.values()),
        "all_finite": all(row["all_finite"] for row in by_depth.values()),
        "ood_logits_finite": all(row["ood_logits_finite"] for row in by_depth.values()),
    }


# ---------------------------------------------------------------------------
# Sections 9/10: representation counterfactual + timestamp-origin invariance.
# ---------------------------------------------------------------------------


def _golden_context():
    network = build_wntr_network()
    simulator = HydraulicSimulator(network)
    raw = simulator.calculate_state(3600)
    estimated = HydraulicStateEstimator().estimate(raw, OperationalTelemetry())
    graph = simulator.build_dynamic_graph(estimated.as_hydraulic_state())
    return network, graph, estimated


def _series(node_id: str, timestamps: tuple[float, ...], concentrations: tuple[float, ...]) -> SensorSeries:
    n = len(timestamps)
    return SensorSeries(
        node_id=node_id, timestamps_seconds=timestamps, concentration_mg_l=concentrations,
        pressure_m=tuple(25.0 for _ in range(n)), health=tuple(1.0 for _ in range(n)),
        missing=tuple(False for _ in range(n)), drift=tuple(False for _ in range(n)), delayed=tuple(False for _ in range(n)),
    )


def _build_batch(network, graph, state, series, node_ids, feature_kwargs) -> dict[str, torch.Tensor]:
    prior = {n: 1.0 / len(node_ids) for n in node_ids}
    built = HydraulicFeatureBuilder().build(network, graph, state, series, classical_prior=prior, **feature_kwargs)
    return built.batch


def _posterior(model, batch) -> np.ndarray:
    with torch.no_grad():
        output = model(batch)
    return torch.softmax(output["source_node_logits"][0], dim=-1).numpy()


def run_origin_invariance_and_counterfactual(model, feature_kwargs: dict[str, Any], *, tolerance: float, network_factory, label: str) -> dict[str, Any]:
    network, graph, state = network_factory()
    node_ids = tuple(sorted(network.node_name_list))

    # Section 10: timestamp-origin translation, sparse + no-event cases.
    sparse_series = [_series(node_ids[0] if node_ids[0] in network.junction_name_list else next(n for n in node_ids if n in network.junction_name_list), (0.0, 3600.0, 7200.0), (0.0, 0.3, 0.6))]
    no_event_series = [_series(n, (0.0, 3600.0, 7200.0), (0.0, 0.0, 0.0)) for n in network.junction_name_list]
    origin_results = {}
    for case_name, series in (("sparse", sparse_series), ("no_event", no_event_series)):
        reference = _posterior(model, _build_batch(network, graph, state, series, node_ids, feature_kwargs))
        offsets = {}
        for offset_label, offset in TIMESTAMP_OFFSETS_SECONDS.items():
            shifted = [replace(s, timestamps_seconds=tuple(t + offset for t in s.timestamps_seconds)) for s in series]
            candidate = _posterior(model, _build_batch(network, graph, state, shifted, node_ids, feature_kwargs))
            max_abs_diff = float(np.abs(reference - candidate).max())
            offsets[offset_label] = {"max_abs_diff": max_abs_diff, "pass": max_abs_diff <= tolerance}
        origin_results[case_name] = offsets

    # Section 9: representation-sensitivity counterfactual (tight vs wide spacing).
    junction = next(n for n in node_ids if n in network.junction_name_list)
    counterfactual = {}
    for n_reports, tight_spacing, wide_spacing in ((2, (0.0, 600.0), (0.0, 36_000.0)), (3, (0.0, 600.0, 1_200.0), (0.0, 36_000.0, 72_000.0))):
        tight_series = [_series(junction, tight_spacing, tuple(0.1 * i for i in range(n_reports)))]
        wide_series = [_series(junction, wide_spacing, tuple(0.1 * i for i in range(n_reports)))]
        tight_posterior = _posterior(model, _build_batch(network, graph, state, tight_series, node_ids, feature_kwargs))
        wide_posterior = _posterior(model, _build_batch(network, graph, state, wide_series, node_ids, feature_kwargs))
        max_abs_diff = float(np.abs(tight_posterior - wide_posterior).max())
        counterfactual[f"N{n_reports}"] = {
            "max_abs_diff": max_abs_diff, "label": "REPRESENTATION_SENSITIVITY_COUNTERFACTUAL",
        }

    any_origin_failure = any(entry["pass"] is False for case in origin_results.values() for entry in case.values())
    return {
        "network_label": label, "origin_invariance": origin_results, "counterfactual": counterfactual,
        "any_origin_invariance_failure": any_origin_failure,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed"

    runs = _load_run_records()
    missing = [f"{arm}-seed{seed}" for arm in ARM_DEFINITIONS for seed in (SCREENING_REPRESENTATIVE_SEED, 20260814) if seed not in runs.get(arm, {})]
    if missing:
        raise RuntimeError(f"missing required M8.7 training runs before evaluation: {missing}")

    dev_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    dev_grid_network, dev_grid_names = build_grid_network(25)

    results: dict[str, Any] = {"arms": {}}
    temporal_diagnostics: dict[str, Any] = {"arms": {}}
    calibration_report: dict[str, Any] = {"arms": {}}

    for arm, definition in ARM_DEFINITIONS.items():
        feature_kwargs = definition["feature_kwargs"]
        origin_tolerance = STRUCTURAL_TOLERANCE if arm == "CURRENT_CONTROL" else ORIGIN_TOLERANCE_ARMS_BC
        seed_results: dict[str, Any] = {}
        for seed, record in runs[arm].items():
            model = _load_checkpoint(record)
            localization = run_standard_localization(model, dev_records, library, feature_kwargs)
            golden_check = run_origin_invariance_and_counterfactual(
                model, feature_kwargs, tolerance=origin_tolerance, network_factory=_golden_context, label="golden-reference",
            )
            seed_results[str(seed)] = {
                "checkpoint_sha256": record["checkpoint_sha256"], "param_count": record["model_architecture"]["param_count"],
                "standard_localization": localization, "origin_invariance_and_counterfactual": {"golden-reference": golden_check},
            }
        results["arms"][arm] = {"per_seed": seed_results}

        # Large-network (dev-grid-25) origin/counterfactual check -- one representative seed only.
        representative_record = runs[arm].get(SCREENING_REPRESENTATIVE_SEED) or next(iter(runs[arm].values()))
        representative_model = _load_checkpoint(representative_record)

        def _dev_grid_context(network=dev_grid_network):
            simulator = HydraulicSimulator(network)
            raw = simulator.calculate_state(3600)
            estimated = HydraulicStateEstimator().estimate(raw, OperationalTelemetry())
            graph = simulator.build_dynamic_graph(estimated.as_hydraulic_state())
            return network, graph, estimated

        large_network_check = run_origin_invariance_and_counterfactual(
            representative_model, feature_kwargs, tolerance=origin_tolerance, network_factory=_dev_grid_context, label="dev-grid-25",
        )
        results["arms"][arm]["large_network_check_seed"] = representative_record["seed"]
        results["arms"][arm]["large_network_check"] = large_network_check

        # Sections 7/8/11: M6-style diagnostics, representative seed only.
        junctions = tuple(sorted(build_wntr_network().junction_name_list))
        incidents = _generate_cadence_incident_pool(junctions)
        # Matches run_m6_temporal.py's own main(): the TRAIN split's hourly
        # grid (what the signature library was fit against), NOT the
        # cadence pool's own 15-minute-resolution timestamps.
        target_timestamps = build_sensor_series(train_records[0].scenario, train_records[0].feature_context)[0].timestamps_seconds
        started = time.perf_counter()
        cadence_report = run_cadence_experiment(representative_model, library, target_timestamps, incidents, feature_kwargs=feature_kwargs)
        irregular_report = run_irregular_telemetry_experiment(representative_model, library, target_timestamps, incidents, feature_kwargs=feature_kwargs)
        elapsed = time.perf_counter() - started
        temporal_diagnostics["arms"][arm] = {
            "representative_seed": representative_record["seed"], "wall_seconds": elapsed,
            "cadence_and_post_onset": cadence_report, "irregular_telemetry": irregular_report,
        }

        # Section 12: calibration, representative seed only.
        calibrator = _fit_frozen_calibrator(representative_model, library, calibration_records, feature_kwargs=feature_kwargs)
        calibration_examples = []
        with torch.no_grad():
            for depth in CAUSAL_PREFIX_DEPTHS:
                for dev_record in dev_records:
                    scenario = dev_record.scenario
                    example = scenario_to_prefix_example(scenario, dev_record.network, library, depth, feature_context=dev_record.feature_context, **feature_kwargs)
                    output = representative_model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                    probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                    truth = int(example.targets["source_node"].item())
                    full_series = build_sensor_series(scenario, dev_record.feature_context)
                    truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                    condition = classify_runtime_condition(truncated_series)
                    candidate_indices = calibrator.candidate_set(probs, condition=condition, network_id=f"m8-7-eval:{depth}")
                    calibration_examples.append({"depth": depth, "covered": truth in candidate_indices, "set_size": len(candidate_indices)})
        marginal_coverage = statistics.fmean(row["covered"] for row in calibration_examples)
        mean_set_size = statistics.fmean(row["set_size"] for row in calibration_examples)
        singleton_rate = statistics.fmean(row["set_size"] == 1 for row in calibration_examples)
        by_maturity = {}
        for label, depths in (("early", EARLY_DEPTHS), ("mid", MID_DEPTHS), ("mature", MATURE_DEPTHS)):
            subset = [row for row in calibration_examples if row["depth"] in depths]
            by_maturity[label] = {
                "coverage": statistics.fmean(row["covered"] for row in subset) if subset else None,
                "mean_set_size": statistics.fmean(row["set_size"] for row in subset) if subset else None,
            }
        calibration_report["arms"][arm] = {
            "representative_seed": representative_record["seed"], "alpha": ALPHA,
            "marginal_coverage": marginal_coverage, "mean_candidate_set_size": mean_set_size,
            "singleton_rate": singleton_rate, "by_maturity": by_maturity,
        }

    locked_after = locked_test_opened(ROOT)
    results["locked_test_opened_before"] = locked_before
    results["locked_test_opened_after"] = locked_after
    temporal_diagnostics["locked_test_opened_before"] = locked_before
    temporal_diagnostics["locked_test_opened_after"] = locked_after
    calibration_report["locked_test_opened_before"] = locked_before
    calibration_report["locked_test_opened_after"] = locked_after

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUTPUT_TEMPORAL.write_text(json.dumps(temporal_diagnostics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    OUTPUT_CALIBRATION.write_text(json.dumps(calibration_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("wrote", OUTPUT_RESULTS, OUTPUT_TEMPORAL, OUTPUT_CALIBRATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
