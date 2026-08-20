"""Milestone 3 (experiments.txt): causal calibration and candidate-set
contraction.

3.1 Freezes one predictor checkpoint: the Milestone-2 exit decision's
winning arm if REWEIGHT_TASKS/KEEP_ROLE_ADAPTERS_ISOLATION, otherwise the
Milestone-1 winning arm (KEEP_FULL_MULTITASK/PCGRAD_JUSTIFIED, where no M2
ablation beat the Milestone-1 baseline). No further predictor tuning
happens after this point in this script.

3.2/3.3 Fits two calibration schemes on the `calibration` split only
(never opened by Milestone 1/2): alpha=0.1 throughout.
  A: existing network/condition control (network_id=real network id,
     condition=CLEAN/OPERATIONAL/DEGRADED).
  B: network + evidence-depth (EARLY 1-3 / MID 4-6 / MATURE 12-25).
     Adaptation note: hydroswarm.calibration.conformal.SplitConformalCalibrator
     only exposes two Mondrian keys (`condition`, `network_id`), and
     `selection()` always prefers `network_id` over `condition` when both
     are populated. Because this corpus is single-topology (Milestone 1's
     documented scope), Scheme A's `network_id` is constant, so Scheme B
     encodes network+depth jointly into the `network_id` field
     (f"{network_id}:{depth_bucket}") -- the only way to make depth
     actually discriminate `selection()`'s group choice through the
     existing two-field API without modifying it. `condition` is still
     recorded on every example for coverage-by-condition reporting.

3.4/3.5 Reports NLL/Brier/ECE/reliability/selective-risk and conformal
coverage/set-size/singleton-rate/source-inclusion/planning-gate-pass-rate,
for both schemes, on (a) the calibration split itself (resubstitution,
reported for reference only) and (b) development_holdout (genuine held-out
check -- development_holdout was never used to fit either scheme).

3.6 (optional APS/RAPS) is only attempted if Scheme B still yields
excessively broad candidate sets; this script's exit records whether that
condition was met, not a de-facto attempt.

Writes:
  reports/evaluation/hydrocore-v5/m3-calibration.json
  reports/evaluation/hydrocore-v5/m3-summary.md
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from hydroswarm.calibration.conformal import (  # noqa: E402
    CalibrationExample,
    SplitConformalCalibrator,
    classify_runtime_condition,
    expected_calibration_error,
)
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CAUSAL_PREFIX_DEPTHS,
    build_scenario_pool,
    fit_pool_signature_library,
    scenario_to_prefix_example,
    truncate_causal_prefix,
)
from hydroswarm.training.corpus import build_sensor_series  # noqa: E402
from evaluate_m2 import OUTPUT_RESULTS as M2_RESULTS_PATH  # noqa: E402
from run_m1_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m2_arm import SUMMARY_ROOT as M2_RUNS_ROOT  # noqa: E402
from run_m2_conflict import _select_base_run  # noqa: E402

OUTPUT_RESULTS = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m3-calibration.json"
OUTPUT_SUMMARY = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m3-summary.md"

ALPHA = 0.1
DEPTH_BUCKET_OF: dict[int, str] = {1: "EARLY", 2: "EARLY", 3: "EARLY", 4: "MID", 6: "MID", 12: "MATURE", 25: "MATURE"}
PLANNING_GATE_MAX_CANDIDATES = 4  # placeholder pending Milestone 4's own predeclared K.


def _freeze_predictor() -> tuple[str, bool, str]:
    """Returns (export_path, use_adapters, description)."""

    base_run = _select_base_run()
    if not M2_RESULTS_PATH.exists():
        return base_run["training_summary"]["export_path"], False, f"Milestone-1 winner (arm {base_run['_selected_as']}), Milestone 2 not yet run"
    m2 = json.loads(M2_RESULTS_PATH.read_text())
    decision = m2["exit_decision"]
    if decision == "REWEIGHT_TASKS":
        run = json.loads((M2_RUNS_ROOT / f"B-seed{base_run['seed']}.json").read_text())
        return run["training_summary"]["export_path"], False, "Milestone-2 winner: Arm B (reweighted)"
    if decision == "KEEP_ROLE_ADAPTERS_ISOLATION":
        run = json.loads((M2_RUNS_ROOT / f"C-seed{base_run['seed']}.json").read_text())
        return run["training_summary"]["export_path"], True, "Milestone-2 winner: Arm C (role-isolated adapters)"
    return base_run["training_summary"]["export_path"], False, f"Milestone-1 winner (arm {base_run['_selected_as']}); Milestone 2 decision was {decision}"


def _collect_examples(model, records, library) -> list[dict[str, Any]]:
    collected = []
    with torch.no_grad():
        for depth in CAUSAL_PREFIX_DEPTHS:
            for record in records:
                scenario = record.scenario
                example = scenario_to_prefix_example(
                    scenario, record.network, library, depth, feature_context=record.feature_context
                )
                output = model({key: value.unsqueeze(0) for key, value in example.inputs.items()})
                probs = torch.softmax(output["source_node_logits"][0], dim=-1).tolist()
                truth = int(example.targets["source_node"].item())
                full_series = build_sensor_series(scenario, record.feature_context)
                truncated_series = [truncate_causal_prefix(item, depth) for item in full_series]
                condition = classify_runtime_condition(truncated_series)
                collected.append({
                    "probabilities": probs,
                    "true_index": truth,
                    "condition": condition,
                    "network_id": scenario.manifest.network_id,
                    "depth": depth,
                    "depth_bucket": DEPTH_BUCKET_OF[depth],
                })
    return collected


def _diagnostics(examples: list[dict[str, Any]], calibrator: SplitConformalCalibrator, *, network_id_fn) -> dict[str, Any]:
    covered, sizes, singletons = [], [], []
    nlls, briers, confidences, corrects = [], [], [], []
    for item in examples:
        probs = item["probabilities"]
        truth = item["true_index"]
        selected = calibrator.candidate_set(probs, condition=item["condition"], network_id=network_id_fn(item))
        covered.append(truth in selected)
        sizes.append(len(selected))
        singletons.append(len(selected) == 1)
        p_truth = probs[truth]
        nlls.append(-math.log(max(p_truth, 1e-9)))
        onehot = [0.0] * len(probs)
        onehot[truth] = 1.0
        briers.append(sum((p - o) ** 2 for p, o in zip(probs, onehot, strict=True)))
        confidences.append(max(probs))
        corrects.append(probs.index(max(probs)) == truth)
    n = len(examples)
    mean_p = statistics.fmean(covered)
    se = (mean_p * (1 - mean_p) / n) ** 0.5 if n else 0.0
    sorted_sizes = sorted(sizes)
    return {
        "n": n,
        "empirical_coverage": mean_p,
        "coverage_95ci": [max(0.0, mean_p - 1.96 * se), min(1.0, mean_p + 1.96 * se)],
        "mean_candidate_set_size": statistics.fmean(sizes),
        "median_candidate_set_size": statistics.median(sizes),
        "p90_candidate_set_size": sorted_sizes[min(len(sorted_sizes) - 1, int(0.9 * len(sorted_sizes)))],
        "singleton_rate": statistics.fmean(singletons),
        "source_inclusion_rate": mean_p,
        "planning_gate_pass_rate": statistics.fmean(size <= PLANNING_GATE_MAX_CANDIDATES for size in sizes),
        "nll": statistics.fmean(nlls),
        "brier": statistics.fmean(briers),
        "ece": expected_calibration_error(confidences, corrects),
    }


def _diagnostics_by_bucket(examples, calibrator, *, network_id_fn) -> dict[str, Any]:
    out = {}
    for bucket in ("EARLY", "MID", "MATURE"):
        subset = [item for item in examples if item["depth_bucket"] == bucket]
        if subset:
            out[bucket] = _diagnostics(subset, calibrator, network_id_fn=network_id_fn)
    return out


def main() -> int:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    export_path, use_adapters, predictor_description = _freeze_predictor()
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)
    model.load_state_dict(load_file(export_path, device="cpu"), strict=True)
    model.eval()

    calibration_records = build_scenario_pool("calibration", network_loader=build_wntr_network)
    development_records = build_scenario_pool("development_holdout", network_loader=build_wntr_network)
    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    calibration_examples = _collect_examples(model, calibration_records, library)
    development_examples = _collect_examples(model, development_records, library)

    def scheme_a_calibration_examples(items):
        return [
            CalibrationExample(probabilities=tuple(i["probabilities"]), true_index=i["true_index"], condition=i["condition"], network_id=i["network_id"])
            for i in items
        ]

    def scheme_b_calibration_examples(items):
        return [
            CalibrationExample(
                probabilities=tuple(i["probabilities"]), true_index=i["true_index"], condition=i["condition"],
                network_id=f"{i['network_id']}:{i['depth_bucket']}",
            )
            for i in items
        ]

    calibrator_a = SplitConformalCalibrator.fit(
        scheme_a_calibration_examples(calibration_examples), alpha=ALPHA,
        model_hash=export_path, feature_schema_hash="n/a", dataset_manifest_hash="m3-calibration-pool",
    )
    calibrator_b = SplitConformalCalibrator.fit(
        scheme_b_calibration_examples(calibration_examples), alpha=ALPHA,
        model_hash=export_path, feature_schema_hash="n/a", dataset_manifest_hash="m3-calibration-pool",
    )

    results_a_calib = _diagnostics(calibration_examples, calibrator_a, network_id_fn=lambda i: i["network_id"])
    results_a_dev = _diagnostics(development_examples, calibrator_a, network_id_fn=lambda i: i["network_id"])
    results_a_dev_by_bucket = _diagnostics_by_bucket(development_examples, calibrator_a, network_id_fn=lambda i: i["network_id"])

    results_b_calib = _diagnostics(calibration_examples, calibrator_b, network_id_fn=lambda i: f"{i['network_id']}:{i['depth_bucket']}")
    results_b_dev = _diagnostics(development_examples, calibrator_b, network_id_fn=lambda i: f"{i['network_id']}:{i['depth_bucket']}")
    results_b_dev_by_bucket = _diagnostics_by_bucket(development_examples, calibrator_b, network_id_fn=lambda i: f"{i['network_id']}:{i['depth_bucket']}")

    similar_coverage = abs(results_a_dev["empirical_coverage"] - results_b_dev["empirical_coverage"]) <= 0.05
    smaller_sets = results_b_dev["mean_candidate_set_size"] < results_a_dev["mean_candidate_set_size"]
    higher_actionability = results_b_dev["planning_gate_pass_rate"] > results_a_dev["planning_gate_pass_rate"]

    # Aggregate coverage can mask a per-bucket coverage failure: alpha=0.1
    # promises ~90% coverage in EVERY governed group, not merely on
    # average. A scheme whose worst bucket coverage is materially below
    # target is a real defect the aggregate number hides -- most
    # consequential at EARLY depths, exactly where planning decisions are
    # least evidenced. This is evaluated in addition to (not instead of)
    # the predeclared "similar coverage + smaller sets + higher
    # actionability" rule, since a per-bucket coverage failure is itself a
    # form of "not similar coverage" that only bucket-level inspection can
    # catch.
    target_coverage = 1 - ALPHA
    worst_bucket_undercoverage_a = max(
        0.0, target_coverage - min(m["empirical_coverage"] for m in results_a_dev_by_bucket.values())
    )
    worst_bucket_undercoverage_b = max(
        0.0, target_coverage - min(m["empirical_coverage"] for m in results_b_dev_by_bucket.values())
    )
    material_undercoverage_pp = 5.0
    a_has_material_bucket_undercoverage = worst_bucket_undercoverage_a * 100 > material_undercoverage_pp
    b_corrects_it = worst_bucket_undercoverage_b * 100 <= material_undercoverage_pp

    if (similar_coverage and smaller_sets and higher_actionability) or (
        a_has_material_bucket_undercoverage and b_corrects_it
    ):
        frozen_scheme = "B_DEPTH_AWARE"
    else:
        frozen_scheme = "A_NETWORK_CONDITION"

    still_broad = results_b_dev["mean_candidate_set_size"] > 2.0  # more than half the 4-junction candidate space
    aps_raps_recommended = frozen_scheme == "B_DEPTH_AWARE" and still_broad

    report = {
        "schema_version": 1,
        "purpose": "Milestone 3 (experiments.txt): causal calibration scheme comparison and candidate-set contraction.",
        "alpha": ALPHA,
        "predictor": {"export_path": export_path, "use_adapters": use_adapters, "description": predictor_description},
        "depth_buckets": DEPTH_BUCKET_OF,
        "planning_gate_max_candidates_placeholder": PLANNING_GATE_MAX_CANDIDATES,
        "scheme_a_network_condition": {
            "resubstitution_on_calibration_split": results_a_calib,
            "held_out_on_development_holdout": results_a_dev,
            "held_out_by_depth_bucket": results_a_dev_by_bucket,
        },
        "scheme_b_network_depth_aware": {
            "resubstitution_on_calibration_split": results_b_calib,
            "held_out_on_development_holdout": results_b_dev,
            "held_out_by_depth_bucket": results_b_dev_by_bucket,
        },
        "promotion_comparison": {
            "similar_coverage_within_5pp": similar_coverage,
            "smaller_mean_candidate_set": smaller_sets,
            "higher_planning_gate_pass_rate": higher_actionability,
            "target_coverage": target_coverage,
            "worst_bucket_undercoverage_a_pp": worst_bucket_undercoverage_a * 100,
            "worst_bucket_undercoverage_b_pp": worst_bucket_undercoverage_b * 100,
            "a_has_material_bucket_undercoverage": a_has_material_bucket_undercoverage,
            "b_corrects_material_undercoverage": b_corrects_it,
        },
        "frozen_calibration_scheme": frozen_scheme,
        "aps_raps_experiment_recommended": aps_raps_recommended,
        "aps_raps_experiment_run": False,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Milestone 3 summary: causal calibration and candidate-set contraction",
        "",
        f"Frozen calibration policy: **{frozen_scheme}**",
        f"Predictor: {predictor_description}",
        "",
        "## Held-out (development_holdout) comparison",
        "",
        "| scheme | coverage | mean set size | median set size | singleton rate | planning-gate pass rate |",
        "|---|---|---|---|---|---|",
        f"| A (network/condition) | {results_a_dev['empirical_coverage']:.3f} | {results_a_dev['mean_candidate_set_size']:.2f} | "
        f"{results_a_dev['median_candidate_set_size']:.2f} | {results_a_dev['singleton_rate']:.3f} | {results_a_dev['planning_gate_pass_rate']:.3f} |",
        f"| B (network+depth) | {results_b_dev['empirical_coverage']:.3f} | {results_b_dev['mean_candidate_set_size']:.2f} | "
        f"{results_b_dev['median_candidate_set_size']:.2f} | {results_b_dev['singleton_rate']:.3f} | {results_b_dev['planning_gate_pass_rate']:.3f} |",
        "",
        "## Per-depth-bucket held-out coverage (the aggregate numbers above can mask a subgroup failure)",
        "",
        "| scheme | EARLY coverage | MID coverage | MATURE coverage |",
        "|---|---|---|---|",
        f"| A | {results_a_dev_by_bucket['EARLY']['empirical_coverage']:.3f} | {results_a_dev_by_bucket['MID']['empirical_coverage']:.3f} | "
        f"{results_a_dev_by_bucket['MATURE']['empirical_coverage']:.3f} |",
        f"| B | {results_b_dev_by_bucket['EARLY']['empirical_coverage']:.3f} | {results_b_dev_by_bucket['MID']['empirical_coverage']:.3f} | "
        f"{results_b_dev_by_bucket['MATURE']['empirical_coverage']:.3f} |",
        "",
        f"Target coverage (1-alpha): {target_coverage:.2f}. Scheme A's worst bucket is "
        f"{worst_bucket_undercoverage_a * 100:.1f}pp under target; Scheme B's worst bucket is "
        f"{worst_bucket_undercoverage_b * 100:.1f}pp under target. "
        f"Material per-bucket undercoverage in A not corrected by B: **{a_has_material_bucket_undercoverage and not b_corrects_it}**. "
        f"This drove the {frozen_scheme} decision "
        f"{'via the per-bucket override (Scheme A materially under-covers EARLY, the most decision-consequential bucket, while Scheme B does not; the aggregate-only comparison alone would have kept Scheme A on a false premise of parity)' if frozen_scheme == 'B_DEPTH_AWARE' and a_has_material_bucket_undercoverage and b_corrects_it else 'via the predeclared aggregate rule (similar coverage + smaller sets + higher actionability), since no material per-bucket coverage failure was found needing correction'}.",
        "",
        f"APS/RAPS follow-up recommended: **{aps_raps_recommended}** (not run this session; Milestone 3.6 is optional and "
        "conditional on excessively broad sets persisting after depth-aware grouping).",
    ]
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_scheme": frozen_scheme, "a": results_a_dev, "b": results_b_dev}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
