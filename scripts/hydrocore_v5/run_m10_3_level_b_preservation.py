"""M10.3A Strategist refit -- Level-B M9 preservation check (only run
because the Level-B competence result requires it under the frozen
protocol's own "Level B promotion additionally requires ... M9
preservation" rule).

Development-only, NEVER locked: re-evaluates the nine `TRAINED_WITH_REAL_
TARGETS` Sentinel tasks (per `HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_
AMENDMENT.md`) plus frozen-calibration coverage, comparing each Level-B
checkpoint against the SAME unmodified M9.6 teacher it was warm-started
from, on the SAME development population `run_m10_3_level_a_gate.py`
already used (golden-reference, `VALIDATION_SEED_BASE`/`VALIDATION_COUNT`)
-- cheap to rebuild here since NO Strategist candidate/WNTR-verification
work is needed for this check (pure Sentinel evaluation).

Does NOT refit calibration -- applies the frozen M9.6 B_DEPTH_AWARE
calibrator via the SAME frozen-support-refit pattern
`run_m10_1_decide.py`/the TRUE M10.2 evaluation already established.

Writes:
  reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-preservation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import m10_3_refit_protocol as proto  # noqa: E402
import m10_common as m10  # noqa: E402
from run_m7_topology import TRAINED_FAMILIES, _family_scenario_pool  # noqa: E402
from run_m9_0a_evaluate import SHARED_MODEL_CONFIG  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.causal_prefix import fit_pool_signature_library, scenario_to_prefix_example  # noqa: E402

M10_3_REFIT_DIR = m10.M10_DIR / "m10-3-refit"

CALIBRATION_ALPHA = 0.1
CALIBRATION_COVERAGE_FLOOR = 0.85
CALIBRATION_SUPPORT_PATH = "reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-calibration.jsonl"
CALIBRATION_SUPPORT_ARM = "ARM_B_M9_6"
CALIBRATION_MINIMUM_GROUP_SIZE = 10


def _fit_frozen_calibrator() -> SplitConformalCalibrator:
    path = ROOT / CALIBRATION_SUPPORT_PATH
    examples: list[CalibrationExample] = []
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            if record["arm"] != CALIBRATION_SUPPORT_ARM:
                continue
            examples.append(CalibrationExample(
                probabilities=tuple(record["probabilities"]), true_index=record["true_index"],
                condition=record["condition"], network_id=f"{record['family']}:{record['depth_bucket']}",
            ))
    return SplitConformalCalibrator.fit(
        examples, alpha=CALIBRATION_ALPHA, model_hash="m9-6-arm-b-frozen-S", feature_schema_hash="m9-6-frozen",
        dataset_manifest_hash="m9-6-canonical-calibration", minimum_group_size=CALIBRATION_MINIMUM_GROUP_SIZE,
    )


def _paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, *, resamples: int, ci: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(a[idx]) - np.mean(b[idx]))
    tail = (1.0 - ci) / 2.0
    return float(np.percentile(diffs, tail * 100)), float(np.percentile(diffs, (1.0 - tail) * 100))


def _predict_sentinel(model: HydroCore, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    with torch.no_grad():
        output = model(inputs)
    return {
        "source_node_probs": torch.softmax(output["source_node_logits"][0], dim=-1).numpy(),
        "source_region_pred": int(torch.argmax(output["source_region_logits"][0])),
        "start_time_pred": int(torch.argmax(output["start_time_logits"][0])),
        "duration_pred": int(torch.argmax(output["duration_logits"][0])),
        "relative_strength_pred": int(torch.argmax(output["relative_strength_logits"][0])),
        "event_presence_pred": float(torch.sigmoid(output["event_presence_logits"][0])) if output["event_presence_logits"].dim() == 1 else float(torch.sigmoid(output["event_presence_logits"][0])),
        "event_cause_pred": int(torch.argmax(output["event_cause_logits"][0])),
        "sensor_fault_probs": torch.sigmoid(output["sensor_fault_logits"][0]).numpy(),
        "evidence_sufficiency_pred": float(output["evidence_sufficiency"][0]),
    }


def _evaluate_model(model: HydroCore, examples: list[tuple[dict, dict]], node_ids_per_example: list[tuple[str, ...]]) -> dict[str, Any]:
    metrics: dict[str, list] = {name: [] for name in (
        "source_node_correct", "source_region_correct", "start_time_correct", "duration_correct",
        "relative_strength_correct", "event_presence_correct", "event_cause_correct", "sensor_fault_correct",
        "evidence_sufficiency_abs_error",
    )}
    calibration_probs: list[np.ndarray] = []
    calibration_true_index: list[int] = []

    for (inputs, targets), node_ids in zip(examples, node_ids_per_example):
        pred = _predict_sentinel(model, inputs)
        true_source_index = int(targets["source_node"])
        source_top1 = int(np.argmax(pred["source_node_probs"])) == true_source_index
        if bool(targets["source_node_mask"]):
            metrics["source_node_correct"].append(source_top1)
            calibration_probs.append(pred["source_node_probs"])
            calibration_true_index.append(true_source_index)
        if bool(targets["source_region_mask"]):
            metrics["source_region_correct"].append(pred["source_region_pred"] == int(targets["source_region"]))
        if bool(targets["start_time_mask"]):
            metrics["start_time_correct"].append(pred["start_time_pred"] == int(targets["start_time"]))
        if bool(targets["duration_mask"]):
            metrics["duration_correct"].append(pred["duration_pred"] == int(targets["duration"]))
        if bool(targets["relative_strength_mask"]):
            metrics["relative_strength_correct"].append(pred["relative_strength_pred"] == int(targets["relative_strength"]))
        metrics["event_presence_correct"].append(
            int(pred["event_presence_pred"] >= 0.5) == int(targets["event_presence"])
        )
        metrics["event_cause_correct"].append(pred["event_cause_pred"] == int(targets["event_cause"]))
        if bool(targets["sensor_fault_mask"].any()):
            fault_target = targets["sensor_fault"].numpy()
            fault_mask = targets["sensor_fault_mask"].numpy().astype(bool)
            fault_pred = (pred["sensor_fault_probs"] >= 0.5).astype(int)
            metrics["sensor_fault_correct"].extend((fault_pred[fault_mask] == fault_target[fault_mask].astype(int)).tolist())
        metrics["evidence_sufficiency_abs_error"].append(
            abs(pred["evidence_sufficiency_pred"] - float(targets["evidence_sufficiency"]))
        )

    summary = {name: (float(np.mean(values)) if values else None) for name, values in metrics.items()}
    summary["n_per_metric"] = {name: len(values) for name, values in metrics.items()}
    return {"summary": summary, "raw": metrics, "calibration_probs": calibration_probs, "calibration_true_index": calibration_true_index}


def main() -> None:
    locked_before = m10.assert_locked_test_closed()
    family, loader = TRAINED_FAMILIES[0]

    print("rebuilding VALIDATION population (Sentinel-only, no WNTR verification needed)...", flush=True)
    pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=proto.VALIDATION_SEED_BASE,
        count=proto.VALIDATION_COUNT, source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    train_pool = _family_scenario_pool(
        "train", network_loader=loader, family=family, seed_base=proto.TRAIN_SEED_BASE, count=proto.TRAIN_COUNT,
        source_round_robin=proto.SOURCE_ROUND_ROBIN,
    )
    input_library = fit_pool_signature_library(train_pool)

    examples: list[tuple[dict, dict]] = []
    node_ids_per_example: list[tuple[str, ...]] = []
    for record in pool:
        example = scenario_to_prefix_example(
            record.scenario, record.network, input_library, proto.DEPTH, feature_context=record.feature_context,
        )
        inputs = {key: value.unsqueeze(0) for key, value in example.inputs.items()}
        examples.append((inputs, dict(example.targets)))
        node_ids_per_example.append(example.topology.node_ids if example.topology else ())
    print(f"built {len(examples)} Sentinel-only evaluation examples", flush=True)

    calibrator = _fit_frozen_calibrator()
    network_id_key = f"{proto.FAMILY}:{m10.depth_bucket_of(proto.DEPTH)}"

    per_seed: dict[str, Any] = {}
    for seed in m10.SEEDS:
        print(f"=== M9 preservation check, seed {seed} ===", flush=True)
        teacher_record = m10.canonical_s_checkpoint(seed)
        teacher_model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
        teacher_model.load_state_dict(load_file(teacher_record["canonical_export_path"], device="cpu"), strict=True)
        teacher_model.eval()

        level_b_checkpoint = M10_3_REFIT_DIR / "checkpoints" / f"level-b-seed{seed}" / "model.safetensors"
        level_b_model = HydroCore.from_variant(m10.S_VARIANT, use_adapters=False, **SHARED_MODEL_CONFIG)
        level_b_model.load_state_dict(load_file(level_b_checkpoint, device="cpu"), strict=True)
        level_b_model.eval()

        teacher_eval = _evaluate_model(teacher_model, examples, node_ids_per_example)
        level_b_eval = _evaluate_model(level_b_model, examples, node_ids_per_example)

        teacher_coverage = np.mean([
            true_idx in calibrator.candidate_set(probs, condition=None, network_id=network_id_key)
            for probs, true_idx in zip(teacher_eval["calibration_probs"], teacher_eval["calibration_true_index"])
        ])
        level_b_coverage = np.mean([
            true_idx in calibrator.candidate_set(probs, condition=None, network_id=network_id_key)
            for probs, true_idx in zip(level_b_eval["calibration_probs"], level_b_eval["calibration_true_index"])
        ])

        regressions: dict[str, Any] = {}
        for metric_name in ("source_node_correct", "source_region_correct", "start_time_correct", "duration_correct",
                             "relative_strength_correct", "event_presence_correct", "event_cause_correct",
                             "sensor_fault_correct"):
            teacher_values = np.array(teacher_eval["raw"][metric_name], dtype=float)
            level_b_values = np.array(level_b_eval["raw"][metric_name], dtype=float)
            if len(teacher_values) == 0 or len(level_b_values) == 0 or len(teacher_values) != len(level_b_values):
                regressions[metric_name] = {"comparable": False}
                continue
            diff_lower, diff_upper = _paired_bootstrap_ci(
                level_b_values, teacher_values, resamples=proto.BOOTSTRAP_RESAMPLES, ci=proto.BOOTSTRAP_CI,
                seed=proto.BOOTSTRAP_SEED,
            )
            regressions[metric_name] = {
                "comparable": True,
                "teacher_mean": float(teacher_values.mean()),
                "level_b_mean": float(level_b_values.mean()),
                "diff_ci_lower": diff_lower, "diff_ci_upper": diff_upper,
                "ci_confident_regression": bool(diff_upper < 0.0),
            }

        calibration_regressed = bool(level_b_coverage < CALIBRATION_COVERAGE_FLOOR)
        any_ci_confident_regression = any(
            entry.get("ci_confident_regression") for entry in regressions.values() if entry.get("comparable")
        )

        per_seed[str(seed)] = {
            "teacher_metrics": teacher_eval["summary"],
            "level_b_metrics": level_b_eval["summary"],
            "teacher_calibration_coverage": float(teacher_coverage),
            "level_b_calibration_coverage": float(level_b_coverage),
            "calibration_coverage_floor": CALIBRATION_COVERAGE_FLOOR,
            "calibration_below_floor": calibration_regressed,
            "per_metric_regression": regressions,
            "any_ci_confident_regression": any_ci_confident_regression,
            "m9_preservation_passed": bool(not any_ci_confident_regression and not calibration_regressed),
        }

    locked_after = m10.assert_locked_test_closed()
    doc = {
        "kind": "M10_3_REFIT_LEVEL_B_PRESERVATION",
        "protocol_hash": proto.protocol_hash(),
        "n_examples": len(examples),
        "per_seed": per_seed,
        "all_seeds_preserve_m9": all(entry["m9_preservation_passed"] for entry in per_seed.values()),
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
    }
    (M10_3_REFIT_DIR / "m10-3-refit-preservation.json").write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(json.dumps({seed: entry["m9_preservation_passed"] for seed, entry in per_seed.items()}, indent=2))


if __name__ == "__main__":
    main()
