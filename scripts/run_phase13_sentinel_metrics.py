"""core-issues3.txt Phase 13: required Sentinel metrics not yet computed anywhere.

Gap this closes (see reports/results/v4/phase13-metrics-and-baselines.md for the
full accounting): the Stage-A checkpoint (scripts/run_stage3_finalist_training.py)
never constructs event_presence_head/event_cause_head (event_control_heads=False
by default) and no script anywhere computes classification-quality metrics
(precision/recall/F1) for event_presence, event_cause, or sensor_fault, or
ordinal accuracy/error for the profile heads (start_time/duration/
relative_strength) -- only raw training losses exist for these in
reports/results/v4/stage-f-adapters-comparison.json's per_task_mean_loss.

Uses the already-trained, already-on-disk Stage-F finalist checkpoints (both
have event_control_heads=True; see scripts/run_stage_f_training.py's
SHARED_MODEL_CONFIG) evaluated on validation and development_holdout only --
no locked test access, no retraining, no new corpus generation. Also reports
a v4-architecture localization/calibration-style cross-check (source_top1/
top3/mrr) on the same splits, since that number has never been measured
against a Stage-F checkpoint before (only against the plain Stage-A
checkpoint).

Known, inherited data-quality caveat (documented previously in this file's
Phase 8 section of the handoff, restated here because it directly bears on
this script's event_cause numbers): data/learning-v2/cycle-b2 -- which
cycle-b2-joint-v4 folds in unchanged, being a protected immutable artifact --
still contains ~5% HYDRAULIC_MISMATCH-mislabeled event_cause examples from
before the Phase 6.4 fix. This script reports that caveat inline; it does not
attempt to filter or correct the affected rows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from safetensors.torch import load_file
import torch

from hydroswarm.classical.metrics import candidate_set_metrics, localization_top_k, mean_reciprocal_rank
from hydroswarm.calibration.conformal import expected_calibration_error
from hydroswarm.model import HydroCore
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology
from hydroswarm.training.corpus import EVENT_CAUSE_INDEX

ROOT = Path(__file__).resolve().parents[1]
JOINT_CORPUS_ROOT = Path("data/learning-v2/cycle-b2-joint-v4/tensors-normalized")
BATCH_SIZE = 16

#: Matches scripts/run_stage_f_training.py's SHARED_MODEL_CONFIG exactly --
#: this script only ever reads pre-trained Stage-F checkpoints, so the
#: constructed model's architecture must match bit-for-bit or strict
#: load_state_dict fails closed (the intended behavior on any drift).
SHARED_MODEL_CONFIG: dict[str, Any] = {
    "prior_mode": "feature_only",
    "event_control_heads": True,
    "scout_control_heads": True,
    "strategist_mode": "candidate_conditioned",
    "action_vocabulary_size": 9,
    "consequence_prescreening_heads": True,
    "ood_category_head": True,
}

EVENT_CAUSE_NAMES = [cause.value for cause, _ in sorted(EVENT_CAUSE_INDEX.items(), key=lambda item: item[1])]
PROFILE_TARGETS: dict[str, int] = {"start_time": 4, "duration": 3, "relative_strength": 3}


#: The ood-SEVERE_MISSINGNESS/ood-UNSEEN_TOPOLOGY populations (and, more
#: generally, any population outside {train,validation,calibration,
#: development_holdout}) structurally never carry Strategist candidate-plan
#: fields (plan_template_ids/plan_target_type/plan_mask/plan_features) --
#: HydroCore.forward's own strategist_mode="candidate_conditioned" check
#: requires those fields batch-wide, so evaluating these populations needs
#: a model built WITHOUT strategist_mode, loaded strict=False against the
#: same checkpoint (only the Strategist-specific parameters go unused; see
#: scripts/run_stage_f_joint_corpus_gates.py's OOD_CLASS_MODEL_OVERRIDES
#: for the identical, previously-established pattern this mirrors).
#: action_vocabulary_size is kept: the legacy action_head is sized by it
#: unconditionally (independent of strategist_mode), so dropping it back to
#: HydroCore's own default would size-mismatch against the real 9-template
#: Stage-F checkpoint's action_head weights (see
#: scripts/run_phase13_ood_control_metrics.py's identical comment).
NO_STRATEGIST_MODEL_CONFIG: dict[str, Any] = {
    key: value for key, value in SHARED_MODEL_CONFIG.items() if key not in {"strategist_mode", "consequence_prescreening_heads"}
}


def load_model(checkpoint_path: Path, *, use_adapters: bool, strategist_fields_available: bool) -> HydroCore:
    config = SHARED_MODEL_CONFIG if strategist_fields_available else NO_STRATEGIST_MODEL_CONFIG
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **config)
    state_dict = load_file(str(checkpoint_path), device="cpu")
    if strategist_fields_available:
        model.load_state_dict(state_dict, strict=True)
    else:
        report = model.load_state_dict(state_dict, strict=False)
        if report.missing_keys:
            raise RuntimeError(f"{checkpoint_path}: unexpected missing_keys when loading non-Strategist model: {report.missing_keys}")
    model.eval()
    return model


def _load_split(name: str) -> ShardedScenarioDataset:
    dataset = ShardedScenarioDataset(JOINT_CORPUS_ROOT / name, expected_split=name if not name.startswith("ood-") else "development_holdout")
    dataset.verify_shard_checksums()
    return dataset


def _binary_prf1(predictions: list[bool], truths: list[bool]) -> dict[str, float]:
    tp = sum(1 for p, t in zip(predictions, truths) if p and t)
    fp = sum(1 for p, t in zip(predictions, truths) if p and not t)
    fn = sum(1 for p, t in zip(predictions, truths) if not p and t)
    tn = sum(1 for p, t in zip(predictions, truths) if not p and not t)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "support": len(predictions), "positive_support": tp + fn}


def _macro_multiclass(predictions: list[int], truths: list[int], class_names: list[str]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for class_index, name in enumerate(class_names):
        pred_bool = [p == class_index for p in predictions]
        truth_bool = [t == class_index for t in truths]
        stats = _binary_prf1(pred_bool, truth_bool)
        per_class[name] = stats
        # Only include classes with real support in the macro average --
        # an entirely-unsupported class's undefined F1=0.0 would otherwise
        # silently drag the macro average down for a reason unrelated to
        # model quality (see UNSUPPORTED_OOD_CATEGORIES for the analogous
        # OOD convention, reused here for event_cause).
        if stats["support"] and stats["positive_support"]:
            f1s.append(stats["f1"])
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    accuracy = sum(1 for p, t in zip(predictions, truths) if p == t) / len(predictions) if predictions else 0.0
    return {"macro_f1": macro_f1, "accuracy": accuracy, "per_class": per_class, "examples": len(predictions)}


@torch.no_grad()
def evaluate_split(model: HydroCore, dataset: ShardedScenarioDataset, *, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    latencies: list[float] = []

    localization_predictions: list[dict[int, float]] = []
    localization_truths: list[int] = []
    localization_confidences: list[float] = []
    localization_correct: list[bool] = []

    event_presence_pred: list[bool] = []
    event_presence_truth: list[bool] = []

    event_cause_pred: list[int] = []
    event_cause_truth: list[int] = []

    profile_pred: dict[str, list[int]] = {name: [] for name in PROFILE_TARGETS}
    profile_truth: dict[str, list[int]] = {name: [] for name in PROFILE_TARGETS}

    sensor_fault_pred: list[bool] = []
    sensor_fault_truth: list[bool] = []

    total = len(dataset)
    for start in range(0, total, batch_size):
        batch_examples = [dataset[index] for index in range(start, min(start + batch_size, total))]
        inputs, targets = collate_variable_topology(batch_examples)
        started = time.perf_counter()
        output = model(inputs)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed / len(batch_examples))
        peak_rss = max(peak_rss, process.memory_info().rss)

        # --- localization cross-check ---
        probabilities = torch.softmax(output["source_node_logits"], dim=-1)
        source_mask = targets.get("source_node_mask")
        for row in range(probabilities.shape[0]):
            if source_mask is not None and not bool(source_mask[row]):
                continue
            row_probs = {position: float(value) for position, value in enumerate(probabilities[row].numpy()) if value > 0}
            truth = int(targets["source_node"][row].item())
            localization_predictions.append(row_probs)
            localization_truths.append(truth)
            localization_correct.append(bool(truth in row_probs and localization_top_k(row_probs, truth, k=1)))
            localization_confidences.append(max(row_probs.values()) if row_probs else 0.0)

        # --- event_presence (never masked) ---
        if "event_presence" in targets:
            presence_pred = (torch.sigmoid(output["event_presence_logits"]) >= 0.5).tolist()
            presence_truth = targets["event_presence"].bool().tolist()
            event_presence_pred.extend(presence_pred)
            event_presence_truth.extend(presence_truth)

        # --- event_cause (never masked) ---
        if "event_cause" in targets:
            cause_pred = torch.argmax(output["event_cause_logits"], dim=-1).tolist()
            cause_truth = targets["event_cause"].long().tolist()
            event_cause_pred.extend(cause_pred)
            event_cause_truth.extend(cause_truth)

        # --- profile heads (masked: event_presence & event_cause==CONTAMINATION) ---
        for name in PROFILE_TARGETS:
            mask_key = f"{name}_mask"
            if name not in targets or mask_key not in targets:
                continue
            mask = targets[mask_key].bool()
            if not bool(mask.any()):
                continue
            logits = output[f"{name}_logits"]
            preds = torch.argmax(logits, dim=-1)[mask].tolist()
            truths = targets[name].long()[mask].tolist()
            profile_pred[name].extend(preds)
            profile_truth[name].extend(truths)

        # --- sensor_fault (masked per-node) ---
        if "sensor_fault" in targets and "sensor_fault_mask" in targets:
            mask = targets["sensor_fault_mask"].bool()
            if bool(mask.any()):
                fault_pred = (torch.sigmoid(output["sensor_fault_logits"]) >= 0.5)[mask].tolist()
                fault_truth = targets["sensor_fault"].bool()[mask].tolist()
                sensor_fault_pred.extend(fault_pred)
                sensor_fault_truth.extend(fault_truth)

    result: dict[str, Any] = {
        "examples": total,
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency_seconds": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_seconds": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "peak_rss_bytes": int(peak_rss),
    }

    if localization_truths:
        coverage = candidate_set_metrics(
            (sorted(pred, key=lambda k: -pred[k])[:3] for pred in localization_predictions), localization_truths
        )
        result["localization"] = {
            "source_top1": float(np.mean(localization_correct)),
            "source_top3": float(
                np.mean(
                    [localization_top_k(pred, truth, k=3) for pred, truth in zip(localization_predictions, localization_truths)]
                )
            ),
            "mrr": mean_reciprocal_rank(localization_predictions, localization_truths),
            "candidate_coverage_at_3": coverage.coverage,
            "ece": expected_calibration_error(localization_confidences, localization_correct),
            "examples": len(localization_truths),
        }

    if event_presence_truth:
        result["event_presence"] = _binary_prf1(event_presence_pred, event_presence_truth)

    if event_cause_truth:
        result["event_cause"] = _macro_multiclass(event_cause_pred, event_cause_truth, EVENT_CAUSE_NAMES)
        result["event_cause"]["known_data_quality_caveat"] = (
            "data/learning-v2/cycle-b2 (folded unchanged into cycle-b2-joint-v4) contains "
            "~5% (633/12750) HYDRAULIC_MISMATCH-mislabeled event_cause examples predating "
            "the Phase 6.4 fix; this checkpoint's event_cause head was trained on that data."
        )

    profile_result: dict[str, Any] = {}
    for name in PROFILE_TARGETS:
        if not profile_truth[name]:
            continue
        preds = profile_pred[name]
        truths = profile_truth[name]
        accuracy = sum(1 for p, t in zip(preds, truths) if p == t) / len(preds)
        ordinal_mae = float(np.mean([abs(p - t) for p, t in zip(preds, truths)]))
        profile_result[name] = {"accuracy": accuracy, "ordinal_mean_absolute_error_bins": ordinal_mae, "examples": len(preds)}
    if profile_result:
        result["profile"] = profile_result

    if sensor_fault_truth:
        result["sensor_fault"] = _binary_prf1(sensor_fault_pred, sensor_fault_truth)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "experiments/runs/stage-f/no_adapters-seed20260810/20260808T041727Z-de5f4b0e/model-export.safetensors",
            "experiments/runs/stage-f/no_adapters-seed20260811/20260808T050419Z-b1d15d98/model-export.safetensors",
        ],
        help="checkpoints to evaluate, one entry per seed/arm",
    )
    parser.add_argument("--use-adapters", action="store_true", help="checkpoints were trained with use_adapters=True")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "development_holdout", "ood-SEVERE_MISSINGNESS", "ood-UNSEEN_TOPOLOGY"],
    )
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/phase13-sentinel-classification-metrics.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    results: dict[str, Any] = {}
    strategist_bearing_splits = {"train", "validation", "calibration", "development_holdout"}
    for checkpoint in args.checkpoints:
        checkpoint_path = Path(checkpoint)
        full_model = load_model(checkpoint_path, use_adapters=args.use_adapters, strategist_fields_available=True)
        reduced_model = None
        per_split: dict[str, Any] = {}
        for split in args.splits:
            if split in strategist_bearing_splits:
                model = full_model
            else:
                if reduced_model is None:
                    reduced_model = load_model(checkpoint_path, use_adapters=args.use_adapters, strategist_fields_available=False)
                model = reduced_model
            dataset = _load_split(split)
            per_split[split] = evaluate_split(model, dataset)
            print(f"[{checkpoint_path.parent.parent.name}] {split}: {json.dumps({k: v for k, v in per_split[split].items() if not isinstance(v, dict)})}")
        results[str(checkpoint_path)] = per_split

    report = {
        "schema_version": 1,
        "checkpoints": args.checkpoints,
        "use_adapters": args.use_adapters,
        "corpus": str(JOINT_CORPUS_ROOT),
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
