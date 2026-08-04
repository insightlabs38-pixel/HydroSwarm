"""Bundle F, Stage 2: architecture screening for E0-E8 (overnight-plan.txt)
against the Cycle B corpus.

Per the plan: "For E0-E8 or a pruned matrix: use Cycle B; 3-6 epochs or a
comparable fixed step budget; one seed; use validation and development
holdout; rank configurations by a predeclared score." The score is
declared here, before any run's results are seen, exactly as required:

    score = 0.30 * source_top1(validation)
          + 0.15 * source_top3(validation)
          + 0.10 * mean_reciprocal_rank(validation)
          + 0.15 * (1 - expected_calibration_error(validation))
          + 0.10 * candidate_coverage_at_3(validation)
          + 0.10 * source_top1(development_holdout)   # topology-dev-holdout behavior
          + 0.05 * sensor_fault_accuracy(validation)
          + 0.05 * normalized_inverse_latency

"Do not let weak profile heads dominate the screening score": start_time/
duration/relative_strength are deliberately excluded from the score
entirely, not down-weighted.

Scope note: Scout quality (plan's own criterion) is not included -- no
Scout labels exist in Cycle B (scout_labels.py's generator was never run
against Cycle A or B; documented in both corpora's dataset-report.json).
E7's "other task weights set to zero" is implemented via
TrainingConfig.task_weights. E8's "complete Sentinel, Scout, Strategist,
control, and auxiliary targets" is implemented as event_control_heads=True
only -- Cycle B has no Scout/Strategist/auxiliary-objective labels, so
those loss terms cannot fire regardless of which heads are enabled; this
is the same "Full targets_v2 multitask run" one can honestly run today.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from hydroswarm.classical.metrics import candidate_set_metrics, localization_top_k, mean_reciprocal_rank
from hydroswarm.calibration.conformal import expected_calibration_error
from hydroswarm.model import HydroCore
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training import (
    GovernedScenarioDataset,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION

CYCLE_B_ROOT = Path("data/learning-v2/cycle-b")
SEED = 20260805
EPOCHS = 4
BATCH_SIZE = 16

#: overnight-plan.txt's E0-E8. E7/E8 are training-regime variants of E0's
#: architecture, not new architecture flags, so their "overrides" entry is
#: empty and they carry a separate task_weights/event_control_heads knob
#: applied directly in run_experiment.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E0": {},
    "E1": {"prior_mode": "feature_only"},
    "E2": {"prior_mode": "logit_only"},
    "E3": {"incident_pooling": "source_conditioned"},
    "E4": {"message_direction": "dual_gated"},
    "E5": {"incident_pooling": "source_conditioned", "message_direction": "dual_gated"},
    "E6": {"incident_pooling": "source_conditioned", "message_direction": "dual_gated", "auxiliary_heads": True},
    "E7": {},  # source-only diagnostic: architecture is E0's, task_weights differ
    "E8": {"event_control_heads": True},  # full-multitask diagnostic
}
SOURCE_ONLY_TASK_WEIGHTS = {
    "source_region": 0.0,
    "start_time": 0.0,
    "duration": 0.0,
    "relative_strength": 0.0,
    "sensor_fault": 0.0,
    "evidence_sufficiency": 0.0,
}


def _load_dataset(split: str) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(CYCLE_B_ROOT / "tensors" / split, expected_split=split)
    examples = [dataset[index] for index in range(len(dataset))]
    return GovernedScenarioDataset(examples, expected_split=split)


@torch.no_grad()
def _evaluate_source_localization(model: HydroCore, dataset: GovernedScenarioDataset, *, batch_size: int = 16) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[int, float]] = []
    truths: list[int] = []
    correct_top1: list[bool] = []
    confidences: list[float] = []
    fault_correct = 0
    fault_total = 0
    latencies: list[float] = []
    examples = [dataset[index] for index in range(len(dataset))]
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        inputs, targets = collate_variable_topology(batch_examples)
        started = time.perf_counter()
        output = model(inputs)
        latencies.append((time.perf_counter() - started) / len(batch_examples))
        probabilities = torch.softmax(output["source_node_logits"], dim=-1)
        source_mask = targets.get("source_node_mask")
        for row in range(probabilities.shape[0]):
            # A masked source_node (NORMAL/SENSOR_FAULT_ONLY placeholder,
            # per targets_v2's masking convention -- see
            # hydroswarm.training.losses._apply_target_mask) has no real
            # source to localize; scoring against it would penalize the
            # model for not "finding" a source that was never injected.
            if source_mask is not None and not bool(source_mask[row]):
                continue
            valid_positions = torch.isfinite(output["source_node_logits"][row]) & (
                output["source_node_logits"][row] > torch.finfo(output["source_node_logits"].dtype).min
            )
            row_probabilities = {
                position: float(probabilities[row, position])
                for position in range(probabilities.shape[1])
                if bool(valid_positions[position])
            }
            truth = int(targets["source_node"][row].item())
            predictions.append(row_probabilities)
            truths.append(truth)
            correct_top1.append(bool(truth in row_probabilities and localization_top_k(row_probabilities, truth, k=1)))
            confidences.append(max(row_probabilities.values()) if row_probabilities else 0.0)
        if "sensor_fault_logits" in output and "sensor_fault" in targets:
            fault_predictions = (torch.sigmoid(output["sensor_fault_logits"]) > 0.5).float()
            fault_target = targets["sensor_fault"].float()
            valid = torch.isfinite(fault_target)
            fault_correct += int((fault_predictions[valid] == fault_target[valid]).sum())
            fault_total += int(valid.sum())

    top1 = float(sum(correct_top1) / len(correct_top1)) if correct_top1 else 0.0
    top3 = float(sum(localization_top_k(pred, truth, k=3) for pred, truth in zip(predictions, truths)) / len(predictions)) if predictions else 0.0
    mrr = mean_reciprocal_rank(predictions, truths) if predictions else 0.0
    coverage = candidate_set_metrics(
        (sorted(pred, key=lambda key: -pred[key])[:3] for pred in predictions), truths
    ) if predictions else None
    ece = expected_calibration_error(confidences, correct_top1) if confidences else 1.0
    return {
        "source_top1": top1,
        "source_top3": top3,
        "mrr": mrr,
        "candidate_coverage_at_3": coverage.coverage if coverage else 0.0,
        "ece": ece,
        "sensor_fault_accuracy": (fault_correct / fault_total) if fault_total else None,
        "mean_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
    }


def _score(validation_metrics: dict[str, Any], dev_holdout_metrics: dict[str, Any], all_latencies: list[float]) -> float:
    # Latency normalized against the slowest config in this sweep so the
    # score stays comparable across a single run's matrix (not an absolute
    # latency threshold, which the plan does not predeclare a value for).
    slowest = max(all_latencies) if all_latencies else validation_metrics["mean_latency_seconds"]
    inverse_latency = 1.0 - (validation_metrics["mean_latency_seconds"] / slowest if slowest > 0 else 0.0)
    fault_accuracy = validation_metrics["sensor_fault_accuracy"] or 0.0
    return (
        0.30 * validation_metrics["source_top1"]
        + 0.15 * validation_metrics["source_top3"]
        + 0.10 * validation_metrics["mrr"]
        + 0.15 * (1.0 - validation_metrics["ece"])
        + 0.10 * validation_metrics["candidate_coverage_at_3"]
        + 0.10 * dev_holdout_metrics["source_top1"]
        + 0.05 * fault_accuracy
        + 0.05 * inverse_latency
    )


def run_experiment(
    name: str,
    overrides: dict[str, Any],
    *,
    train: GovernedScenarioDataset,
    validation: GovernedScenarioDataset,
    development_holdout: GovernedScenarioDataset,
    run_root: Path,
    registry: ExperimentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_weights = dict(SOURCE_ONLY_TASK_WEIGHTS) if name == "E7" else {}
    config = TrainingConfig(
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=20,
        checkpoint_every_epochs=EPOCHS,
        maximum_runtime_seconds=3600.0 * 2,
        task_weights=task_weights,
    )
    model = HydroCore.from_variant("small", **overrides)

    handle = registry.open_run(
        kind="training",
        purpose=f"Bundle F Stage 2 architecture screening: {name}",
        architecture="hydrocore",
        variant="small",
        seed=SEED,
        resolved_config={"experiment": name, "overrides": overrides, **config.as_dict()},
        manifest_hashes={
            "train": train.manifest_hash,
            "validation": validation.manifest_hash,
            "development_holdout": development_holdout.manifest_hash,
        },
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        workdir=".",
    )

    trainer = Trainer(
        model, train, validation_dataset=validation, config=config,
        run_root=run_root, workdir=".", collate_fn=collate_variable_topology,
    )
    summary = trainer.fit()
    if not math.isfinite(summary.best_validation_loss):
        handle.close(exit_status="failed", notes="non-finite validation loss")
        raise RuntimeError(f"{name}: non-finite validation loss")

    validation_metrics = _evaluate_source_localization(model, validation)
    dev_holdout_metrics = _evaluate_source_localization(model, development_holdout)

    handle.close(
        exit_status="success",
        checkpoint_paths=[summary.final_checkpoint],
        selected_checkpoint=summary.final_checkpoint,
        selection_metric={"validation": validation_metrics, "development_holdout": dev_holdout_metrics},
    )

    return {
        "experiment": name,
        "overrides": overrides,
        "task_weights": task_weights,
        "run_id": handle.run_id,
        "checkpoint": summary.final_checkpoint,
        "best_validation_loss": summary.best_validation_loss,
        "validation_metrics": validation_metrics,
        "development_holdout_metrics": dev_holdout_metrics,
        "parameter_count": model.parameter_count(),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/bundle-f-stage2"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/bundle-f-stage2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v3/stage2-architecture-screening.json"))
    args = parser.parse_args()

    train = _load_dataset("train")
    validation = _load_dataset("validation")
    development_holdout = _load_dataset("development_holdout")
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name, overrides in EXPERIMENTS.items():
        try:
            results[name] = run_experiment(
                name, overrides, train=train, validation=validation,
                development_holdout=development_holdout, run_root=args.run_root, registry=registry,
            )
            print(f"{name}: OK ({results[name]['wall_seconds']:.1f}s)")
        except Exception as error:  # noqa: BLE001 -- terminate this config, continue the sweep
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"{name}: FAILED ({failures[name]})")

    all_latencies = [result["validation_metrics"]["mean_latency_seconds"] for result in results.values()]
    for result in results.values():
        result["score"] = _score(result["validation_metrics"], result["development_holdout_metrics"], all_latencies)
    ranking = sorted(results, key=lambda name: results[name]["score"], reverse=True)

    report = {
        "schema_version": 1,
        "stage": "Bundle F Stage 2 -- architecture screening (E0-E8)",
        "corpus": str(CYCLE_B_ROOT),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "score_formula": (
            "0.30*source_top1(val) + 0.15*source_top3(val) + 0.10*mrr(val) "
            "+ 0.15*(1-ece(val)) + 0.10*candidate_coverage_at_3(val) "
            "+ 0.10*source_top1(dev_holdout) + 0.05*sensor_fault_accuracy(val) "
            "+ 0.05*normalized_inverse_latency"
        ),
        "wall_seconds": time.perf_counter() - started,
        "ranking": ranking,
        "results": results,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
