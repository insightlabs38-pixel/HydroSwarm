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
    ScenarioDatasetView,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION

CYCLE_B_ROOT = Path("data/learning-v2/cycle-b")
TENSORS_DIRNAME = "tensors"
SEED = 20260805
EPOCHS = 4
BATCH_SIZE = 16
#: core-issues.txt repair item 11: see run_stage3_finalist_training.py's
#: identical constant/rationale.
GRADNORM_LOG_EVERY_N_BATCHES = 25

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


def _load_dataset(split: str, *, corpus_root: Path, tensors_dirname: str) -> ScenarioDatasetView:
    # core-issues.txt repair item 12: return the lazy, disk-backed dataset
    # directly -- see run_stage3_finalist_training.py's identical fix.
    # Verify shard checksums once, explicitly, before training.
    dataset = ShardedScenarioDataset(corpus_root / tensors_dirname / split, expected_split=split)
    dataset.verify_shard_checksums()
    return dataset


def _topology_hashes(*datasets: ScenarioDatasetView) -> tuple[str, ...]:
    """core-issues.txt repair item 5: real topology_hash provenance for the
    experiment registry. Empty until Cycle B is regenerated with the
    populated TopologyMetadata that repair item 5 also added (Phase 3, not
    this repair pass) -- every example generated before that has
    topology=None."""

    hashes = {
        example.topology.topology_hash
        for dataset in datasets
        for example in (dataset[index] for index in range(len(dataset)))
        if example.topology is not None
    }
    return tuple(sorted(hashes))


@torch.no_grad()
def _evaluate_source_localization(model: HydroCore, dataset: ScenarioDatasetView, *, batch_size: int = 16) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[int, float]] = []
    truths: list[int] = []
    correct_top1: list[bool] = []
    confidences: list[float] = []
    fault_correct = 0
    fault_total = 0
    latencies: list[float] = []
    # core-issues.txt repair item 12: only ever materialize one batch's
    # worth of examples at a time.
    total = len(dataset)
    for start in range(0, total, batch_size):
        batch_examples = [dataset[index] for index in range(start, min(start + batch_size, total))]
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
    train: ScenarioDatasetView,
    validation: ScenarioDatasetView,
    development_holdout: ScenarioDatasetView,
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
        gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES,
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
        topology_hashes=_topology_hashes(train, validation, development_holdout),
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

    # core-issues.txt repair item 11: use summary.export_path (unconditionally
    # populated) as the selected checkpoint, never summary.final_checkpoint
    # (empty whenever this run hit the runtime budget ceiling before a clean
    # end-of-run save).
    handle.close(
        exit_status="success",
        checkpoint_paths=tuple(dict.fromkeys(
            path
            for path in (summary.final_checkpoint, summary.last_resumable_checkpoint, summary.export_path)
            if path
        )),
        checkpoint_hashes={summary.export_path: summary.export_sha256},
        selected_checkpoint=summary.export_path,
        selection_metric={"validation": validation_metrics, "development_holdout": dev_holdout_metrics},
    )

    return {
        "experiment": name,
        "overrides": overrides,
        "task_weights": task_weights,
        "run_id": handle.run_id,
        "checkpoint": summary.export_path,
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
    parser.add_argument("--corpus-root", type=Path, default=CYCLE_B_ROOT)
    parser.add_argument(
        "--tensors-dirname",
        default=TENSORS_DIRNAME,
        help="subdirectory of --corpus-root holding sharded tensors (default: tensors; use "
        "tensors-normalized for a corpus with governed normalization applied)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=list(EXPERIMENTS),
        help="subset of E0-E8 to run (default: all). e.g. --experiments E0 E1 E2 for a "
        "targeted re-screen after a corpus/pipeline correction.",
    )
    args = parser.parse_args()
    experiments = {name: EXPERIMENTS[name] for name in args.experiments}

    train = _load_dataset("train", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname)
    validation = _load_dataset("validation", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname)
    development_holdout = _load_dataset(
        "development_holdout", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname
    )
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name, overrides in experiments.items():
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
        "corpus": str(args.corpus_root / args.tensors_dirname),
        "experiments_run": list(experiments),
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
