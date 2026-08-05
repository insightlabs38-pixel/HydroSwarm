"""Bundle F, Stage 3: HydroCore-S finalist training (overnight-plan.txt).

Per the plan: "Select up to three finalist configurations. For each finalist: train to
early stopping or a documented maximum runtime; run at least two seeds when time permits;
... fit calibration separately per checkpoint or documented ensemble; evaluate on
validation, development holdout, and OOD development sets; run full-trajectory evaluation
where supported."

Finalists are the top three configurations from Stage 2's predeclared score (see
`reports/results/v3/stage2-architecture-screening.json`'s `ranking`), taken mechanically
-- E2 > E0 > E1 (E2's 0.7794 margin over E0's 0.7783 and E1's 0.7768 is small enough that
all three are carried forward rather than trusting a single-seed screening pick).

Each finalist is trained with two seeds ("at least two ... when time permits"; three-seed
"final candidate" training is deferred to Stage 6 once one architecture is actually
selected, not attempted for all three finalists here). Early stopping
(patience=3, min_delta from TrainingConfig's default) or a documented 2-hour-per-run
ceiling bounds each run.

Calibration is fit per checkpoint via SplitConformalCalibrator against the calibration
split only (never validation/development_holdout/train), per the plan's explicit
constraint. OOD development-set evaluation uses Cycle B's two governed OOD-holdout
categories (UNSEEN_TOPOLOGY, SEVERE_MISSINGNESS); this reports raw source-localization
and calibrated-coverage metrics under real distribution shift, not the full live
OODDetector pipeline (which needs a reconstructed hydraulic graph/state per scenario, not
just precomputed tensors) -- documented as a scope limitation, not silently skipped.
Full-trajectory evaluation (Task 7.3) is out of scope for this script; it requires
Scout/Strategist trajectory labels Cycle B does not generate (see
data/learning-v2/cycle-b/dataset-report.json's limitations).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator, expected_calibration_error
from hydroswarm.classical.metrics import candidate_set_metrics, localization_top_k, mean_reciprocal_rank
from hydroswarm.inference import HybridInferencePipeline
from hydroswarm.inference.fusion import fixed_weight_fusion, fixed_weight_fusion_config
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
BATCH_SIZE = 16
MAX_EPOCHS = 16
EARLY_STOPPING_PATIENCE = 3
MAXIMUM_RUNTIME_SECONDS = 7200.0
CALIBRATION_ALPHA = 0.1
#: core-issues.txt repair item 10: the fixed_weight_fusion weighting used
#: to approximate the deployed hybrid predictor when fitting calibration
#: from stored governed tensors (see fixed_weight_fusion's docstring for
#: why the full dynamic-trust fusion cannot be reconstructed here).
CALIBRATION_FUSION_NEURAL_WEIGHT = 0.6
#: core-issues.txt repair item 11: task_gradient_norms costs one extra
#: torch.autograd.grad call per task loss (7-9 tasks) every batch it runs
#: on -- computing it every 25th batch instead of every batch still gives
#: a usable per-epoch trend at a fraction of the CPU cost.
GRADNORM_LOG_EVERY_N_BATCHES = 25

#: Top three from Stage 2's predeclared score, taken mechanically (no retuning after
#: viewing results): E2 (0.7794) > E0 (0.7783) > E1 (0.7768).
FINALISTS: dict[str, dict[str, Any]] = {
    "E2": {"prior_mode": "logit_only"},
    "E0": {},
    "E1": {"prior_mode": "feature_only"},
}
SEEDS: tuple[int, ...] = (20260810, 20260811)


def _load_dataset(split: str) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(CYCLE_B_ROOT / "tensors" / split, expected_split=split)
    examples = [dataset[index] for index in range(len(dataset))]
    return GovernedScenarioDataset(examples, expected_split=split)


def topology_hashes(*datasets: GovernedScenarioDataset) -> tuple[str, ...]:
    """core-issues.txt repair item 5: real topology_hash provenance for the
    experiment registry. Empty for any corpus generated before that repair
    landed (topology is None on every example until Cycle B is
    regenerated) -- a real, non-empty result requires no code change here,
    only regenerated data (Phase 3, not this repair pass)."""

    hashes = {
        example.topology.topology_hash
        for dataset in datasets
        for example in (dataset[index] for index in range(len(dataset)))
        if example.topology is not None
    }
    return tuple(sorted(hashes))


def _load_ood_dataset(category: str) -> GovernedScenarioDataset:
    # OOD-holdout shards are tagged split="development_holdout" (see
    # data/learning-v2/cycle-b/tensors/ood-*/index.jsonl) -- they are a
    # governed-OOD-category subset of that split, not a distinct split name.
    directory = CYCLE_B_ROOT / "tensors" / f"ood-{category}"
    dataset = ShardedScenarioDataset(directory, expected_split="development_holdout")
    examples = [dataset[index] for index in range(len(dataset))]
    return GovernedScenarioDataset(examples, expected_split="development_holdout")


@torch.no_grad()
def _predict_rows(model: HydroCore, dataset: GovernedScenarioDataset, *, batch_size: int = BATCH_SIZE):
    """Yield (example, node_ids, neural_probabilities, classical_probabilities,
    latency_seconds) for every example with a real source to localize -- masked
    NORMAL/SENSOR_FAULT_ONLY placeholders are skipped, exactly as Stage 2's
    screening evaluation already does (see
    hydroswarm.training.losses._apply_target_mask for the masking convention).

    classical_probabilities is the same governed classical_prior tensor the
    model itself receives as an input feature -- yielded alongside neural
    probabilities so callers that need the fused hybrid prediction (see
    core-issues.txt repair item 10) don't have to re-derive it."""

    model.eval()
    examples = [dataset[index] for index in range(len(dataset))]
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        inputs, targets = collate_variable_topology(batch_examples)
        started = time.perf_counter()
        output = model(inputs)
        latency = (time.perf_counter() - started) / len(batch_examples)
        probabilities = torch.softmax(output["source_node_logits"], dim=-1)
        classical_prior = inputs["classical_prior"]
        source_mask = targets.get("source_node_mask")
        for row in range(probabilities.shape[0]):
            if source_mask is not None and not bool(source_mask[row]):
                continue
            yield (
                batch_examples[row],
                int(targets["source_node"][row].item()),
                probabilities[row].numpy(),
                classical_prior[row].numpy(),
                latency,
            )


def _localization_metrics(rows: list[tuple[Any, int, Any, Any, float]]) -> dict[str, Any]:
    predictions: list[dict[int, float]] = []
    truths: list[int] = []
    correct_top1: list[bool] = []
    confidences: list[float] = []
    latencies: list[float] = []
    for _example, truth, probability_row, _classical_row, latency in rows:
        row_probabilities = {position: float(value) for position, value in enumerate(probability_row) if value > 0}
        predictions.append(row_probabilities)
        truths.append(truth)
        correct_top1.append(bool(truth in row_probabilities and localization_top_k(row_probabilities, truth, k=1)))
        confidences.append(max(row_probabilities.values()) if row_probabilities else 0.0)
        latencies.append(latency)
    top1 = float(sum(correct_top1) / len(correct_top1)) if correct_top1 else 0.0
    top3 = float(sum(localization_top_k(pred, truth, k=3) for pred, truth in zip(predictions, truths)) / len(predictions)) if predictions else 0.0
    mrr = mean_reciprocal_rank(predictions, truths) if predictions else 0.0
    coverage = candidate_set_metrics((sorted(pred, key=lambda key: -pred[key])[:3] for pred in predictions), truths) if predictions else None
    ece = expected_calibration_error(confidences, correct_top1) if confidences else 1.0
    return {
        "source_top1": top1,
        "source_top3": top3,
        "mrr": mrr,
        "candidate_coverage_at_3": coverage.coverage if coverage else 0.0,
        "ece": ece,
        "examples": len(predictions),
        "mean_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
    }


def _calibrated_metrics(calibrator: SplitConformalCalibrator, rows: list[tuple[Any, int, Any, Any, float]]) -> dict[str, Any]:
    # core-issues.txt repair item 10: candidate_set must be queried against
    # the same fused hybrid probability vector the calibrator was fit on,
    # not the raw neural probability_row -- otherwise the calibrator's own
    # measured coverage/set-size here would silently disagree with what it
    # was actually fit against.
    covered = 0
    set_sizes: list[int] = []
    for example, truth, probability_row, classical_row, _latency in rows:
        fused_row = fixed_weight_fusion(classical_row, probability_row)
        indices = calibrator.candidate_set(
            fused_row, condition=example.stage.name, network_id=example.network_id,
        )
        set_sizes.append(len(indices))
        if truth in indices:
            covered += 1
    return {
        "measured_coverage": (covered / len(rows)) if rows else 0.0,
        "mean_set_size": (sum(set_sizes) / len(set_sizes)) if set_sizes else 0.0,
        "examples": len(rows),
    }


def run_finalist_seed(
    name: str,
    overrides: dict[str, Any],
    seed: int,
    *,
    train: GovernedScenarioDataset,
    validation: GovernedScenarioDataset,
    calibration: GovernedScenarioDataset,
    development_holdout: GovernedScenarioDataset,
    ood_datasets: dict[str, GovernedScenarioDataset],
    run_root: Path,
    registry: ExperimentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = TrainingConfig(
        seed=seed,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=30,
        checkpoint_every_epochs=4,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        maximum_runtime_seconds=MAXIMUM_RUNTIME_SECONDS,
        gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES,
    )
    model = HydroCore.from_variant("small", **overrides)

    handle = registry.open_run(
        kind="training",
        purpose=f"Bundle F Stage 3 finalist training: {name} seed {seed}",
        architecture="hydrocore",
        variant="small",
        seed=seed,
        resolved_config={"finalist": name, "overrides": overrides, **config.as_dict()},
        manifest_hashes={
            "train": train.manifest_hash,
            "validation": validation.manifest_hash,
            "calibration": calibration.manifest_hash,
            "development_holdout": development_holdout.manifest_hash,
        },
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        topology_hashes=topology_hashes(train, validation, calibration, development_holdout),
        workdir=".",
    )

    run_dir = run_root / f"{name}-seed{seed}"
    trainer = Trainer(
        model, train, validation_dataset=validation, config=config,
        run_root=run_dir, workdir=".", collate_fn=collate_variable_topology,
    )
    summary = trainer.fit()
    if not math.isfinite(summary.best_validation_loss):
        handle.close(exit_status="failed", notes="non-finite validation loss")
        raise RuntimeError(f"{name}/seed{seed}: non-finite validation loss")

    model_hash = HybridInferencePipeline._fingerprint_model(model)

    # core-issues.txt repair item 10: fit conformal calibration on the fused
    # hybrid probability vector (fixed_weight_fusion of the stored classical
    # prior and the model's own neural probabilities), not on neural
    # probabilities alone -- the deployed HybridInferencePipeline always
    # thresholds a fused vector, never raw neural output.
    calibration_rows = list(_predict_rows(model, calibration))
    calibration_examples = [
        CalibrationExample(
            probabilities=tuple(
                float(value)
                for value in fixed_weight_fusion(
                    classical_row, probability_row, neural_weight=CALIBRATION_FUSION_NEURAL_WEIGHT
                )
            ),
            true_index=truth,
            condition=example.stage.name,
            network_id=example.network_id,
        )
        for example, truth, probability_row, classical_row, _latency in calibration_rows
    ]
    calibration_topology_hashes = tuple(
        sorted(
            {
                example.topology.topology_hash
                for example, *_rest in calibration_rows
                if example.topology is not None
            }
        )
    )
    calibrator = SplitConformalCalibrator.fit(
        calibration_examples,
        alpha=CALIBRATION_ALPHA,
        model_hash=model_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash=calibration.manifest_hash,
        fusion_config_hash=fixed_weight_fusion_config(CALIBRATION_FUSION_NEURAL_WEIGHT),
        topology_hashes=calibration_topology_hashes,
    )
    calibrator.save(run_dir / "calibration.json")

    validation_rows = list(_predict_rows(model, validation))
    development_rows = list(_predict_rows(model, development_holdout))
    validation_metrics = _localization_metrics(validation_rows)
    development_metrics = _localization_metrics(development_rows)
    validation_calibrated = _calibrated_metrics(calibrator, validation_rows)
    development_calibrated = _calibrated_metrics(calibrator, development_rows)

    ood_metrics: dict[str, Any] = {}
    for category, dataset in ood_datasets.items():
        rows = list(_predict_rows(model, dataset))
        ood_metrics[category] = {
            "localization": _localization_metrics(rows),
            "calibrated": _calibrated_metrics(calibrator, rows),
        }

    # core-issues.txt repair item 11: summary.final_checkpoint is
    # intentionally "" whenever the run was cut off before a clean
    # end-of-run resumable save (see Trainer.fit's runtime_budget_reached
    # handling) -- every one of Stage 3's finalist runs hits the 2-hour
    # ceiling (see finalist-selection-recommendation.md), so this was
    # closing every "success" run with an empty selected_checkpoint.
    # summary.export_path is unconditional: it always points at the best
    # validated model this run produced, whether or not training was cut
    # short, so it is what "selected" must mean here.
    handle.close(
        exit_status="success",
        checkpoint_paths=tuple(dict.fromkeys(
            path
            for path in (summary.final_checkpoint, summary.last_resumable_checkpoint, summary.export_path)
            if path
        )),
        checkpoint_hashes={summary.export_path: summary.export_sha256},
        selected_checkpoint=summary.export_path,
        selection_metric={
            "validation": validation_metrics,
            "development_holdout": development_metrics,
            "calibration_report": {
                "coverage": calibrator.artifact.report.coverage,
                "mean_set_size": calibrator.artifact.report.mean_set_size,
                "expected_calibration_error": calibrator.artifact.report.expected_calibration_error,
            },
        },
    )

    return {
        "finalist": name,
        "seed": seed,
        "overrides": overrides,
        "run_id": handle.run_id,
        "checkpoint": summary.final_checkpoint,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "stop_reason": summary.stop_reason,
        "best_validation_loss": summary.best_validation_loss,
        "parameter_count": model.parameter_count(),
        "validation_metrics": validation_metrics,
        "validation_calibrated": validation_calibrated,
        "development_holdout_metrics": development_metrics,
        "development_holdout_calibrated": development_calibrated,
        "calibration_artifact_hash": calibrator.artifact.artifact_hash,
        "calibration_report": {
            "coverage": calibrator.artifact.report.coverage,
            "mean_set_size": calibrator.artifact.report.mean_set_size,
            "expected_calibration_error": calibrator.artifact.report.expected_calibration_error,
        },
        "ood_development": ood_metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/bundle-f-stage3"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/bundle-f-stage3.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v3/stage3-finalist-training.json"))
    args = parser.parse_args()

    train = _load_dataset("train")
    validation = _load_dataset("validation")
    calibration = _load_dataset("calibration")
    development_holdout = _load_dataset("development_holdout")
    ood_datasets = {
        "UNSEEN_TOPOLOGY": _load_ood_dataset("UNSEEN_TOPOLOGY"),
        "SEVERE_MISSINGNESS": _load_ood_dataset("SEVERE_MISSINGNESS"),
    }
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for name, overrides in FINALISTS.items():
        results[name] = {}
        for seed in SEEDS:
            key = f"{name}-seed{seed}"
            try:
                results[name][str(seed)] = run_finalist_seed(
                    name, overrides, seed,
                    train=train, validation=validation, calibration=calibration,
                    development_holdout=development_holdout, ood_datasets=ood_datasets,
                    run_root=args.run_root, registry=registry,
                )
                print(f"{key}: OK ({results[name][str(seed)]['wall_seconds']:.1f}s, "
                      f"val_top1={results[name][str(seed)]['validation_metrics']['source_top1']:.3f}, "
                      f"stopped_early={results[name][str(seed)]['stopped_early']})")
            except Exception as error:  # noqa: BLE001 -- terminate this run, continue the sweep
                failures[key] = f"{type(error).__name__}: {error}"
                print(f"{key}: FAILED ({failures[key]})")

    report = {
        "schema_version": 1,
        "stage": "Bundle F Stage 3 -- finalist training",
        "corpus": str(CYCLE_B_ROOT),
        "finalists": FINALISTS,
        "seeds": list(SEEDS),
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "maximum_runtime_seconds_per_run": MAXIMUM_RUNTIME_SECONDS,
        "calibration_alpha": CALIBRATION_ALPHA,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
