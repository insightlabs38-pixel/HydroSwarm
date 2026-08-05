"""Bundle F, Stage 4: HydroMono and no-adapter controls (overnight-plan.txt).

Per the plan: "Train comparable controls under the same data and budget: HydroMono-S;
no-adapter HydroCore-S; current architecture baseline; classical-only baseline. Equalize:
training examples; optimization steps; feature inputs; candidate budgets; evaluation
scenarios."

Of the four required controls, this script trains exactly one:

- **HydroMono-S and no-adapter HydroCore-S are the same control**, not two. Verified
  directly: ``HydroMono`` (hydroswarm.model.core) is an unmodified subclass of
  ``NoAdapterHydroCore`` with no overriding ``__init__`` or methods of its own --
  constructing both from the same seed produces bit-identical state dicts (same keys, same
  values, same parameter count). Training both would be pure duplicate compute for zero
  additional information; this script trains the control once (as ``HydroCore.from_variant
  ("small", use_adapters=False)``, i.e. the no-adapter ablation) and the results apply to
  both names.
- **current architecture baseline is E0**, already trained with 2 seeds in Stage 3
  (see reports/results/v3/stage3-finalist-training.json) -- not repeated here.
- **classical-only baseline** requires no training at all: ``classical_prior`` is already a
  precomputed per-example tensor feature (the classical signature-localization posterior),
  so its accuracy is computed directly from the corpus with no model. Computed separately
  (see reports/results/v3/finalist-selection-recommendation.md's classical-baseline note);
  not repeated here either.

The one control this script actually trains (no-adapter HydroCore-S) uses the identical
budget as Stage 3's finalists (same corpus, same epochs/early-stopping/runtime ceiling,
same seeds, same batch size) so the comparison is apples-to-apples per the plan's
"equalize" requirement. Reuses Stage 3's proven training/calibration/evaluation pipeline
directly (imported, not duplicated) to avoid re-introducing bugs Stage 2/3 already worked
through.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage3_finalist_training as stage3  # noqa: E402

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator  # noqa: E402
from hydroswarm.inference import HybridInferencePipeline  # noqa: E402
from hydroswarm.inference.fusion import fixed_weight_fusion, fixed_weight_fusion_config  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA  # noqa: E402
from hydroswarm.training import Trainer, TrainingConfig, collate_variable_topology  # noqa: E402
from hydroswarm.training.registry import ExperimentRegistry  # noqa: E402
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION  # noqa: E402

SEEDS: tuple[int, ...] = stage3.SEEDS
CONTROL_NAME = "no-adapter-S"


def run_control_seed(
    seed: int,
    *,
    train, validation, calibration, development_holdout, ood_datasets,
    run_root: Path,
    registry: ExperimentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = TrainingConfig(
        seed=seed,
        epochs=stage3.MAX_EPOCHS,
        batch_size=stage3.BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=30,
        checkpoint_every_epochs=4,
        early_stopping_patience=stage3.EARLY_STOPPING_PATIENCE,
        maximum_runtime_seconds=stage3.MAXIMUM_RUNTIME_SECONDS,
        gradnorm_log_every_n_batches=stage3.GRADNORM_LOG_EVERY_N_BATCHES,
    )
    model = HydroCore.from_variant("small", use_adapters=False)

    handle = registry.open_run(
        kind="training",
        purpose=f"Bundle F Stage 4 control: {CONTROL_NAME} seed {seed}",
        architecture="hydrocore-no-adapter",
        variant="small",
        seed=seed,
        resolved_config={"control": CONTROL_NAME, "overrides": {"use_adapters": False}, **config.as_dict()},
        manifest_hashes={
            "train": train.manifest_hash,
            "validation": validation.manifest_hash,
            "calibration": calibration.manifest_hash,
            "development_holdout": development_holdout.manifest_hash,
        },
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        topology_hashes=stage3.topology_hashes(train, validation, calibration, development_holdout),
        workdir=".",
    )

    run_dir = run_root / f"{CONTROL_NAME}-seed{seed}"
    trainer = Trainer(
        model, train, validation_dataset=validation, config=config,
        run_root=run_dir, workdir=".", collate_fn=collate_variable_topology,
    )
    summary = trainer.fit()
    if not math.isfinite(summary.best_validation_loss):
        handle.close(exit_status="failed", notes="non-finite validation loss")
        raise RuntimeError(f"{CONTROL_NAME}/seed{seed}: non-finite validation loss")

    model_hash = HybridInferencePipeline._fingerprint_model(model)

    # core-issues.txt repair item 10: fit on the fused hybrid probability
    # vector, not raw neural probabilities -- see stage3's identical fix.
    calibration_rows = list(stage3._predict_rows(model, calibration))
    calibration_examples = [
        CalibrationExample(
            probabilities=tuple(
                float(value)
                for value in fixed_weight_fusion(
                    classical_row, probability_row, neural_weight=stage3.CALIBRATION_FUSION_NEURAL_WEIGHT
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
        alpha=stage3.CALIBRATION_ALPHA,
        model_hash=model_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash=calibration.manifest_hash,
        fusion_config_hash=fixed_weight_fusion_config(stage3.CALIBRATION_FUSION_NEURAL_WEIGHT),
        topology_hashes=calibration_topology_hashes,
    )
    calibrator.save(run_dir / "calibration.json")

    validation_rows = list(stage3._predict_rows(model, validation))
    development_rows = list(stage3._predict_rows(model, development_holdout))
    validation_metrics = stage3._localization_metrics(validation_rows)
    development_metrics = stage3._localization_metrics(development_rows)
    validation_calibrated = stage3._calibrated_metrics(calibrator, validation_rows)
    development_calibrated = stage3._calibrated_metrics(calibrator, development_rows)

    ood_metrics: dict[str, Any] = {}
    for category, dataset in ood_datasets.items():
        rows = list(stage3._predict_rows(model, dataset))
        ood_metrics[category] = {
            "localization": stage3._localization_metrics(rows),
            "calibrated": stage3._calibrated_metrics(calibrator, rows),
        }

    # core-issues.txt repair item 11: see stage3's identical fix -- use
    # summary.export_path (unconditionally populated) as the selected
    # checkpoint, never summary.final_checkpoint (empty when this run hit
    # the runtime budget ceiling before a clean end-of-run save).
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
        "control": CONTROL_NAME,
        "seed": seed,
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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/bundle-f-stage4"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/bundle-f-stage4.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v3/stage4-controls-training.json"))
    parser.add_argument("--corpus-root", type=Path, default=stage3.CYCLE_B_ROOT)
    parser.add_argument(
        "--tensors-dirname",
        default=stage3.TENSORS_DIRNAME,
        help="subdirectory of --corpus-root holding sharded tensors (default: tensors; use "
        "tensors-normalized for a corpus with governed normalization applied)",
    )
    args = parser.parse_args()

    train = stage3._load_dataset("train", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname)
    validation = stage3._load_dataset(
        "validation", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname
    )
    calibration = stage3._load_dataset(
        "calibration", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname
    )
    development_holdout = stage3._load_dataset(
        "development_holdout", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname
    )
    ood_datasets = {
        "UNSEEN_TOPOLOGY": stage3._load_ood_dataset(
            "UNSEEN_TOPOLOGY", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname
        ),
        "SEVERE_MISSINGNESS": stage3._load_ood_dataset(
            "SEVERE_MISSINGNESS", corpus_root=args.corpus_root, tensors_dirname=args.tensors_dirname
        ),
    }
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {CONTROL_NAME: {}}
    failures: dict[str, str] = {}
    for seed in SEEDS:
        key = f"{CONTROL_NAME}-seed{seed}"
        try:
            results[CONTROL_NAME][str(seed)] = run_control_seed(
                seed, train=train, validation=validation, calibration=calibration,
                development_holdout=development_holdout, ood_datasets=ood_datasets,
                run_root=args.run_root, registry=registry,
            )
            r = results[CONTROL_NAME][str(seed)]
            print(f"{key}: OK ({r['wall_seconds']:.1f}s, val_top1={r['validation_metrics']['source_top1']:.3f}, "
                  f"stopped_early={r['stopped_early']})")
        except Exception as error:  # noqa: BLE001
            failures[key] = f"{type(error).__name__}: {error}"
            print(f"{key}: FAILED ({failures[key]})")

    report = {
        "schema_version": 1,
        "stage": "Bundle F Stage 4 -- HydroMono / no-adapter control",
        "note": (
            "HydroMono-S and no-adapter HydroCore-S are architecturally identical in this "
            "codebase (verified: bit-identical state dicts from the same seed) -- trained "
            "once, applies to both names. Current-architecture baseline is E0 from Stage 3 "
            "(reports/results/v3/stage3-finalist-training.json), not repeated. Classical-only "
            "baseline requires no training (see finalist-selection-recommendation.md)."
        ),
        "corpus": str(args.corpus_root / args.tensors_dirname),
        "seeds": list(SEEDS),
        "max_epochs": stage3.MAX_EPOCHS,
        "early_stopping_patience": stage3.EARLY_STOPPING_PATIENCE,
        "maximum_runtime_seconds_per_run": stage3.MAXIMUM_RUNTIME_SECONDS,
        "calibration_alpha": stage3.CALIBRATION_ALPHA,
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
