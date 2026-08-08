"""core-issues3.txt Phase 12 Stage F: the planned HydroCore-S "shared
backbone + adapters" vs. "shared backbone without adapters" comparison on
the real joint multitask corpus (data/learning-v2/cycle-b2-joint-v4),
at least two seeds per arm, same corpus/training budget/evaluation
protocol/seeds across both arms.

Both arms train the SAME full Stage-F config the pre-freeze pass's
corpus+gradient-smoke gates already verified end-to-end (21 real
simultaneous tasks, every one with a positive valid count and nonzero
gradient) -- the only thing that varies between arms is
``use_adapters`` (hydroswarm.model.core.HydroCore's own existing
per-role BottleneckAdapter-vs-nn.Identity ablation flag). Training
budget (epochs/patience/runtime ceiling/batch size/seeds) matches this
project's own established Bundle-F-Stage-3 convention
(scripts/run_stage3_finalist_training.py) for direct comparability
across the whole project's HydroCore-S run history, not just within
this comparison.

Evaluation protocol, identical for both arms: best-validation-loss
checkpoint selection (never development_holdout, matching this
project's split policy), then one no-grad multitask-loss pass over
development_holdout (real out-of-training-distribution comparison;
joint-v4's development_holdout population is documented to omit
Scout/next_step targets entirely -- compute_multitask_loss already
skips any task whose target key is simply absent, so this reports
exactly the tasks development_holdout actually supervises, not a
padded/fabricated set).

Runs sequentially, one seed at a time (overnight-plan.txt Task 6.0:
"avoid simultaneously running multiple all-core training processes") --
NOT four parallel processes competing for the same CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from hydroswarm.inference import HybridInferencePipeline
from hydroswarm.model import HydroCore
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training import (
    ScenarioDatasetView,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.losses import compute_multitask_loss
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION

JOINT_CORPUS_ROOT = Path("data/learning-v2/cycle-b2-joint-v4")
TENSORS_DIRNAME = "tensors-normalized"

#: Matches scripts/run_stage3_finalist_training.py's own established
#: Bundle-F-Stage-3 budget exactly -- "same training budget ... across
#: arms" means across this whole project's HydroCore-S run history, not
#: a bespoke Stage-F-only number.
BATCH_SIZE = 16
MAX_EPOCHS = 16
EARLY_STOPPING_PATIENCE = 3
MAXIMUM_RUNTIME_SECONDS = 7200.0
GRADNORM_LOG_EVERY_N_BATCHES = 25
SEEDS: tuple[int, ...] = (20260810, 20260811)

#: The exact full Stage-F config scripts/run_stage_f_joint_corpus_gates.py
#: already verified end-to-end (21/21 real tasks, positive valid counts,
#: nonzero gradients) -- shared by both arms; only use_adapters varies.
SHARED_MODEL_CONFIG: dict[str, Any] = {
    "prior_mode": "feature_only",
    "event_control_heads": True,
    "scout_control_heads": True,
    "strategist_mode": "candidate_conditioned",
    "action_vocabulary_size": ACTION_TEMPLATE_COUNT,
    "consequence_prescreening_heads": True,
}
ARMS: dict[str, bool] = {"adapters": True, "no_adapters": False}


def _load_dataset(split: str) -> ScenarioDatasetView:
    dataset = ShardedScenarioDataset(JOINT_CORPUS_ROOT / TENSORS_DIRNAME / split, expected_split=split)
    dataset.verify_shard_checksums()
    return dataset


def _topology_hashes(*datasets: ScenarioDatasetView) -> tuple[str, ...]:
    hashes = {
        example.topology.topology_hash
        for dataset in datasets
        for example in (dataset[index] for index in range(len(dataset)))
        if example.topology is not None
    }
    return tuple(sorted(hashes))


def _evaluate(model: torch.nn.Module, dataset: ScenarioDatasetView, *, batch_size: int) -> dict[str, Any]:
    """No-grad multitask-loss pass over ``dataset`` -- mirrors
    Trainer._validate's own logic exactly (same collate_fn, same
    compute_multitask_loss call, same per-task averaging) so both arms'
    development_holdout numbers are produced by the identical evaluation
    path used for validation-loss checkpoint selection during training."""

    model.eval()
    losses: list[float] = []
    task_loss_sums: dict[str, float] = {}
    task_loss_counts: dict[str, int] = {}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            examples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            inputs, targets = collate_variable_topology(examples)
            output = model(inputs)
            result = compute_multitask_loss(output, targets)
            losses.append(float(result.total))
            for name, value in result.tasks.items():
                task_loss_sums[name] = task_loss_sums.get(name, 0.0) + float(value)
                task_loss_counts[name] = task_loss_counts.get(name, 0) + 1
    per_task = {name: task_loss_sums[name] / task_loss_counts[name] for name in task_loss_sums}
    return {
        "mean_loss": float(sum(losses) / len(losses)) if losses else math.nan,
        "per_task_mean_loss": per_task,
        "examples": len(dataset),
    }


def run_arm_seed(
    arm: str,
    use_adapters: bool,
    seed: int,
    *,
    train: ScenarioDatasetView,
    validation: ScenarioDatasetView,
    development_holdout: ScenarioDatasetView,
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
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **SHARED_MODEL_CONFIG)

    handle = registry.open_run(
        kind="training",
        purpose=f"Phase 12 Stage F: {arm} seed {seed}",
        architecture="hydrocore",
        variant="small",
        seed=seed,
        resolved_config={"arm": arm, "use_adapters": use_adapters, **SHARED_MODEL_CONFIG, **config.as_dict()},
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

    run_dir = run_root / f"{arm}-seed{seed}"
    trainer = Trainer(
        model, train, validation_dataset=validation, config=config,
        run_root=run_dir, workdir=".", collate_fn=collate_variable_topology,
    )
    summary = trainer.fit()
    if not math.isfinite(summary.best_validation_loss):
        handle.close(exit_status="failed", notes="non-finite validation loss")
        raise RuntimeError(f"{arm}/seed{seed}: non-finite validation loss")

    development_metrics = _evaluate(model, development_holdout, batch_size=BATCH_SIZE)
    model_hash = HybridInferencePipeline._fingerprint_model(model)

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
            "best_validation_loss": summary.best_validation_loss,
            "development_holdout": development_metrics,
        },
    )

    return {
        "arm": arm,
        "use_adapters": use_adapters,
        "seed": seed,
        "run_id": handle.run_id,
        "model_hash": model_hash,
        "checkpoint": summary.export_path,
        "checkpoint_sha256": summary.export_sha256,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "stop_reason": summary.stop_reason,
        "best_validation_loss": summary.best_validation_loss,
        "development_holdout": development_metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/stage-f"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/stage-f.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/stage-f-adapters-comparison.json"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train = _load_dataset("train")
    validation = _load_dataset("validation")
    development_holdout = _load_dataset("development_holdout")

    registry = ExperimentRegistry(args.registry)
    results: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for arm in args.arms:
        use_adapters = ARMS[arm]
        for seed in args.seeds:
            key = f"{arm}-seed{seed}"
            result = run_arm_seed(
                arm, use_adapters, seed,
                train=train, validation=validation, development_holdout=development_holdout,
                run_root=args.run_root, registry=registry,
            )
            results[key] = result
            print(
                f"{key}: OK ({result['wall_seconds']:.1f}s, "
                f"best_validation_loss={result['best_validation_loss']:.4f}, "
                f"dev_holdout_loss={result['development_holdout']['mean_loss']:.4f}, "
                f"stopped_early={result['stopped_early']})"
            )

    report = {
        "schema_version": 1,
        "corpus": str(JOINT_CORPUS_ROOT),
        "shared_model_config": SHARED_MODEL_CONFIG,
        "training_budget": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
        },
        "arms": args.arms,
        "seeds": args.seeds,
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
