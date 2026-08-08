"""core-issues3.txt Phase 12 Stage G: conditional HydroCore-M scaling screen.

Stage G's own gate text: run HydroCore-M ONLY if HydroCore-S shows a
capacity-limited case -- "train and validation losses both remain high; no
major overfitting; persistent underfitting on important tasks; short
scaling screen improves operational metrics; CPU/RAM cost remains
acceptable."

Pre-screen analysis (see reports/results/v4/phase12-stage-g-decision.md for
the full writeup) of the already-completed Stage F `no_adapters` runs'
per-batch metrics.jsonl found NO evidence of a plateaued/capacity-limited
S: every one of the 21 real tasks' mean per-epoch loss was still declining
at a healthy rate through the final (16th) epoch, and none of Stage F's 4
runs early-stopped (all hit the epoch/time ceiling still improving) --
this looks like an epoch/time-budget-limited regime, not a capacity
ceiling, which argues AGAINST running HydroCore-M. This script performs the
gate's own explicitly-named empirical check ("short scaling screen
improves operational metrics") to confirm or overturn that reading with
real evidence rather than resting on the reasoning alone -- a short,
CHEAP screen (few epochs, single seed, small runtime ceiling), not a
promotion-quality run.

Same joint-v4 corpus, same SHARED_MODEL_CONFIG, same `use_adapters=False`
arm (Stage F's own established winner) as scripts/run_stage_f_training.py
-- only `variant` ("small" vs "medium") and the epoch/runtime budget
differ, so the comparison is apples-to-apples at a matched, small step
budget.
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
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training import ScenarioDatasetView, ShardedScenarioDataset, Trainer, TrainingConfig, collate_variable_topology
from hydroswarm.training.losses import compute_multitask_loss
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION

JOINT_CORPUS_ROOT = Path("data/learning-v2/cycle-b2-joint-v4")
TENSORS_DIRNAME = "tensors-normalized"

BATCH_SIZE = 16
SCREEN_EPOCHS = 3
SCREEN_RUNTIME_CEILING_SECONDS = 2400.0
SEED = 20260810

SHARED_MODEL_CONFIG: dict[str, Any] = {
    "prior_mode": "feature_only",
    "event_control_heads": True,
    "scout_control_heads": True,
    "strategist_mode": "candidate_conditioned",
    "action_vocabulary_size": 9,
    "consequence_prescreening_heads": True,
    "ood_category_head": True,
}
VARIANTS: tuple[str, ...] = ("small", "medium")


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


@torch.no_grad()
def _evaluate(model: torch.nn.Module, dataset: ScenarioDatasetView, *, batch_size: int) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    task_loss_sums: dict[str, float] = {}
    task_loss_counts: dict[str, int] = {}
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
    return {"mean_loss": float(sum(losses) / len(losses)) if losses else math.nan, "per_task_mean_loss": per_task, "examples": len(dataset)}


def run_variant(
    variant: str,
    seed: int,
    *,
    train: ScenarioDatasetView,
    validation: ScenarioDatasetView,
    run_root: Path,
    registry: ExperimentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = TrainingConfig(
        seed=seed,
        epochs=SCREEN_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=30,
        checkpoint_every_epochs=SCREEN_EPOCHS,
        early_stopping_patience=SCREEN_EPOCHS,  # a 3-epoch screen must not early-stop before finishing
        maximum_runtime_seconds=SCREEN_RUNTIME_CEILING_SECONDS,
        gradnorm_log_every_n_batches=1_000_000,  # screen: effectively disabled, no diagnostics overhead
    )
    model = HydroCore.from_variant(variant, use_adapters=False, **SHARED_MODEL_CONFIG)
    parameter_count = sum(p.numel() for p in model.parameters())

    handle = registry.open_run(
        kind="training",
        purpose=f"Phase 12 Stage G scaling screen: {variant}",
        architecture="hydrocore",
        variant=variant,
        seed=seed,
        resolved_config={"variant": variant, "use_adapters": False, **SHARED_MODEL_CONFIG, **config.as_dict()},
        manifest_hashes={"train": train.manifest_hash, "validation": validation.manifest_hash},
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        topology_hashes=_topology_hashes(train, validation),
        workdir=".",
    )

    run_dir = run_root / f"{variant}-seed{seed}"
    trainer = Trainer(model, train, validation_dataset=validation, config=config, run_root=run_dir, workdir=".", collate_fn=collate_variable_topology)
    summary = trainer.fit()
    if not math.isfinite(summary.best_validation_loss):
        handle.close(exit_status="failed", notes="non-finite validation loss")
        raise RuntimeError(f"{variant}: non-finite validation loss")

    validation_metrics = _evaluate(model, validation, batch_size=BATCH_SIZE)
    model_hash = HybridInferencePipeline._fingerprint_model(model)

    handle.close(
        exit_status="success",
        checkpoint_paths=tuple(dict.fromkeys(p for p in (summary.final_checkpoint, summary.last_resumable_checkpoint, summary.export_path) if p)),
        checkpoint_hashes={summary.export_path: summary.export_sha256},
        selected_checkpoint=summary.export_path,
        selection_metric={"best_validation_loss": summary.best_validation_loss},
    )

    return {
        "variant": variant,
        "parameter_count": int(parameter_count),
        "seed": seed,
        "run_id": handle.run_id,
        "model_hash": model_hash,
        "checkpoint": summary.export_path,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "best_validation_loss": summary.best_validation_loss,
        "validation_full_pass": validation_metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/stage-g-scaling-screen"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/stage-g-scaling-screen.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/stage-g-scaling-screen.json"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train = _load_dataset("train")
    validation = _load_dataset("validation")

    registry = ExperimentRegistry(args.registry)
    results: dict[str, Any] = {}
    started = time.perf_counter()
    for variant in args.variants:
        result = run_variant(variant, args.seed, train=train, validation=validation, run_root=args.run_root, registry=registry)
        results[variant] = result
        print(
            f"{variant}: params={result['parameter_count']:,} best_validation_loss={result['best_validation_loss']:.4f} "
            f"validation_full_pass_mean_loss={result['validation_full_pass']['mean_loss']:.4f} wall={result['wall_seconds']:.1f}s"
        )

    report = {
        "schema_version": 1,
        "corpus": str(JOINT_CORPUS_ROOT),
        "shared_model_config": SHARED_MODEL_CONFIG,
        "screen_budget": {"epochs": SCREEN_EPOCHS, "batch_size": BATCH_SIZE, "runtime_ceiling_seconds": SCREEN_RUNTIME_CEILING_SECONDS, "seed": args.seed},
        "results": results,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
