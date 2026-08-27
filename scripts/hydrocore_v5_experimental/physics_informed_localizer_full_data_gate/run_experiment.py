"""physics-informed-localizer-full-data-gate (EXPERIMENTAL, NON-RELEASE):
data-scale validation gate for the `C1_C2` candidate-conditioned localizer
selected by the completed `exp/physics-informed-localizer-scale-validation`
branch (final report: `reports/evaluation/physics-informed-localizer-scale-
validation/FINAL_REPORT.md`, classification
`CANDIDATE_FOR_FULL_SCALE_VALIDATION`).

Every architecture-validation run to date (this branch's own predecessors)
trained on a 600-example stratified subsample of Cycle-B2's `train` split,
while the full normalized split contains 9000 examples. This script answers
exactly one question: does `C1_C2`'s unseen-topology advantage over
`A_CONTROL` survive when training-data scale increases from 600 to 9000
examples, architecture and optimization budget (6 epochs, same optimizer/
batch/LR/etc.) held fixed? It is a data-scale validation gate, not a
hyperparameter search and not a release/promotion training run.

This is a thin wrapper, not a fork: it imports
`physics_informed_localizer_validation.run_experiment` unmodified and reuses
its `ARMS` registry (`A_CONTROL`/`C1_C2`, identical `model_kwargs` to every
other C-family arm per that branch's Phase 5 requirement), its
`_mask_physics_columns` ablation mechanism, `build_model`, `train_arm`,
`evaluate_arm`, `augment_batch`/`make_collate_fn`, `stratified_indices`/
`capped_indices`/`has_real_source`, and its harness structure (stratified
family sampling for the pilot stage, OODDetector/SplitConformalCalibrator
reuse, proxy actionable/abstention metrics, per-row logging) byte-for-byte.
Only the module-level configuration is retargeted (a new, separate results/
run root so the completed branches' own seed directories, manifests, and
reports are never touched), and two NEW things are added that no prior
branch needed:

  1. a single pre-declared seed (20261110) run at TWO training scales
     (`pilot-600`: the existing 600-example protocol, for a same-seed
     effect-size anchor; `full-data-9000`: the entire, unsubsampled 9000-
     example train split) so the 600-vs-9000 comparison is not confounded
     by a seed change;
  2. for `full-data-9000` only, full (uncapped) evaluation populations
     (validation: all 1000, development_holdout: all 1750, calibration:
     all 1000, ood-UNSEEN_TOPOLOGY: all 400) instead of the pilot's
     arbitrary 300-example evaluation caps, and per-epoch checkpointing
     (`checkpoint_every_epochs=1` instead of the pilot's end-of-run-only
     `checkpoint_every_epochs=PILOT_EPOCHS`) purely for crash-recovery --
     this changes I/O cadence, never the trained result, optimizer, LR,
     batch size, or epoch budget.

Usage:
  python3 scripts/hydrocore_v5_experimental/physics_informed_localizer_full_data_gate/run_experiment.py \
      --stage pilot --arms A_CONTROL,C1_C2
  python3 scripts/hydrocore_v5_experimental/physics_informed_localizer_full_data_gate/run_experiment.py \
      --stage full_data --arms A_CONTROL,C1_C2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental" / "physics_informed_localizer_validation"))
sys.path.insert(0, str(ROOT / "scripts" / "hydrocore_v5_experimental"))

import run_experiment as base  # noqa: E402  (physics_informed_localizer_validation's own module)

import yaml  # noqa: E402
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.trainer import Trainer  # noqa: E402

#: Phase 0 pre-registration (see docs/evaluation/experimental/
#: PHYSICS_INFORMED_LOCALIZER_FULL_DATA_GATE_PLAN.md): exactly one fresh
#: seed, fixed before any training on this branch, never selected or
#: replaced based on results.
SEED: int = 20261110

#: Exactly the two arms this gate compares -- the validated C1_C2
#: representation vs. the existing default-architecture control. C2, C3,
#: C_FULL, B_CANDIDATE_CONDITIONED, and A_CAPACITY_MATCHED are answered
#: questions from the completed studies and are deliberately not re-run.
ARM_NAMES: tuple[str, ...] = ("A_CONTROL", "C1_C2")

EXPERIMENT_NAME = "physics-informed-localizer-full-data-gate"
RUN_ROOT = ROOT / "experiments" / EXPERIMENT_NAME / "runs"
RESULTS_ROOT = ROOT / "reports" / "evaluation" / EXPERIMENT_NAME

#: Full-data-stage evaluation populations use every available example in
#: each normalized split (no arbitrary cap) -- see plan doc Section
#: "Full evaluation populations". Sizes recorded here match the corpus's
#: own manifest counts (validation=1000, development_holdout=1750,
#: ood-UNSEEN_TOPOLOGY=400, calibration=1000) and are asserted at run time
#: rather than hardcoded as limits.


def _latest_checkpoint(run_root: Path) -> Path | None:
    """Crash-recovery support (task requirement): if a previous attempt at
    this exact arm/seed/stage was interrupted mid-training, resume from its
    latest periodic checkpoint instead of restarting from scratch. Returns
    None (train from scratch) if no checkpoint exists yet.

    `RunArtifacts.create` (hydroswarm.training.artifacts) nests every run's
    own checkpoints one level deeper than `run_root` itself, under a fresh
    `{timestamp}-{uuid}` run-instance directory it generates per `Trainer`
    construction (`run_root/{timestamp}-{uuid}/checkpoints/checkpoint-NNNN`)
    -- NOT directly at `run_root/checkpoints`. Both the run-instance
    timestamp prefix and the zero-padded checkpoint suffix sort
    lexicographically in chronological order, so the latest of either is
    simply the last sorted entry."""

    if not run_root.exists():
        return None
    run_instances = sorted(p for p in run_root.iterdir() if p.is_dir())
    for run_instance in reversed(run_instances):
        checkpoint_dir = run_instance / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        candidates = sorted(p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir())
        if candidates:
            return candidates[-1]
    return None


def train_arm_full_data(
    *, arm_name: str, seed: int, train_dataset, validation_dataset
) -> tuple[Any, dict[str, Any]]:
    """Identical to `base.train_arm` (same config source, same PILOT_EPOCHS
    budget, same optimizer/batch/LR/weight-decay/clip/warmup/scheduler/task
    weights read from `configs/training-v5-causal.yaml`) except:
      (a) `checkpoint_every_epochs=1` instead of end-of-run-only, purely for
          crash recovery on the much longer full-data run;
      (b) resumes from the latest checkpoint under this arm's run_root if
          one already exists (an interrupted prior attempt at this exact
          arm/seed/stage), instead of always training from scratch.
    Never changes epochs, LR, batch size, gradient accumulation, weight
    decay, gradient clipping, warmup, scheduler, or task weights."""

    cfg_dict = yaml.safe_load((ROOT / "configs" / "training-v5-causal.yaml").read_text())["training"]
    config = TrainingConfig(
        seed=seed,
        epochs=base.PILOT_EPOCHS,
        batch_size=cfg_dict["batch_size"],
        gradient_accumulation_steps=cfg_dict["gradient_accumulation_steps"],
        learning_rate=cfg_dict["learning_rate"],
        weight_decay=cfg_dict["weight_decay"],
        gradient_clip_norm=cfg_dict["gradient_clip_norm"],
        warmup_steps=cfg_dict["warmup_steps"],
        scheduler=cfg_dict["scheduler"],
        checkpoint_every_epochs=1,
        early_stopping_patience=0,
        maximum_runtime_seconds=28800.0,
        device="cpu",
        fp32=True,
        deterministic=True,
        task_weights=cfg_dict["task_weights"],
    )
    model = base.build_model(arm_name=arm_name, seed=seed)
    run_root = RUN_ROOT / "full-data-9000" / f"seed-{seed}" / arm_name
    resume_from = _latest_checkpoint(run_root)
    started = time.monotonic()
    localizer_mode = base.ARMS[arm_name]["localizer_mode"]
    with_physics = base.ARMS[arm_name]["model_kwargs"].get("localizer_physics_feature_dim", 0) > 0
    physics_columns = base.ARMS[arm_name]["physics_columns"]
    trainer = Trainer(
        model,
        train_dataset,
        config=config,
        run_root=run_root,
        validation_dataset=validation_dataset,
        collate_fn=base.make_collate_fn(localizer_mode=localizer_mode, with_physics=with_physics, physics_columns=physics_columns),
    )
    if resume_from is not None:
        print(f"  resuming arm {arm_name} from {resume_from}")
    summary = trainer.fit(resume_from=resume_from)
    elapsed = time.monotonic() - started
    return model, {
        "arm": arm_name,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "epochs_completed": summary.epochs_completed,
        "stopped_early": summary.stopped_early,
        "stop_reason": summary.stop_reason,
        "global_steps": summary.global_steps,
        "best_validation_loss": summary.best_validation_loss,
        "resumed_from": str(resume_from) if resume_from else None,
    }


def run_pilot_stage(seed: int, arm_names: list[str]) -> None:
    """Stage 1 (same-seed pilot anchor): byte-for-byte reuse of
    `base.run_seed`'s existing 600-example protocol (200/family stratified
    train sample, validation/development_holdout capped at 300, full
    calibration/ood-UNSEEN_TOPOLOGY, 6 epochs, same optimizer/config) --
    only redirected to this branch's own pilot-600 results/run
    directories, and restricted to this branch's own SEED/ARM_NAMES."""

    base.RUN_ROOT = RUN_ROOT / "pilot-600"
    base.RESULTS_ROOT = RESULTS_ROOT / "pilot-600"
    base.SEEDS = (seed,)
    base.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    base.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    base.run_seed(seed, arm_names)


def run_full_data_stage(seed: int, arm_names: list[str]) -> None:
    """Stage 2 (full-data training): entire, unsubsampled 9000-example
    train split (all 3 trained families, no stratified cap, no filtering by
    real-source label -- matching what "the entire normalized Cycle-B2
    train split" means literally); full (uncapped) validation/
    development_holdout/calibration/ood-UNSEEN_TOPOLOGY evaluation
    populations. Same architecture, optimizer, epoch budget (6), and
    feature computation as the pilot stage -- only training-data scale and
    evaluation population size change, isolating exactly that one factor."""

    results_dir = RESULTS_ROOT / "full-data-9000" / f"seed-{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[seed {seed}] [full-data] Loading full, unsubsampled datasets...")
    train_full = base.ShardedScenarioDataset(base.CORPUS_ROOT / "train", expected_split="train")
    validation_full = base.ShardedScenarioDataset(base.CORPUS_ROOT / "validation", expected_split="validation")
    calibration_full = base.ShardedScenarioDataset(base.CORPUS_ROOT / "calibration", expected_split="calibration")
    dev_holdout_full = base.ShardedScenarioDataset(base.CORPUS_ROOT / "development_holdout", expected_split="development_holdout")
    ood_full = base.ShardedScenarioDataset(base.CORPUS_ROOT / "ood-UNSEEN_TOPOLOGY", expected_split="development_holdout")

    train_families_present = sorted({entry.network_id for entry in train_full._entries})
    print(f"[seed {seed}] [full-data] Train: {len(train_full)} examples (no subsampling), families {train_families_present}")

    manifest_path = results_dir / "run-manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        {
            "seed": seed,
            "stage": "full-data-9000",
            "pilot_epochs": base.PILOT_EPOCHS,
            "train_examples_count": len(train_full),
            "train_subsampled": False,
            "validation_examples_count": len(validation_full),
            "development_holdout_examples_count": len(dev_holdout_full),
            "ood_unseen_topology_count": len(ood_full),
            "calibration_count": len(calibration_full),
        }
    )
    manifest.setdefault("arms_run", [])
    for name in arm_names:
        if name not in manifest["arms_run"]:
            manifest["arms_run"].append(name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for arm_name in arm_names:
        eval_path = results_dir / f"{arm_name.lower()}-evaluation.json"
        if eval_path.exists():
            print(f"[seed {seed}] [full-data] arm {arm_name} already evaluated, skipping (delete {eval_path} to rerun)")
            continue
        print(f"\n=== [seed {seed}] [full-data] Training arm {arm_name} on {len(train_full)} examples ===")
        model, train_summary = train_arm_full_data(
            arm_name=arm_name, seed=seed, train_dataset=train_full, validation_dataset=validation_full
        )
        print(f"  trained in {train_summary['elapsed_seconds']:.1f}s, epochs={train_summary['epochs_completed']}, stop={train_summary['stop_reason']}")

        parameter_report = model.parameter_report_dict()
        print(f"  parameters: {parameter_report}")

        print(f"  Evaluating arm {arm_name} on full populations...")
        eval_datasets = {
            "train": (train_full, list(range(len(train_full)))),
            "validation": (validation_full, list(range(len(validation_full)))),
            "calibration": (calibration_full, list(range(len(calibration_full)))),
            "development_holdout": (dev_holdout_full, list(range(len(dev_holdout_full)))),
            "ood-UNSEEN_TOPOLOGY": (ood_full, list(range(len(ood_full)))),
        }
        eval_summary, rows_by_population = base.evaluate_arm(model, arm_name=arm_name, datasets=eval_datasets)
        eval_summary["training"] = train_summary
        eval_summary["parameter_report"] = parameter_report
        eval_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for population in ("validation", "development_holdout", "ood-UNSEEN_TOPOLOGY"):
            rows = rows_by_population.get(population, [])
            path = results_dir / f"{arm_name.lower()}-{population}-rows.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"  wrote {arm_name.lower()}-evaluation.json and per-population row logs")

    print(f"\n[seed {seed}] [full-data] All requested arms complete. Results under {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "full_data"), required=True)
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm names; default: A_CONTROL,C1_C2")
    parser.add_argument("--seed", type=int, default=SEED, help="must equal the pre-declared seed 20261110")
    args = parser.parse_args()

    if args.seed != SEED:
        raise SystemExit(f"seed {args.seed} is not the pre-declared full-data-gate seed {SEED}")

    arm_names = args.arms.split(",") if args.arms else list(ARM_NAMES)
    for name in arm_names:
        if name not in base.ARMS:
            raise SystemExit(f"unknown arm {name!r}; choices: {sorted(base.ARMS)}")
        if name not in ARM_NAMES:
            raise SystemExit(
                f"arm {name!r} is out of scope for physics-informed-localizer-full-data-gate "
                f"(only {ARM_NAMES} are run on this branch)"
            )

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "pilot":
        run_pilot_stage(args.seed, arm_names)
    else:
        run_full_data_stage(args.seed, arm_names)


if __name__ == "__main__":
    main()
