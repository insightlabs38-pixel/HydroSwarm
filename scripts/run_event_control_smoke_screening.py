"""core-issues2.txt Phase 8 Stage 1: smoke and failure screening for the
full targets_v2 multitask configuration (event_control_heads=True,
auxiliary_heads=True), against a trajectory-target-enriched corpus
(scripts/generate_trajectory_corpus.py + scripts/merge_trajectory_targets.py).

Mirrors scripts/run_architecture_smoke_jobs.py's exact Bundle E Stage 1
pattern (1-2 epoch gradient/finite-loss check, checkpoint/resume
verification, reload-under-the-same-architecture-config check) rather than
inventing a new one. The one real difference from the earlier E8 run
(scripts/run_stage2_architecture_screening.py's docstring: "Cycle B has no
Scout/Strategist/auxiliary-objective labels, so those loss terms cannot
fire regardless of which heads are enabled") is that a trajectory-enriched
corpus DOES carry real ood_class/next_step/evidence_sufficiency/
sensor_reconstruction/future_concentration/travel_time labels now -- this
is the first run where those loss terms can genuinely fire, not merely
exist unused. Scout/Strategist sequential targets still cannot fire here:
they are not flat per-example tensors and are not merged into these
shards (see merge_trajectory_targets.py's own docstring) -- a distinct,
larger training-loop extension, not attempted by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from hydroswarm.model import HydroCore, verify_architecture_compatibility
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training import (
    GovernedScenarioDataset,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.losses import compute_multitask_loss, task_gradient_norms
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION

TRAIN_SUBSET = 200
VALIDATION_SUBSET = 50
GRADIENT_CHECK_BATCH = 8
SEED = 20260806

#: The configuration this script screens: full event/control + auxiliary
#: heads enabled together (overnight-plan.txt's E8, extended -- "add
#: validated auxiliary objectives" is Phase 8 stage 6, folded in here since
#: this smoke pass is specifically about whether the new corpus's labels
#: light up every new head, not about ranking architectures).
OVERRIDES: dict[str, Any] = {"event_control_heads": True, "auxiliary_heads": True}

#: The exact new targets this run exists to confirm receive gradients --
#: the previous E8 run (against Cycle B, no trajectory labels) could not
#: check any of these; they were simply absent from every batch.
NEW_TASKS = (
    "event_presence",
    "event_cause",
    "next_step",
    "ood_class",
    "evidence_sufficiency",
    "sensor_reconstruction",
    "future_concentration",
    "travel_time",
)


def _load_subset(corpus_root: Path, tensors_dirname: str, split: str, count: int) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(corpus_root / tensors_dirname / split, expected_split=split)
    examples = [dataset[index] for index in range(min(count, len(dataset)))]
    return GovernedScenarioDataset(examples, expected_split=split)


def _gradient_check(model: HydroCore, train: GovernedScenarioDataset) -> dict[str, float]:
    inputs, targets = collate_variable_topology([train[index] for index in range(min(GRADIENT_CHECK_BATCH, len(train)))])
    present_new_tasks = sorted(set(NEW_TASKS) & set(targets))
    model.train()
    output = model(inputs)
    result = compute_multitask_loss(output, targets)
    if not torch.isfinite(result.total):
        raise RuntimeError("non-finite loss during pre-training gradient check")
    norms = task_gradient_norms(result.tasks, model)
    zero = [task for task, norm in norms.items() if norm == 0.0]
    if zero:
        raise RuntimeError(f"tasks present in this batch received zero gradient: {zero}")
    missing_new_tasks = [task for task in present_new_tasks if task not in norms]
    if missing_new_tasks:
        raise RuntimeError(
            f"new Phase 4-6 targets present in the batch but never reached compute_multitask_loss: "
            f"{missing_new_tasks} (a head-gating or loss-key-naming defect)"
        )
    model.zero_grad(set_to_none=True)
    return norms


def run_smoke_screening(
    *, train: GovernedScenarioDataset, validation: GovernedScenarioDataset, run_root: Path, registry: ExperimentRegistry
) -> dict[str, Any]:
    started = time.perf_counter()
    model = HydroCore.from_variant("small", **OVERRIDES)
    gradient_norms = _gradient_check(model, train)

    base_config = dict(
        seed=SEED,
        batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=2,
        checkpoint_every_epochs=1,
        maximum_runtime_seconds=600.0,
    )
    first_config = TrainingConfig(epochs=2, **base_config)

    handle = registry.open_run(
        kind="training",
        purpose="core-issues2.txt Phase 8 Stage 1 smoke screening: event_control_heads + auxiliary_heads "
        "against a trajectory-target-enriched corpus",
        architecture="hydrocore",
        variant="small",
        seed=SEED,
        resolved_config={"overrides": OVERRIDES, **first_config.as_dict()},
        manifest_hashes={"train": train.manifest_hash, "validation": validation.manifest_hash},
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        workdir=".",
    )

    first_trainer = Trainer(
        HydroCore.from_variant("small", **OVERRIDES),
        train,
        validation_dataset=validation,
        config=first_config,
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    first_summary = first_trainer.fit()
    if not math.isfinite(first_summary.best_validation_loss):
        handle.close(exit_status="failed", notes="non-finite validation loss")
        raise RuntimeError(f"non-finite validation loss after {first_summary.epochs_completed} epochs")

    resumed_config = TrainingConfig(epochs=3, **base_config)
    resumed_trainer = Trainer(
        HydroCore.from_variant("small", **OVERRIDES),
        train,
        validation_dataset=validation,
        config=resumed_config,
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    resumed_summary = resumed_trainer.fit(resume_from=Path(first_summary.final_checkpoint))
    resume_ok = (
        resumed_summary.epochs_completed == 3
        and resumed_summary.global_steps > first_summary.global_steps
        and math.isfinite(resumed_summary.best_validation_loss)
    )
    if not resume_ok:
        handle.close(exit_status="failed", notes="resume did not advance training as expected")
        raise RuntimeError(f"resume check failed ({resumed_summary!r})")

    reload_model = HydroCore.from_variant("small", **OVERRIDES)
    state_dict = load_file(str(Path(resumed_summary.final_checkpoint) / "model.safetensors"))
    if any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
        handle.close(exit_status="failed", notes="checkpoint contains non-finite weights")
        raise RuntimeError("exported checkpoint contains non-finite weights")
    reload_model.load_state_dict(state_dict, strict=True)
    verify_architecture_compatibility(reload_model, reload_model.architecture_config())

    handle.close(
        exit_status="success",
        checkpoint_paths=[first_summary.final_checkpoint, resumed_summary.final_checkpoint],
        selected_checkpoint=resumed_summary.final_checkpoint,
        selection_metric={"best_validation_loss": resumed_summary.best_validation_loss},
    )

    return {
        "overrides": OVERRIDES,
        "run_id": handle.run_id,
        "gradient_norms": gradient_norms,
        "new_tasks_with_gradient": sorted(set(NEW_TASKS) & set(gradient_norms)),
        "first_pass_epochs": first_summary.epochs_completed,
        "first_pass_best_validation_loss": first_summary.best_validation_loss,
        "resumed_epochs": resumed_summary.epochs_completed,
        "resumed_global_steps": resumed_summary.global_steps,
        "resumed_best_validation_loss": resumed_summary.best_validation_loss,
        "resume_ok": resume_ok,
        "checkpoint_reload_ok": True,
        "wall_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--tensors-dirname", type=str, default="tensors")
    parser.add_argument("--train-subset", type=int, default=TRAIN_SUBSET)
    parser.add_argument("--validation-subset", type=int, default=VALIDATION_SUBSET)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/event-control-smoke"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/event-control-smoke.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v3/event-control-smoke-screening.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    train = _load_subset(args.corpus_root, args.tensors_dirname, "train", args.train_subset)
    validation = _load_subset(args.corpus_root, args.tensors_dirname, "validation", args.validation_subset)
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    try:
        result = run_smoke_screening(train=train, validation=validation, run_root=args.run_root, registry=registry)
        failure: str | None = None
        print(f"OK ({result['wall_seconds']:.1f}s); new tasks with gradient: {result['new_tasks_with_gradient']}")
    except Exception as error:  # noqa: BLE001 -- record and report, matching Bundle E's own smoke-job pattern
        result = None
        failure = f"{type(error).__name__}: {error}"
        print(f"FAILED: {failure}")

    report = {
        "schema_version": 1,
        "stage": "core-issues2.txt Phase 8 Stage 1 -- event/control/auxiliary smoke screening",
        "corpus": str(args.corpus_root / args.tensors_dirname),
        "train_subset": args.train_subset,
        "validation_subset": args.validation_subset,
        "seed": SEED,
        "wall_seconds": time.perf_counter() - started,
        "result": result,
        "failure": failure,
        "stop_gate": {
            "no_nans": result is not None and result["checkpoint_reload_ok"],
            "labels_receive_gradients": result is not None and all(norm > 0 for norm in result["gradient_norms"].values()),
            "new_tasks_all_receive_gradients": result is not None
            and set(result["new_tasks_with_gradient"]) == set(NEW_TASKS) & set(result["gradient_norms"]),
            "reproducible": result is not None and result["resume_ok"],
            "passed": failure is None and result is not None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["stop_gate"], indent=2, sort_keys=True))
    return 0 if report["stop_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
