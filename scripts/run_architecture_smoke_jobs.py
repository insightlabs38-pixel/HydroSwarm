"""Bundle E, Stage 1: smoke and failure screening for E0, E3, E4, and E9
(overnight-plan.txt's experiment matrix), against the Cycle A corpus.

Per the plan: 1-2 epochs or a small fixed step budget, one seed, Cycle A
or a bounded Cycle B subset (a bounded Cycle A subset is used here).
Verifies loss decreases/finishes finite, every supervised head present in
this batch receives a nonzero gradient, checkpoint/resume works, and the
exported checkpoint reloads under the same architecture config it was
trained with (inference/calibration compatibility, in the narrower sense
of "the checkpoint schema itself is self-consistent" -- this script does
not run the full calibration pipeline, which requires more examples than
this smoke scale provides).

E9 ("prior/fusion diagnosis... no prior inside HydroCore; prior feature
only; log-prior residual only; both current pathways; external fusion
only") is covered by three of its four internal-pathway comparisons
(none/feature_only/logit_only) -- the fourth ("both current pathways",
i.e. prior_mode=feature_and_logit) is E0's own configuration and is not
re-run; "external fusion only" is an operational pipeline setting outside
HydroCore's own prior_mode axis, not itself a new architecture to smoke
test here.
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

CYCLE_A_ROOT = Path("data/learning-v2/cycle-a")
TRAIN_SUBSET = 200
VALIDATION_SUBSET = 50
GRADIENT_CHECK_BATCH = 8
SEED = 20260804

#: overnight-plan.txt's experiment matrix, restricted to the four
#: Bundle E asks Stage 1 to smoke-screen. Each maps directly onto a
#: Bundle D architecture-config flag; the missing values in each dict
#: use HydroCore's own defaults, which are E0's baseline by construction.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E0": {},
    "E3": {"incident_pooling": "source_conditioned"},
    "E4": {"message_direction": "dual_gated"},
    "E9-none": {"prior_mode": "none"},
    "E9-feature_only": {"prior_mode": "feature_only"},
    "E9-logit_only": {"prior_mode": "logit_only"},
}


def _load_subset(split: str, count: int) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(CYCLE_A_ROOT / "tensors" / split, expected_split=split)
    examples = [dataset[index] for index in range(min(count, len(dataset)))]
    return GovernedScenarioDataset(examples, expected_split=split)


def _gradient_check(model: HydroCore, train: GovernedScenarioDataset) -> dict[str, float]:
    inputs, targets = collate_variable_topology([train[index] for index in range(GRADIENT_CHECK_BATCH)])
    model.train()
    output = model(inputs)
    result = compute_multitask_loss(output, targets)
    if not torch.isfinite(result.total):
        raise RuntimeError("non-finite loss during pre-training gradient check")
    norms = task_gradient_norms(result.tasks, model)
    zero = [task for task, norm in norms.items() if norm == 0.0]
    if zero:
        raise RuntimeError(f"tasks present in this batch received zero gradient: {zero}")
    model.zero_grad(set_to_none=True)
    return norms


def run_experiment(
    name: str,
    overrides: dict[str, Any],
    *,
    train: GovernedScenarioDataset,
    validation: GovernedScenarioDataset,
    run_root: Path,
    registry: ExperimentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = HydroCore.from_variant("small", **overrides)
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
        purpose=f"Bundle E Stage 1 smoke screening: {name}",
        architecture="hydrocore",
        variant="small",
        seed=SEED,
        resolved_config={"experiment": name, "overrides": overrides, **first_config.as_dict()},
        manifest_hashes={"train": train.manifest_hash, "validation": validation.manifest_hash},
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        workdir=".",
    )

    first_trainer = Trainer(
        HydroCore.from_variant("small", **overrides),
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
        raise RuntimeError(f"{name}: non-finite validation loss after {first_summary.epochs_completed} epochs")

    resumed_config = TrainingConfig(epochs=3, **base_config)
    resumed_trainer = Trainer(
        HydroCore.from_variant("small", **overrides),
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
        raise RuntimeError(f"{name}: resume check failed ({resumed_summary!r})")

    reload_model = HydroCore.from_variant("small", **overrides)
    state_dict = load_file(str(Path(resumed_summary.final_checkpoint) / "model.safetensors"))
    if any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
        handle.close(exit_status="failed", notes="checkpoint contains non-finite weights")
        raise RuntimeError(f"{name}: exported checkpoint contains non-finite weights")
    reload_model.load_state_dict(state_dict, strict=True)
    verify_architecture_compatibility(reload_model, reload_model.architecture_config())

    handle.close(
        exit_status="success",
        checkpoint_paths=[first_summary.final_checkpoint, resumed_summary.final_checkpoint],
        selected_checkpoint=resumed_summary.final_checkpoint,
        selection_metric={"best_validation_loss": resumed_summary.best_validation_loss},
    )

    return {
        "experiment": name,
        "overrides": overrides,
        "run_id": handle.run_id,
        "gradient_norms": gradient_norms,
        "first_pass_epochs": first_summary.epochs_completed,
        "first_pass_best_validation_loss": first_summary.best_validation_loss,
        "resumed_epochs": resumed_summary.epochs_completed,
        "resumed_global_steps": resumed_summary.global_steps,
        "resumed_best_validation_loss": resumed_summary.best_validation_loss,
        "resume_ok": resume_ok,
        "checkpoint_reload_ok": True,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/bundle-e-smoke"))
    parser.add_argument(
        "--registry", type=Path, default=Path("experiments/registry/bundle-e-smoke.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/results/v3/architecture-smoke-jobs.json"))
    args = parser.parse_args()

    train = _load_subset("train", TRAIN_SUBSET)
    validation = _load_subset("validation", VALIDATION_SUBSET)
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name, overrides in EXPERIMENTS.items():
        try:
            results[name] = run_experiment(
                name, overrides, train=train, validation=validation, run_root=args.run_root, registry=registry
            )
            print(f"{name}: OK ({results[name]['wall_seconds']:.1f}s)")
        except Exception as error:  # noqa: BLE001 -- terminate this config, continue the sweep
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"{name}: FAILED ({failures[name]})")

    report = {
        "schema_version": 1,
        "stage": "Bundle E Stage 1 -- smoke and failure screening",
        "corpus": str(CYCLE_A_ROOT),
        "train_subset": TRAIN_SUBSET,
        "validation_subset": VALIDATION_SUBSET,
        "seed": SEED,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
        "failures": failures,
        "stop_gate": {
            # bool(results) guards against every check vacuously reporting
            # True when every experiment in the sweep failed before
            # producing a result -- "no results" must never read as "no
            # problems found".
            "no_nans": bool(results) and all(result["checkpoint_reload_ok"] for result in results.values()),
            "labels_receive_gradients": bool(results)
            and all(
                all(norm > 0 for norm in result["gradient_norms"].values()) for result in results.values()
            ),
            "all_jobs_reproducible": bool(results) and all(result["resume_ok"] for result in results.values()),
            "passed": not failures and bool(results) and len(results) == len(EXPERIMENTS),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["stop_gate"], indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
