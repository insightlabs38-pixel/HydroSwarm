"""core-issues3.txt Phase 12, Stage B -- flat control/OOD/auxiliary.

"Add: supported OOD category; preliminary control targets; corrected
auxiliaries. Run ablations: no auxiliaries; each auxiliary individually;
all validated auxiliaries. Retain an auxiliary only if it improves or does
not materially degrade primary tasks."

Scoping decisions, made explicit rather than left implicit:

- Corpus: `data/learning-v2/cycle-b2-control-v2` was considered first but
  REJECTED after actually inspecting its real target keys -- it carries
  only the SECOND-pass calibrated evidence_sufficiency/next_step labels
  (Phase 8/Section F/Stage C) and corrected event_cause, with NO
  ood_class/sensor_reconstruction/travel_time targets at all. Training
  against it would have made this ablation vacuous (the aux/no-aux arms
  would be computationally identical -- those loss terms would never
  reach `compute_multitask_loss` regardless of `task_weights`, since the
  targets are simply absent). The actually-correct corpus for "preliminary
  control targets [+] supported OOD category [+] corrected auxiliaries"
  is `data/learning-v2/cycle-b2` merged with `cycle-b2-trajectories-v3`'s
  per-example flat targets via `scripts/merge_trajectory_targets.py`
  (Phase 10.5's own mechanism) -- its evidence_sufficiency/next_step are
  the FIRST-pass (trajectory-derived) labels, i.e. genuinely "preliminary"
  relative to Stage C's second-pass ones, and it carries real ood_class/
  sensor_reconstruction/travel_time targets. The merge is a fast,
  deterministic join (no re-simulation) -- run once into
  `experiments/runs/v4-stage-b-control-ood-auxiliary/merged-corpus/
  {train,validation}` (gitignored, reproducible on demand from two
  already-committed inputs) before launching any ablation arm.
- One consistent architecture across all 4 ablation arms
  (event_control_heads=True, ood_category_head=True, auxiliary_heads=True
  -- every new head physically exists in every arm's model), ablated via
  `task_weights` alone (sensor_reconstruction/travel_time zeroed or not
  per arm) -- "Use identical manifests, seeds, budgets, and policies"
  (Phase 12's own requirement) is easiest to guarantee this way, and it
  avoids a fourth, subtly-different `verify_architecture_compatibility`
  identity per arm.
- Every arm initializes from the same Stage-A Sentinel teacher checkpoint
  (compatible-weight init, strict=False + fail-closed on any unexpected
  missing/extra key -- same discipline as scripts/train_control_heads.py's
  build_model_from_teacher) and trains WITH the backbone unfrozen (unlike
  train_control_heads.py's frozen-backbone ablation baseline): Stage B's
  whole point is joint multitask integration of brand-new heads
  (event_presence/event_cause/next_step/ood_category/sensor_reconstruction/
  travel_time all initialize from scratch), which needs real gradient flow
  through the shared backbone to mean anything -- a frozen backbone would
  only be testing whether a linear probe on frozen Stage-A features can
  approximate these targets, not whether joint training is safe/beneficial.
- Base task_weights come from configs/training.yaml via
  TrainingConfig.from_yaml(require_complete_task_weights=True) (Phase
  11.1) -- every ablation arm's task_weights differs from every other
  ONLY in sensor_reconstruction/travel_time, everything else identical.
"""

from __future__ import annotations

import argparse
import json
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
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2_SCHEMA_VERSION

VARIANT = "small"
OVERRIDES: dict[str, Any] = dict(
    prior_mode="feature_only", event_control_heads=True, ood_category_head=True, auxiliary_heads=True
)
SEED = 20260807
BATCH_SIZE = 16
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 3
MAXIMUM_RUNTIME_SECONDS = 2400.0
GRADNORM_LOG_EVERY_N_BATCHES = 20

#: New head parameter prefixes the Stage-A pure-Sentinel teacher checkpoint
#: never had -- verified empirically against the actual checkpoint
#: (experiments/runs/v4-stage-a-sentinel/.../checkpoint-0016/model.safetensors
#: has no event_presence_head/event_cause_head/next_step_head/
#: ood_category_head/sensor_reconstruction_head/future_concentration_head/
#: travel_time_head prefix at all), not merely assumed from the OVERRIDES
#: dict used to train it.
EXPECTED_NEW_HEAD_PREFIXES = (
    "event_presence_head.",
    "event_cause_head.",
    "next_step_head.",
    "ood_category_head.",
    "sensor_reconstruction_head.",
    "future_concentration_head.",
    "travel_time_head.",
)

#: Ablation arms (core-issues3.txt Phase 12 Stage B's exact required set):
#: no auxiliaries; each of the 2 validated auxiliaries individually; all
#: validated auxiliaries together. future_concentration is excluded from
#: every arm -- it is not a "validated auxiliary" (core-issues3.txt Phase
#: 7.4 / checkpoint_identity.py Section D item 11: its target generator
#: always returns an all-masked placeholder, so no arm's weight for it
#: would ever multiply a real loss regardless of value).
ABLATION_ARMS: dict[str, dict[str, float]] = {
    "no_aux": {"sensor_reconstruction": 0.0, "travel_time": 0.0},
    "aux_sensor_reconstruction_only": {"travel_time": 0.0},  # sensor_reconstruction keeps its configs/training.yaml weight
    "aux_travel_time_only": {"sensor_reconstruction": 0.0},  # travel_time keeps its configs/training.yaml weight
    "all_aux": {},  # both keep their configs/training.yaml weights, unmodified
}

#: Sentinel tasks this run's "does an auxiliary degrade primary tasks"
#: verdict is judged against -- the governed HydroSentinel objectives
#: (core-issues3.txt MISSION section), not every task_weights entry.
PRIMARY_TASK_VALIDATION_LOSS_KEYS = (
    "source_node",
    "source_region",
    "start_time",
    "duration",
    "relative_strength",
    "event_presence",
    "event_cause",
    "sensor_fault",
)


def _load_dataset(
    corpus_root: Path, tensors_dirname: str, split: str, *, limit: int | None = None
) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(corpus_root / tensors_dirname / split, expected_split=split)
    dataset.verify_shard_checksums()
    count = min(limit, len(dataset)) if limit is not None else len(dataset)
    examples = [dataset[index] for index in range(count)]
    return GovernedScenarioDataset(examples, expected_split=split)


def build_model_from_teacher(teacher_checkpoint: Path) -> tuple[HydroCore, dict[str, Any]]:
    model = HydroCore.from_variant(VARIANT, **OVERRIDES)
    teacher_state = load_file(str(teacher_checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(teacher_state, strict=False)
    unexpected_missing = [key for key in missing if not key.startswith(EXPECTED_NEW_HEAD_PREFIXES)]
    if unexpected_missing:
        raise RuntimeError(
            f"loading the Stage-A teacher checkpoint left unexpected missing keys "
            f"(not one of the known new v4 Stage-B heads): {unexpected_missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Stage-A teacher checkpoint has keys this v4 model does not: {unexpected} "
            "(architecture mismatch -- refusing to silently partial-load)"
        )
    return model, {"missing_keys": missing, "unexpected_keys": unexpected}


def _accuracy(predicted: list[int], truth: list[int]) -> float:
    return sum(1 for p, t in zip(predicted, truth, strict=True) if p == t) / len(truth) if truth else 0.0


def evaluate(model: HydroCore, validation: GovernedScenarioDataset, *, batch_size: int = 32) -> dict[str, Any]:
    model.eval()
    source_node_pred: list[int] = []
    source_node_truth: list[int] = []
    event_cause_pred: list[int] = []
    event_cause_truth: list[int] = []
    sensor_fault_correct = 0
    sensor_fault_total = 0
    sensor_reconstruction_squared_error = 0.0
    sensor_reconstruction_count = 0
    travel_time_squared_error = 0.0
    travel_time_count = 0

    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            batch_examples = [validation[index] for index in range(start, min(start + batch_size, len(validation)))]
            inputs, targets = collate_variable_topology(batch_examples)
            output = model(inputs)

            source_node_pred.extend(torch.argmax(output["source_node_logits"], dim=-1).tolist())
            source_node_truth.extend(targets["source_node"].tolist())

            if "event_cause_logits" in output and "event_cause" in targets:
                event_cause_pred.extend(torch.argmax(output["event_cause_logits"], dim=-1).tolist())
                event_cause_truth.extend(targets["event_cause"].tolist())

            if "sensor_fault_logits" in output and "sensor_fault" in targets:
                fault_target = targets["sensor_fault"].float()
                fault_mask = targets.get("sensor_fault_mask")
                valid = torch.isfinite(fault_target) & (fault_target >= 0)
                if fault_mask is not None:
                    valid = valid & fault_mask.bool()
                predicted = (torch.sigmoid(output["sensor_fault_logits"]) >= 0.5).float()
                sensor_fault_correct += int((predicted[valid] == fault_target[valid]).sum())
                sensor_fault_total += int(valid.sum())

            if "sensor_reconstruction_prediction" in output and "sensor_reconstruction" in targets:
                prediction = output["sensor_reconstruction_prediction"].float()
                target = targets["sensor_reconstruction"].float()
                mask = targets.get("sensor_reconstruction_mask")
                valid = torch.isfinite(target)
                if mask is not None:
                    valid = valid & mask.bool()
                sensor_reconstruction_squared_error += float(((prediction[valid] - target[valid]) ** 2).sum())
                sensor_reconstruction_count += int(valid.sum())

            if "travel_time_prediction" in output and "travel_time" in targets:
                prediction = output["travel_time_prediction"].float()
                target = targets["travel_time"].float()
                mask = targets.get("travel_time_mask")
                valid = torch.isfinite(target)
                if mask is not None:
                    valid = valid & mask.bool()
                travel_time_squared_error += float(((prediction[valid] - target[valid]) ** 2).sum())
                travel_time_count += int(valid.sum())

    return {
        "examples_evaluated": len(source_node_truth),
        "source_node_accuracy": _accuracy(source_node_pred, source_node_truth),
        "event_cause_accuracy": _accuracy(event_cause_pred, event_cause_truth) if event_cause_truth else None,
        "sensor_fault_accuracy": (sensor_fault_correct / sensor_fault_total) if sensor_fault_total else None,
        "sensor_reconstruction_mse": (
            sensor_reconstruction_squared_error / sensor_reconstruction_count if sensor_reconstruction_count else None
        ),
        "sensor_reconstruction_valid_count": sensor_reconstruction_count,
        "travel_time_mse": (travel_time_squared_error / travel_time_count if travel_time_count else None),
        "travel_time_valid_count": travel_time_count,
    }


def run_arm(
    arm_name: str,
    task_weight_overrides: dict[str, float],
    *,
    base_task_weights: dict[str, float],
    train: GovernedScenarioDataset,
    validation: GovernedScenarioDataset,
    teacher_checkpoint: Path,
    run_root: Path,
    registry: ExperimentRegistry,
    max_epochs: int = MAX_EPOCHS,
    maximum_runtime_seconds: float = MAXIMUM_RUNTIME_SECONDS,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_weights = {**base_task_weights, **task_weight_overrides}
    model, load_report = build_model_from_teacher(teacher_checkpoint)

    config = TrainingConfig(
        seed=SEED,
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,  # lower than train_control_heads.py's 3e-4 -- unfrozen backbone, avoid disturbing Stage-A's already-good Sentinel weights too fast
        warmup_steps=20,
        checkpoint_every_epochs=2,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        maximum_runtime_seconds=maximum_runtime_seconds,
        gradnorm_logging=True,
        gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES,
        task_weights=task_weights,
    )

    handle = registry.open_run(
        kind="training",
        purpose=f"core-issues3.txt Phase 12 Stage B ablation arm: {arm_name}",
        architecture="hydrocore",
        variant=VARIANT,
        seed=SEED,
        resolved_config={
            "arm": arm_name,
            "overrides": OVERRIDES,
            "teacher_checkpoint": str(teacher_checkpoint),
            **config.as_dict(),
        },
        manifest_hashes={"train": train.manifest_hash, "validation": validation.manifest_hash},
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        target_schema_hash=TARGETS_V2_SCHEMA_VERSION,
        workdir=".",
    )

    trainer = Trainer(
        model,
        train,
        validation_dataset=validation,
        config=config,
        run_root=run_root / arm_name,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    summary = trainer.fit()

    reload_model, _ = build_model_from_teacher(teacher_checkpoint)
    state_dict = load_file(str(Path(summary.final_checkpoint) / "model.safetensors"))
    if any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
        handle.close(exit_status="failed", notes="exported checkpoint contains non-finite weights")
        raise RuntimeError(f"[{arm_name}] exported checkpoint contains non-finite weights")
    reload_model.load_state_dict(state_dict, strict=True)
    verify_architecture_compatibility(reload_model, reload_model.architecture_config())

    metrics = evaluate(reload_model, validation)

    handle.close(
        exit_status="success",
        checkpoint_paths=[summary.final_checkpoint],
        selected_checkpoint=summary.final_checkpoint,
        selection_metric={"best_validation_loss": summary.best_validation_loss},
    )

    return {
        "arm": arm_name,
        "task_weight_overrides": task_weight_overrides,
        "run_id": handle.run_id,
        "load_report": load_report,
        "epochs_completed": summary.epochs_completed,
        "best_validation_loss": summary.best_validation_loss,
        "final_checkpoint": summary.final_checkpoint,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def summarize_verdicts(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """core-issues3.txt Phase 12 Stage B: "Retain an auxiliary only if it
    improves or does not materially degrade primary tasks." Compares each
    auxiliary-enabled arm's primary-task metrics against the no_aux
    baseline -- a simple, honest, non-statistical comparison (single seed,
    single run per arm; not a substitute for a real multi-seed significance
    test, which core-issues3.txt does not require at this stage)."""

    if "no_aux" not in results:
        return {"verdict": "inconclusive", "reason": "no_aux baseline arm did not complete"}
    baseline = results["no_aux"]["metrics"]
    verdicts: dict[str, Any] = {}
    for arm_name in ("aux_sensor_reconstruction_only", "aux_travel_time_only", "all_aux"):
        if arm_name not in results:
            verdicts[arm_name] = {"verdict": "inconclusive", "reason": "arm did not complete"}
            continue
        arm_metrics = results[arm_name]["metrics"]
        # "Materially degrade" threshold: primary-task accuracy dropping by
        # more than 2 percentage points relative to no_aux. Deliberately
        # conservative and documented, not tuned to make a particular
        # answer come out favorably.
        degradations = []
        for key in ("source_node_accuracy", "event_cause_accuracy", "sensor_fault_accuracy"):
            base_value, arm_value = baseline.get(key), arm_metrics.get(key)
            if base_value is None or arm_value is None:
                continue
            if arm_value < base_value - 0.02:
                degradations.append({"metric": key, "no_aux": base_value, arm_name: arm_value})
        verdicts[arm_name] = {
            "verdict": "retain" if not degradations else "do_not_retain",
            "degradations": degradations,
        }
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("experiments/runs/v4-stage-b-control-ood-auxiliary/merged-corpus"),
        help="a cycle-b2 + cycle-b2-trajectories-v3 merge (scripts/merge_trajectory_targets.py) -- "
        "see the module docstring for why cycle-b2-control-v2 is NOT the right corpus here",
    )
    parser.add_argument("--tensors-dirname", default="")
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path(
            "experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/"
            "checkpoints/checkpoint-0016/model.safetensors"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/v4-stage-b-control-ood-auxiliary"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/v4-stage-b.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/stage-b-training.json"))
    parser.add_argument(
        "--arms", nargs="+", default=list(ABLATION_ARMS), choices=list(ABLATION_ARMS),
        help="run only a subset of ablation arms (resumability: an already-written report's completed arms can be skipped by re-invoking with the remaining ones)",
    )
    parser.add_argument(
        "--train-limit", type=int, default=None,
        help="load only the first N train examples -- smoke-testing only, never for a real/promotion-quality run",
    )
    parser.add_argument(
        "--validation-limit", type=int, default=None,
        help="load only the first N validation examples -- smoke-testing only",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=MAX_EPOCHS,
        help="override MAX_EPOCHS -- smoke-testing only",
    )
    parser.add_argument(
        "--maximum-runtime-seconds", type=float, default=MAXIMUM_RUNTIME_SECONDS,
        help="override MAXIMUM_RUNTIME_SECONDS per arm -- smoke-testing only",
    )
    args = parser.parse_args()

    registry = ExperimentRegistry(args.registry)
    base_config = TrainingConfig.from_yaml("configs/training.yaml", require_complete_task_weights=True)
    base_task_weights = dict(base_config.task_weights)

    train = _load_dataset(args.corpus_root, args.tensors_dirname, "train", limit=args.train_limit)
    validation = _load_dataset(args.corpus_root, args.tensors_dirname, "validation", limit=args.validation_limit)

    started = time.perf_counter()
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for arm_name in args.arms:
        print(f"=== Stage B arm: {arm_name} ===", flush=True)
        try:
            results[arm_name] = run_arm(
                arm_name,
                ABLATION_ARMS[arm_name],
                base_task_weights=base_task_weights,
                train=train,
                validation=validation,
                teacher_checkpoint=args.teacher_checkpoint,
                run_root=args.run_root,
                registry=registry,
                max_epochs=args.max_epochs,
                maximum_runtime_seconds=args.maximum_runtime_seconds,
            )
            print(f"OK ({results[arm_name]['wall_seconds']:.1f}s): {json.dumps(results[arm_name]['metrics'])}")
        except Exception as error:  # noqa: BLE001 -- record and continue with remaining arms
            failures[arm_name] = f"{type(error).__name__}: {error}"
            print(f"FAILED [{arm_name}]: {failures[arm_name]}")

    report = {
        "schema_version": 1,
        "stage": "core-issues3.txt Phase 12 Stage B: flat control/OOD/auxiliary ablations",
        "corpus": str(args.corpus_root / args.tensors_dirname),
        "seed": SEED,
        "base_task_weights_source": "configs/training.yaml",
        "wall_seconds": time.perf_counter() - started,
        "results": results,
        "failures": failures,
        "verdicts": summarize_verdicts(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
