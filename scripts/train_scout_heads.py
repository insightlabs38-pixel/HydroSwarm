"""core-issues3.txt Phase 12 Stage D prep: train HydroScout's heads
(sample_node, information_gain, candidate_reduction, should_continue_sampling)
for the first time on real data.

Why this is needed before Stage D's policy comparison can include a
"learned Scout" arm at all: `sample_node_head`/`information_gain_head` are
unconditional base heads (constructed regardless of any flag), so the
existing Stage-A Sentinel checkpoint HAS them -- but Stage-A was trained
against `data/learning-v2/cycle-b2`, which carries no Scout targets at all
(confirmed by inspecting its real target keys), so those two heads never
received a real gradient there. `candidate_reduction_head`/
`should_continue_sampling_head` are gated behind `scout_control_heads=True`
and did not exist in the Stage-A checkpoint at all.

Staging (matches scripts/train_control_heads.py's established pattern):

1. v4 model with scout_control_heads=True, initialized from the Stage-A
   teacher checkpoint (strict=False -- sample_node_head/information_gain_head
   reload their EXISTING but untrained Stage-A weights;
   candidate_reduction_head/should_continue_sampling_head are the only
   genuinely new/missing parameters).
2. Freeze backbone: only the 4 Scout heads' own parameters are trainable
   (same conservative choice train_control_heads.py made for
   evidence_head/next_step_head -- this is a narrow supervised-head
   training pass, not a joint backbone fine-tune).
3. Train on data/learning-v2/cycle-b2-trajectories-v3/
   scout-tensors-normalized (Phase 10.2's real Scout-state dataset --
   ~30% of examples carry a real step-0 recommendation; the rest are
   correctly masked "no useful candidate" examples, per Phase 2 item 4).
4. Evaluate on validation: sample_node top-1 accuracy (among masked/valid
   examples only), information_gain/candidate_reduction MSE,
   should_continue_sampling accuracy.
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
OVERRIDES: dict[str, Any] = dict(prior_mode="feature_only", scout_control_heads=True)
SEED = 20260807
BATCH_SIZE = 16
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
MAXIMUM_RUNTIME_SECONDS = 2400.0

TRAINABLE_HEAD_PREFIXES = (
    "sample_node_head.",
    "information_gain_head.",
    "candidate_reduction_head.",
    "should_continue_sampling_head.",
)
#: sample_node_head/information_gain_head already exist (unconditional
#: base heads) in the Stage-A checkpoint -- only these two are genuinely
#: new/missing.
EXPECTED_NEW_HEAD_PREFIXES = ("candidate_reduction_head.", "should_continue_sampling_head.")

TASK_WEIGHTS: dict[str, float] = {
    "sample_node": 1.0,
    "information_gain": 0.5,
    "candidate_reduction": 0.5,
    "should_continue_sampling": 0.5,
}


def _load_dataset(corpus_root: Path, split: str) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(corpus_root / split, expected_split=split)
    dataset.verify_shard_checksums()
    examples = [dataset[index] for index in range(len(dataset))]
    return GovernedScenarioDataset(examples, expected_split=split)


def build_model_from_teacher(teacher_checkpoint: Path) -> tuple[HydroCore, dict[str, Any]]:
    model = HydroCore.from_variant(VARIANT, **OVERRIDES)
    teacher_state = load_file(str(teacher_checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(teacher_state, strict=False)
    unexpected_missing = [key for key in missing if not key.startswith(EXPECTED_NEW_HEAD_PREFIXES)]
    if unexpected_missing:
        raise RuntimeError(
            f"loading the Stage-A teacher checkpoint left unexpected missing keys "
            f"(not one of the known new Scout-control heads): {unexpected_missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Stage-A teacher checkpoint has keys this v4 model does not: {unexpected} "
            "(architecture mismatch -- refusing to silently partial-load)"
        )
    return model, {"missing_keys": missing, "unexpected_keys": unexpected}


def freeze_backbone(model: HydroCore) -> list[str]:
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_HEAD_PREFIXES):
            parameter.requires_grad_(True)
            trainable.append(name)
        else:
            parameter.requires_grad_(False)
    if not trainable:
        raise RuntimeError("no trainable parameters found under the 4 Scout head prefixes -- naming mismatch?")
    return trainable


def evaluate(model: HydroCore, validation: GovernedScenarioDataset, *, batch_size: int = 32) -> dict[str, Any]:
    model.eval()
    sample_node_correct = 0
    sample_node_total = 0
    ig_squared_error = 0.0
    ig_count = 0
    cr_squared_error = 0.0
    cr_count = 0
    continue_correct = 0
    continue_total = 0

    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            batch_examples = [validation[index] for index in range(start, min(start + batch_size, len(validation)))]
            inputs, targets = collate_variable_topology(batch_examples)
            output = model(inputs)

            sample_mask = targets["sample_node_mask"].bool()
            if sample_mask.any():
                predicted = torch.argmax(output["sample_node_logits"], dim=-1)
                truth = targets["sample_node"]
                sample_node_correct += int((predicted[sample_mask] == truth[sample_mask]).sum())
                sample_node_total += int(sample_mask.sum())

            ig_mask = targets["information_gain_mask"].bool()
            if ig_mask.any():
                prediction = output["expected_information_gain"].float()
                truth = targets["information_gain"].float()
                ig_squared_error += float(((prediction[ig_mask] - truth[ig_mask]) ** 2).sum())
                ig_count += int(ig_mask.sum())

            if "candidate_reduction_prediction" in output and "candidate_reduction" in targets:
                cr_mask = targets["candidate_reduction_mask"].bool()
                if cr_mask.any():
                    prediction = output["candidate_reduction_prediction"].float()
                    truth = targets["candidate_reduction"].float()
                    cr_squared_error += float(((prediction[cr_mask] - truth[cr_mask]) ** 2).sum())
                    cr_count += int(cr_mask.sum())

            if "should_continue_sampling_logits" in output and "should_continue_sampling" in targets:
                truth = targets["should_continue_sampling"].bool()
                predicted = (torch.sigmoid(output["should_continue_sampling_logits"]) >= 0.5).bool()
                continue_correct += int((predicted == truth).sum())
                continue_total += int(truth.numel())

    return {
        "examples_evaluated": len(validation),
        "sample_node_top1_accuracy": (sample_node_correct / sample_node_total) if sample_node_total else None,
        "sample_node_valid_count": sample_node_total,
        "information_gain_mse": (ig_squared_error / ig_count) if ig_count else None,
        "candidate_reduction_mse": (cr_squared_error / cr_count) if cr_count else None,
        "should_continue_sampling_accuracy": (continue_correct / continue_total) if continue_total else None,
    }


def run(
    *,
    corpus_root: Path,
    teacher_checkpoint: Path,
    run_root: Path,
    registry: ExperimentRegistry,
    max_epochs: int = MAX_EPOCHS,
    maximum_runtime_seconds: float = MAXIMUM_RUNTIME_SECONDS,
) -> dict[str, Any]:
    started = time.perf_counter()
    train = _load_dataset(corpus_root, "train")
    validation = _load_dataset(corpus_root, "validation")

    model, load_report = build_model_from_teacher(teacher_checkpoint)
    trainable_parameters = freeze_backbone(model)

    config = TrainingConfig(
        seed=SEED,
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=20,
        checkpoint_every_epochs=2,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        maximum_runtime_seconds=maximum_runtime_seconds,
        gradnorm_logging=False,
        task_weights=TASK_WEIGHTS,
    )

    handle = registry.open_run(
        kind="training",
        purpose="core-issues3.txt Phase 12 Stage D prep: train Scout heads "
        "(sample_node, information_gain, candidate_reduction, should_continue_sampling) "
        "from cycle-b2-trajectories-v3's scout-tensors-normalized, frozen Sentinel backbone "
        "initialized from the Stage-A teacher checkpoint",
        architecture="hydrocore",
        variant=VARIANT,
        seed=SEED,
        resolved_config={
            "overrides": OVERRIDES,
            "teacher_checkpoint": str(teacher_checkpoint),
            "trainable_parameters": trainable_parameters,
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
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    summary = trainer.fit()

    reload_model, _ = build_model_from_teacher(teacher_checkpoint)
    state_dict = load_file(str(Path(summary.final_checkpoint) / "model.safetensors"))
    if any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
        handle.close(exit_status="failed", notes="exported checkpoint contains non-finite weights")
        raise RuntimeError("exported checkpoint contains non-finite weights")
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
        "run_id": handle.run_id,
        "teacher_checkpoint": str(teacher_checkpoint),
        "trainable_parameters": trainable_parameters,
        "load_report": load_report,
        "epochs_completed": summary.epochs_completed,
        "best_validation_loss": summary.best_validation_loss,
        "final_checkpoint": summary.final_checkpoint,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized"),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path(
            "experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/"
            "checkpoints/checkpoint-0016/model.safetensors"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/v4-scout-heads"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/v4-scout-heads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/scout-heads-training.json"))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="smoke-testing only")
    parser.add_argument("--maximum-runtime-seconds", type=float, default=MAXIMUM_RUNTIME_SECONDS, help="smoke-testing only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    try:
        result = run(
            corpus_root=args.corpus_root,
            teacher_checkpoint=args.teacher_checkpoint,
            run_root=args.run_root,
            registry=registry,
            max_epochs=args.max_epochs,
            maximum_runtime_seconds=args.maximum_runtime_seconds,
        )
        failure: str | None = None
        print(f"OK ({result['wall_seconds']:.1f}s); metrics: {json.dumps(result['metrics'], indent=2)}")
    except Exception as error:  # noqa: BLE001 -- record and report, matching this repo's established smoke-job pattern
        result = None
        failure = f"{type(error).__name__}: {error}"
        print(f"FAILED: {failure}")

    report = {
        "schema_version": 1,
        "stage": "core-issues3.txt Phase 12 Stage D prep: Scout head training",
        "corpus": str(args.corpus_root),
        "seed": SEED,
        "wall_seconds": time.perf_counter() - started,
        "result": result,
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
