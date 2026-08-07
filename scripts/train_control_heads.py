"""core-issues4.txt Section F step 6b / core-issues3.txt Phase 8 step 6:
train v4 event_control_heads (evidence_sufficiency, next_step) from the
corrected data/learning-v2/cycle-b2-control-v2 corpus, with the Sentinel
backbone frozen and initialized from the Stage-A checkpoint
(experiments/runs/v4-stage-a-sentinel/E1-seed20260810's selected
checkpoint).

Staging (exactly Section F's explicit order):

1. v4 model with event/control heads enabled
   (HydroCore.from_variant("small", prior_mode="feature_only",
   event_control_heads=True)).
2. Initialize compatible Sentinel weights explicitly: load the Stage-A
   checkpoint's state dict with strict=False, then assert the ONLY missing
   keys are the new event/control head parameters the Stage-A checkpoint
   never had (fail closed on anything else -- a silent partial load would
   be exactly the kind of unverified checkpoint-identity gap Phase 9 exists
   to close).
3. Freeze backbone: every parameter except evidence_head.*/next_step_head.*
   gets requires_grad_(False). task_weights zeroes every task loss except
   evidence_sufficiency/next_step as a second, independent guarantee (not
   merely relying on frozen parameters to make other losses inert).
4. Train.
5. Evaluate on validation: evidence-sufficiency accuracy/F1, next-step
   macro F1 + per-class recall, policy agreement (next_step_head's own
   prediction vs. hydroswarm.training.control_labels.classify_next_step
   re-derived from THIS model's own evidence_sufficiency prediction --
   checks the two jointly-trained-but-architecturally-independent heads
   stay internally consistent with the governed policy, not just each
   individually accurate), unsafe non-abstention count and GENERATE_PLANS-
   with-empty-candidate-set count (joined against the persisted second-pass
   labels' calibrated_candidate_set_size by scenario_id), evidence_sufficiency
   output calibration (ECE).
6. (not run automatically here -- a separate, explicit follow-up invocation
   with --joint-finetune, only if step 5's metrics justify it, per Section
   F's own "optionally run a low-learning-rate joint fine-tune only if
   validation and safety metrics justify it").
7. The frozen-backbone result IS retained as the ablation baseline (this
   run's own report -- there is nothing else to separately produce for
   that requirement).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from hydroswarm.calibration.conformal import expected_calibration_error
from hydroswarm.model import HydroCore, verify_architecture_compatibility
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.training import (
    GovernedScenarioDataset,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.control_labels import MAXIMUM_SAMPLING_ROUNDS, classify_next_step
from hydroswarm.training.registry import ExperimentRegistry
from hydroswarm.training.targets_v2 import TARGETS_V2, TARGETS_V2_SCHEMA_VERSION, EventCause, NextStep

VARIANT = "small"
OVERRIDES: dict[str, Any] = {"prior_mode": "feature_only", "event_control_heads": True}
SEED = 20260807
BATCH_SIZE = 16
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
MAXIMUM_RUNTIME_SECONDS = 3600.0

#: Only these two heads' own parameters are unfrozen -- see module
#: docstring step 3. Both are the exact heads the labels this script trains
#: from actually supervise (evidence_head is unconditional in HydroCore,
#: shape [d_model] -> 1 probability; next_step_head only exists when
#: event_control_heads=True).
TRAINABLE_HEAD_PREFIXES = ("evidence_head.", "next_step_head.")

#: Every other head's checkpoint parameters are frozen but Stage-A never
#: trained event_presence_head/event_cause_head/next_step_head at all (they
#: did not exist in a model built without event_control_heads) -- these are
#: the only keys allowed to be "missing" from the Stage-A state dict.
EXPECTED_NEW_HEAD_PREFIXES = ("event_presence_head.", "event_cause_head.", "next_step_head.")

TASK_WEIGHTS: dict[str, float] = {name: 0.0 for name in TARGETS_V2}
TASK_WEIGHTS["evidence_sufficiency"] = 1.0
TASK_WEIGHTS["next_step"] = 1.0

_NEXT_STEP_NAMES = [member.value for member in NextStep]
_EVENT_CAUSE_BY_INDEX = list(EventCause)


def _load_dataset(corpus_root: Path, tensors_dirname: str, split: str) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(corpus_root / tensors_dirname / split, expected_split=split)
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
            f"(not one of the known new v4 control heads): {unexpected_missing}"
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
        raise RuntimeError("no trainable parameters found under evidence_head./next_step_head. -- naming mismatch?")
    return trainable


def _binary_f1(predicted: list[bool], truth: list[bool]) -> dict[str, float]:
    tp = sum(1 for p, t in zip(predicted, truth, strict=True) if p and t)
    fp = sum(1 for p, t in zip(predicted, truth, strict=True) if p and not t)
    fn = sum(1 for p, t in zip(predicted, truth, strict=True) if not p and t)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = sum(1 for p, t in zip(predicted, truth, strict=True) if p == t) / len(truth) if truth else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _macro_f1_and_per_class_recall(
    predicted: list[int], truth: list[int], *, class_count: int, class_names: list[str]
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    f1_scores = []
    for class_index in range(class_count):
        tp = sum(1 for p, t in zip(predicted, truth, strict=True) if p == class_index and t == class_index)
        fp = sum(1 for p, t in zip(predicted, truth, strict=True) if p == class_index and t != class_index)
        fn = sum(1 for p, t in zip(predicted, truth, strict=True) if p != class_index and t == class_index)
        support = sum(1 for t in truth if t == class_index)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[class_names[class_index]] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
        if support > 0:
            f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    accuracy = sum(1 for p, t in zip(predicted, truth, strict=True) if p == t) / len(truth) if truth else 0.0
    return {"macro_f1": macro_f1, "accuracy": accuracy, "per_class": per_class}


def _load_second_pass_candidate_set_sizes(second_pass_dir: Path, split: str) -> dict[str, int]:
    path = second_pass_dir / f"{split}.jsonl"
    sizes: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sizes[record["scenario_id"]] = record["calibrated_candidate_set_size"]
    return sizes


def evaluate(
    model: HydroCore,
    validation: GovernedScenarioDataset,
    *,
    second_pass_candidate_set_sizes: dict[str, int],
    batch_size: int = 32,
) -> dict[str, Any]:
    model.eval()
    evidence_pred: list[bool] = []
    evidence_truth: list[bool] = []
    evidence_confidence: list[float] = []
    next_step_pred: list[int] = []
    next_step_truth: list[int] = []
    policy_agreement_matches = 0
    unsafe_non_abstention_count = 0
    generate_plans_with_empty_candidate_set = 0
    scenario_ids: list[str] = []

    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            batch_examples = [validation[index] for index in range(start, min(start + batch_size, len(validation)))]
            inputs, targets = collate_variable_topology(batch_examples)
            output = model(inputs)

            evidence_prob = output["evidence_sufficiency"]
            next_step_logits = output["next_step_logits"]
            next_step_predicted = torch.argmax(next_step_logits, dim=-1)

            for row, example in enumerate(batch_examples):
                scenario_ids.append(example.scenario_id)
                p_sufficient = float(evidence_prob[row])
                predicted_sufficient = p_sufficient >= 0.5
                truth_sufficient = bool(targets["evidence_sufficiency"][row])
                evidence_pred.append(predicted_sufficient)
                evidence_truth.append(truth_sufficient)
                evidence_confidence.append(max(p_sufficient, 1.0 - p_sufficient))

                predicted_step = int(next_step_predicted[row])
                truth_step = int(targets["next_step"][row])
                next_step_pred.append(predicted_step)
                next_step_truth.append(truth_step)

                event_cause = _EVENT_CAUSE_BY_INDEX[int(targets["event_cause"][row])]
                candidate_set_size = second_pass_candidate_set_sizes.get(example.scenario_id)
                calibration_valid = candidate_set_size is not None and candidate_set_size > 0
                # Recomputing calibration_valid strictly from candidate_set_size > 0
                # slightly understates "calibration invalid but a stale candidate
                # set size of 0" -- but calibrated_candidate_set_size is already 0
                # in exactly that case (second_pass_control_labels.py forces
                # effective_candidate_set_size=0 when calibration_valid is False),
                # so this is an exact, not approximate, reconstruction.
                policy_step = classify_next_step(
                    ood_level_outside_validated_range=not calibration_valid,
                    evidence_sufficient=predicted_sufficient,
                    sample_count=0,
                    event_cause=event_cause,
                    maximum_sampling_rounds=MAXIMUM_SAMPLING_ROUNDS,
                )
                if _NEXT_STEP_NAMES[predicted_step] == policy_step.value:
                    policy_agreement_matches += 1

                if _NEXT_STEP_NAMES[predicted_step] == NextStep.GENERATE_PLANS.value:
                    if candidate_set_size is not None and candidate_set_size == 0:
                        unsafe_non_abstention_count += 1
                        generate_plans_with_empty_candidate_set += 1

    evidence_metrics = _binary_f1(evidence_pred, evidence_truth)
    next_step_metrics = _macro_f1_and_per_class_recall(
        next_step_pred, next_step_truth, class_count=len(NextStep), class_names=_NEXT_STEP_NAMES
    )
    evidence_ece = expected_calibration_error(
        evidence_confidence, [p == t for p, t in zip(evidence_pred, evidence_truth, strict=True)]
    )

    return {
        "examples_evaluated": len(scenario_ids),
        "evidence_sufficiency": evidence_metrics,
        "evidence_sufficiency_ece": evidence_ece,
        "next_step": next_step_metrics,
        "policy_agreement": policy_agreement_matches / len(scenario_ids) if scenario_ids else 0.0,
        "unsafe_non_abstention_count": unsafe_non_abstention_count,
        "generate_plans_with_empty_candidate_set_count": generate_plans_with_empty_candidate_set,
    }


def run(
    *,
    corpus_root: Path,
    tensors_dirname: str,
    teacher_checkpoint: Path,
    teacher_checkpoint_hash: str,
    second_pass_dir: Path,
    run_root: Path,
    registry: ExperimentRegistry,
) -> dict[str, Any]:
    started = time.perf_counter()
    train = _load_dataset(corpus_root, tensors_dirname, "train")
    validation = _load_dataset(corpus_root, tensors_dirname, "validation")

    model, load_report = build_model_from_teacher(teacher_checkpoint)
    trainable_parameters = freeze_backbone(model)

    config = TrainingConfig(
        seed=SEED,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=20,
        checkpoint_every_epochs=2,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        maximum_runtime_seconds=MAXIMUM_RUNTIME_SECONDS,
        gradnorm_logging=False,
        task_weights=TASK_WEIGHTS,
    )

    handle = registry.open_run(
        kind="training",
        purpose="core-issues4.txt Section F step 6b: train v4 event_control_heads "
        "(evidence_sufficiency, next_step) from cycle-b2-control-v2, frozen Sentinel "
        "backbone initialized from the Stage-A teacher checkpoint",
        architecture="hydrocore",
        variant=VARIANT,
        seed=SEED,
        resolved_config={
            "overrides": OVERRIDES,
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_checkpoint_hash": teacher_checkpoint_hash,
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

    second_pass_candidate_set_sizes = _load_second_pass_candidate_set_sizes(second_pass_dir, "validation")
    metrics = evaluate(reload_model, validation, second_pass_candidate_set_sizes=second_pass_candidate_set_sizes)

    handle.close(
        exit_status="success",
        checkpoint_paths=[summary.final_checkpoint],
        selected_checkpoint=summary.final_checkpoint,
        selection_metric={"best_validation_loss": summary.best_validation_loss},
    )

    return {
        "run_id": handle.run_id,
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_hash": teacher_checkpoint_hash,
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
    parser.add_argument("--corpus-root", type=Path, default=Path("data/learning-v2/cycle-b2-control-v2"))
    parser.add_argument("--tensors-dirname", default="tensors-normalized")
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path(
            "experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/"
            "checkpoints/checkpoint-0016/model.safetensors"
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint-hash",
        default="ca31dd665908fd6e7c2797c22ffc708bfd436162ca0367dba3ddc670df5ad9de",
    )
    parser.add_argument(
        "--second-pass-dir", type=Path, default=Path("data/learning-v2/cycle-b2-control-v2/second-pass-labels")
    )
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/v4-control-heads"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/v4-control-heads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/control-heads-training.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ExperimentRegistry(args.registry)

    started = time.perf_counter()
    try:
        result = run(
            corpus_root=args.corpus_root,
            tensors_dirname=args.tensors_dirname,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_checkpoint_hash=args.teacher_checkpoint_hash,
            second_pass_dir=args.second_pass_dir,
            run_root=args.run_root,
            registry=registry,
        )
        failure: str | None = None
        print(f"OK ({result['wall_seconds']:.1f}s); metrics: {json.dumps(result['metrics'], indent=2)}")
    except Exception as error:  # noqa: BLE001 -- record and report, matching this repo's established smoke-job pattern
        result = None
        failure = f"{type(error).__name__}: {error}"
        print(f"FAILED: {failure}")

    report = {
        "schema_version": 1,
        "stage": "core-issues4.txt Section F step 6b / core-issues3.txt Phase 8 step 6: "
        "frozen-backbone control-head training",
        "corpus": str(args.corpus_root / args.tensors_dirname),
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
