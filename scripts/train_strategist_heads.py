"""core-issues3.txt Phase 12 Stage E prep: train HydroCore's
candidate-conditioned Strategist heads (plan_validity, plan_value, and the
five consequence proxies) for the first time on real data.

Why this is needed before Stage E's policy comparison can include a
"learned candidate ordering"/"learned validity/value/proxy ranking" arm at
all: the Stage-A teacher checkpoint was trained with the default
`strategist_mode="anonymous_queries"` and the default (buggy,
core-issues3.txt Phase 3.5) `action_vocabulary_size=8` -- so
`plan_value_head`/`plan_validity_head` exist in that checkpoint but never
saw a real candidate-plan representation (only the anonymous fixed-position
plan-query tokens), and `candidate_plan_encoder`/`consequence_proxy_heads`
do not exist in it at all (candidate_conditioned-only / consequence-
prescreening-only parameters, matching train_scout_heads.py's identical
"which heads are genuinely new" analysis for Scout).

Staging (matches scripts/train_scout_heads.py's established pattern):

1. v4 model with strategist_mode="candidate_conditioned",
   action_vocabulary_size=ACTION_TEMPLATE_COUNT (the canonical 9, not the
   stale 8 default -- core-issues3.txt Phase 3.5/9.4), and
   consequence_prescreening_heads=True, initialized from the Stage-A
   teacher checkpoint with strict=False.
2. The teacher's `action_head.*` weights are DROPPED before loading
   (not reloaded, not left as an unexpected-key error): action_head's
   final projection shape depends on action_vocabulary_size (8 in the
   teacher, 9 here), and action_logits/action_pointer_logits are excluded
   from the v4 output vocabulary entirely regardless
   (checkpoint_identity.py's Section D item 3: "deterministic candidate
   plans own action-template and target identity; the learned model only
   ranks/validates/prescreens"), so reloading or training this head would
   be pure waste even if the shapes happened to match.
3. Freeze backbone: only candidate_plan_encoder/plan_value_head/
   plan_validity_head/consequence_proxy_heads parameters are trainable
   (same conservative choice as Scout's frozen-backbone head training --
   this is a narrow supervised-head pass, not a joint backbone fine-tune).
4. Train on data/learning-v2/cycle-b2-trajectories-v3/
   strategist-tensors-normalized (Phase 10.3's real candidate-conditioned
   dataset -- every real scenario's full, exactly-WNTR-verified bounded
   candidate set, Phase 3.1 compliant: never heuristically prescreened
   before labeling).
5. Evaluate on validation: plan_validity accuracy/F1 (matching WNTR's own
   exact verification), plan_value MSE, and each consequence-proxy MSE,
   all masked by their real f"{task}_mask" companion (which is already
   False for every padded/no-step position -- see
   build_strategist_candidate_dataset.py's `_masked_placeholder`).
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
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
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
    prior_mode="feature_only",
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
)
SEED = 20260807
BATCH_SIZE = 16
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
MAXIMUM_RUNTIME_SECONDS = 2400.0

TRAINABLE_HEAD_PREFIXES = (
    "candidate_plan_encoder.",
    "plan_value_head.",
    "plan_validity_head.",
    "consequence_proxy_heads.",
)
#: plan_value_head/plan_validity_head already exist (unconditional base
#: heads) in the Stage-A checkpoint -- only these two are genuinely
#: new/missing. action_head is deliberately dropped before loading (see
#: module docstring point 2), not "new" in the same sense -- it is excluded
#: from `_load_report` as its own separate, explicit category so a missing
#: action_head.* key is never confused with an unexpected real gap.
EXPECTED_NEW_HEAD_PREFIXES = ("candidate_plan_encoder.", "consequence_proxy_heads.")
DROPPED_TEACHER_PREFIXES = ("action_head.",)

#: Matches configs/training.yaml's Strategist section exactly (same source
#: of truth used by every other governed training entry point in this
#: repo) -- deliberately not re-derived here.
TASK_WEIGHTS: dict[str, float] = {
    "plan_validity": 1.0,
    "plan_value": 0.5,
    "exposure_proxy": 0.3,
    "pressure_risk_proxy": 0.3,
    "service_loss_proxy": 0.3,
    "containment_time_proxy": 0.3,
    "plan_regret_proxy": 0.3,
}


def _load_dataset(corpus_root: Path, split: str) -> GovernedScenarioDataset:
    dataset = ShardedScenarioDataset(corpus_root / split, expected_split=split)
    dataset.verify_shard_checksums()
    examples = [dataset[index] for index in range(len(dataset))]
    return GovernedScenarioDataset(examples, expected_split=split)


def build_model_from_teacher(teacher_checkpoint: Path) -> tuple[HydroCore, dict[str, Any]]:
    model = HydroCore.from_variant(VARIANT, **OVERRIDES)
    teacher_state = load_file(str(teacher_checkpoint), device="cpu")
    dropped = [key for key in teacher_state if key.startswith(DROPPED_TEACHER_PREFIXES)]
    for key in dropped:
        del teacher_state[key]
    missing, unexpected = model.load_state_dict(teacher_state, strict=False)
    unexpected_missing = [
        key
        for key in missing
        if not key.startswith(EXPECTED_NEW_HEAD_PREFIXES) and not key.startswith(DROPPED_TEACHER_PREFIXES)
    ]
    if unexpected_missing:
        raise RuntimeError(
            f"loading the Stage-A teacher checkpoint left unexpected missing keys "
            f"(not one of the known new Strategist heads or the deliberately dropped "
            f"action_head): {unexpected_missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Stage-A teacher checkpoint has keys this v4 model does not: {unexpected} "
            "(architecture mismatch -- refusing to silently partial-load)"
        )
    return model, {"missing_keys": missing, "unexpected_keys": unexpected, "dropped_teacher_keys": dropped}


def freeze_backbone(model: HydroCore) -> list[str]:
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_HEAD_PREFIXES):
            parameter.requires_grad_(True)
            trainable.append(name)
        else:
            parameter.requires_grad_(False)
    if not trainable:
        raise RuntimeError("no trainable parameters found under the 4 Strategist head prefixes -- naming mismatch?")
    return trainable


def evaluate(model: HydroCore, validation: GovernedScenarioDataset, *, batch_size: int = 32) -> dict[str, Any]:
    model.eval()
    validity_correct = 0
    validity_total = 0
    validity_true_positive = 0
    validity_predicted_positive = 0
    validity_actual_positive = 0
    squared_error: dict[str, float] = {
        key: 0.0
        for key in (
            "plan_value",
            "exposure_proxy",
            "pressure_risk_proxy",
            "service_loss_proxy",
            "containment_time_proxy",
            "plan_regret_proxy",
        )
    }
    valid_counts: dict[str, int] = {key: 0 for key in squared_error}

    with torch.no_grad():
        for start in range(0, len(validation), batch_size):
            batch_examples = [validation[index] for index in range(start, min(start + batch_size, len(validation)))]
            inputs, targets = collate_variable_topology(batch_examples)
            output = model(inputs)

            validity_mask = targets["plan_validity_mask"].bool()
            if validity_mask.any():
                predicted = torch.argmax(output["plan_validity_logits"], dim=-1).bool()
                truth = targets["plan_validity"].bool()
                predicted_v = predicted[validity_mask]
                truth_v = truth[validity_mask]
                validity_correct += int((predicted_v == truth_v).sum())
                validity_total += int(validity_mask.sum())
                validity_true_positive += int((predicted_v & truth_v).sum())
                validity_predicted_positive += int(predicted_v.sum())
                validity_actual_positive += int(truth_v.sum())

            for key in squared_error:
                mask = targets.get(f"{key}_mask")
                if mask is None or key not in output:
                    continue
                mask = mask.bool()
                if not mask.any():
                    continue
                prediction = output[key].float()
                truth = targets[key].float()
                squared_error[key] += float(((prediction[mask] - truth[mask]) ** 2).sum())
                valid_counts[key] += int(mask.sum())

    precision = (validity_true_positive / validity_predicted_positive) if validity_predicted_positive else None
    recall = (validity_true_positive / validity_actual_positive) if validity_actual_positive else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None

    metrics: dict[str, Any] = {
        "examples_evaluated": len(validation),
        "plan_validity_accuracy": (validity_correct / validity_total) if validity_total else None,
        "plan_validity_precision": precision,
        "plan_validity_recall": recall,
        "plan_validity_f1": f1,
        "plan_validity_valid_count": validity_total,
    }
    for key in squared_error:
        metrics[f"{key}_mse"] = (squared_error[key] / valid_counts[key]) if valid_counts[key] else None
        metrics[f"{key}_valid_count"] = valid_counts[key]
    return metrics


def run(
    *,
    corpus_root: Path,
    teacher_checkpoint: Path,
    run_root: Path,
    registry: ExperimentRegistry,
    max_epochs: int = MAX_EPOCHS,
    maximum_runtime_seconds: float = MAXIMUM_RUNTIME_SECONDS,
    seed: int = SEED,
) -> dict[str, Any]:
    started = time.perf_counter()
    train = _load_dataset(corpus_root, "train")
    validation = _load_dataset(corpus_root, "validation")

    model, load_report = build_model_from_teacher(teacher_checkpoint)
    trainable_parameters = freeze_backbone(model)

    config = TrainingConfig(
        seed=seed,
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
        purpose="core-issues5.txt Section 7: a second corrected-input Strategist seed, same governed "
        "dataset/evaluation protocol as the first (core-issues3.txt Phase 12 Stage E prep), to test "
        "outcome consistency before promoting learned Strategist prescreening/ordering",
        architecture="hydrocore",
        variant=VARIANT,
        seed=seed,
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
        default=Path("data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized"),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path(
            "experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/"
            "checkpoints/checkpoint-0016/model.safetensors"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/v4-strategist-heads"))
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry/v4-strategist-heads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/strategist-heads-training.json"))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="smoke-testing only")
    parser.add_argument("--maximum-runtime-seconds", type=float, default=MAXIMUM_RUNTIME_SECONDS, help="smoke-testing only")
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="core-issues5.txt Section 7: pass a second, different seed to train an independent "
        "finalist for the promotion-consistency comparison",
    )
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
            seed=args.seed,
        )
        failure: str | None = None
        print(f"OK ({result['wall_seconds']:.1f}s); metrics: {json.dumps(result['metrics'], indent=2)}")
    except Exception as error:  # noqa: BLE001 -- record and report, matching this repo's established smoke-job pattern
        result = None
        failure = f"{type(error).__name__}: {error}"
        print(f"FAILED: {failure}")

    report = {
        "schema_version": 1,
        "stage": "core-issues3.txt Phase 12 Stage E prep: Strategist head training",
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
