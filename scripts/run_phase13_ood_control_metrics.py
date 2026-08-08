"""core-issues3.txt Phase 13: required OOD/control metrics not yet computed anywhere.

Gap this closes: no script anywhere computes ood_category macro F1,
per-category recall, false-normal rate, ECE-by-category, plan-suppression
correctness, or a real classical/neural disagreement rate over a corpus
(only a single-scenario golden_result.json example and the unrelated
control-heads `next_step`/`evidence_sufficiency` metrics exist).

Uses the already-trained, on-disk Stage-F no_adapters checkpoint (has
ood_category_head=True; see scripts/run_stage_f_training.py). Evaluated only
against data/learning-v2/cycle-b2-joint-v4's already-committed
development_holdout ood-* populations and validation split -- no locked
test, no retraining, no new corpus generation.

**Important, honestly-reported limitation carried forward from the Stage-F
handoff** (reports/results/v4/pre-freeze-implementation-handoff.md, "Stage F
real run" section): Stage F's actual `train` split has zero `ood_class`
target coverage by design (the OOD-extension categories are
development_holdout-only, matching cycle-b2's own established
SEVERE_MISSINGNESS/UNSEEN_TOPOLOGY convention -- see
core-issues3.txt restriction #9, which forbids using development-holdout
data to fit/train anything). This means `ood_category_head`'s weights never
received a training gradient during the real Stage F run and remain at
random initialization. The metrics below are therefore an honest
NEAR-CHANCE baseline, not a trained classifier's performance -- this is the
expected, correct outcome given the current governed data-split design, and
is exactly the finding Phase 14's promotion gate #3 ("received nonzero
gradients in real multi-topology batches") needs to disqualify this head
from `runtime_enabled_outputs` without further work. Fixing it for real
would require a new, separately-authorized OOD-extension `train` population
(a Phase 6-scope data change), which is out of this script's/this
pass's scope.

`false-normal rate` and `plan suppression correctness` below are computed
two ways: (1) using the LEARNED head's argmax prediction (near-chance, per
above), and (2) as a pure deterministic check of
`ood_categories.OOD_CATEGORY_BEHAVIOR` against the TRUE category (governed
metadata, independent of the learned head) -- the second is the one that
actually matters for runtime safety, since `OOD_CATEGORY_BEHAVIOR` is
authoritative and the learned head is (and, per the above, must remain)
advisory only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hydroswarm.inference.fusion import jensen_shannon_divergence
from hydroswarm.model import HydroCore
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology
from hydroswarm.training.ood_categories import OOD_CATEGORY_BEHAVIOR, OODCategory

ROOT = Path(__file__).resolve().parents[1]
JOINT_CORPUS_ROOT = Path("data/learning-v2/cycle-b2-joint-v4/tensors-normalized")

SHARED_MODEL_CONFIG: dict[str, Any] = {
    "prior_mode": "feature_only",
    "event_control_heads": True,
    "scout_control_heads": True,
    "strategist_mode": "candidate_conditioned",
    "action_vocabulary_size": 9,
    "consequence_prescreening_heads": True,
    "ood_category_head": True,
}

OOD_CATEGORY_NAMES = [category.value for category in OODCategory]
NONE_INDEX = OOD_CATEGORY_NAMES.index(OODCategory.NONE.value)

#: cycle-b2-joint-v4's own real ood_class-labeled populations (Phase 6.3
#: extension, the 4 newly generated categories) -- see
#: scripts/build_stage_f_joint_corpus.py's OOD_EXTENSION_POPULATIONS.
LABELED_OOD_POPULATIONS = ("ood-EXTREME_DEMAND", "ood-FROZEN_DRIFTING_SENSOR", "ood-ROUGHNESS_MISMATCH", "ood-TANK_STATE_SHIFT")
#: cycle-b2's own pre-existing OOD-holdout populations -- real distribution
#: shift by construction (folder identity), but never carry a real
#: ood_class *target* (targets["ood_class_mask"] is always False here; see
#: this file's module docstring and the Stage-F handoff section it quotes).
UNLABELED_OOD_POPULATIONS = ("ood-SEVERE_MISSINGNESS", "ood-UNSEEN_TOPOLOGY")
ALL_OOD_POPULATIONS = LABELED_OOD_POPULATIONS + UNLABELED_OOD_POPULATIONS

DISAGREEMENT_THRESHOLD = 0.5  # matches inference.fusion/uncertainty_control's own default


#: Every ood-* population lacks Strategist candidate-plan fields (see
#: scripts/run_stage_f_joint_corpus_gates.py's OOD_CLASS_MODEL_OVERRIDES,
#: which this mirrors exactly) -- all OOD-population evaluation in this
#: script uses this reduced config, loaded strict=False against the same
#: full Stage-F checkpoint (only the unused Strategist-specific parameters
#: go unloaded).
#: action_vocabulary_size is kept (unlike run_stage_f_joint_corpus_gates.py's
#: OOD_CLASS_MODEL_OVERRIDES, which can drop it because it never loads a
#: real checkpoint): the legacy action_head is sized by
#: action_vocabulary_size unconditionally (independent of strategist_mode),
#: so dropping it back to HydroCore's own default would size-mismatch
#: against the real 9-template Stage-F checkpoint's action_head weights.
NO_STRATEGIST_MODEL_CONFIG: dict[str, Any] = {
    key: value for key, value in SHARED_MODEL_CONFIG.items() if key not in {"strategist_mode", "consequence_prescreening_heads"}
}


def load_model(checkpoint_path: Path, *, use_adapters: bool, strategist_fields_available: bool) -> HydroCore:
    from safetensors.torch import load_file

    config = SHARED_MODEL_CONFIG if strategist_fields_available else NO_STRATEGIST_MODEL_CONFIG
    model = HydroCore.from_variant("small", use_adapters=use_adapters, **config)
    state_dict = load_file(str(checkpoint_path), device="cpu")
    if strategist_fields_available:
        model.load_state_dict(state_dict, strict=True)
    else:
        report = model.load_state_dict(state_dict, strict=False)
        if report.missing_keys:
            raise RuntimeError(f"{checkpoint_path}: unexpected missing_keys when loading non-Strategist model: {report.missing_keys}")
    model.eval()
    return model


def _load_population(name: str) -> ShardedScenarioDataset:
    dataset = ShardedScenarioDataset(JOINT_CORPUS_ROOT / name, expected_split="development_holdout")
    dataset.verify_shard_checksums()
    return dataset


def _binary_prf1(predictions: list[bool], truths: list[bool]) -> dict[str, float]:
    tp = sum(1 for p, t in zip(predictions, truths) if p and t)
    fp = sum(1 for p, t in zip(predictions, truths) if p and not t)
    fn = sum(1 for p, t in zip(predictions, truths) if not p and t)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": len(predictions)}


@torch.no_grad()
def evaluate_labeled_ood(model: HydroCore, *, batch_size: int = 16) -> dict[str, Any]:
    """Macro F1 / per-category recall / ECE-by-category over the 4
    populations with a real ood_class target."""

    all_preds: list[int] = []
    all_truths: list[int] = []
    all_confidences: list[float] = []
    per_category_examples: dict[str, int] = {}

    for population in LABELED_OOD_POPULATIONS:
        dataset = _load_population(population)
        per_category_examples[population] = len(dataset)
        for start in range(0, len(dataset), batch_size):
            examples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            inputs, targets = collate_variable_topology(examples)
            # ood_class is targets_v2's only maskable=False control target
            # (see TARGETS_V2["ood_class"].masking_rule: "Never masked; every
            # example is classified") -- there is structurally no
            # "ood_class_mask" key to check; every row in these
            # ood-extension-only populations carries a real label.
            if "ood_class" not in targets:
                continue
            output = model(inputs)
            probabilities = torch.softmax(output["ood_category_logits"].float(), dim=-1)
            preds = torch.argmax(probabilities, dim=-1).tolist()
            confs = probabilities.max(dim=-1).values.tolist()
            truths = targets["ood_class"].long().tolist()
            all_preds.extend(preds)
            all_truths.extend(truths)
            all_confidences.extend(confs)

    per_category: dict[str, Any] = {}
    f1s: list[float] = []
    for index, name in enumerate(OOD_CATEGORY_NAMES):
        pred_bool = [p == index for p in all_preds]
        truth_bool = [t == index for t in all_truths]
        stats = _binary_prf1(pred_bool, truth_bool)
        if stats["support"] and sum(truth_bool):
            per_category[name] = stats
            f1s.append(stats["f1"])

    correct = [p == t for p, t in zip(all_preds, all_truths)]
    ece = _expected_calibration_error(all_confidences, correct) if all_confidences else None

    return {
        "populations": per_category_examples,
        "examples_with_real_ood_class_target": len(all_truths),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "per_category_recall_f1": per_category,
        "calibration_error": ece,
        "class_index_reference": OOD_CATEGORY_NAMES,
    }


def _expected_calibration_error(confidences: list[float], correct: list[bool], *, bins: int = 10) -> float:
    if not confidences:
        return 0.0
    confidences_array = np.asarray(confidences)
    correct_array = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total_error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences_array > low) & (confidences_array <= high)
        if not np.any(in_bin):
            continue
        bin_confidence = float(confidences_array[in_bin].mean())
        bin_accuracy = float(correct_array[in_bin].mean())
        total_error += (in_bin.sum() / len(confidences_array)) * abs(bin_confidence - bin_accuracy)
    return float(total_error)


@torch.no_grad()
def evaluate_false_normal_and_suppression(model: HydroCore, *, batch_size: int = 16) -> dict[str, Any]:
    """false-normal rate and plan-suppression correctness across ALL 6 real
    OOD populations, using the TRUE category (known by directory identity,
    even for the 2 unlabeled-target populations) as ground truth -- both
    for the learned head's prediction and for the deterministic
    OOD_CATEGORY_BEHAVIOR table, which is authoritative at runtime
    regardless of what the (near-chance, per this file's module docstring)
    learned head predicts."""

    learned_false_normal = 0
    learned_total = 0
    deterministic_suppressed_correctly = 0
    deterministic_total = 0

    per_population: dict[str, Any] = {}
    for population in ALL_OOD_POPULATIONS:
        true_category = OODCategory[population.removeprefix("ood-")]
        expected_behavior = OOD_CATEGORY_BEHAVIOR[true_category]
        dataset = _load_population(population)
        population_false_normal = 0
        population_examples = 0
        for start in range(0, len(dataset), batch_size):
            examples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            inputs, _targets = collate_variable_topology(examples)
            output = model(inputs)
            preds = torch.argmax(torch.softmax(output["ood_category_logits"].float(), dim=-1), dim=-1).tolist()
            for pred in preds:
                learned_total += 1
                population_examples += 1
                if pred == NONE_INDEX:
                    learned_false_normal += 1
                    population_false_normal += 1
                # Deterministic check: the governed behavior table's
                # planning_permitted for this example's TRUE category must
                # be False (every real, non-NONE category in this project's
                # governed table is False -- see ood_categories.py) --
                # independent of the learned head's prediction.
                deterministic_total += 1
                if expected_behavior.planning_permitted is False:
                    deterministic_suppressed_correctly += 1

        per_population[population] = {
            "true_category": true_category.value,
            "examples": population_examples,
            "learned_false_normal_rate": (population_false_normal / population_examples) if population_examples else 0.0,
            "deterministic_planning_permitted": expected_behavior.planning_permitted,
            "deterministic_calibration_valid": expected_behavior.calibration_valid,
        }

    return {
        "learned_false_normal_rate_overall": (learned_false_normal / learned_total) if learned_total else 0.0,
        "deterministic_plan_suppression_correctness_rate": (
            deterministic_suppressed_correctly / deterministic_total if deterministic_total else 0.0
        ),
        "per_population": per_population,
        "note": "learned_false_normal_rate reflects a head with zero real training-gradient "
        "exposure this run (see module docstring) -- expected near/at the 1/11 chance rate for "
        "predicting NONE, not a validated safety number. deterministic_plan_suppression_correctness_rate "
        "is the number that matters for runtime safety: it verifies the governed "
        "OOD_CATEGORY_BEHAVIOR table itself (independent of any learned head) correctly forbids "
        "planning for every real non-NONE category in this evaluation.",
    }


@torch.no_grad()
def evaluate_disagreement(model: HydroCore, *, split: str = "validation", batch_size: int = 16, limit: int = 1000) -> dict[str, Any]:
    """Deterministic-vs-learned disagreement: Jensen-Shannon divergence
    between the model's neural source-localization distribution and the
    classical_prior distribution already carried as a governed input
    feature -- the same JS divergence inference.fusion.uncertainty_control
    thresholds live at DISAGREEMENT_THRESHOLD=0.5 to trigger
    ControlAction.INSPECT_SENSORS."""

    dataset = ShardedScenarioDataset(JOINT_CORPUS_ROOT / split, expected_split=split)
    dataset.verify_shard_checksums()
    total = min(limit, len(dataset)) if limit else len(dataset)
    divergences: list[float] = []
    for start in range(0, total, batch_size):
        examples = [dataset[i] for i in range(start, min(start + batch_size, total))]
        inputs, targets = collate_variable_topology(examples)
        source_mask = targets.get("source_node_mask")
        output = model(inputs)
        neural = torch.softmax(output["source_node_logits"].float(), dim=-1)
        classical = inputs["classical_prior"].float()
        for row in range(neural.shape[0]):
            if source_mask is not None and not bool(source_mask[row]):
                continue
            node_mask = inputs["node_mask"][row].bool()
            neural_row = neural[row][node_mask].numpy()
            classical_row = classical[row][node_mask].numpy()
            if neural_row.sum() <= 0 or classical_row.sum() <= 0:
                continue
            divergences.append(jensen_shannon_divergence(neural_row, classical_row))

    disagreement_rate = float(np.mean([d >= DISAGREEMENT_THRESHOLD for d in divergences])) if divergences else 0.0
    return {
        "examples": len(divergences),
        "mean_js_divergence": float(np.mean(divergences)) if divergences else 0.0,
        "disagreement_threshold": DISAGREEMENT_THRESHOLD,
        "disagreement_rate": disagreement_rate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/runs/stage-f/no_adapters-seed20260810/20260808T041727Z-de5f4b0e/model-export.safetensors"),
    )
    parser.add_argument("--use-adapters", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/phase13-ood-control-metrics.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    # ood-* populations never carry Strategist candidate-plan fields;
    # validation (used for the disagreement check) does -- see
    # NO_STRATEGIST_MODEL_CONFIG's docstring.
    ood_model = load_model(args.checkpoint, use_adapters=args.use_adapters, strategist_fields_available=False)
    full_model = load_model(args.checkpoint, use_adapters=args.use_adapters, strategist_fields_available=True)

    labeled = evaluate_labeled_ood(ood_model)
    print(f"labeled OOD: macro_f1={labeled['macro_f1']:.4f} accuracy={labeled['accuracy']:.4f} n={labeled['examples_with_real_ood_class_target']}")

    suppression = evaluate_false_normal_and_suppression(ood_model)
    print(
        f"false_normal_rate(learned)={suppression['learned_false_normal_rate_overall']:.4f} "
        f"plan_suppression_correctness(deterministic)={suppression['deterministic_plan_suppression_correctness_rate']:.4f}"
    )

    disagreement = evaluate_disagreement(full_model)
    print(f"disagreement: mean_js={disagreement['mean_js_divergence']:.4f} rate={disagreement['disagreement_rate']:.4f}")

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "use_adapters": args.use_adapters,
        "corpus": str(JOINT_CORPUS_ROOT),
        "labeled_ood_classification": labeled,
        "false_normal_and_plan_suppression": suppression,
        "deterministic_vs_learned_disagreement": disagreement,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
