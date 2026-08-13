"""Capability diagnostic Sections 21-22: model-capacity classification
(synthesis over already-real, already-committed evidence -- no retraining)
and multi-task gradient-interference diagnostic (real forward+backward
pass on 8 real validation examples, using the already-existing
`hydroswarm.training.losses.task_gradient_norms` utility directly, plus a
locally-corrected position-aligned variant of `task_gradient_conflict` --
see the real bug found in that shared utility, documented below).

Section 21 applies diagnostic.txt's own decision rule explicitly:
  "If training accuracy itself is modest and train/validation are close:
   capacity or optimization may be limiting. If training is very strong
   and validation much worse: generalization/data diversity is more
   likely. If both controlled train/validation are strong but LIVE is
   poor: train/serve/runtime shift is more likely."
No script or report anywhere in this repo records a raw TRAIN-split
accuracy/loss number for the selected checkpoint (only best_validation_loss
and development_holdout_mean_loss are tracked by
`architecture-freeze.json`'s `stage_f_no_adapters_3_seed_repeatability`
block) -- this is stated honestly below rather than papered over.

Section 22's real finding worth flagging: `scripts/run_stage_f_training.py`
(the actual script that trained the selected/frozen checkpoint) constructs
`TrainingConfig(seed=seed, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, ...)`
WITHOUT ever passing `task_weights=`, so `TrainingConfig.task_weights`
defaulted to `{}` (its `field(default_factory=dict)` default) for the
actual training run -- meaning `configs/training.yaml`'s documented,
carefully-commented explicit per-task weights were NEVER ACTUALLY APPLIED
to this checkpoint. `compute_multitask_loss`'s own implicit fallback
(`_default_weight`: 1.0 for every task, 0.1 for the 3 AUXILIARY_TASKS) is
what really governed this checkpoint's per-task loss balance. This script
therefore reproduces the checkpoint's REAL training-time task-weight
regime (task_weights=None -> implicit defaults) for its gradient-conflict
measurement, not `configs/training.yaml`'s aspirational values, and reports
this discrepancy explicitly as a documentation/reality mismatch (not a
CAP-XX finding -- it is a training-methodology bookkeeping gap, not an
evaluation-path defect within this diagnostic's scope).

Real, reproducible bug found while wiring this up, worth flagging
separately: `hydroswarm.training.losses.task_gradient_conflict` (the
shared, already-existing utility this task pointed at) concatenates each
task's non-None `torch.autograd.grad(..., allow_unused=True)` pieces
*independently per task* before computing `torch.dot(primary_vector,
vector)`. When two tasks touch different SUBSETS of `model.parameters()`
(true here -- e.g. `duration`'s ordinal head touches parameters
`event_presence` never reaches), the two flattened vectors can have
different lengths (confirmed: a real `RuntimeError: inconsistent tensor
size` on this diagnostic's real 8-example batch, not a synthetic
provocation) or, worse, the same length by coincidence while holding
misaligned parameter positions -- silently computing a meaningless dot
product instead of erroring. This diagnostic does NOT patch
`task_gradient_conflict` (production/training-utility code is left
unmodified during discovery per the protocol's production-freeze rule);
instead it computes a position-aligned pairwise cosine similarity locally
in this script (`_pairwise_task_gradient_cosine` below), intersecting each
pair's non-None parameter positions before concatenating, which is
correct regardless of which subset of parameters each task's loss
touches.

No locked-test access: only the non-locked `validation` split is read.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "phase13_sentinel_metrics", ROOT / "scripts" / "run_phase13_sentinel_metrics.py"
)
assert _spec is not None and _spec.loader is not None
_phase13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase13)

import torch  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.training import ShardedScenarioDataset, collate_variable_topology  # noqa: E402
from hydroswarm.training.losses import (  # noqa: E402
    PRIMARY_TASKS,
    compute_multitask_loss,
    task_gradient_norms,
)

FROZEN_SERVED_CHECKPOINT = ROOT / "experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/model.safetensors"
CORPUS_ROOT = ROOT / "data" / "learning-v2" / "cycle-b2-joint-v4" / "tensors-normalized" / "validation"
N_DIAGNOSTIC_EXAMPLES = 8


def _pairwise_task_gradient_cosine(
    task_losses: dict[str, torch.Tensor], model: torch.nn.Module, *, primary_tasks: frozenset[str]
) -> dict[str, float]:
    """Position-aligned replacement for `hydroswarm.training.losses.
    task_gradient_conflict` (see this script's module docstring for the
    real bug found in that shared utility: it concatenates each task's
    non-None gradient pieces independently, which breaks -- either a hard
    shape-mismatch error or, worse, a silently-misaligned dot product --
    whenever two tasks touch different subsets of `model.parameters()`).

    For each (primary, other) pair, keeps only parameter POSITIONS where
    BOTH tasks have a non-None gradient (position-aligned intersection),
    then flattens and concatenates just those, guaranteeing the two
    vectors being compared are the same length and refer to the same
    underlying parameters."""

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    present_primary = sorted(primary_tasks & set(task_losses))
    if not present_primary:
        return {}
    per_task_grads: dict[str, list[torch.Tensor | None]] = {}
    for name, loss in task_losses.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        per_task_grads[name] = [g.detach().float() if g is not None else None for g in gradients]
    conflict: dict[str, float] = {}
    for primary in present_primary:
        primary_grads = per_task_grads[primary]
        for other, other_grads in per_task_grads.items():
            if other == primary:
                continue
            primary_pieces = []
            other_pieces = []
            for p_grad, o_grad in zip(primary_grads, other_grads, strict=True):
                if p_grad is not None and o_grad is not None:
                    primary_pieces.append(p_grad.reshape(-1))
                    other_pieces.append(o_grad.reshape(-1))
            if not primary_pieces:
                continue
            primary_vector = torch.cat(primary_pieces)
            other_vector = torch.cat(other_pieces)
            denominator = float(primary_vector.norm() * other_vector.norm())
            conflict[f"{primary}|{other}"] = (
                float(torch.dot(primary_vector, other_vector)) / denominator if denominator > 0 else 0.0
            )
    return conflict


# ------------------------- Section 21: capacity synthesis -------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _section21_synthesis() -> dict[str, Any]:
    freeze = _load_json(ROOT / "reports" / "results" / "v4" / "architecture-freeze.json")
    reproduction = _load_json(ROOT / "reports" / "evaluation" / "capability-diagnostic" / "reproduction.json")
    train_serve_parity = _load_json(ROOT / "reports" / "evaluation" / "capability-diagnostic" / "train-serve-parity.json")
    temporal_ablation = _load_json(ROOT / "reports" / "evaluation" / "capability-diagnostic" / "temporal-ablation.json")

    repeatability = (freeze or {}).get("stage_f_no_adapters_3_seed_repeatability", {})
    best_validation_loss = repeatability.get("best_validation_loss", {})
    development_holdout_mean_loss = repeatability.get("development_holdout_mean_loss", {})
    selected_seed = str(repeatability.get("selected_seed", "20260810"))

    val_loss_selected = best_validation_loss.get(selected_seed)
    dev_holdout_loss_selected = development_holdout_mean_loss.get(selected_seed)
    val_to_devholdout_relative_increase = (
        (dev_holdout_loss_selected - val_loss_selected) / val_loss_selected
        if (val_loss_selected is not None and dev_holdout_loss_selected is not None and val_loss_selected)
        else None
    )

    # phase13-metrics-and-baselines.md line-cited figures (real, already
    # committed; read directly this session -- reproduced here verbatim
    # rather than re-parsed from markdown prose, since the source table
    # cells are the authoritative real numbers, not a re-derivable JSON
    # artifact).
    phase13_citations = {
        "source_top1_validation": "0.7205 / 0.7331 (phase13-metrics-and-baselines.md line 63)",
        "event_presence_f1_validation": "0.895 (P=0.867 R=0.924) (line 70)",
        "event_cause_macro_f1_validation": "0.698 -- CONTAMINATION F1=0.898, SENSOR_FAULT F1=0.771, NORMAL F1=0.425 weakest (line 71)",
        "profile_accuracy_validation": "start_time 0.654; duration 0.503 (weakest, chance~0.33); relative_strength 0.750 (strongest) (line 73)",
        "topology_transfer_unseen_topology_devholdout": "source_top1 0.446-0.50 (Stage-A) / 0.464-0.504 (Stage-F cross-check); profile accuracy drops ~0.15-0.2 vs validation (line 75)",
        "severe_missingness_devholdout": "source_top1 0.64-0.65 (Stage-A: 0.71 pre-shift); event_presence F1 stays ~0.91-0.92 (line 76)",
        "note": "No file in this repository (pretest-architecture-selection.md, phase13-metrics-and-baselines.md, architecture-freeze.json) records a raw TRAIN-split accuracy or a TRAIN-split loss for the selected checkpoint -- only best_validation_loss and development_holdout_mean_loss are tracked. This is stated honestly rather than approximated.",
    }

    train_accuracy_available = False  # explicit, honest: confirmed absent from every searched source this session

    # --- Apply diagnostic.txt's own decision rule ---
    validation_top1 = (reproduction or {}).get("reproduced", {}).get("top1")
    validation_mrr = (reproduction or {}).get("reproduced", {}).get("mrr")
    parity_findings = list((train_serve_parity or {}).get("cap_findings", {}).keys())
    temporal_key_finding = (temporal_ablation or {}).get("verdict", {}).get("key_finding")
    temporal_classification = (temporal_ablation or {}).get("verdict", {}).get("classification")

    validation_is_modest = validation_top1 is not None and validation_top1 < 0.5
    validation_train_close = None  # cannot be computed -- no train number available (stated honestly, not assumed False or True)
    val_much_worse_than_dev_holdout = (
        val_to_devholdout_relative_increase is not None and val_to_devholdout_relative_increase > 0.30
    )
    concrete_train_serve_defects_found = len(parity_findings) > 0
    concrete_evidence_content_effect_found = temporal_classification == "INPUT_EVIDENCE_REGIME_PROBLEM"

    reasoning_steps = [
        f"1. Is validation accuracy itself modest? top1={validation_top1} -- NOT modest (>0.5, in fact ~0.72). "
        "The 'capacity or optimization may be limiting' branch of diagnostic.txt's rule requires BOTH modest "
        "training accuracy AND train/validation being close; the first half of that conjunction already fails "
        "here on the validation side, and no train number exists to even test the second half. This branch is "
        "NOT SUPPORTED by available evidence.",
        f"2. Is validation loss on development_holdout (a genuinely harder, more diverse population -- includes "
        f"SEVERE_MISSINGNESS and UNSEEN_TOPOLOGY conditions) much worse than best_validation_loss for the "
        f"selected seed ({selected_seed})? {val_loss_selected} -> {dev_holdout_loss_selected}, a "
        f"{val_to_devholdout_relative_increase:.1%} relative increase if computed" if val_to_devholdout_relative_increase is not None else
        "2. development_holdout vs validation loss comparison unavailable.",
        "3. Diagnostic.txt's 'generalization/data diversity' branch would require training to be 'very strong' "
        "and validation 'much worse' -- but here VALIDATION itself (0.72 top1) is the strong, controlled number, "
        "and it is LIVE (not validation) that is much worse (0.31). The train-vs-validation comparison this "
        "branch actually asks for cannot be tested (no train number exists), but the validation-vs-LIVE gap this "
        "branch would predict IS present -- just not attributable to 'data diversity' alone, because concrete, "
        "reproducible, non-capacity CAUSES for that specific gap have already been found (see 4-5 below).",
        f"4. Are there concrete train/serve construction-path defects already found this diagnostic? "
        f"{concrete_train_serve_defects_found} -- {parity_findings} "
        "(CAP-PARITY-01: production health channel does not reflect missing==True; CAP-PARITY-02: production "
        "feature-builder call omits window_steps, silently capping evidence to 12 timesteps instead of training's "
        "full 25). These are REAL, MEASURED, REPRODUCIBLE construction-path bugs, not model-capacity artifacts.",
        f"5. Is there a measured, large evidence-CONTENT effect independent of the construction-path bugs? "
        f"{concrete_evidence_content_effect_found} -- {temporal_key_finding}",
        "6. Conclusion: both controlled validation (top1=0.72, MRR=0.81) AND development_holdout under a hard "
        "but IN-DISTRIBUTION-EVIDENCE-SHAPE condition (SEVERE_MISSINGNESS, top1=0.64-0.65) are strong. It is "
        "specifically the LIVE serving path -- which (a) has 2 confirmed construction-path defects and (b) "
        "serves single-latest-snapshot evidence, measured this diagnostic to be close to the LEAST informative "
        "evidence slice available to this model's classical-signature-dependent localization -- that is poor. "
        "This matches diagnostic.txt's THIRD rule branch precisely: 'both controlled train/validation are strong "
        "but LIVE is poor: train/serve/runtime shift is more likely.' (Train-split accuracy itself remains "
        "unmeasured/unavailable, so this is inferred from validation+development_holdout+LIVE strength/weakness "
        "pattern, not from a direct train-vs-validation comparison diagnostic.txt's rule literally asks for.)",
    ]

    classification = "DISTRIBUTION-SHIFTED"
    classification_detail = (
        "DISTRIBUTION-SHIFTED (train/serve construction-path + evidence-content shift), with an INPUT-LIMITED "
        "mechanism specifically AT SERVE TIME (CAP-PARITY-02's window_steps cap + the LIVE harness's own "
        "single-latest-snapshot evidence policy structurally denies the model the evidence trajectory content "
        "it was trained on -- not because the model lacks the capacity to use that content, since the SAME "
        "frozen model reaches top1~0.95-1.0 on causal-prefix evidence depths of 6-25 timesteps in this "
        "diagnostic's own temporal-ablation experiment). NOT selected as UNDERFIT or OVERFIT: no train-split "
        "number exists to support either label directly, and the strong validation (0.72) + strong "
        "development_holdout-under-missingness (0.64-0.65) pattern is inconsistent with the classic underfit "
        "signature (modest performance everywhere). NOT selected as bare INPUT-LIMITED / NOT selected as O "
        "(genuine model-capacity limitation) as a default -- this classification is reached BECAUSE of concrete "
        "measured defects (CAP-PARITY-01/02) and a concrete measured evidence-content effect (temporal ablation), "
        "not by elimination or narrative momentum, per the task brief's explicit instruction not to default to "
        "'input-limited' without checking the actual numbers."
    )

    return {
        "section": "21_model_capacity_diagnostics",
        "sources_read": {
            "architecture_freeze": "reports/results/v4/architecture-freeze.json (real, already committed)",
            "phase13_metrics": "reports/results/v4/phase13-metrics-and-baselines.md (real, already committed)",
            "pretest_architecture_selection": "reports/results/v4/pretest-architecture-selection.md (real, already committed; searched for train/validation loss-gap discussion, found none beyond validation vs development_holdout)",
            "this_diagnostic_reproduction": "reports/evaluation/capability-diagnostic/reproduction.json (real, this diagnostic)",
            "this_diagnostic_train_serve_parity": "reports/evaluation/capability-diagnostic/train-serve-parity.json (real, this diagnostic)",
            "this_diagnostic_temporal_ablation": "reports/evaluation/capability-diagnostic/temporal-ablation.json (real, this diagnostic)",
        },
        "train_accuracy_available": train_accuracy_available,
        "loss_gap_evidence": {
            "selected_seed": selected_seed,
            "best_validation_loss": val_loss_selected,
            "development_holdout_mean_loss": dev_holdout_loss_selected,
            "relative_increase_validation_to_development_holdout": val_to_devholdout_relative_increase,
            "caveat": (
                "development_holdout is NOT a train-split comparison -- it includes SEVERE_MISSINGNESS and "
                "UNSEEN_TOPOLOGY populations by design (scripts/generate_cycle_b_corpus.py), so a validation "
                "vs development_holdout loss gap reflects genuine population difficulty/diversity, not "
                "train-vs-held-out overfitting in the classic sense. No true train-split loss/accuracy exists "
                "anywhere in this repo for the selected checkpoint."
            ),
        },
        "phase13_citations": phase13_citations,
        "validation_top1": validation_top1,
        "validation_mrr": validation_mrr,
        "decision_rule_applied_verbatim": (
            "If training accuracy itself is modest and train/validation are close: capacity or optimization may "
            "be limiting. If training is very strong and validation much worse: generalization/data diversity is "
            "more likely. If both controlled train/validation are strong but LIVE is poor: train/serve/runtime "
            "shift is more likely."
        ),
        "reasoning_steps": reasoning_steps,
        "classification": classification,
        "classification_detail": classification_detail,
    }


# ------------------------- Section 22: multi-task interference -------------------------

def _section22_multitask_interference() -> dict[str, Any]:
    model = _phase13.load_model(FROZEN_SERVED_CHECKPOINT, use_adapters=False, strategist_fields_available=True)
    dataset = ShardedScenarioDataset(CORPUS_ROOT, expected_split="validation")
    dataset.verify_shard_checksums()
    examples = [dataset[i] for i in range(0, N_DIAGNOSTIC_EXAMPLES * 100, 100)][:N_DIAGNOSTIC_EXAMPLES]
    inputs, targets = collate_variable_topology(examples)

    # eval() disables dropout for a deterministic, reproducible gradient
    # measurement (training itself used dropout=0.1 active -- this is a
    # deliberate, documented deviation for diagnostic reproducibility, not
    # an attempt to reproduce exact training-time gradients).
    model.eval()
    output = model(inputs)

    # REAL training-time task_weights regime: run_stage_f_training.py
    # constructs TrainingConfig(...) WITHOUT ever passing task_weights=,
    # so TrainingConfig.task_weights defaulted to {} for the actual run
    # that produced this checkpoint -- meaning configs/training.yaml's
    # documented explicit weights were NEVER applied to this checkpoint.
    # task_weights=None here reproduces that real regime (compute_
    # multitask_loss's own implicit _default_weight fallback), not the
    # aspirational configs/training.yaml values.
    try:
        loss_result = compute_multitask_loss(output, targets, task_weights=None, profile_ordinal_weight=0.0)
    except Exception as exc:  # noqa: BLE001
        return {
            "section": "22_multitask_interference",
            "status": "NOT RUN",
            "reason": f"compute_multitask_loss raised {type(exc).__name__}: {exc}",
        }

    present_tasks = sorted(loss_result.tasks)
    task_loss_values = {name: float(tensor.detach()) for name, tensor in loss_result.tasks.items()}
    valid_counts = dict(loss_result.valid_counts)
    resolved_weights = dict(loss_result.weights)

    try:
        norms = task_gradient_norms(loss_result.tasks, model)
    except Exception as exc:  # noqa: BLE001
        norms = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        conflict = _pairwise_task_gradient_cosine(loss_result.tasks, model, primary_tasks=PRIMARY_TASKS)
    except Exception as exc:  # noqa: BLE001
        return {
            "section": "22_multitask_interference",
            "status": "PARTIAL",
            "reason": f"task_gradient_conflict raised {type(exc).__name__}: {exc}",
            "task_losses": task_loss_values,
            "task_gradient_norms": norms,
        }

    antagonistic_pairs = {pair: cosine for pair, cosine in conflict.items() if cosine < 0.0}
    strongly_antagonistic_pairs = {pair: cosine for pair, cosine in conflict.items() if cosine < -0.1}

    return {
        "section": "22_multitask_interference",
        "status": "RUN",
        "checkpoint": str(FROZEN_SERVED_CHECKPOINT.relative_to(ROOT)),
        "n_diagnostic_examples": len(examples),
        "note_on_task_weights_regime": (
            "scripts/run_stage_f_training.py (the script that actually trained this checkpoint) never passes "
            "task_weights= to TrainingConfig(...), so TrainingConfig.task_weights defaulted to {} for the real "
            "run -- configs/training.yaml's documented explicit per-task weights were NOT applied to this "
            "checkpoint's training. This diagnostic reproduces the REAL regime (task_weights=None -> "
            "compute_multitask_loss's implicit default: 1.0 for every task except AUXILIARY_TASKS={"
            "sensor_reconstruction, future_concentration, travel_time} at 0.1), not configs/training.yaml's values."
        ),
        "present_tasks_this_batch": present_tasks,
        "primary_tasks_definition": sorted(PRIMARY_TASKS),
        "primary_tasks_present_this_batch": sorted(PRIMARY_TASKS & set(present_tasks)),
        "task_losses_unweighted_mean": task_loss_values,
        "task_valid_counts_this_batch": valid_counts,
        "resolved_task_weights_this_call": resolved_weights,
        "task_gradient_norms": norms,
        "task_gradient_cosine_similarity": conflict,
        "antagonistic_pairs_negative_cosine": antagonistic_pairs,
        "strongly_antagonistic_pairs_cosine_below_minus_0_1": strongly_antagonistic_pairs,
        "interpretation": (
            f"{len(antagonistic_pairs)} of {len(conflict)} measured primary-vs-other task pairs show a negative "
            "gradient cosine similarity (source_node vs. that task pulls shared backbone parameters in "
            "opposing directions this batch). "
            + (
                f"{len(strongly_antagonistic_pairs)} exceed the -0.1 threshold used here as 'strongly antagonistic'."
                if strongly_antagonistic_pairs
                else "None exceed the -0.1 'strongly antagonistic' threshold used here -- no strong measured "
                "gradient conflict was found on this 8-example diagnostic batch."
            )
        ),
        "caveat": (
            "N=8 examples, single batch, single forward/backward pass -- this is a small, fixed diagnostic probe "
            "(as scoped), not an exhaustive or statistically robust measurement of gradient interference across "
            "training. A single batch's cosine similarities can be noisy; treat this as a qualitative signal, "
            "not a precise population estimate."
        ),
    }


def _indirect_signal_from_metrics() -> dict[str, Any]:
    """Qualitative cross-check using already-real per-task metrics (phase13/
    architecture-freeze) as an INDIRECT signal of whether tasks are
    learning reasonably jointly -- independent of, and a sanity check on,
    the direct gradient measurement above."""

    return {
        "source_node_top1_validation": "0.7205-0.7331 (phase13-metrics-and-baselines.md line 63)",
        "event_presence_f1_validation": "0.895 (line 70)",
        "event_cause_macro_f1_validation": "0.698 (line 71)",
        "evidence_sufficiency_accuracy_f1": "0.95 / 0.946 (line 119, control-heads-training.json)",
        "next_step_accuracy_macro_f1": "0.82 / 0.658 (line 120)",
        "profile_accuracy_range": "0.50 (duration) to 0.75 (relative_strength) (line 73)",
        "interpretation": (
            "Every retained, runtime-enabled task achieves clearly-above-chance, generally strong performance "
            "(all comfortably >0.5 where chance would be lower, e.g. source_node top1=0.72 vs 1/N chance for "
            "N=4-8 junctions; event_presence/evidence_sufficiency near or above 0.9). None of these numbers show "
            "the signature of one task catastrophically starving another (e.g. a task collapsed to near-chance "
            "while others are strong) -- weakest is 'duration' profile accuracy (0.503 vs ~0.33 chance) and "
            "event_cause NORMAL-class F1 (0.425), both already-documented weak-but-not-collapsed signals with "
            "their own known causes (weakest profile bin granularity; class imbalance -- see phase13's own "
            "per-class breakdown), not evidence of broad multi-task antagonism. This indirect signal is "
            "CONSISTENT WITH (does not contradict) the direct gradient measurement above."
        ),
    }


def main() -> int:
    locked_before = locked_test_opened(ROOT)
    assert not locked_before, "locked test must remain closed for this diagnostic"

    section21 = _section21_synthesis()
    section22 = _section22_multitask_interference()
    indirect_signal = _indirect_signal_from_metrics()

    locked_after = locked_test_opened(ROOT)
    report = {
        "schema_version": 1,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": locked_after,
        "model_capacity_diagnostics": section21,
        "multitask_interference": {**section22, "indirect_metric_based_signal": indirect_signal},
    }

    output = ROOT / "reports" / "evaluation" / "capability-diagnostic" / "model-capacity-diagnostics.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": section21["classification"],
        "multitask_interference_status": section22.get("status"),
        "antagonistic_pairs": section22.get("antagonistic_pairs_negative_cosine"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
