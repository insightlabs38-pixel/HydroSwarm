"""Gates required before any real Stage F joint-multitask training run may
start against `data/learning-v2/cycle-b2-joint-v4/` (core-issues3.txt Phase
12 Stage F prerequisite; see `scripts/build_stage_f_joint_corpus.py` for the
merge itself).

Per this pass's explicit scope, this script runs ONLY the pre-training
gates -- it does not start the full Stage F training run:

1. corpus_integrity   -- merge-report.json's own requirement_status is all
                          true (zero missing joins/duplicates/conflicts),
                          and every emitted population directory's shard
                          checksums verify independently (defense in depth:
                          a corrupted/edited shard written after the merge
                          completed would still be caught here even if the
                          merge report itself looked fine).
2. leakage            -- leakage-report.json recorded zero cross-population
                          scenario_id/seed_family collisions, re-verified
                          directly against the emitted index.jsonl files
                          (not merely re-reading the stored report).
3. batch_load          -- one real batch collated from the train population
                          spans more than one distinct topology
                          (network_id) -- proof this is genuinely a
                          multi-topology Stage-F batch, not an accidental
                          single-topology slice.
4. gradient_smoke      -- a full-config HydroCore ("small", every Stage-F-
                          relevant head enabled: event/control heads, Scout
                          control heads, candidate-conditioned Strategist
                          with consequence prescreening) forward+backward
                          pass on that batch reaches every retained task
                          group with a nonzero gradient and a finite loss:
                          Sentinel (source_node), control (event_cause,
                          next_step), Scout (sample_node), Strategist
                          (plan_validity, plan_value).
5. checkpoint_resume   -- Trainer.fit() for a handful of steps, then
                          fit(resume_from=<checkpoint>) for a few more,
                          exactly Bundle E's own established smoke pattern
                          (scripts/run_architecture_smoke_jobs.py), against
                          the real joint corpus instead of Cycle A.

Exits nonzero if any gate fails. Does not start the full Stage F run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from hydroswarm.model import HydroCore, verify_architecture_compatibility
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.training import (
    GovernedScenarioDataset,
    ShardedScenarioDataset,
    Trainer,
    TrainingConfig,
    collate_variable_topology,
)
from hydroswarm.training.losses import compute_multitask_loss, task_gradient_norms

JOINT_CORPUS_ROOT = Path("data/learning-v2/cycle-b2-joint-v4")
GRADIENT_CHECK_BATCH = 16
SEED = 20260808

#: "Every retained task group" (core-issues3.txt Phase 12 Stage F /
#: user-directed scope): one representative, real governed target per role.
#: Not ALL_TASK_NAMES -- a subset chosen so a batch drawn only from `train`
#: (the one population with 100% coverage of every role, per the merge
#: report) is guaranteed to exercise each group at least once.
REQUIRED_TASK_GROUPS: dict[str, str] = {
    "sentinel": "source_node",
    "control": "event_cause",
    "control_next_step": "next_step",
    "scout": "sample_node",
    "strategist_validity": "plan_validity",
    "strategist_value": "plan_value",
}

FULL_STAGE_F_MODEL_OVERRIDES: dict[str, Any] = dict(
    prior_mode="feature_only",
    event_control_heads=True,
    scout_control_heads=True,
    strategist_mode="candidate_conditioned",
    action_vocabulary_size=ACTION_TEMPLATE_COUNT,
    consequence_prescreening_heads=True,
)


def gate_corpus_integrity(root: Path) -> dict[str, Any]:
    merge_report = json.loads((root / "merge-report.json").read_text(encoding="utf-8"))
    requirement_status = merge_report["requirement_status"]
    if not all(requirement_status.values()):
        raise RuntimeError(f"merge-report.json requirement_status has a failing gate: {requirement_status}")

    verified_populations = []
    for population in merge_report["populations"]:
        split_dir = root / "tensors-normalized" / population["population"]
        dataset = ShardedScenarioDataset(split_dir, expected_split=population["output_split"])
        dataset.verify_shard_checksums()
        if len(dataset) != population["joined_examples"]:
            raise RuntimeError(
                f"{population['population']}: on-disk example count {len(dataset)} != "
                f"merge report's joined_examples {population['joined_examples']}"
            )
        verified_populations.append(population["population"])

    return {
        "status": "passed",
        "requirement_status": requirement_status,
        "populations_reverified": verified_populations,
    }


def gate_leakage(root: Path) -> dict[str, Any]:
    leakage_report = json.loads((root / "leakage-report.json").read_text(encoding="utf-8"))
    if leakage_report["scenario_id_leaks"] or leakage_report["seed_family_leaks"]:
        raise RuntimeError(f"leakage-report.json recorded real leaks: {leakage_report}")

    seen_scenarios: dict[str, str] = {}
    seen_families: dict[tuple[str, str], str] = {}
    for population in leakage_report["populations_checked"]:
        index_path = root / "tensors-normalized" / population / "index.jsonl"
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                scenario_id = record["scenario_id"]
                family = (record["network_id"], record["seed_family"])
                if scenario_id in seen_scenarios and seen_scenarios[scenario_id] != population:
                    raise RuntimeError(f"re-verified scenario_id leak: {scenario_id}")
                seen_scenarios[scenario_id] = population
                if family in seen_families and seen_families[family] != population:
                    raise RuntimeError(f"re-verified seed_family leak: {family}")
                seen_families[family] = population

    return {
        "status": "passed",
        "populations_reverified": leakage_report["populations_checked"],
        "total_examples_reverified": len(seen_scenarios),
    }


def _load_train_batch(root: Path, batch_size: int) -> list:
    dataset = ShardedScenarioDataset(root / "tensors-normalized" / "train", expected_split="train")
    dataset.verify_shard_checksums()
    # Deterministic spread across the shard file (not just the first
    # contiguous block) so the batch is not accidentally single-topology --
    # cycle-b2's own generator writes examples topology-by-topology within a
    # split (verified: shard-00000 is entirely golden-reference), so a
    # purely-sequential slice would silently fail the multi-topology check
    # this gate exists to enforce.
    stride = max(1, len(dataset) // batch_size)
    indices = [(index * stride) % len(dataset) for index in range(batch_size)]
    return [dataset[index] for index in indices]


def gate_batch_load(root: Path) -> dict[str, Any]:
    examples = _load_train_batch(root, GRADIENT_CHECK_BATCH)
    topologies = {example.network_id for example in examples}
    if len(topologies) < 2:
        raise RuntimeError(f"batch is not genuinely multi-topology: only {topologies} present")
    inputs, targets = collate_variable_topology(examples)
    return {
        "status": "passed",
        "batch_size": len(examples),
        "topologies_in_batch": sorted(topologies),
        "input_keys": sorted(inputs),
        "target_keys": sorted(targets),
    }


def gate_gradient_smoke(root: Path) -> dict[str, Any]:
    examples = _load_train_batch(root, GRADIENT_CHECK_BATCH)
    inputs, targets = collate_variable_topology(examples)

    missing_targets = [key for key in REQUIRED_TASK_GROUPS.values() if key not in targets]
    if missing_targets:
        raise RuntimeError(f"batch is missing required task-group target(s): {missing_targets}")

    model = HydroCore.from_variant("small", **FULL_STAGE_F_MODEL_OVERRIDES)
    model.train()
    output = model(inputs)
    result = compute_multitask_loss(output, targets)
    if not torch.isfinite(result.total):
        raise RuntimeError("non-finite total loss during Stage F gradient smoke test")

    for group, task in REQUIRED_TASK_GROUPS.items():
        if task not in result.tasks:
            raise RuntimeError(
                f"required task group {group!r} (target {task!r}) did not reach compute_multitask_loss "
                "-- missing model output or target key"
            )
        if result.valid_counts.get(task, 0) <= 0:
            raise RuntimeError(f"required task group {group!r} (target {task!r}) had zero valid positions")

    norms = task_gradient_norms(result.tasks, model)
    zero_gradient_tasks = [task for task, norm in norms.items() if norm == 0.0]
    required_zero = [task for task in zero_gradient_tasks if task in REQUIRED_TASK_GROUPS.values()]
    if required_zero:
        raise RuntimeError(f"required task(s) received zero gradient: {required_zero}")
    model.zero_grad(set_to_none=True)

    return {
        "status": "passed",
        "tasks_present": sorted(result.tasks),
        "valid_counts": {task: int(count) for task, count in result.valid_counts.items()},
        "gradient_norms": norms,
        "total_loss": float(result.total.detach()),
    }


def gate_checkpoint_resume(root: Path, run_root: Path) -> dict[str, Any]:
    train_dataset = ShardedScenarioDataset(root / "tensors-normalized" / "train", expected_split="train")
    validation_dataset = ShardedScenarioDataset(root / "tensors-normalized" / "validation", expected_split="validation")
    train_subset = GovernedScenarioDataset(
        [train_dataset[index] for index in range(0, min(64, len(train_dataset)), 1)], expected_split="train"
    )
    validation_subset = GovernedScenarioDataset(
        [validation_dataset[index] for index in range(0, min(16, len(validation_dataset)), 1)],
        expected_split="validation",
    )

    base_config = dict(
        seed=SEED,
        batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=2,
        checkpoint_every_epochs=1,
        maximum_runtime_seconds=900.0,
    )
    first_config = TrainingConfig(epochs=1, **base_config)
    first_trainer = Trainer(
        HydroCore.from_variant("small", **FULL_STAGE_F_MODEL_OVERRIDES),
        train_subset,
        validation_dataset=validation_subset,
        config=first_config,
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    first_summary = first_trainer.fit()
    if not math.isfinite(first_summary.best_validation_loss):
        raise RuntimeError("non-finite validation loss in Stage F checkpoint smoke test, first pass")

    resumed_config = TrainingConfig(epochs=2, **base_config)
    resumed_trainer = Trainer(
        HydroCore.from_variant("small", **FULL_STAGE_F_MODEL_OVERRIDES),
        train_subset,
        validation_dataset=validation_subset,
        config=resumed_config,
        run_root=run_root,
        workdir=".",
        collate_fn=collate_variable_topology,
    )
    resumed_summary = resumed_trainer.fit(resume_from=Path(first_summary.final_checkpoint))
    resume_ok = (
        resumed_summary.epochs_completed == 2
        and resumed_summary.global_steps > first_summary.global_steps
        and math.isfinite(resumed_summary.best_validation_loss)
    )
    if not resume_ok:
        raise RuntimeError(f"Stage F checkpoint resume did not advance training as expected: {resumed_summary!r}")

    reload_model = HydroCore.from_variant("small", **FULL_STAGE_F_MODEL_OVERRIDES)
    state_dict = load_file(str(Path(resumed_summary.final_checkpoint) / "model.safetensors"))
    if any(not torch.isfinite(tensor).all() for tensor in state_dict.values()):
        raise RuntimeError("Stage F resumed checkpoint contains non-finite weights")
    reload_model.load_state_dict(state_dict, strict=True)
    verify_architecture_compatibility(reload_model, reload_model.architecture_config())

    return {
        "status": "passed",
        "first_pass_epochs": first_summary.epochs_completed,
        "first_pass_best_validation_loss": first_summary.best_validation_loss,
        "resumed_epochs": resumed_summary.epochs_completed,
        "resumed_global_steps": resumed_summary.global_steps,
        "resumed_best_validation_loss": resumed_summary.best_validation_loss,
        "final_checkpoint": resumed_summary.final_checkpoint,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-root", type=Path, default=JOINT_CORPUS_ROOT)
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/stage-f-joint-corpus-smoke"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/results/v4/stage-f-joint-corpus-gates.json")
    )
    args = parser.parse_args(argv)

    gates: dict[str, Any] = {}
    overall_status = "passed"
    for name, gate in (
        ("corpus_integrity", lambda: gate_corpus_integrity(args.corpus_root)),
        ("leakage", lambda: gate_leakage(args.corpus_root)),
        ("batch_load", lambda: gate_batch_load(args.corpus_root)),
        ("gradient_smoke", lambda: gate_gradient_smoke(args.corpus_root)),
        ("checkpoint_resume", lambda: gate_checkpoint_resume(args.corpus_root, args.run_root)),
    ):
        try:
            gates[name] = gate()
        except Exception as error:  # noqa: BLE001 -- deliberately broad: record and continue to next gate
            gates[name] = {"status": "failed", "error": str(error)}
            overall_status = "failed"

    report = {
        "corpus_root": str(args.corpus_root),
        "overall_status": overall_status,
        "gates": gates,
        "note": "Pre-training gates only -- the full Stage F joint-multitask training run was not started.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: gate["status"] for name, gate in gates.items()}, indent=2, sort_keys=True))
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
