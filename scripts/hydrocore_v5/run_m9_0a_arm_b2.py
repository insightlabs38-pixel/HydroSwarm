"""Milestone 9.0a: train the STEP_MATCHED_INTERLEAVED_MULTI_FAMILY arm
(Arm B2) under the frozen M9.0a protocol
(docs/evaluation/HYDROCORE_V5_M9_0A_PROTOCOL.md).

Confound-resolution follow-up to Milestone 9.0's `run_m9_0_arm_b.py`: that
script used one microbatch per family (3 microbatches/optimizer update),
which gave Arm B ~33% more optimizer updates than Arm A
(gradient_accumulation_steps=4, 1 family) and a different scheduler
trajectory -- so M9.0's observed unseen-topology gain could not be cleanly
attributed to topology diversity alone. Arm B2 instead uses 4
family-pure microbatches per optimizer update (matching Arm A's own
accumulation window exactly) via a fixed 3-update rotation
(golden/branched/loop/EXTRA, cycling which family gets the 4th "extra"
slot), so every family still receives equal (1/3) long-run weight while the
optimizer-step COUNT and scheduler TRAJECTORY match Arm A exactly (protocol
Section 3; verified to require zero remainder against the real
curriculum-filtered per-family pools).

Reuses run_m9_0_arm_b.py's own helpers unmodified where the design is
identical (family-pool construction, per-epoch per-family DataLoader
construction, validation, AGE_FIX_ONLY feature kwargs, signature-library
handling) -- only the per-update family SCHEDULE, the loss normalizer (/4,
not /3), and the scheduler `total_steps` (Arm A's own 1500, not a per-family
recomputation) differ.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_0a_arm_b2.py --seed 20260814

Writes:
  reports/evaluation/hydrocore-v5/m9-0a-runs/ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.artifacts import RunArtifacts, atomic_json  # noqa: E402
from hydroswarm.training.causal_prefix import (  # noqa: E402
    CausalPrefixDatasetView,
    fit_pool_signature_library,
    full_history_policy,
)
from hydroswarm.training.checkpoint import export_model, save_checkpoint  # noqa: E402
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.data import CurriculumSchedule, collate_scenarios  # noqa: E402
from hydroswarm.training.losses import compute_multitask_loss  # noqa: E402
from hydroswarm.training.trainer import _scheduler, set_deterministic_seed  # noqa: E402

from run_m7_topology import TRAINED_FAMILIES  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0_arm_b import (  # noqa: E402
    FAMILY_SEED_OFFSET,
    FEATURE_KWARGS,
    _build_family_pools,
    _combined_manifest_hash,
    _to_cpu_float,
    _validate_family,
)

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
RUN_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m9-0a"
SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-runs"

GRADNORM_LOG_EVERY_N_BATCHES = 5  # identical compute-cost knob to run_m9_0_arm_b.py.
FAMILY_NAMES: tuple[str, ...] = tuple(name for name, _ in TRAINED_FAMILIES)
NUM_FAMILIES = len(FAMILY_NAMES)
assert NUM_FAMILIES == 3
MICROBATCHES_PER_UPDATE = 4  # matches Arm A's gradient_accumulation_steps=4 (protocol Section 3).

#: Arm A's REAL (not static-estimate) per-epoch optimizer-step counts,
#: measured directly from
#: experiments/runs/hydrocore-v5-causal-m8-7/AGE_FIX_ONLY-seed*/*/metrics.jsonl
#: (identical across all 3 seeds -- curriculum-stage filtering is
#: seed-independent). Frozen BEFORE Arm B2 training (M9.0a protocol Section 2).
ARM_A_OPTIMIZER_STEPS_PER_EPOCH: tuple[int, ...] = (15, 30, 45, 60) + (75,) * 16
ARM_A_TOTAL_OPTIMIZER_STEPS = sum(ARM_A_OPTIMIZER_STEPS_PER_EPOCH)
assert ARM_A_TOTAL_OPTIMIZER_STEPS == 1350
assert len(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == 20  # config.epochs.

#: Arm A's own static scheduler total_steps estimate
#: (Trainer.__init__'s formula, ceil(600/2/4)*20) -- Arm B2 uses the SAME
#: value (protocol Section 5), not a per-family recomputation (M9.0's Arm B
#: used ceil(200/2/1)*20=2000, the source of its scheduler-trajectory
#: mismatch).
ARM_A_SCHEDULER_TOTAL_STEPS = math.ceil(600 / 2 / 4) * 20
assert ARM_A_SCHEDULER_TOTAL_STEPS == 1500


def step_matched_interleaved_optimizer_step(
    model: HydroCore,
    optimizer: torch.optim.Optimizer,
    slot_batches: list[tuple[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]],
    *,
    config: TrainingConfig,
    step: bool = True,
    clip: bool = True,
) -> dict[str, Any]:
    """One M9.0a step-matched interleaved optimizer step (protocol Section
    3): zero_grad, then one family-pure micro-batch forward+backward per
    SLOT in `slot_batches` (a LIST, not a dict -- unlike M9.0's own
    `interleaved_optimizer_step`, because one family legitimately appears
    TWICE in the same update under the 3-update rotation's "extra" slot), no
    zero_grad between slots (gradients accumulate onto the SAME model
    parameters across every slot), each loss normalized by `len(slot_batches)`
    (4 in every real call site, NOT the distinct-family count 3 -- protocol
    Section 4), then clip+step. `step=False`/`clip=False` are test-only,
    identical rationale to M9.0's own function (see
    tests/scientific/test_step_matched_interleaving_m9_0a.py)."""

    optimizer.zero_grad(set_to_none=True)
    slot_losses: list[tuple[str, float]] = []
    family_batch_sizes: dict[str, int] = {}
    num_slots = len(slot_batches)
    for family, (inputs, targets) in slot_batches:
        output = model(_to_cpu_float(inputs))
        result = compute_multitask_loss(output, targets, task_weights=config.task_weights, profile_ordinal_weight=config.profile_ordinal_weight)
        if not torch.isfinite(result.total):
            raise FloatingPointError(f"non-finite multitask loss (family={family})")
        (result.total / num_slots).backward()
        slot_losses.append((family, float(result.total.detach())))
        family_batch_sizes[family] = family_batch_sizes.get(family, 0) + int(next(iter(targets.values())).shape[0])
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)) if clip else float("nan")
    if step:
        optimizer.step()
    family_losses_mean: dict[str, float] = {}
    family_loss_counts: dict[str, int] = {}
    for family, loss in slot_losses:
        family_losses_mean[family] = family_losses_mean.get(family, 0.0) + loss
        family_loss_counts[family] = family_loss_counts.get(family, 0) + 1
    for family in family_losses_mean:
        family_losses_mean[family] /= family_loss_counts[family]
    return {
        "slot_losses": slot_losses,
        "family_losses_mean": family_losses_mean,
        "family_batch_sizes": family_batch_sizes,
        "gradient_norm": gradient_norm,
    }


def _extra_family_for_cycle_position(position: int) -> str:
    """Protocol Section 3's fixed 3-update rotation: cycle position 0 gives
    the 4th 'extra' microbatch slot to FAMILY_NAMES[0] (golden-reference),
    position 1 to FAMILY_NAMES[1] (branched-loop), position 2 to
    FAMILY_NAMES[2] (loop-grid)."""
    return FAMILY_NAMES[position % NUM_FAMILIES]


def _family_loader(dataset: CausalPrefixDatasetView, *, base_seed: int, epoch: int, family_index: int) -> DataLoader:
    generator = torch.Generator().manual_seed(base_seed + epoch + family_index * FAMILY_SEED_OFFSET)
    return DataLoader(
        dataset, batch_size=dataset._batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_scenarios, generator=generator,
    )


def _build_update_slots(epoch: int, target_updates: int, iterators: dict[str, Any]) -> list[list[tuple[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]]]:
    """Builds the exact ordered list of per-update slot lists for one
    epoch, consuming `iterators` in the fixed rotation order. Separated from
    the training loop so the exact schedule shape is independently
    unit-testable without a real model/optimizer
    (tests/scientific/test_step_matched_interleaving_m9_0a.py)."""
    updates: list[list[tuple[str, Any]]] = []
    for update_index in range(target_updates):
        extra_family = _extra_family_for_cycle_position(update_index % NUM_FAMILIES)
        slots: list[tuple[str, Any]] = []
        for family in FAMILY_NAMES:
            slots.append((family, next(iterators[family])))
        slots.append((extra_family, next(iterators[extra_family])))
        updates.append(slots)
    return updates


def train_arm_b2(seed: int) -> dict[str, Any]:
    assert not locked_test_opened(ROOT), "locked test must remain closed"

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=seed, gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES)
    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    pools = _build_family_pools()
    libraries = {family: fit_pool_signature_library(pools[family]["train"]) for family in FAMILY_NAMES}
    train_views = {
        family: CausalPrefixDatasetView(
            pools[family]["train"], expected_split="train", signature_library=libraries[family],
            depth_policy=full_history_policy, base_seed=config.seed, batch_size=config.batch_size, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }
    validation_views = {
        family: CausalPrefixDatasetView(
            pools[family]["validation"], expected_split="validation", signature_library=libraries[family],
            depth_policy=full_history_policy, base_seed=config.seed, batch_size=config.batch_size, **FEATURE_KWARGS,
        )
        for family in FAMILY_NAMES
    }

    model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    param_count = sum(p.numel() for p in model.parameters())
    model = model.cpu().float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, foreach=False)
    #: Matches Arm A's own scheduler total_steps EXACTLY (protocol Section 5)
    #: -- not a per-family recomputation.
    scheduler = _scheduler(optimizer, config, ARM_A_SCHEDULER_TOTAL_STEPS)

    run_root = RUN_ROOT / f"ARM_B2-seed{seed}"
    artifacts = RunArtifacts.create(
        run_root, config=asdict(config), manifest_hash=_combined_manifest_hash(train_views), workdir=".",
    )
    curriculum = CurriculumSchedule.progressive()

    global_step = 0
    best = math.inf
    best_epoch = -1
    best_global_step = 0
    stale_epochs = 0
    stopped_early = False
    epochs_completed = 0
    final_checkpoint = ""
    last_resumable_checkpoint = ""
    family_exposure_counts = dict.fromkeys(FAMILY_NAMES, 0)
    per_epoch_optimizer_steps: list[int] = []
    per_epoch_family_microbatches: list[dict[str, int]] = []
    started = time.monotonic()

    try:
        for epoch in range(config.epochs):
            stage = curriculum.stage_for_epoch(epoch)
            epoch_datasets = {family: view.stages_through(stage) for family, view in train_views.items()}
            for family in FAMILY_NAMES:
                assert epoch_datasets[family]._unobserved_age_sentinel == "fixed"
                assert epoch_datasets[family]._include_relative_gap_feature is False

            loaders = {
                family: _family_loader(epoch_datasets[family], base_seed=config.seed, epoch=epoch, family_index=index)
                for index, family in enumerate(FAMILY_NAMES)
            }
            family_loader_lengths = {family: len(loader) for family, loader in loaders.items()}
            iterators = {family: iter(loader) for family, loader in loaders.items()}
            target_updates = ARM_A_OPTIMIZER_STEPS_PER_EPOCH[epoch]

            model.train()
            epoch_loss_total = 0.0
            epoch_steps = 0
            epoch_family_mb = dict.fromkeys(FAMILY_NAMES, 0)
            update_slot_lists = _build_update_slots(epoch, target_updates, iterators)
            for slot_batches in update_slot_lists:
                for family, _batch in slot_batches:
                    epoch_family_mb[family] += 1
                step_result = step_matched_interleaved_optimizer_step(model, optimizer, slot_batches, config=config)
                scheduler.step()
                global_step += 1
                epoch_steps += 1
                for family, batch_size in step_result["family_batch_sizes"].items():
                    family_exposure_counts[family] += batch_size
                step_total = sum(loss for _family, loss in step_result["slot_losses"])
                epoch_loss_total += step_total
                artifacts.append_metric({
                    "epoch": epoch, "step": epoch_steps, "global_step": global_step,
                    "slot_losses": step_result["slot_losses"], "family_losses_mean": step_result["family_losses_mean"],
                    "step_total_loss": step_total, "gradient_norm": step_result["gradient_norm"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                })

            for family in FAMILY_NAMES:
                if epoch_family_mb[family] != family_loader_lengths[family]:
                    raise AssertionError(
                        f"epoch {epoch}: family {family} consumed {epoch_family_mb[family]} microbatches, "
                        f"but its curriculum-filtered pool offered {family_loader_lengths[family]} -- "
                        "step-matched schedule did not exactly consume the family's available pool "
                        "(protocol Section 3's zero-remainder guarantee is violated)"
                    )
            train_loss = epoch_loss_total / max(epoch_steps, 1)
            per_epoch_optimizer_steps.append(epoch_steps)
            per_epoch_family_microbatches.append(dict(epoch_family_mb))

            per_family_validation_loss = {
                family: _validate_family(model, validation_views[family], base_seed=config.seed, epoch=epoch, family_index=index, config=config)
                for index, family in enumerate(FAMILY_NAMES)
            }
            validation_loss = sum(per_family_validation_loss.values()) / NUM_FAMILIES
            artifacts.append_jsonl("validation_history.jsonl", {
                "epoch": epoch, "curriculum_stage": stage.name, "validation_loss": validation_loss,
                "per_family_validation_loss": per_family_validation_loss,
            })

            improved = validation_loss < best - config.minimum_delta
            if improved:
                best = validation_loss
                best_epoch = epoch
                best_global_step = global_step
                stale_epochs = 0
                export_model(
                    model, artifacts.path / "best-model.safetensors",
                    metadata={"epoch": str(epoch), "validation_loss": str(validation_loss), "global_step": str(global_step)},
                )
            else:
                stale_epochs += 1
            epochs_completed = epoch + 1

            if (epoch + 1) % config.checkpoint_every_epochs == 0:
                checkpoint = artifacts.path / "checkpoints" / f"checkpoint-{epoch + 1:04d}"
                final_checkpoint = str(save_checkpoint(
                    checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, global_step=global_step, best_validation_loss=best,
                ))
                last_resumable_checkpoint = final_checkpoint

            atomic_json(artifacts.path / "epoch_summary.json", {
                "epoch": epoch, "curriculum_stage": stage.name, "train_loss": train_loss,
                "validation_loss": validation_loss, "per_family_validation_loss": per_family_validation_loss,
                "best_validation_loss": best, "optimizer_steps_this_epoch": epoch_steps,
                "family_microbatches_this_epoch": epoch_family_mb,
            })

            if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
                stopped_early = True
                break

        if not final_checkpoint:
            checkpoint = artifacts.path / "checkpoints" / "checkpoint-final"
            final_checkpoint = str(save_checkpoint(
                checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=max(epochs_completed - 1, 0), global_step=global_step, best_validation_loss=best,
            ))
            last_resumable_checkpoint = final_checkpoint

        best_model_path = artifacts.path / "best-model.safetensors"
        if best_model_path.exists():
            from safetensors.torch import load_file
            model.load_state_dict(load_file(best_model_path, device="cpu"), strict=True)
        export_path = export_model(
            model, artifacts.path / "model-export.safetensors",
            metadata={"manifest_hash": _combined_manifest_hash(train_views), "global_steps": str(best_global_step or global_step)},
        )
        export_sha256 = hashlib.sha256(Path(export_path).read_bytes()).hexdigest()
        wall_seconds = time.monotonic() - started

        training_summary = {
            "run_directory": str(artifacts.path), "epochs_completed": epochs_completed,
            "global_steps": best_global_step or global_step, "best_validation_loss": best,
            "best_epoch": best_epoch, "stopped_early": stopped_early,
            "stop_reason": "validation_convergence" if stopped_early else "maximum_epochs",
            "final_checkpoint": final_checkpoint, "export_path": str(export_path),
            "export_sha256": export_sha256, "last_resumable_checkpoint": last_resumable_checkpoint,
        }
        atomic_json(artifacts.path / "summary.json", training_summary)
        artifacts.status("COMPLETED", stop_reason=training_summary["stop_reason"])
    except Exception as error:
        atomic_json(artifacts.path / "failure.json", {"type": type(error).__name__, "message": str(error)})
        artifacts.status("FAILED", error=str(error))
        raise

    record = {
        "schema_version": 1,
        "purpose": "Milestone 9.0a: STEP_MATCHED_INTERLEAVED_MULTI_FAMILY (Arm B2) training run under the frozen M9.0a protocol.",
        "arm": "ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY",
        "seed": seed,
        "feature_kwargs": FEATURE_KWARGS,
        "trained_families": list(FAMILY_NAMES),
        "family_weighting": {family: 1.0 / NUM_FAMILIES for family in FAMILY_NAMES},
        "microbatches_per_optimizer_update": MICROBATCHES_PER_UPDATE,
        "model_architecture": {"variant": "small", "use_adapters": False, **SHARED_MODEL_CONFIG, "param_count": param_count},
        "training_config": asdict(config),
        "training_config_source": str(CONFIG_PATH.relative_to(ROOT)),
        "scheduler_total_steps": ARM_A_SCHEDULER_TOTAL_STEPS,
        "arm_a_target_optimizer_steps_total": ARM_A_TOTAL_OPTIMIZER_STEPS,
        "arm_a_target_optimizer_steps_per_epoch": list(ARM_A_OPTIMIZER_STEPS_PER_EPOCH),
        "actual_optimizer_steps_total": global_step,
        "actual_optimizer_steps_per_epoch": per_epoch_optimizer_steps,
        "matches_arm_a_total_optimizer_steps": global_step == ARM_A_TOTAL_OPTIMIZER_STEPS,
        "matches_arm_a_per_epoch_optimizer_steps": per_epoch_optimizer_steps == list(ARM_A_OPTIMIZER_STEPS_PER_EPOCH),
        "family_exposure_counts": family_exposure_counts,
        "per_epoch_family_microbatches": per_epoch_family_microbatches,
        "train_scenario_count_per_family": 200,
        "total_train_scenario_count": 200 * NUM_FAMILIES,
        "train_manifest_hash_per_family": {family: view.manifest_hash for family, view in train_views.items()},
        "validation_manifest_hash_per_family": {family: view.manifest_hash for family, view in validation_views.items()},
        "signature_library_manifest_hash_per_family": {family: library.manifest_hash for family, library in libraries.items()},
        "combined_manifest_hash": _combined_manifest_hash(train_views),
        "wall_seconds": wall_seconds,
        "training_summary": training_summary,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_ROOT / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return record


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    record = train_arm_b2(args.seed)
    print(json.dumps({
        "arm": record["arm"], "seed": record["seed"], "wall_seconds": record["wall_seconds"],
        "param_count": record["model_architecture"]["param_count"],
        "checkpoint_sha256": record["training_summary"]["export_sha256"],
        "matches_arm_a_total_optimizer_steps": record["matches_arm_a_total_optimizer_steps"],
        "matches_arm_a_per_epoch_optimizer_steps": record["matches_arm_a_per_epoch_optimizer_steps"],
        "family_exposure_counts": record["family_exposure_counts"],
        "summary": record["training_summary"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
