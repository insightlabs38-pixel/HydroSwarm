"""Milestone 9.6: freshly train ARM_B_M9_6 (INTERLEAVED_MULTI_FAMILY) under
the exact-compute-parity protocol frozen in `m9-6-protocol.json`.

Mirrors `run_m9_0a_arm_b2.py`'s `train_arm_b2` orchestration exactly (same
`_build_family_pools`, `step_matched_interleaved_optimizer_step`,
`_build_update_slots`, fixed 3-update family rotation, ARM_A-matched
per-epoch optimizer-step targets and scheduler `total_steps`) -- reusing
every low-level helper UNMODIFIED, imported directly from
`run_m9_0a_arm_b2`/`run_m9_0_arm_b`. The only differences from that
historical script:
  1. `early_stopping_patience=0` (Section 9) so every seed always completes
     all 20 epochs = 1350 optimizer steps; and
  2. after training, the LAST periodic checkpoint (always epoch 20 now) is
     reloaded into a FRESH model instance and exported separately as the
     canonical, promotion-authoritative M9.6 checkpoint (FINAL STEP 1350,
     Section 14), alongside the untouched best-validation export.

Does not modify `run_m9_0a_arm_b2.py` or `run_m9_0_arm_b.py`.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_6_train_arm_b.py --seed 20260814

Writes:
  experiments/runs/hydrocore-v5-causal-m9-6/ARM_B-seed{seed}/...
  reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed{seed}.json
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

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.training.artifacts import RunArtifacts, atomic_json  # noqa: E402
from hydroswarm.training.causal_prefix import CausalPrefixDatasetView, fit_pool_signature_library, full_history_policy  # noqa: E402
from hydroswarm.training.checkpoint import export_model, load_checkpoint, save_checkpoint  # noqa: E402
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.data import CurriculumSchedule  # noqa: E402
from hydroswarm.training.trainer import _scheduler, set_deterministic_seed  # noqa: E402

import m9_6_common as m6  # noqa: E402
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402
from run_m9_0_arm_b import FAMILY_SEED_OFFSET, FEATURE_KWARGS, _build_family_pools, _combined_manifest_hash, _to_cpu_float, _validate_family  # noqa: E402
from run_m9_0a_arm_b2 import (  # noqa: E402
    ARM_A_OPTIMIZER_STEPS_PER_EPOCH,
    ARM_A_SCHEDULER_TOTAL_STEPS,
    FAMILY_NAMES,
    NUM_FAMILIES,
    _build_update_slots,
    _family_loader,
    step_matched_interleaved_optimizer_step,
)

assert list(ARM_A_OPTIMIZER_STEPS_PER_EPOCH) == list(m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH)
assert ARM_A_SCHEDULER_TOTAL_STEPS == m6.SCHEDULER_TOTAL_STEPS

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
GRADNORM_LOG_EVERY_N_BATCHES = 5


def train_arm_b_m9_6(seed: int) -> dict[str, Any]:
    assert not m6.assert_locked_test_closed()

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=seed, gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES, early_stopping_patience=0)
    assert config.early_stopping_patience == 0
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
    scheduler = _scheduler(optimizer, config, ARM_A_SCHEDULER_TOTAL_STEPS)

    run_root = m6.RUN_ROOT_M9_6 / f"ARM_B-seed{seed}"
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
                        f"but its curriculum-filtered pool offered {family_loader_lengths[family]}"
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

            # Reused unmodified from train_arm_b2 -- inert here because
            # config.early_stopping_patience == 0 (falsy), per protocol Section 9.
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

    assert epochs_completed == config.epochs, f"seed{seed}: ARM_B did not complete all {config.epochs} epochs (completed {epochs_completed})"
    assert stopped_early is False, f"seed{seed}: ARM_B early-stopping fired despite patience=0"
    assert global_step == m6.TOTAL_OPTIMIZER_STEPS, f"seed{seed}: ARM_B total optimizer steps {global_step} != required {m6.TOTAL_OPTIMIZER_STEPS}"

    # --- Section 14: canonical FINAL-STEP (1350) checkpoint. ---
    final_checkpoint_dir = Path(final_checkpoint)
    trainer_state = json.loads((final_checkpoint_dir / "trainer_state.json").read_text())
    true_total_optimizer_steps = int(trainer_state["global_step"])
    assert true_total_optimizer_steps == global_step
    assert int(trainer_state["epoch"]) == config.epochs - 1, f"seed{seed}: final checkpoint is not the last epoch"

    final_model = HydroCore.from_variant("small", use_adapters=False, **SHARED_MODEL_CONFIG)
    load_checkpoint(final_checkpoint_dir, model=final_model)
    canonical_export_path = export_model(
        final_model, run_root / "model-export-final-step.safetensors",
        metadata={"manifest_hash": _combined_manifest_hash(train_views), "global_steps": str(true_total_optimizer_steps), "epoch": str(trainer_state["epoch"])},
    )
    canonical_sha256 = hashlib.sha256(Path(canonical_export_path).read_bytes()).hexdigest()

    record = {
        "schema_version": 1,
        "purpose": "Milestone 9.6: ARM_B_M9_6 (INTERLEAVED_MULTI_FAMILY) exact-compute-parity training run.",
        "milestone": "M9.6", "arm": "ARM_B_M9_6", "seed": seed,
        "feature_kwargs": FEATURE_KWARGS, "trained_families": list(FAMILY_NAMES),
        "family_weighting": {family: 1.0 / NUM_FAMILIES for family in FAMILY_NAMES},
        "microbatches_per_optimizer_update": 4,
        "model_architecture": {"variant": "small", "use_adapters": False, **SHARED_MODEL_CONFIG, "param_count": param_count},
        "training_config": asdict(config), "training_config_source": str(CONFIG_PATH.relative_to(ROOT)),
        "scheduler_total_steps_required": m6.SCHEDULER_TOTAL_STEPS,
        "total_optimizer_steps_required": m6.TOTAL_OPTIMIZER_STEPS,
        "actual_total_optimizer_steps": global_step,
        "matches_required_total_optimizer_steps": global_step == m6.TOTAL_OPTIMIZER_STEPS,
        "actual_optimizer_steps_per_epoch": per_epoch_optimizer_steps,
        "matches_required_per_epoch_optimizer_steps": per_epoch_optimizer_steps == list(m6.ARM_A_OPTIMIZER_STEPS_PER_EPOCH),
        "epochs_completed": epochs_completed, "stopped_early": stopped_early, "stop_reason": training_summary["stop_reason"],
        "family_exposure_counts": family_exposure_counts, "per_epoch_family_microbatches": per_epoch_family_microbatches,
        "train_scenario_count_per_family": 200, "total_train_scenario_count": 200 * NUM_FAMILIES,
        "train_manifest_hash_per_family": {family: view.manifest_hash for family, view in train_views.items()},
        "validation_manifest_hash_per_family": {family: view.manifest_hash for family, view in validation_views.items()},
        "signature_library_manifest_hash_per_family": {family: library.manifest_hash for family, library in libraries.items()},
        "combined_manifest_hash": _combined_manifest_hash(train_views),
        "wall_seconds": wall_seconds,
        "best_validation_export_path": training_summary["export_path"], "best_validation_export_sha256": training_summary["export_sha256"],
        "best_validation_global_steps": training_summary["global_steps"], "best_epoch": best_epoch,
        "canonical_checkpoint_policy": m6.CANONICAL_CHECKPOINT_POLICY,
        "canonical_export_path": str(canonical_export_path), "canonical_export_sha256": canonical_sha256,
        "canonical_global_step": true_total_optimizer_steps, "canonical_epoch": int(trainer_state["epoch"]),
        "final_checkpoint": final_checkpoint,
        "locked_test_opened_after": m6.assert_locked_test_closed(),
    }
    m6.M9_6_TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = m6.M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return record


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    record = train_arm_b_m9_6(args.seed)
    print(json.dumps({
        "arm": record["arm"], "seed": record["seed"], "wall_seconds": record["wall_seconds"],
        "param_count": record["model_architecture"]["param_count"],
        "actual_total_optimizer_steps": record["actual_total_optimizer_steps"],
        "matches_required_total_optimizer_steps": record["matches_required_total_optimizer_steps"],
        "canonical_export_sha256": record["canonical_export_sha256"],
        "family_exposure_counts": record["family_exposure_counts"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
