"""Milestone 9.6: freshly train ARM_A_M9_6 (CURRENT / SINGLE_FAMILY,
golden-reference only) under the exact-compute-parity protocol frozen in
`m9-6-protocol.json`.

Byte-for-byte the same training entry point `run_m8_7_arm.py`'s AGE_FIX_ONLY
arm uses (same `configs/training-v5-causal.yaml`, same `Trainer`, same
`build_scenario_pool`-derived train/validation pools, same
`ARM_POLICIES["A"]` full-history depth policy, same AGE_FIX_ONLY feature/
model kwargs) -- the ONLY differences are:
  1. `early_stopping_patience=0` (an existing TrainingConfig field, not new
     code) so every seed always completes all 20 epochs = 1350 optimizer
     steps, per M9.6 protocol Section 9; and
  2. after `Trainer.fit()` returns, the LAST periodic checkpoint (always
     epoch 20 now that early stopping cannot fire) is reloaded into a FRESH
     model instance and exported separately as the canonical,
     promotion-authoritative M9.6 checkpoint (FINAL STEP 1350 -- Section
     14), while the best-validation export `Trainer.fit()` itself already
     writes is left untouched for the record.

Does not modify `hydroswarm/training/trainer.py` or `run_m8_7_arm.py`.

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_6_train_arm_a.py --seed 20260814

Writes:
  experiments/runs/hydrocore-v5-causal-m9-6/ARM_A-seed{seed}/...
  reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_A_M9_6-seed{seed}.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.simulation.network import build_wntr_network  # noqa: E402
from hydroswarm.training.causal_prefix import ARM_POLICIES, CausalPrefixDatasetView, build_scenario_pool, fit_pool_signature_library  # noqa: E402
from hydroswarm.training.checkpoint import export_model, load_checkpoint  # noqa: E402
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.trainer import Trainer, set_deterministic_seed  # noqa: E402

import m9_6_common as m6  # noqa: E402
from run_m8_7_arm import ARM_DEFINITIONS, SHARED_MODEL_CONFIG  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
GRADNORM_LOG_EVERY_N_BATCHES = 5


def train_arm_a_m9_6(seed: int) -> dict:
    assert not m6.assert_locked_test_closed()

    definition = ARM_DEFINITIONS["AGE_FIX_ONLY"]
    feature_kwargs = definition["feature_kwargs"]
    model_kwargs = definition["model_kwargs"]

    config = TrainingConfig.from_yaml(str(CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=seed, gradnorm_log_every_n_batches=GRADNORM_LOG_EVERY_N_BATCHES, early_stopping_patience=0)
    assert config.early_stopping_patience == 0

    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    train_records = build_scenario_pool("train", network_loader=build_wntr_network)
    validation_records = build_scenario_pool("validation", network_loader=build_wntr_network)
    library = fit_pool_signature_library(train_records)

    depth_policy = ARM_POLICIES["A"]
    train_view = CausalPrefixDatasetView(
        train_records, expected_split="train", signature_library=library, depth_policy=depth_policy,
        base_seed=config.seed, batch_size=config.batch_size, **feature_kwargs,
    )
    validation_view = CausalPrefixDatasetView(
        validation_records, expected_split="validation", signature_library=library, depth_policy=depth_policy,
        base_seed=config.seed, batch_size=config.batch_size, **feature_kwargs,
    )

    model = HydroCore.from_variant("small", use_adapters=False, **model_kwargs, **SHARED_MODEL_CONFIG)
    param_count = sum(p.numel() for p in model.parameters())

    run_root = m6.RUN_ROOT_M9_6 / f"ARM_A-seed{seed}"
    started = time.time()
    trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
    summary = trainer.fit()
    wall_seconds = time.time() - started

    assert summary.epochs_completed == config.epochs, f"seed{seed}: ARM_A did not complete all {config.epochs} epochs (completed {summary.epochs_completed})"
    assert summary.stopped_early is False, f"seed{seed}: ARM_A early-stopping fired despite patience=0"
    assert summary.stop_reason == "maximum_epochs"

    # --- Section 14: canonical FINAL-STEP (1350) checkpoint, independent of
    # Trainer.fit()'s own best-validation export. ---
    final_checkpoint_dir = Path(summary.final_checkpoint)
    trainer_state = json.loads((final_checkpoint_dir / "trainer_state.json").read_text())
    true_total_optimizer_steps = int(trainer_state["global_step"])
    assert int(trainer_state["epoch"]) == config.epochs - 1, f"seed{seed}: final checkpoint is not the last epoch"

    final_model = HydroCore.from_variant("small", use_adapters=False, **model_kwargs, **SHARED_MODEL_CONFIG)
    load_checkpoint(final_checkpoint_dir, model=final_model)
    canonical_export_path = export_model(
        final_model, run_root / "model-export-final-step.safetensors",
        metadata={"manifest_hash": train_view.manifest_hash, "global_steps": str(true_total_optimizer_steps), "epoch": str(trainer_state["epoch"])},
    )
    canonical_sha256 = hashlib.sha256(Path(canonical_export_path).read_bytes()).hexdigest()

    record = {
        "schema_version": 1,
        "purpose": "Milestone 9.6: ARM_A_M9_6 (CURRENT/SINGLE_FAMILY) exact-compute-parity training run.",
        "milestone": "M9.6", "arm": "ARM_A_M9_6", "seed": seed,
        "feature_kwargs": feature_kwargs, "model_kwargs": model_kwargs,
        "model_architecture": {"variant": "small", "use_adapters": False, **model_kwargs, **SHARED_MODEL_CONFIG, "param_count": param_count},
        "trained_families": ["golden-reference"], "family_weighting": {"golden-reference": 1.0},
        "training_config": asdict(config), "training_config_source": str(CONFIG_PATH.relative_to(ROOT)),
        "scheduler_total_steps_required": m6.SCHEDULER_TOTAL_STEPS,
        "total_optimizer_steps_required": m6.TOTAL_OPTIMIZER_STEPS,
        "actual_total_optimizer_steps": true_total_optimizer_steps,
        "matches_required_total_optimizer_steps": true_total_optimizer_steps == m6.TOTAL_OPTIMIZER_STEPS,
        "epochs_completed": summary.epochs_completed, "stopped_early": summary.stopped_early, "stop_reason": summary.stop_reason,
        "train_scenario_count": len(train_records), "validation_scenario_count": len(validation_records),
        "train_manifest_hash": train_view.manifest_hash, "validation_manifest_hash": validation_view.manifest_hash,
        "signature_library_manifest_hash": library.manifest_hash,
        "wall_seconds": wall_seconds,
        "best_validation_export_path": summary.export_path, "best_validation_export_sha256": summary.export_sha256,
        "best_validation_global_steps": summary.global_steps, "best_epoch": summary.best_epoch,
        "canonical_checkpoint_policy": m6.CANONICAL_CHECKPOINT_POLICY,
        "canonical_export_path": str(canonical_export_path), "canonical_export_sha256": canonical_sha256,
        "canonical_global_step": true_total_optimizer_steps, "canonical_epoch": int(trainer_state["epoch"]),
        "final_checkpoint": summary.final_checkpoint,
        "locked_test_opened_after": m6.assert_locked_test_closed(),
    }
    m6.M9_6_TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = m6.M9_6_TRAINING_RUNS_DIR / f"ARM_A_M9_6-seed{seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return record


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    record = train_arm_a_m9_6(args.seed)
    print(json.dumps({
        "arm": record["arm"], "seed": record["seed"], "wall_seconds": record["wall_seconds"],
        "param_count": record["model_architecture"]["param_count"],
        "actual_total_optimizer_steps": record["actual_total_optimizer_steps"],
        "matches_required_total_optimizer_steps": record["matches_required_total_optimizer_steps"],
        "canonical_export_sha256": record["canonical_export_sha256"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
