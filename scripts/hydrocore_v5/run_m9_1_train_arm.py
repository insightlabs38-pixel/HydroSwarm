"""Milestone 9.1: train or reuse-checkpoint ONE (arm, seed) under the frozen
protocol (`docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md`, Sections 2/5/6/10).

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_train_arm.py --arm CURRENT --seed 20260814
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_train_arm.py --arm GRAPH_ODE --seed 20260814
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_train_arm.py --arm GRAPH_CDE --seed 31874
    .venv/bin/python scripts/hydrocore_v5/run_m9_1_train_arm.py --arm GRAPH_SDE --seed 20260815

CURRENT: Section 2's checkpoint-reuse procedure -- reads
`m8-7-runs/AGE_FIX_ONLY-seed{seed}.json`, verifies the checkpoint's on-disk
SHA-256 against the recorded `training_summary.export_sha256`, and loads it
(REUSED_M8_7_CHECKPOINT) with zero training. Only if that checkpoint file is
physically absent does this retrain from scratch under the IDENTICAL frozen
M8.7 AGE_FIX_ONLY recipe (REGENERATED_NOT_ORIGINAL_M8_7_CHECKPOINT,
transparently flagged, never silently absorbed).

GRAPH_ODE/GRAPH_CDE/GRAPH_SDE: always trained fresh (no prior checkpoint),
identical `configs/training-v5-causal.yaml` recipe, identical
`ARM_POLICIES["A"]` full-history depth policy, identical AGE_FIX_ONLY
feature kwargs -- differing from CURRENT only in `temporal_dynamics`
(Section 6's frozen parameter-matched width).

Idempotent: if this (arm, seed)'s run record already exists and its
checkpoint's on-disk SHA-256 still matches, this script prints and exits
without retraining -- required for a multi-hour/multi-session-spanning
screening run to be safely re-invoked.

Writes:
  reports/evaluation/hydrocore-v5/m9-1-runs/{ARM}-seed{SEED}.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m9_1_common as common  # noqa: E402
from hydroswarm.training.causal_prefix import ARM_POLICIES, CausalPrefixDatasetView  # noqa: E402
from hydroswarm.training.config import TrainingConfig  # noqa: E402
from hydroswarm.training.trainer import Trainer, set_deterministic_seed  # noqa: E402


def _already_done(arm: str, seed: int) -> bool:
    path = common.run_record_path(arm, seed)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if arm != "CURRENT" and record.get("instability_flag") is not None:
        # Section 5: an UNSTABLE_ARM_SEED/SOLVER_STEP_LIMIT_EXCEEDED result
        # is a final, recorded outcome -- never retried.
        return True
    export_path = record["export_path"] if arm == "CURRENT" else record["training_summary"]["export_path"]
    recorded_hash = record["checkpoint_sha256"] if arm == "CURRENT" else record["training_summary"]["export_sha256"]
    checkpoint = Path(export_path)
    if not checkpoint.exists():
        return False
    return hashlib.sha256(checkpoint.read_bytes()).hexdigest() == recorded_hash


def _reuse_or_regenerate_current(seed: int, code_under_test_commit: str, locked_before: bool) -> dict:
    m8_7_record_path = common.M8_7_RUNS_DIR / f"AGE_FIX_ONLY-seed{seed}.json"
    m8_7_record = json.loads(m8_7_record_path.read_text())
    export_path = Path(m8_7_record["training_summary"]["export_path"])
    recorded_hash = m8_7_record["training_summary"]["export_sha256"]

    if export_path.exists():
        on_disk_hash = hashlib.sha256(export_path.read_bytes()).hexdigest()
        if on_disk_hash != recorded_hash:
            raise AssertionError(
                f"seed {seed}: CURRENT checkpoint at {export_path} exists but its SHA-256 does not match "
                f"m8-7-runs' recorded provenance -- refusing to load a mismatched/corrupted checkpoint"
            )
        source = "REUSED_M8_7_CHECKPOINT"
        checkpoint_sha256 = recorded_hash
        final_export_path = str(export_path)
        wall_seconds = 0.0
        param_count = m8_7_record["model_architecture"]["param_count"]
        training_summary = None
    else:
        print(f"seed {seed}: M8.7 AGE_FIX_ONLY checkpoint missing on disk -- retraining under the identical frozen recipe", flush=True)
        config = TrainingConfig.from_yaml(str(common.CONFIG_PATH), require_complete_task_weights=True)
        config = replace(config, seed=seed, gradnorm_log_every_n_batches=common.GRADNORM_LOG_EVERY_N_BATCHES)
        set_deterministic_seed(config.seed, deterministic=config.deterministic)

        train_records = common.load_pool("train")
        validation_records = common.load_pool("validation")
        library = common.fit_library(train_records)
        depth_policy = ARM_POLICIES["A"]
        train_view = CausalPrefixDatasetView(
            train_records, expected_split="train", signature_library=library, depth_policy=depth_policy,
            base_seed=config.seed, batch_size=config.batch_size, **common.FEATURE_KWARGS,
        )
        validation_view = CausalPrefixDatasetView(
            validation_records, expected_split="validation", signature_library=library, depth_policy=depth_policy,
            base_seed=config.seed, batch_size=config.batch_size, **common.FEATURE_KWARGS,
        )
        model = common.build_current_model()
        param_count = sum(p.numel() for p in model.parameters())
        run_root = common.RUN_ROOT / f"CURRENT-seed{seed}"
        started = time.time()
        trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
        summary = trainer.fit()
        wall_seconds = time.time() - started
        source = "REGENERATED_NOT_ORIGINAL_M8_7_CHECKPOINT"
        checkpoint_sha256 = summary.export_sha256
        final_export_path = summary.export_path
        training_summary = asdict(summary)

    return {
        "schema_version": 1,
        "protocol_frozen_at_commit": common.PROTOCOL_FROZEN_AT_COMMIT,
        "code_under_test_commit": code_under_test_commit,
        "arm": "CURRENT",
        "seed": seed,
        "current_checkpoint_source": source,
        "m8_7_source_record": str(m8_7_record_path.relative_to(common.ROOT)),
        "model_kwargs": common.CURRENT_MODEL_KWARGS,
        "feature_kwargs": common.FEATURE_KWARGS,
        "param_count": param_count,
        "export_path": final_export_path,
        "checkpoint_sha256": checkpoint_sha256,
        "wall_seconds": wall_seconds,
        "training_summary": training_summary,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": common.assert_locked_test_closed(),
    }


def _train_novel_arm(arm: str, seed: int, code_under_test_commit: str, locked_before: bool) -> dict:
    config = TrainingConfig.from_yaml(str(common.CONFIG_PATH), require_complete_task_weights=True)
    config = replace(config, seed=seed, gradnorm_log_every_n_batches=common.GRADNORM_LOG_EVERY_N_BATCHES)
    set_deterministic_seed(config.seed, deterministic=config.deterministic)

    train_records = common.load_pool("train")
    validation_records = common.load_pool("validation")
    library = common.fit_library(train_records)
    depth_policy = ARM_POLICIES["A"]
    train_view = CausalPrefixDatasetView(
        train_records, expected_split="train", signature_library=library, depth_policy=depth_policy,
        base_seed=config.seed, batch_size=config.batch_size, **common.FEATURE_KWARGS,
    )
    validation_view = CausalPrefixDatasetView(
        validation_records, expected_split="validation", signature_library=library, depth_policy=depth_policy,
        base_seed=config.seed, batch_size=config.batch_size, **common.FEATURE_KWARGS,
    )

    model = common.build_novel_model(arm)
    param_count = sum(p.numel() for p in model.parameters())
    delta_pct = (param_count - common.CURRENT_BASELINE_TOTAL_PARAMS) / common.CURRENT_BASELINE_TOTAL_PARAMS * 100.0

    run_root = common.RUN_ROOT / f"{arm}-seed{seed}"
    started = time.time()
    try:
        trainer = Trainer(model, train_view, config=config, run_root=run_root, validation_dataset=validation_view)
        summary = trainer.fit()
        instability = None
    except AssertionError as error:
        # torchdiffeq's dopri5 stepper (rk_common.py) enforces max_num_steps
        # via a bare `assert n_steps < self.max_num_steps, "max_num_steps
        # exceeded (...)"` -- this is the ONLY AssertionError this training
        # loop can raise (protocol Section 7: recorded, never silently
        # raised past, never retried with a larger bound).
        if "max_num_steps exceeded" in str(error):
            instability = "SOLVER_STEP_LIMIT_EXCEEDED"
            summary = None
        else:
            raise
    except FloatingPointError as error:
        # Trainer._step (trainer.py) raises FloatingPointError("non-finite
        # multitask loss") the moment any batch's total loss is non-finite
        # -- protocol Section 5: recorded as UNSTABLE_ARM_SEED, never
        # retried, disqualifies this arm from PROMOTION_CANDIDATE at Step 1
        # regardless of the other seed's result.
        if "non-finite" in str(error):
            instability = "UNSTABLE_ARM_SEED"
            summary = None
        else:
            raise
    wall_seconds = time.time() - started

    record = {
        "schema_version": 1,
        "protocol_frozen_at_commit": common.PROTOCOL_FROZEN_AT_COMMIT,
        "code_under_test_commit": code_under_test_commit,
        "arm": arm,
        "seed": seed,
        "mlp_width": common.ARM_MLP_WIDTH[arm],
        "param_count": param_count,
        "param_count_delta_vs_current_percent": delta_pct,
        "feature_kwargs": common.FEATURE_KWARGS,
        "training_config": asdict(config),
        "training_config_source": str(common.CONFIG_PATH.relative_to(common.ROOT)),
        "train_manifest_hash": train_view.manifest_hash,
        "validation_manifest_hash": validation_view.manifest_hash,
        "train_scenario_count": len(train_records),
        "validation_scenario_count": len(validation_records),
        "wall_seconds": wall_seconds,
        "instability_flag": instability,
        "training_summary": asdict(summary) if summary is not None else None,
        "locked_test_opened_before": locked_before,
        "locked_test_opened_after": common.assert_locked_test_closed(),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=common.ALL_ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--force", action="store_true", help="retrain/reload even if a verified run record already exists")
    args = parser.parse_args()

    code_under_test_commit = common.assert_code_under_test_commit()
    locked_before = common.assert_locked_test_closed()

    if not args.force and _already_done(args.arm, args.seed):
        print(f"{args.arm}-seed{args.seed}: already done (verified checkpoint on disk) -- skipping")
        return 0

    if args.arm == "CURRENT":
        record = _reuse_or_regenerate_current(args.seed, code_under_test_commit, locked_before)
    else:
        record = _train_novel_arm(args.arm, args.seed, code_under_test_commit, locked_before)

    common.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = common.run_record_path(args.arm, args.seed)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps({k: v for k, v in record.items() if k != "training_config"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
