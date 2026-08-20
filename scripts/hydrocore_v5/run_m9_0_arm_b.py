"""Milestone 9.0: train the INTERLEAVED_MULTI_FAMILY arm (Arm B) under the
frozen M9.0 protocol (docs/evaluation/HYDROCORE_V5_M9_0_PROTOCOL.md).

Retests Milestone 7's topology-diversity question with TRUE interleaving
instead of M7's sequential 3-phase curriculum: one optimizer step consumes
exactly one family-pure micro-batch from EACH of golden-reference,
branched-loop, and loop-grid, in fixed order, every step (zero_grad ->
golden backward -> branched-loop backward -> loop-grid backward -> clip ->
optimizer.step() -> scheduler.step()) -- see the M9.0 protocol document,
Section 4, for the full frozen design and its rationale.

No interleaved-family training loop exists anywhere else in this repo
(`Trainer._train_epoch` is built around exactly one dataset/`DataLoader`
with no multi-dataset extension point -- verified before writing this),
so this script manually replicates `Trainer.fit()`'s epoch/optimizer/
scheduler/checkpoint/early-stopping machinery for three concurrently
family-pure data sources sharing ONE model/optimizer/scheduler, using
`Trainer`'s own helpers (`RunArtifacts`, `export_model`, `save_checkpoint`,
`_scheduler`, `CurriculumSchedule`, `compute_multitask_loss`) unmodified.

Training families/pools/seed-bases are IMPORTED, unmodified, from
`run_m7_topology.py` (`_family_scenario_pool`, `SEED_BASES`,
`TRAINED_FAMILIES`, `TRAIN_PER_FAMILY`, `VALIDATION_PER_FAMILY`,
`CALIBRATION_PER_FAMILY`) so Arm B's training corpus is the SAME
600-train/100-validation/150-calibration, 200-per-family pool Milestone
7's own EXPANDED arm would have used -- the only material difference from
M7 is the family SCHEDULE (interleaved here vs sequential there).

Usage:
    .venv/bin/python scripts/hydrocore_v5/run_m9_0_arm_b.py --seed 20260814

Writes:
  reports/evaluation/hydrocore-v5/m9-0-runs/ARM_B-seed{seed}.json
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

from run_m7_topology import (  # noqa: E402
    CALIBRATION_PER_FAMILY,
    SEED_BASES,
    TRAIN_PER_FAMILY,
    TRAINED_FAMILIES,
    VALIDATION_PER_FAMILY,
    _family_scenario_pool,
)
from run_m8_7_arm import SHARED_MODEL_CONFIG  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "training-v5-causal.yaml"
RUN_ROOT = ROOT / "experiments" / "runs" / "hydrocore-v5-causal-m9-0"
SUMMARY_ROOT = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0-runs"

GRADNORM_LOG_EVERY_N_BATCHES = 5  # identical compute-cost knob to run_m8_7_arm.py.
FEATURE_KWARGS = {"unobserved_age_sentinel": "fixed", "include_relative_gap_feature": False}  # AGE_FIX_ONLY, frozen M8.7.
FAMILY_NAMES: tuple[str, ...] = tuple(name for name, _ in TRAINED_FAMILIES)
NUM_FAMILIES = len(FAMILY_NAMES)
assert NUM_FAMILIES == 3
#: Milestone-1 total-budget parity, matching run_m7_topology's own convention:
#: only the LAST family gets the +1 (33+33+34 = 100).
VALIDATION_COUNTS: tuple[int, ...] = (VALIDATION_PER_FAMILY, VALIDATION_PER_FAMILY, VALIDATION_PER_FAMILY + 1)
#: Deterministic per-family DataLoader-shuffle seed offset (a large prime)
#: so the three families never draw identical shuffles from the same
#: (config.seed + epoch) base -- frozen in the M9.0 protocol, Section 4.
FAMILY_SEED_OFFSET = 7919


def _to_cpu_float(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.float() if value.is_floating_point() else value for key, value in inputs.items()}


def _build_family_pools() -> dict[str, dict[str, list]]:
    pools: dict[str, dict[str, list]] = {}
    for family_index, (family, loader) in enumerate(TRAINED_FAMILIES):
        pools[family] = {
            "train": _family_scenario_pool(
                "train", network_loader=loader, family=family,
                seed_base=SEED_BASES[(family, "train")], count=TRAIN_PER_FAMILY,
            ),
            "validation": _family_scenario_pool(
                "validation", network_loader=loader, family=family,
                seed_base=SEED_BASES[(family, "validation")], count=VALIDATION_COUNTS[family_index],
            ),
            "calibration": _family_scenario_pool(
                "calibration", network_loader=loader, family=family,
                seed_base=SEED_BASES[(family, "calibration")], count=CALIBRATION_PER_FAMILY,
            ),
        }
    return pools


def _combined_manifest_hash(views: dict[str, CausalPrefixDatasetView]) -> str:
    payload = json.dumps({family: view.manifest_hash for family, view in sorted(views.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _family_loader(dataset: CausalPrefixDatasetView, *, base_seed: int, epoch: int, family_index: int) -> DataLoader:
    generator = torch.Generator().manual_seed(base_seed + epoch + family_index * FAMILY_SEED_OFFSET)
    return DataLoader(
        dataset, batch_size=dataset._batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_scenarios, generator=generator,
    )


def interleaved_optimizer_step(
    model: HydroCore, optimizer: torch.optim.Optimizer, family_batches: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    *, config: TrainingConfig, step: bool = True, clip: bool = True,
) -> dict[str, Any]:
    """One M9.0 interleaved optimizer step (protocol Section 4): zero_grad,
    then one family-pure micro-batch forward+backward per family IN THE
    ORDER `family_batches` iterates (no zero_grad between families, so
    gradients accumulate across all of them onto the SAME model
    parameters), then clip+step. Returns per-family losses and the
    post-clip gradient norm; does not touch the scheduler or global-step
    counter (left to the caller, which also owns those across validation/
    checkpoint bookkeeping) -- extracted as its own function so the exact
    accumulation mechanics are independently unit-testable
    (tests/scientific/test_interleaved_topology_m9_0.py). `step=False`
    (test-only; every production call site uses the default) skips the
    final `optimizer.step()` so a test can inspect the accumulated
    `.grad` tensors before the weights actually move. `clip=False`
    (test-only) skips `clip_grad_norm_`, whose rescale factor depends on
    the TOTAL gradient norm across every family passed in -- a 1-family
    call and a 3-family call clip by different factors even when the
    underlying per-family gradient contributions are identical, so a test
    comparing "combined accumulation" against "sum of independent
    per-family gradients" must disable it to isolate pure accumulation
    correctness from this unrelated, expected nonlinearity."""

    optimizer.zero_grad(set_to_none=True)
    family_losses: dict[str, float] = {}
    family_batch_sizes: dict[str, int] = {}
    num_families = len(family_batches)
    for family, (inputs, targets) in family_batches.items():
        output = model(_to_cpu_float(inputs))
        result = compute_multitask_loss(output, targets, task_weights=config.task_weights, profile_ordinal_weight=config.profile_ordinal_weight)
        if not torch.isfinite(result.total):
            raise FloatingPointError(f"non-finite multitask loss (family={family})")
        (result.total / num_families).backward()
        family_losses[family] = float(result.total.detach())
        family_batch_sizes[family] = int(next(iter(targets.values())).shape[0])
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)) if clip else float("nan")
    if step:
        optimizer.step()
    return {"family_losses": family_losses, "family_batch_sizes": family_batch_sizes, "gradient_norm": gradient_norm}


@torch.no_grad()
def _validate_family(model: HydroCore, dataset: CausalPrefixDatasetView, *, base_seed: int, epoch: int, family_index: int, config: TrainingConfig) -> float:
    model.eval()
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_scenarios, generator=torch.Generator().manual_seed(base_seed + epoch + family_index * FAMILY_SEED_OFFSET),
    )
    losses = []
    for inputs, targets in loader:
        output = model(_to_cpu_float(inputs))
        result = compute_multitask_loss(output, targets, task_weights=config.task_weights, profile_ordinal_weight=config.profile_ordinal_weight)
        losses.append(float(result.total))
    return sum(losses) / max(len(losses), 1)


def train_arm_b(seed: int) -> dict[str, Any]:
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
    #: Static scheduler total_steps estimate mirrors Trainer.__init__'s own
    #: formula (ceil(N/batch_size/accum) * epochs), with N = one family's
    #: own TRAIN_PER_FAMILY and accum=1 (one micro-batch per family per
    #: step) -- see M9.0 protocol Section 6 for the full accounting.
    optimizer_steps_estimate = math.ceil(TRAIN_PER_FAMILY / config.batch_size / 1) * config.epochs
    scheduler = _scheduler(optimizer, config, max(optimizer_steps_estimate, 1))

    run_root = RUN_ROOT / f"ARM_B-seed{seed}"
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
    started = time.monotonic()

    try:
        for epoch in range(config.epochs):
            stage = curriculum.stage_for_epoch(epoch)
            epoch_datasets = {family: view.stages_through(stage) for family, view in train_views.items()}
            for family in FAMILY_NAMES:
                assert epoch_datasets[family]._unobserved_age_sentinel == "fixed"
                assert epoch_datasets[family]._include_relative_gap_feature is False

            loaders = [
                _family_loader(epoch_datasets[family], base_seed=config.seed, epoch=epoch, family_index=index)
                for index, family in enumerate(FAMILY_NAMES)
            ]
            model.train()
            epoch_loss_total = 0.0
            epoch_steps = 0
            for batch_triplet in zip(*loaders, strict=False):
                family_batches = dict(zip(FAMILY_NAMES, batch_triplet, strict=True))
                step_result = interleaved_optimizer_step(model, optimizer, family_batches, config=config)
                scheduler.step()
                global_step += 1
                epoch_steps += 1
                for family, batch_size in step_result["family_batch_sizes"].items():
                    family_exposure_counts[family] += batch_size
                step_total = sum(step_result["family_losses"].values())
                epoch_loss_total += step_total
                artifacts.append_metric({
                    "epoch": epoch, "step": epoch_steps, "global_step": global_step,
                    "family_losses": step_result["family_losses"], "step_total_loss": step_total,
                    "gradient_norm": step_result["gradient_norm"], "learning_rate": optimizer.param_groups[0]["lr"],
                })
            train_loss = epoch_loss_total / max(epoch_steps, 1)

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
        "purpose": "Milestone 9.0: INTERLEAVED_MULTI_FAMILY (Arm B) training run under the frozen M9.0 protocol.",
        "arm": "ARM_B_INTERLEAVED_MULTI_FAMILY",
        "seed": seed,
        "feature_kwargs": FEATURE_KWARGS,
        "trained_families": list(FAMILY_NAMES),
        "family_weighting": {family: 1.0 / NUM_FAMILIES for family in FAMILY_NAMES},
        "model_architecture": {"variant": "small", "use_adapters": False, **SHARED_MODEL_CONFIG, "param_count": param_count},
        "training_config": asdict(config),
        "training_config_source": str(CONFIG_PATH.relative_to(ROOT)),
        "accumulation_window_micro_batches_per_step": NUM_FAMILIES,
        "optimizer_steps_estimate_static": optimizer_steps_estimate,
        "family_exposure_counts": family_exposure_counts,
        "train_scenario_count_per_family": TRAIN_PER_FAMILY,
        "total_train_scenario_count": TRAIN_PER_FAMILY * NUM_FAMILIES,
        "validation_scenario_count_per_family": dict(zip(FAMILY_NAMES, VALIDATION_COUNTS, strict=True)),
        "calibration_scenario_count_per_family": CALIBRATION_PER_FAMILY,
        "train_manifest_hash_per_family": {family: view.manifest_hash for family, view in train_views.items()},
        "validation_manifest_hash_per_family": {family: view.manifest_hash for family, view in validation_views.items()},
        "signature_library_manifest_hash_per_family": {family: library.manifest_hash for family, library in libraries.items()},
        "combined_manifest_hash": _combined_manifest_hash(train_views),
        "wall_seconds": wall_seconds,
        "training_summary": training_summary,
        "locked_test_opened_after": locked_test_opened(ROOT),
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SUMMARY_ROOT / f"ARM_B_INTERLEAVED_MULTI_FAMILY-seed{seed}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return record


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    record = train_arm_b(args.seed)
    print(json.dumps({
        "arm": record["arm"], "seed": record["seed"], "wall_seconds": record["wall_seconds"],
        "param_count": record["model_architecture"]["param_count"],
        "checkpoint_sha256": record["training_summary"]["export_sha256"],
        "family_exposure_counts": record["family_exposure_counts"],
        "summary": record["training_summary"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
