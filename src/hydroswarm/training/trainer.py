"""Bounded, deterministic FP32 CPU trainer with exact resume semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import signal
import time
import traceback
from typing import Mapping

import numpy as np
from safetensors.torch import load_file
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .artifacts import RunArtifacts, atomic_json
from .checkpoint import export_model, load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import (
    CurriculumSchedule,
    GovernedScenarioDataset,
    collate_scenarios,
)
from .losses import compute_multitask_loss, task_gradient_norms


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    run_directory: str
    epochs_completed: int
    global_steps: int
    best_validation_loss: float
    best_epoch: int
    stopped_early: bool
    stop_reason: str
    final_checkpoint: str
    export_path: str


def set_deterministic_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max((step + 1) / config.warmup_steps, 1e-8)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        if config.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if config.scheduler == "linear":
            return 1.0 - progress
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: GovernedScenarioDataset,
        *,
        config: TrainingConfig,
        run_root: str | Path,
        validation_dataset: GovernedScenarioDataset | None = None,
        curriculum: CurriculumSchedule | None = None,
        workdir: str | Path = ".",
    ) -> None:
        if config.pcgrad_enabled and config.gradient_accumulation_steps != 1:
            raise ValueError("PCGrad requires gradient_accumulation_steps=1")
        self.model = model.cpu().float()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.curriculum = curriculum or CurriculumSchedule.progressive()
        self.artifacts = RunArtifacts.create(
            run_root,
            config=config.as_dict(),
            manifest_hash=train_dataset.manifest_hash,
            workdir=workdir,
        )
        set_deterministic_seed(config.seed, deterministic=config.deterministic)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        optimizer_steps = math.ceil(
            len(train_dataset)
            / config.batch_size
            / config.gradient_accumulation_steps
        ) * config.epochs
        self.scheduler = _scheduler(self.optimizer, config, max(optimizer_steps, 1))
        self.global_step = 0
        self._shutdown_requested = False

    def install_signal_handlers(self) -> None:
        """Opt-in graceful-shutdown support for long unattended jobs (Task 0.3).

        A SIGTERM (e.g. from a job supervisor or `systemctl stop`) sets a flag
        that is observed at the same point the runtime budget is checked, so
        the run stops after the current batch, saves a resumable checkpoint,
        and exits through the ordinary `stop_reason="runtime_budget"` path
        instead of being killed mid-write. Must be called from the main
        thread; not installed automatically so tests and multi-Trainer
        processes are unaffected unless they opt in.
        """
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        del signum, frame
        self._shutdown_requested = True

    def _loader(
        self, dataset: GovernedScenarioDataset, *, epoch: int, shuffle: bool
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.seed + epoch)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_scenarios,
            generator=generator,
            worker_init_fn=_worker_seed if self.config.num_workers else None,
            persistent_workers=bool(self.config.num_workers),
        )

    @staticmethod
    def _to_cpu_float(inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {
            key: value.float() if value.is_floating_point() else value
            for key, value in inputs.items()
        }

    def _pcgrad_backward(self, task_losses: Mapping[str, Tensor]) -> None:
        parameters = tuple(parameter for parameter in self.model.parameters() if parameter.requires_grad)
        gradients = [
            torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            for loss in task_losses.values()
        ]
        projected = [list(task) for task in gradients]
        for first in range(len(projected)):
            for second in range(len(gradients)):
                if first == second:
                    continue
                dot = sum(
                    torch.sum(left * right)
                    for left, right in zip(projected[first], gradients[second])
                    if left is not None and right is not None
                )
                norm = sum(
                    torch.sum(right * right)
                    for right in gradients[second]
                    if right is not None
                )
                if dot < 0 and norm > 0:
                    coefficient = dot / norm
                    projected[first] = [
                        left - coefficient * right
                        if left is not None and right is not None
                        else left
                        for left, right in zip(projected[first], gradients[second])
                    ]
        for index, parameter in enumerate(parameters):
            values = [task[index] for task in projected if task[index] is not None]
            parameter.grad = None if not values else torch.stack(values).mean(dim=0)

    def _train_epoch(
        self, dataset: GovernedScenarioDataset, *, epoch: int, started: float
    ) -> float:
        self.model.train()
        loader = self._loader(dataset, epoch=epoch, shuffle=True)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        for batch_index, (inputs, targets) in enumerate(loader):
            if time.monotonic() - started > self.config.maximum_runtime_seconds:
                raise TimeoutError("maximum training runtime exceeded")
            if self._shutdown_requested:
                raise TimeoutError("graceful shutdown requested via SIGTERM")
            self.artifacts.check_disk(self.config.minimum_free_disk_gb)
            output = self.model(self._to_cpu_float(inputs))
            result = compute_multitask_loss(
                output,
                targets,
                task_weights=self.config.task_weights,
                profile_ordinal_weight=self.config.profile_ordinal_weight,
            )
            if not torch.isfinite(result.total):
                raise FloatingPointError("non-finite multitask loss")
            gradnorm = (
                task_gradient_norms(result.tasks, self.model)
                if self.config.gradnorm_logging
                else {}
            )
            if self.config.pcgrad_enabled:
                self._pcgrad_backward(result.tasks)
            else:
                (result.total / self.config.gradient_accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % self.config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            gradient_norm = None
            if should_step:
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            loss_value = float(result.total.detach())
            total_loss += loss_value
            batches += 1
            self.artifacts.append_metric(
                {
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": self.global_step,
                    "loss": loss_value,
                    "task_losses": {
                        name: float(value.detach()) for name, value in result.tasks.items()
                    },
                    "task_gradient_norms": gradnorm,
                    "gradient_norm": gradient_norm,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def _validate(self, *, epoch: int) -> float:
        if self.validation_dataset is None:
            return math.nan
        self.model.eval()
        losses = []
        for inputs, targets in self._loader(
            self.validation_dataset, epoch=epoch, shuffle=False
        ):
            output = self.model(self._to_cpu_float(inputs))
            losses.append(
                float(
                    compute_multitask_loss(
                        output,
                        targets,
                        task_weights=self.config.task_weights,
                        profile_ordinal_weight=self.config.profile_ordinal_weight,
                    ).total
                )
            )
        return float(np.mean(losses))

    def fit(self, *, resume_from: str | Path | None = None) -> TrainingSummary:
        started = time.monotonic()
        start_epoch = 0
        best = math.inf
        best_epoch = -1
        best_global_step = 0
        stale_epochs = 0
        stopped_early = False
        runtime_budget_reached = False
        final_checkpoint = ""
        try:
            if resume_from is not None:
                state = load_checkpoint(
                    resume_from,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                start_epoch = int(state["epoch"]) + 1
                self.global_step = int(state["global_step"])
                best = float(state["best_validation_loss"])
                self.artifacts.status("RUNNING", resumed_from=str(resume_from))
            epochs_completed = start_epoch
            for epoch in range(start_epoch, self.config.epochs):
                stage = self.curriculum.stage_for_epoch(epoch)
                epoch_dataset = self.train_dataset.stages_through(stage)
                try:
                    train_loss = self._train_epoch(
                        epoch_dataset, epoch=epoch, started=started
                    )
                except TimeoutError:
                    runtime_budget_reached = True
                    break
                validation_loss = self._validate(epoch=epoch)
                criterion = train_loss if math.isnan(validation_loss) else validation_loss
                improved = criterion < best - self.config.minimum_delta
                if improved:
                    best = criterion
                    best_epoch = epoch
                    best_global_step = self.global_step
                    stale_epochs = 0
                    export_model(
                        self.model,
                        self.artifacts.path / "best-model.safetensors",
                        metadata={
                            "epoch": str(epoch),
                            "validation_loss": str(criterion),
                            "global_step": str(self.global_step),
                            "manifest_hash": self.train_dataset.manifest_hash,
                        },
                    )
                else:
                    stale_epochs += 1
                epochs_completed = epoch + 1
                if (epoch + 1) % self.config.checkpoint_every_epochs == 0:
                    checkpoint = self.artifacts.path / "checkpoints" / f"checkpoint-{epoch + 1:04d}"
                    final_checkpoint = str(
                        save_checkpoint(
                            checkpoint,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            global_step=self.global_step,
                            best_validation_loss=best,
                        )
                    )
                atomic_json(
                    self.artifacts.path / "epoch_summary.json",
                    {
                        "epoch": epoch,
                        "curriculum_stage": stage.name,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "best_validation_loss": best,
                    },
                )
                if (
                    self.config.early_stopping_patience
                    and stale_epochs >= self.config.early_stopping_patience
                ):
                    stopped_early = True
                    break
            if runtime_budget_reached:
                # Periodic checkpoints remain available for resuming, but none is
                # misrepresented as a clean end-of-run optimizer checkpoint.
                final_checkpoint = ""
            if not final_checkpoint and not runtime_budget_reached:
                checkpoint = self.artifacts.path / "checkpoints" / "checkpoint-final"
                final_checkpoint = str(
                    save_checkpoint(
                        checkpoint,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=max(epochs_completed - 1, 0),
                        global_step=self.global_step,
                        best_validation_loss=best,
                    )
                )
            best_model_path = self.artifacts.path / "best-model.safetensors"
            if best_model_path.exists():
                self.model.load_state_dict(load_file(best_model_path, device="cpu"), strict=True)
            export_path = export_model(
                self.model,
                self.artifacts.path / "model-export.safetensors",
                metadata={
                    "manifest_hash": self.train_dataset.manifest_hash,
                    "global_steps": str(best_global_step or self.global_step),
                },
            )
            summary = TrainingSummary(
                run_directory=str(self.artifacts.path),
                epochs_completed=epochs_completed,
                global_steps=best_global_step or self.global_step,
                best_validation_loss=best,
                best_epoch=best_epoch,
                stopped_early=stopped_early,
                stop_reason=(
                    "runtime_budget"
                    if runtime_budget_reached
                    else "validation_convergence"
                    if stopped_early
                    else "maximum_epochs"
                ),
                final_checkpoint=final_checkpoint,
                export_path=str(export_path),
            )
            atomic_json(self.artifacts.path / "summary.json", asdict(summary))
            self.artifacts.status("COMPLETED", stop_reason=summary.stop_reason)
            return summary
        except Exception as error:
            failure = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_json(self.artifacts.path / "failure.json", failure)
            self.artifacts.status("FAILED", error=failure["message"])
            raise
