"""Safe model weights plus exact optimizer/scheduler resume state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import nn


def save_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(tensors, path / "model.safetensors")
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        path / "optimizer_state.pt",
    )
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_validation_loss": best_validation_loss,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def load_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    path = Path(directory)
    model.load_state_dict(load_file(path / "model.safetensors", device="cpu"), strict=True)
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if optimizer is not None or scheduler is not None:
        resume = torch.load(path / "optimizer_state.pt", map_location="cpu", weights_only=True)
        if optimizer is not None:
            optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resume["scheduler"])
    return state


def export_model(model: nn.Module, path: str | Path, *, metadata: dict[str, str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(tensors, target, metadata=metadata)
    return target

