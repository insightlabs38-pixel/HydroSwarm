"""Safe model weights plus exact optimizer/scheduler resume state.

This is the legacy v3 checkpoint format/loader (no architecture identity
beyond what the caller separately records, e.g. in a `.metadata.json`
sidecar -- see hydroswarm.runtime.defaults.DefaultPipelineFactory). It
stays exactly as-is for existing v3 checkpoints (core-issues4.txt's
PRIMARY DESIGN RULE: preserve legacy v3 loading behavior unchanged). See
hydroswarm.training.checkpoint_identity for the new, separate v4 format
(save_v4_checkpoint/load_v4_checkpoint), which writes an additional
checkpoint_identity.json a v3-format directory never has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
from torch import nn

#: hydroswarm.training.checkpoint_identity.IDENTITY_FILENAME, duplicated as
#: a plain string constant (not imported) so this leaf module does not
#: depend on the newer v4 module -- only the filename itself needs to be
#: known here, purely to refuse loading a v4 checkpoint through this path.
_V4_IDENTITY_FILENAME = "checkpoint_identity.json"


class LegacyLoaderRejectedV4CheckpointError(Exception):
    """Raised by load_checkpoint when `directory` is a v4 checkpoint
    (identifiable by the presence of checkpoint_identity.json) --
    core-issues4.txt's explicit requirement that a v4 checkpoint must never
    silently load through the legacy v3 path, which understands none of
    its architecture-identity/output-governance guarantees."""


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
    if (path / _V4_IDENTITY_FILENAME).exists():
        raise LegacyLoaderRejectedV4CheckpointError(
            f"{path} contains {_V4_IDENTITY_FILENAME} -- this is a v4 checkpoint; use "
            "hydroswarm.training.checkpoint_identity.load_v4_checkpoint instead of the legacy "
            "v3 loader"
        )
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

