"""Native multitask scientific losses and gradient diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class MultiTaskLoss:
    total: Tensor
    tasks: Mapping[str, Tensor]


def _cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    flattened_target = target.long().reshape(-1)
    valid = flattened_target != -100
    if not valid.any():
        return logits.sum() * 0.0
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    return F.cross_entropy(flattened_logits[valid], flattened_target[valid])


def _masked_mse(prediction: Tensor, target: Tensor) -> Tensor:
    target = target.float()
    prediction = prediction.float()
    valid = torch.isfinite(target)
    if not valid.any():
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[valid], target[valid])


def compute_multitask_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    task_weights: Mapping[str, float] | None = None,
) -> MultiTaskLoss:
    """Compute every task for which both a semantic prediction and target exist."""

    weights = task_weights or {}
    losses: dict[str, Tensor] = {}
    classifications = {
        "source_node": "source_node_logits",
        "start_time": "start_time_logits",
        "duration": "duration_logits",
        "relative_strength": "relative_strength_logits",
        "sample_node": "sample_node_logits",
        "action": "action_logits",
        "action_pointer": "action_pointer_logits",
        "plan_validity": "plan_validity_logits",
        "ood": "ood_logits",
    }
    regressions = {
        "plan_value": "plan_value",
        "information_gain": "expected_information_gain",
        "residual": "residual_prediction",
        "reconstruction": "reconstruction_prediction",
        "future_concentration": "future_concentration_prediction",
        "pressure": "pressure_prediction",
        "flow": "flow_prediction",
        "travel_time": "travel_time_prediction",
    }
    for task, output_name in classifications.items():
        if task in targets and output_name in outputs:
            losses[task] = _cross_entropy(outputs[output_name], targets[task])
    for task, output_name in regressions.items():
        if task in targets and output_name in outputs:
            losses[task] = _masked_mse(outputs[output_name], targets[task])
    if "sensor_fault" in targets and "sensor_fault_logits" in outputs:
        fault_target = targets["sensor_fault"].float()
        valid = torch.isfinite(fault_target) & (fault_target >= 0)
        losses["sensor_fault"] = (
            F.binary_cross_entropy_with_logits(
                outputs["sensor_fault_logits"].float()[valid], fault_target[valid]
            )
            if valid.any()
            else outputs["sensor_fault_logits"].sum() * 0.0
        )
    if "evidence_sufficiency" in targets and "evidence_sufficiency" in outputs:
        losses["evidence_sufficiency"] = F.binary_cross_entropy(
            outputs["evidence_sufficiency"].float(), targets["evidence_sufficiency"].float()
        )
    if not losses:
        raise ValueError("no compatible model outputs and training targets")
    weighted = [loss * float(weights.get(name, 1.0)) for name, loss in losses.items()]
    return MultiTaskLoss(total=torch.stack(weighted).sum(), tasks=losses)


def task_gradient_norms(
    task_losses: Mapping[str, Tensor], model: nn.Module
) -> dict[str, float]:
    """Log per-task norms used to diagnose GradNorm-style imbalance."""

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    norms: dict[str, float] = {}
    for name, loss in task_losses.items():
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        squared = sum(
            float(gradient.detach().float().pow(2).sum())
            for gradient in gradients
            if gradient is not None
        )
        norms[name] = squared**0.5
    return norms
