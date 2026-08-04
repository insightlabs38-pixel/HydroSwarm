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


PROFILE_CLASS_COUNTS = {
    "start_time": 4,
    "duration": 3,
    "relative_strength": 3,
}

#: overnight-plan.txt Task 4.5: "Loss weights are explicit" and
#: "Auxiliary predictions are not shown as authoritative product
#: outputs" -- these three tasks are self-supervised/forecasting signal
#: intended to shape the shared backbone representation, not to compete
#: with the primary scientific objectives on equal footing. Applied only
#: as a *default*; task_weights always takes precedence, so a caller that
#: wants equal weighting (e.g. an ablation explicitly studying auxiliary
#: objective strength) can still request it explicitly.
AUXILIARY_TASKS = frozenset({"sensor_reconstruction", "future_concentration", "travel_time"})
AUXILIARY_TASK_DEFAULT_WEIGHT = 0.1


def _cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    flattened_target = target.long().reshape(-1)
    valid = flattened_target != -100
    if not valid.any():
        return logits.sum() * 0.0
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    return F.cross_entropy(flattened_logits[valid], flattened_target[valid])


def _ordinal_classification_loss(
    logits: Tensor,
    target: Tensor,
    *,
    class_count: int,
    ordinal_weight: float,
) -> Tensor:
    """Combine categorical fit with distance-aware supervision over ordered bins."""

    usable_logits = logits[..., :class_count]
    flattened_target = target.long().reshape(-1)
    valid = (flattened_target != -100) & (flattened_target >= 0) & (
        flattened_target < class_count
    )
    if not valid.any():
        return usable_logits.sum() * 0.0
    flattened_logits = usable_logits.reshape(-1, class_count)[valid]
    valid_target = flattened_target[valid]
    categorical = F.cross_entropy(flattened_logits, valid_target)
    if ordinal_weight == 0:
        return categorical
    positions = torch.linspace(
        0.0, 1.0, class_count, device=flattened_logits.device, dtype=flattened_logits.dtype
    )
    expected_position = (torch.softmax(flattened_logits, dim=-1) * positions).sum(dim=-1)
    target_position = valid_target.to(flattened_logits.dtype) / max(class_count - 1, 1)
    ordinal = F.smooth_l1_loss(expected_position, target_position)
    return categorical + ordinal_weight * ordinal


def _apply_target_mask(task: str, target: Tensor, targets: Mapping[str, Tensor]) -> Tensor:
    """Fold a targets_v2 ``f"{task}_mask"`` companion (True = valid/observed,
    per hydroswarm.training.targets_v2's documented convention) into the
    ``-100`` "ignore this position" sentinel _cross_entropy and
    _ordinal_classification_loss already understand.

    Without this, a masked-out placeholder value (e.g. corpus.py sets
    source_node/duration/etc. to 0 for a NORMAL/SENSOR_FAULT_ONLY scenario
    where no real source or timing exists, precisely so it is *not* mistaken
    for a real label) would otherwise be trained against directly, since
    corpus.py's own values never contain -100 -- only the separate mask
    tensor records that a placeholder was used.
    """

    mask = targets.get(f"{task}_mask")
    if mask is None:
        return target
    return target.masked_fill(~mask.bool(), -100)


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
    profile_ordinal_weight: float = 0.0,
) -> MultiTaskLoss:
    """Compute every task for which both a semantic prediction and target exist."""

    weights = task_weights or {}
    losses: dict[str, Tensor] = {}
    classifications = {
        "source_node": "source_node_logits",
        "source_region": "source_region_logits",
        "sample_node": "sample_node_logits",
        "action": "action_logits",
        "action_pointer": "action_pointer_logits",
        "plan_validity": "plan_validity_logits",
        "ood": "ood_logits",
        # overnight-plan.txt Task 4.4: only present in outputs when the
        # model was built with event_control_heads=True, so these are
        # silently skipped (like every other entry here) for models built
        # without them.
        "event_cause": "event_cause_logits",
        "next_step": "next_step_logits",
    }
    regressions = {
        "plan_value": "plan_value",
        "information_gain": "expected_information_gain",
        "residual": "residual_prediction",
        "sensor_reconstruction": "sensor_reconstruction_prediction",
        "future_concentration": "future_concentration_prediction",
        "pressure": "pressure_prediction",
        "flow": "flow_prediction",
        "travel_time": "travel_time_prediction",
        # overnight-plan.txt Task 4.6: only present in outputs when the
        # model was built with consequence_prescreening_heads=True.
        # Output name equals target name here (same convention as
        # plan_value above), not a *_prediction suffix.
        "exposure_proxy": "exposure_proxy",
        "pressure_risk_proxy": "pressure_risk_proxy",
        "service_loss_proxy": "service_loss_proxy",
        "containment_time_proxy": "containment_time_proxy",
        "plan_regret_proxy": "plan_regret_proxy",
    }
    for task, output_name in classifications.items():
        if task in targets and output_name in outputs:
            losses[task] = _cross_entropy(outputs[output_name], _apply_target_mask(task, targets[task], targets))
    for task, class_count in PROFILE_CLASS_COUNTS.items():
        output_name = f"{task}_logits"
        if task in targets and output_name in outputs:
            losses[task] = _ordinal_classification_loss(
                outputs[output_name],
                _apply_target_mask(task, targets[task], targets),
                class_count=class_count,
                ordinal_weight=profile_ordinal_weight,
            )
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
    if "event_presence" in targets and "event_presence_logits" in outputs:
        presence_target = targets["event_presence"].float()
        valid = torch.isfinite(presence_target) & (presence_target >= 0)
        losses["event_presence"] = (
            F.binary_cross_entropy_with_logits(
                outputs["event_presence_logits"].float()[valid], presence_target[valid]
            )
            if valid.any()
            else outputs["event_presence_logits"].sum() * 0.0
        )
    if "evidence_sufficiency" in targets and "evidence_sufficiency" in outputs:
        losses["evidence_sufficiency"] = F.binary_cross_entropy(
            outputs["evidence_sufficiency"].float(), targets["evidence_sufficiency"].float()
        )
    if not losses:
        raise ValueError("no compatible model outputs and training targets")

    def _default_weight(name: str) -> float:
        return AUXILIARY_TASK_DEFAULT_WEIGHT if name in AUXILIARY_TASKS else 1.0

    weighted = [loss * float(weights.get(name, _default_weight(name))) for name, loss in losses.items()]
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
