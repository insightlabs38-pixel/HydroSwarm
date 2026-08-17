"""Reusable, promotion-quality training-coverage validator (M10.2 Scout
supervision/representation refit amendment, Part 2).

The M10.2 preflight's own root-cause finding (`docs/evaluation/
HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`) was that a nonzero
`task_weights` entry does not imply a task was ever actually supervised --
`compute_multitask_loss`'s `if task in targets and output_name in outputs`
guard silently skips a task with no real target, and nothing anywhere
checked that a "trained" task's gradient genuinely reached the parameters it
was supposed to. This module is the reusable, fail-closed check that gap
needed: given one real (`requires_grad`-connected) forward pass and a
task-weight/parameter-group declaration, it produces one
`TaskGradientCoverage` certificate per intended task, verifying -- not
assuming -- every link in the chain from "task is configured" to "the
intended parameters actually moved."

Reusable beyond Scout: any future OOD or Strategist supervision amendment
(`docs/evaluation/HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`'s own
"Reusability" section) should call this same module rather than re-deriving
ad hoc gradient-presence checks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .losses import compute_multitask_loss
from .task_output_names import TASK_OUTPUT_NAMES

ForwardFn = Callable[[nn.Module], Mapping[str, Tensor]]


@dataclass(frozen=True, slots=True)
class TaskGradientCoverage:
    task: str
    output_name: str | None
    task_enabled: bool
    target_present: bool
    output_present: bool
    valid_target_count: int
    loss_present: bool
    loss_weight: float | None
    loss_finite: bool
    trainable_parameter_group: tuple[str, ...]
    gradient_observed: bool
    gradient_norm: float | None
    gradient_norm_finite: bool
    parameter_changed: bool | None
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "output_name": self.output_name,
            "task_enabled": self.task_enabled,
            "target_present": self.target_present,
            "output_present": self.output_present,
            "valid_target_count": self.valid_target_count,
            "loss_present": self.loss_present,
            "loss_weight": self.loss_weight,
            "loss_finite": self.loss_finite,
            "trainable_parameter_group": list(self.trainable_parameter_group),
            "gradient_observed": self.gradient_observed,
            "gradient_norm": self.gradient_norm,
            "gradient_norm_finite": self.gradient_norm_finite,
            "parameter_changed": self.parameter_changed,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


class GradientCoverageError(Exception):
    """Raised by `require_gradient_coverage` when any intended task's
    certificate did not pass."""


def _grad_norm(parameters: Sequence[nn.Parameter]) -> tuple[float | None, bool]:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None, False
    total = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads))
    return float(total), bool(torch.isfinite(total))


def _verify_parameter_updates(
    model: nn.Module,
    forward_fn: ForwardFn,
    targets: Mapping[str, Tensor],
    *,
    task_weights: Mapping[str, float],
    parameter_groups: Mapping[str, Sequence[str]],
    update_lr: float,
) -> dict[str, bool]:
    """Deep-copies `model` so the caller's real model/optimizer state is
    never mutated by this diagnostic. Runs ONE real forward -> loss ->
    backward -> `optimizer.step()` on the copy and compares every allowlisted
    parameter's tensor before/after -- task requirement: "parameter actually
    changes after an optimizer step in a controlled test"."""

    probe = copy.deepcopy(model)
    probe.train()
    all_names = sorted({name for names in parameter_groups.values() for name in names})
    named = dict(probe.named_parameters())
    trainable = [named[name] for name in all_names if name in named]
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    before = {name: named[name].detach().clone() for name in all_names if name in named}

    optimizer = torch.optim.SGD(trainable, lr=update_lr) if trainable else None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    outputs = forward_fn(probe)
    loss_result = compute_multitask_loss(outputs, targets, task_weights=task_weights)
    loss_result.total.backward()
    if optimizer is not None:
        optimizer.step()

    after = {name: named[name].detach() for name in all_names if name in named}
    changed_by_name = {name: not torch.equal(before[name], after[name]) for name in all_names if name in before}

    result: dict[str, bool] = {}
    for task, names in parameter_groups.items():
        real_names = [name for name in names if name in changed_by_name]
        # ANY, not ALL: a group can legitimately contain a parameter with an
        # exactly-zero gradient by mathematical necessity (e.g. a softmax
        # classification head's output bias is shift-invariant -- adding a
        # constant to every class's logit does not change softmax, so its
        # gradient is exactly zero every step regardless of how real the
        # rest of the head's training is). Requiring every name to move
        # would fail a genuinely, correctly trained head.
        result[task] = bool(real_names) and any(changed_by_name[name] for name in real_names)
    return result


def compute_gradient_coverage(
    model: nn.Module,
    forward_fn: ForwardFn,
    targets: Mapping[str, Tensor],
    *,
    task_weights: Mapping[str, float],
    parameter_groups: Mapping[str, Sequence[str]],
    min_valid_target_count: int = 1,
    verify_parameter_update: bool = True,
    update_lr: float = 0.1,
) -> dict[str, TaskGradientCoverage]:
    """Builds one `TaskGradientCoverage` certificate per key in
    `task_weights`. `task_weights[task] == 0.0` marks a task as
    intentionally NOT intended this run (e.g. an ablation) -- such a task is
    recorded (`task_enabled=False`) but never fails the certificate
    regardless of target/gradient state. Every other task in `task_weights`
    is intended and must pass every check to receive `passed=True`.

    `parameter_groups[task]` is the exact, named-parameter allowlist that
    task's loss term is expected to reach -- e.g.
    `{"sample_node": ["sample_node_head.mlp.0.weight", ...]}`. A task with
    no entry (or an empty one) in `parameter_groups` is recorded with an
    empty `trainable_parameter_group` and fails closed if `task_enabled`
    (there is no way to prove "gradient reached the intended parameters" if
    no parameters were named as intended).
    """

    model.zero_grad(set_to_none=True)
    outputs = forward_fn(model)
    loss_result = compute_multitask_loss(outputs, targets, task_weights=task_weights)
    loss_result.total.backward()

    named_params = dict(model.named_parameters())
    updated = (
        _verify_parameter_updates(
            model, forward_fn, targets, task_weights=task_weights, parameter_groups=parameter_groups, update_lr=update_lr
        )
        if verify_parameter_update
        else {}
    )

    certificates: dict[str, TaskGradientCoverage] = {}
    for task, weight in task_weights.items():
        enabled = weight != 0.0
        output_name = TASK_OUTPUT_NAMES.get(task)
        output_present = output_name is not None and output_name in outputs
        target_present = task in targets
        valid_count = int(loss_result.valid_counts.get(task, 0))
        loss_present = task in loss_result.tasks
        loss_weight = loss_result.weights.get(task)
        loss_finite = bool(torch.isfinite(loss_result.tasks[task])) if loss_present else False
        group_names = tuple(parameter_groups.get(task, ()))
        group_params = [named_params[name] for name in group_names if name in named_params]
        missing_names = [name for name in group_names if name not in named_params]
        grad_norm, grad_finite = _grad_norm(group_params)
        gradient_observed = grad_norm is not None and grad_norm > 0.0
        parameter_changed = updated.get(task) if verify_parameter_update else None

        failures: list[str] = []
        if enabled:
            if not target_present:
                failures.append("task enabled but no target present in this batch")
            if not output_present:
                failures.append("task enabled but model output not present")
            if valid_count < min_valid_target_count:
                failures.append(f"valid_target_count={valid_count} below required minimum {min_valid_target_count}")
            if not loss_present:
                failures.append("no loss term for this task in MultiTaskLoss.tasks")
            elif not loss_finite:
                failures.append("loss term is not finite")
            if not group_names:
                failures.append("no trainable parameter group declared for this enabled task")
            if missing_names:
                failures.append(f"parameter group references unknown parameter name(s): {missing_names}")
            if not gradient_observed:
                failures.append("no nonzero gradient reached the intended parameter group")
            elif not grad_finite:
                failures.append("gradient norm is not finite")
            if verify_parameter_update and parameter_changed is False:
                failures.append("parameter values did not change after a controlled optimizer step")

        certificates[task] = TaskGradientCoverage(
            task=task,
            output_name=output_name,
            task_enabled=enabled,
            target_present=target_present,
            output_present=output_present,
            valid_target_count=valid_count,
            loss_present=loss_present,
            loss_weight=loss_weight,
            loss_finite=loss_finite,
            trainable_parameter_group=group_names,
            gradient_observed=gradient_observed,
            gradient_norm=grad_norm,
            gradient_norm_finite=grad_finite,
            parameter_changed=parameter_changed,
            passed=(not enabled) or (not failures),
            failure_reasons=tuple(failures),
        )
    return certificates


def require_gradient_coverage(certificates: Mapping[str, TaskGradientCoverage]) -> None:
    """Fail-closed entry point: raise if any INTENDED (task_enabled=True)
    task's certificate did not pass."""

    failed = {task: cert for task, cert in certificates.items() if cert.task_enabled and not cert.passed}
    if failed:
        details = "; ".join(f"{task}: {cert.failure_reasons}" for task, cert in sorted(failed.items()))
        raise GradientCoverageError(f"gradient coverage failed for {sorted(failed)}: {details}")
