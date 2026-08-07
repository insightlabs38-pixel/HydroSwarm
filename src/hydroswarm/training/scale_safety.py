"""core-issues3.txt Phase 11.3: scale-safety preflight check.

Before a real/promotion-quality training run, run ONE real multi-topology
batch through a real forward+backward pass and assert every governed
property in one place, rather than each training script re-deriving its
own partial version of this check (scripts/run_event_control_smoke_
screening.py's `_gradient_check` and scripts/run_architecture_smoke_jobs.py
each already implement a narrower ad-hoc version of this same idea --
exactly the "two things that must agree independently drift apart" defect
class this project has hit three times already, see
reports/results/v4/pre-freeze-implementation-handoff.md's Phase 10.2
section). This module is the one place that check should live; existing
callers are free to keep their own bespoke logic, but new callers should
prefer this.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .losses import compute_multitask_loss, task_gradient_norms


class ScaleSafetyError(Exception):
    """Raised by run_scale_safety_check when the preflight batch fails any
    governed property. Carries every failure found (not just the first),
    so a caller sees the complete picture in one run rather than
    fixing-and-rerunning failure by failure."""


@dataclass(frozen=True, slots=True)
class ScaleSafetyReport:
    tasks_checked: tuple[str, ...]
    valid_counts: Mapping[str, int]
    gradient_norms: Mapping[str, float]
    total_loss: float


def run_scale_safety_check(
    model: nn.Module,
    inputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    required_tasks: frozenset[str] = frozenset(),
    task_weights: Mapping[str, float] | None = None,
    profile_ordinal_weight: float = 0.0,
) -> ScaleSafetyReport:
    """Run one real forward+backward pass and assert (core-issues3.txt
    Phase 11.3's exact list):

    - every task present in `targets` (with a matching model output) is
      reached by compute_multitask_loss ("every present retained task
      reaches the loss");
    - every name in `required_tasks` has a positive valid count ("every
      retained task has a positive valid count" -- scoped to the tasks
      THIS check batch is meant to actually exercise, not literally every
      task compute_multitask_loss knows about: a structurally-disabled
      target like future_concentration (core-issues3.txt Phase 7.4 --
      always all-masked) would make an unscoped version of this check
      impossible to ever pass on a real full-multitask batch);
    - every task with a positive valid count receives a real, finite,
      nonzero gradient ("every retained task produces nonzero gradient" /
      "no NaN/Inf") -- computed via task_gradient_norms's own per-task
      `torch.autograd.grad(..., retain_graph=True)` calls, NOT via a
      preceding `result.total.backward()` (which would free the graph
      before task_gradient_norms's later per-task calls could run);
    - every task with a ZERO valid count receives EXACTLY zero gradient
      ("padded positions contribute zero" / "masked positions contribute
      zero" -- masked_regression's/the classification masking's fallback
      path is `prediction.sum() * 0.0`, a graph-connected zero; this
      confirms that fallback is still actually gradient-inert, not merely
      loss-value-inert, for whichever tasks this particular batch happens
      not to exercise -- including permanently-disabled ones).

    "No accidental broadcasting" is enforced structurally by
    compute_multitask_loss/masked_regression itself (raises ValueError on
    a prediction/target shape mismatch rather than silently broadcasting)
    -- this function reaching its return means that check already passed.

    Raises ScaleSafetyError listing every failure found (not just the
    first). Does not call `.backward()` or populate `.grad` on `model`'s
    parameters -- gradients are computed via task_gradient_norms's own
    `torch.autograd.grad` calls, which read the graph without consuming it
    for a real optimizer step. Call `.backward()`/`.step()` separately if
    this check is meant to precede (not replace) a real training step on
    the same batch.
    """

    model.train()
    output = model(inputs)
    result = compute_multitask_loss(
        output, targets, task_weights=task_weights, profile_ordinal_weight=profile_ordinal_weight
    )
    failures: list[str] = []
    if not torch.isfinite(result.total):
        failures.append("total loss is not finite (NaN/Inf)")
    for name, loss in result.tasks.items():
        if not torch.isfinite(loss):
            failures.append(f"{name}: loss is not finite (NaN/Inf)")
    missing_required = required_tasks - set(result.tasks)
    if missing_required:
        failures.append(
            f"required_tasks {sorted(missing_required)} never reached compute_multitask_loss at all "
            "-- a head-gating or loss-key-naming defect, or the wrong model/batch was used"
        )
    zero_coverage_required = {
        name for name in required_tasks & set(result.valid_counts) if result.valid_counts[name] <= 0
    }
    if zero_coverage_required:
        failures.append(
            f"required_tasks {sorted(zero_coverage_required)} reached compute_multitask_loss but have "
            "valid_count <= 0 in this batch -- construct a preflight batch that actually exercises them"
        )
    if failures:
        # Fail before backward(): a batch that already fails the
        # loss-level checks should not also pay for/report on a backward
        # pass whose numbers cannot be trusted anyway.
        raise ScaleSafetyError("scale-safety preflight failed:\n" + "\n".join(f"  - {item}" for item in failures))

    norms = task_gradient_norms(result.tasks, model)
    for name, norm in norms.items():
        if not math.isfinite(norm):
            failures.append(f"{name}: gradient norm is not finite (NaN/Inf)")
            continue
        valid_count = result.valid_counts.get(name, 0)
        if valid_count > 0 and norm == 0.0:
            failures.append(
                f"{name}: valid_count={valid_count} but received exactly zero gradient -- "
                "this task is not actually learning from this batch"
            )
        if valid_count == 0 and norm != 0.0:
            failures.append(
                f"{name}: valid_count=0 (fully masked/padded in this batch) but received a "
                f"nonzero gradient ({norm}) -- a masked/padded position is leaking into training"
            )
    if failures:
        raise ScaleSafetyError("scale-safety preflight failed:\n" + "\n".join(f"  - {item}" for item in failures))

    return ScaleSafetyReport(
        tasks_checked=tuple(sorted(result.tasks)),
        valid_counts=dict(result.valid_counts),
        gradient_norms=norms,
        total_loss=float(result.total.detach()),
    )
