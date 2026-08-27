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
    #: Mean UNWEIGHTED loss per task (core-issues3.txt Phase 11.1: "mean
    #: unweighted loss" is a required record, distinct from `total`, which
    #: is the weighted sum actually backpropagated).
    tasks: Mapping[str, Tensor]
    #: Number of positions/examples that actually contributed to each
    #: task's loss this batch (post-mask, post-finite-check) -- Phase
    #: 11.1's "valid target counts". A task present in `tasks` with
    #: `valid_counts[task] == 0` produced a real, graph-connected zero loss
    #: (so backward() still runs) but had nothing to actually learn from in
    #: this particular batch -- distinguishable from a task that reached
    #: real supervision, which a bare loss VALUE of ~0 cannot tell you.
    valid_counts: Mapping[str, int]
    #: The weight actually applied to each task this call (post
    #: `task_weights` override / auxiliary-default fallback) -- Phase
    #: 11.1's "task weights" record, resolved per-task rather than left
    #: implicit in the caller's own config dict.
    weights: Mapping[str, float]
    #: `weights[task] * tasks[task]` -- Phase 11.1's "weighted
    #: contribution", i.e. what each task actually contributed to `total`.
    weighted: Mapping[str, Tensor]


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


def _cross_entropy(logits: Tensor, target: Tensor, *, weight: Tensor | None = None) -> tuple[Tensor, int]:
    flattened_target = target.long().reshape(-1)
    valid = flattened_target != -100
    count = int(valid.sum())
    if not valid.any():
        # Zero BEFORE summing, not after: `logits` can hold a large-magnitude
        # masked-candidate sentinel (e.g. -3.4028e38) at every position for a
        # scenario with several masked-out candidates and no valid target
        # (source_node_mask all False). `logits.sum() * 0.0` sums those huge
        # values FIRST -- two or more such sentinels overflow float32's
        # representable range to +/-inf -- and inf * 0.0 is NaN, poisoning
        # the whole multitask loss. `(logits * 0.0).sum()` zeroes every
        # element first (0.0 * any finite value is exactly +/-0.0, never an
        # overflow) so the sum is always exactly 0.0, still graph-connected
        # to `logits` so backward() runs cleanly through this term.
        return (logits * 0.0).sum(), count
    flattened_logits = logits.reshape(-1, logits.shape[-1])
    return F.cross_entropy(flattened_logits[valid], flattened_target[valid], weight=weight), count


def _ordinal_classification_loss(
    logits: Tensor,
    target: Tensor,
    *,
    class_count: int,
    ordinal_weight: float,
) -> tuple[Tensor, int]:
    """Combine categorical fit with distance-aware supervision over ordered bins."""

    usable_logits = logits[..., :class_count]
    flattened_target = target.long().reshape(-1)
    valid = (flattened_target != -100) & (flattened_target >= 0) & (
        flattened_target < class_count
    )
    count = int(valid.sum())
    if not valid.any():
        # See _cross_entropy's identical fix above: zero before summing to
        # avoid a float32 overflow-to-inf on large-magnitude masked-out
        # sentinel logits producing inf * 0.0 == NaN.
        return (usable_logits * 0.0).sum(), count
    flattened_logits = usable_logits.reshape(-1, class_count)[valid]
    valid_target = flattened_target[valid]
    categorical = F.cross_entropy(flattened_logits, valid_target)
    if ordinal_weight == 0:
        return categorical, count
    positions = torch.linspace(
        0.0, 1.0, class_count, device=flattened_logits.device, dtype=flattened_logits.dtype
    )
    expected_position = (torch.softmax(flattened_logits, dim=-1) * positions).sum(dim=-1)
    target_position = valid_target.to(flattened_logits.dtype) / max(class_count - 1, 1)
    ordinal = F.smooth_l1_loss(expected_position, target_position)
    return categorical + ordinal_weight * ordinal, count


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


def masked_regression(
    prediction: Tensor, target: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, int]:
    """Single governed masked-regression helper (core-issues3.txt Phase 7.1).

    The previous per-call regression path (``_masked_mse``) checked only
    ``torch.isfinite(target)``, silently ignoring the ``f"{task}_mask"``
    companion targets_v2 defines for every maskable regression target
    (plan_value, the consequence proxies, sensor_reconstruction,
    future_concentration, travel_time). Those targets' generators write an
    explicit ``0.0`` placeholder at masked positions -- a *finite* value --
    precisely so the true absence of a label is recorded in the mask, not
    smuggled into the value itself (see targets_v2.py's module docstring).
    Checking only finiteness therefore trained the model directly against
    those placeholder zeros, exactly as _apply_target_mask already existed
    to prevent for classification targets.

    Also enforces `prediction.shape == target.shape` rather than relying on
    torch's broadcasting rules: a per-node prediction ([B, N]) regressed
    against a scalar-per-example target ([B]) broadcasts silently into a
    meaningless loss instead of raising -- the exact shape-mismatch class
    Phase 7.2 describes for Scout's expected_information_gain head. Gather
    or reduce the prediction to the target's shape (or vice versa) before
    calling this rather than depending on broadcasting to paper over a
    shape disagreement.

    Returns ``(loss, valid_count)`` -- core-issues3.txt Phase 11.1's
    required per-task valid-target-count record; `valid_count` is the
    number of positions that actually passed the finite-and-mask check,
    not merely `target.numel()`.
    """

    prediction = prediction.float()
    target = target.float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"masked_regression: prediction shape {tuple(prediction.shape)} disagrees with "
            f"target shape {tuple(target.shape)} -- reduce/gather to a matching shape rather "
            "than relying on broadcasting"
        )
    valid = torch.isfinite(target)
    if mask is not None:
        mask = mask.bool()
        if mask.shape != target.shape:
            raise ValueError(
                f"masked_regression: mask shape {tuple(mask.shape)} disagrees with target "
                f"shape {tuple(target.shape)}"
            )
        valid = valid & mask
    count = int(valid.sum())
    if not valid.any():
        # A graph-connected zero (not a detached Python float) so backward()
        # still runs cleanly through this term in a batch where every
        # example happens to mask this particular task. Zero before
        # summing, not after (see _cross_entropy's identical fix), so a
        # large-magnitude prediction cannot overflow float32 and turn this
        # into inf * 0.0 == NaN.
        return (prediction * 0.0).sum(), count
    return F.mse_loss(prediction[valid], target[valid]), count


def compute_multitask_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    task_weights: Mapping[str, float] | None = None,
    profile_ordinal_weight: float = 0.0,
    class_weights: Mapping[str, Tensor] | None = None,
) -> MultiTaskLoss:
    """Compute every task for which both a semantic prediction and target exist.

    `class_weights` (core-issues3.txt Phase 11.2, distinct from
    `task_weights`): an optional ``{task_name: per-class weight tensor}``
    mapping -- e.g. from hydroswarm.training.class_balance.
    train_owned_class_weights/class_weights_tensor, fit from TRAIN-split
    prevalence only -- passed through to `torch.nn.functional.
    cross_entropy`'s own `weight=` argument for the named classification
    task(s). `task_weights` scales a whole task's contribution to `total`;
    `class_weights` reweights WITHIN one task's classification loss by
    class. Only applies to the multi-class `classifications` tasks below
    (event_cause, ood_class, next_step, plan_validity, and similar) -- the
    binary/BCE-based tasks (event_presence, sensor_fault,
    should_continue_sampling, evidence_sufficiency) would need a
    `pos_weight`, not a per-class weight vector; not wired here, since none
    of them has a documented imbalance problem this pass identified.
    """

    weights = task_weights or {}
    class_weight_map = class_weights or {}
    losses: dict[str, Tensor] = {}
    # Task keys here are hydroswarm.training.targets_v2's governed target
    # names -- not the model's own output-attribute names (the dict values)
    # -- because compute_multitask_loss looks up each task's label by this
    # key in `targets`. Three of these previously used the *output* name
    # for both (action/action_pointer/ood) instead of the governed target
    # name (action_template/target_pointer/ood_class), so a correctly
    # generated governed target would never have matched any key here and
    # would have silently trained nothing (core-issues2.txt Phase 1 audit).
    classifications = {
        "source_node": "source_node_logits",
        "source_region": "source_region_logits",
        "sample_node": "sample_node_logits",
        "action_template": "action_logits",
        "target_pointer": "action_pointer_logits",
        "plan_validity": "plan_validity_logits",
        # core-issues3.txt Phase 6: was "ood_logits" -- a pre-existing
        # 3-logit head for a different, deterministic-severity-adjacent
        # concept (OODLevel), not the governed 11-category ood_class
        # taxonomy (hydroswarm.training.ood_categories.OODCategory).
        # Training this task against the old 3-logit head would raise the
        # moment a real label index >= 3 appeared (SEVERE_MISSINGNESS=7,
        # FROZEN_DRIFTING_SENSOR=8) -- only present in outputs when the
        # model was built with ood_category_head=True, so silently skipped
        # (like every other entry here) for models built without it.
        "ood_class": "ood_category_logits",
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
        # core-issues4.txt Section D item 3: per-node, same masking
        # convention as information_gain (targets_v2's own documented
        # "Same as information_gain" masking rule).
        "candidate_reduction": "candidate_reduction_prediction",
        "sensor_reconstruction": "sensor_reconstruction_prediction",
        "future_concentration": "future_concentration_prediction",
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
    counts: dict[str, int] = {}
    for task, output_name in classifications.items():
        if task in targets and output_name in outputs:
            losses[task], counts[task] = _cross_entropy(
                outputs[output_name],
                _apply_target_mask(task, targets[task], targets),
                weight=class_weight_map.get(task),
            )
    for task, class_count in PROFILE_CLASS_COUNTS.items():
        output_name = f"{task}_logits"
        if task in targets and output_name in outputs:
            losses[task], counts[task] = _ordinal_classification_loss(
                outputs[output_name],
                _apply_target_mask(task, targets[task], targets),
                class_count=class_count,
                ordinal_weight=profile_ordinal_weight,
            )
    for task, output_name in regressions.items():
        if task in targets and output_name in outputs:
            losses[task], counts[task] = masked_regression(
                outputs[output_name], targets[task], targets.get(f"{task}_mask")
            )
    if "sensor_fault" in targets and "sensor_fault_logits" in outputs:
        fault_target = targets["sensor_fault"].float()
        valid = torch.isfinite(fault_target) & (fault_target >= 0)
        fault_mask = targets.get("sensor_fault_mask")
        if fault_mask is not None:
            valid = valid & fault_mask.bool()
        counts["sensor_fault"] = int(valid.sum())
        losses["sensor_fault"] = (
            F.binary_cross_entropy_with_logits(
                outputs["sensor_fault_logits"].float()[valid], fault_target[valid]
            )
            if valid.any()
            # Zero before summing (see _cross_entropy's identical fix) to
            # avoid a float32 overflow-to-inf on this task's own
            # large-magnitude masked-out logits producing inf * 0.0 == NaN.
            else (outputs["sensor_fault_logits"] * 0.0).sum()
        )
    if "event_presence" in targets and "event_presence_logits" in outputs:
        presence_target = targets["event_presence"].float()
        valid = torch.isfinite(presence_target) & (presence_target >= 0)
        counts["event_presence"] = int(valid.sum())
        losses["event_presence"] = (
            F.binary_cross_entropy_with_logits(
                outputs["event_presence_logits"].float()[valid], presence_target[valid]
            )
            if valid.any()
            # Zero before summing -- see _cross_entropy's identical fix.
            else (outputs["event_presence_logits"] * 0.0).sum()
        )
    if "evidence_sufficiency" in targets and "evidence_sufficiency" in outputs:
        losses["evidence_sufficiency"] = F.binary_cross_entropy(
            outputs["evidence_sufficiency"].float(), targets["evidence_sufficiency"].float()
        )
        counts["evidence_sufficiency"] = int(targets["evidence_sufficiency"].numel())
    if "should_continue_sampling" in targets and "should_continue_sampling_logits" in outputs:
        # core-issues4.txt Section D item 3: targets_v2 declares this
        # target "Never masked; always defined by the budget policy", so
        # no f"{task}_mask" companion exists -- same
        # isfinite-and-non-negative validity check as event_presence
        # above (this repo's convention for a boolean target with no
        # separate mask field).
        continue_target = targets["should_continue_sampling"].float()
        valid = torch.isfinite(continue_target) & (continue_target >= 0)
        counts["should_continue_sampling"] = int(valid.sum())
        losses["should_continue_sampling"] = (
            F.binary_cross_entropy_with_logits(
                outputs["should_continue_sampling_logits"].float()[valid], continue_target[valid]
            )
            if valid.any()
            # Zero before summing -- see _cross_entropy's identical fix.
            else (outputs["should_continue_sampling_logits"] * 0.0).sum()
        )
    if not losses:
        raise ValueError("no compatible model outputs and training targets")

    def _default_weight(name: str) -> float:
        return AUXILIARY_TASK_DEFAULT_WEIGHT if name in AUXILIARY_TASKS else 1.0

    resolved_weights = {name: float(weights.get(name, _default_weight(name))) for name in losses}
    weighted = {name: loss * resolved_weights[name] for name, loss in losses.items()}
    return MultiTaskLoss(
        total=torch.stack(list(weighted.values())).sum(),
        tasks=losses,
        valid_counts=counts,
        weights=resolved_weights,
        weighted=weighted,
    )


#: Every task-name key core-issues3.txt Phase 11.1 requires an EXPLICIT
#: `task_weights` entry for -- i.e. every name `compute_multitask_loss`
#: could possibly add to its `losses`/`tasks` dict, given a model/target
#: batch with every gated head/target present. Deliberately the union of
#: every dict/branch above, kept next to `compute_multitask_loss` itself
#: (not hand-copied into configs/training.yaml or a second module) so a
#: future new task is a one-place addition, not a "two lists drift apart"
#: repeat of the class of defect this project has already hit three times
#: (8-vs-9 action templates, ood_head-vs-ood_category_head, the
#: NODE_INDEXED_TARGET_KEYS/permutation.py duplicate lists -- see
#: reports/results/v4/pre-freeze-implementation-handoff.md's Phase 10.2
#: section).
ALL_TASK_NAMES: frozenset[str] = frozenset(
    {
        "source_node",
        "source_region",
        "sample_node",
        "action_template",
        "target_pointer",
        "plan_validity",
        "ood_class",
        "event_cause",
        "next_step",
    }
    | set(PROFILE_CLASS_COUNTS)
    | {
        "plan_value",
        "information_gain",
        "candidate_reduction",
        "sensor_reconstruction",
        "future_concentration",
        "travel_time",
        "exposure_proxy",
        "pressure_risk_proxy",
        "service_loss_proxy",
        "containment_time_proxy",
        "plan_regret_proxy",
        "sensor_fault",
        "event_presence",
        "evidence_sufficiency",
        "should_continue_sampling",
    }
)


class IncompleteTaskWeightsError(Exception):
    """Raised by validate_task_weights_complete when a task_weights mapping
    (typically loaded from configs/training.yaml) omits an explicit entry
    for a retained task -- core-issues3.txt Phase 11.1: "Do not depend on
    hidden defaults for the final experiments." A missing key does not
    fail training (compute_multitask_loss's own `_default_weight` fallback
    still applies), but it fails THIS check, which a training entry point
    is expected to call before a real/promotion-quality run."""


def validate_task_weights_complete(task_weights: Mapping[str, float]) -> None:
    """Fail closed if `task_weights` omits an explicit entry for any task
    in ALL_TASK_NAMES. Extra keys (e.g. a name reserved for a future task)
    are tolerated -- only a missing key for a CURRENTLY known task is an
    error, since that is the specific "hidden default" Phase 11.1 forbids."""

    missing = ALL_TASK_NAMES - set(task_weights)
    if missing:
        raise IncompleteTaskWeightsError(
            f"task_weights is missing explicit weight(s) for: {sorted(missing)} -- "
            "core-issues3.txt Phase 11.1 requires an explicit weight for every retained "
            "target rather than relying on compute_multitask_loss's hidden default"
        )


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


#: Tasks whose gradient direction is the primary safety/scientific concern
#: (core-issues3.txt Phase 11.1's "gradient cosine/conflict diagnostics
#: where practical" / Phase 11.4's "Log gradient norms and primary-task
#: regressions"). Deliberately a small, named subset of ALL_TASK_NAMES, not
#: every task: a full T x T pairwise sweep is O(T^2) extra
#: torch.autograd.grad calls on top of the T calls task_gradient_norms
#: already costs (already documented there as expensive), and would mostly
#: measure pairs nobody is making a promotion decision about. One
#: representative governed head per role (Sentinel/Scout/Strategist/
#: OOD-control) that a newly added or reweighted task must not silently
#: degrade.
PRIMARY_TASKS: frozenset[str] = frozenset(
    {"source_node", "event_presence", "sensor_fault", "sample_node", "plan_validity", "ood_class"}
)


def task_gradient_conflict(
    task_losses: Mapping[str, Tensor],
    model: nn.Module,
    *,
    primary_tasks: frozenset[str] = PRIMARY_TASKS,
) -> dict[str, float]:
    """Cosine similarity between each present PRIMARY task's gradient and
    every other present task's gradient, keyed ``f"{primary}|{other}"``. A
    negative value is a genuine gradient conflict (the two tasks pull
    shared parameters in opposing directions); this function only reports
    the diagnostic -- core-issues3.txt Phase 11.4 explicitly says not to
    enable PCGrad or another automatic intervention by default, only after
    a measured conflict justifies it.

    Deliberately primary-vs-all rather than all-vs-all (see PRIMARY_TASKS'
    own docstring for why). A pair is omitted if either task's gradient
    is entirely `None` (allow_unused -- the loss does not actually depend
    on any shared trainable parameter this batch).

    Every present task's flattened gradient is aligned to the same
    per-parameter coordinate system: a parameter this task's loss does not
    depend on (``allow_unused`` gradient of `None`) contributes a zero
    vector of that parameter's own shape, rather than being dropped
    (Milestone 0.4). Two tasks that use different, unequal-sized parameter
    subsets previously produced flattened vectors of different lengths
    whose components had no defined correspondence to each other --
    `torch.dot`/cosine over them was not a scientifically meaningful
    comparison (it either raised on shape mismatch or, worse, silently
    compared unrelated parameters' gradient components when lengths
    happened to coincide)."""

    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    present_primary = sorted(primary_tasks & set(task_losses))
    if not present_primary:
        return {}
    flattened: dict[str, Tensor | None] = {}
    for name, loss in task_losses.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        if all(gradient is None for gradient in gradients):
            flattened[name] = None
            continue
        pieces = [
            gradient.detach().float().reshape(-1)
            if gradient is not None
            else torch.zeros(parameter.numel(), dtype=torch.float32)
            for gradient, parameter in zip(gradients, parameters)
        ]
        flattened[name] = torch.cat(pieces)
    conflict: dict[str, float] = {}
    for primary in present_primary:
        primary_vector = flattened[primary]
        if primary_vector is None:
            continue
        for other, vector in flattened.items():
            if other == primary or vector is None:
                continue
            denominator = float(primary_vector.norm() * vector.norm())
            conflict[f"{primary}|{other}"] = (
                float(torch.dot(primary_vector, vector)) / denominator if denominator > 0 else 0.0
            )
    return conflict
