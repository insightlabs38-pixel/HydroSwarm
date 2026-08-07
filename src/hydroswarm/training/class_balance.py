"""core-issues3.txt Phase 11.2: class-imbalance reporting and train-owned
class weights for imbalanced classification tasks (event_cause, OOD
categories, plan validity, continue-sampling, and similar).

Deliberately separate from hydroswarm.training.losses: this module only
COUNTS and derives a WEIGHTING POLICY from those counts; losses.py's
compute_multitask_loss consumes the resulting per-task weight tensors (an
optional `class_weights` argument) but has no opinion on how they were
computed. Weights must always be fit from the TRAIN split only -- the same
train-only-fitting discipline this project already applies to
normalization stats and signature artifacts (hydroswarm.classical.
signature_registry) -- never from validation/calibration/development-
holdout, which would leak split-specific class balance into training.

Class weighting changes the LOSS a classification task contributes, never
what hydroswarm.training.losses.MultiTaskLoss.valid_counts reports (that
count is computed purely from the mask/finite check, before any weight is
applied) and never how a downstream evaluation script computes real
metrics -- "evaluate unweighted real-distribution metrics separately" is
therefore a property callers get for free by construction, not something
this module has to separately enforce: a class-weighted training run's
validation/test metrics must still be computed against the real,
unweighted label distribution, exactly as an unweighted run's would be.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

#: Versioned so a future change to the weighting formula/cap is a policy
#: identity change, not a silent behavior change -- same convention as
#: hydroswarm.planning.plan_value_policy.PLAN_VALUE_POLICY_VERSION and
#: hydroswarm.classical.signature_registry.TOPOLOGY_WIDE_REGIME_HASH.
CLASS_WEIGHT_POLICY_VERSION = "inverse-frequency-capped-v1"

#: Default cap on any single class's weight relative to the post-rescale
#: mean of 1.0 -- an extremely rare class (e.g. a handful of examples out
#: of thousands) would otherwise receive an enormous weight that can
#: destabilize training far more than the imbalance itself justifies.
DEFAULT_MAXIMUM_WEIGHT = 10.0


def class_prevalence(
    labels: Tensor, *, mask: Tensor | None = None, ignore_index: int | None = -100
) -> dict[int, int]:
    """Count occurrences of each distinct label in `labels` (any shape;
    flattened before counting, so this works uniformly for a
    scalar-per-example class index like event_cause/next_step/ood_class,
    shape [B], and a per-position array like plan_validity, shape [B, P]).

    `mask` (this project's `f"{task}_mask"` convention, True = real/
    observed) and `ignore_index` (the -100 "ignore this position"
    sentinel `hydroswarm.training.losses._apply_target_mask` already
    writes) are both honored; a position excluded by either is not
    counted. Boolean labels (e.g. plan_validity) count as {0: <False
    count>, 1: <True count>}.
    """

    flat = labels.reshape(-1)
    valid = torch.ones_like(flat, dtype=torch.bool)
    if mask is not None:
        valid = valid & mask.reshape(-1).bool()
    if ignore_index is not None and flat.dtype != torch.bool:
        valid = valid & (flat != ignore_index)
    prevalence: dict[int, int] = {}
    for value in flat[valid].tolist():
        key = int(value)
        prevalence[key] = prevalence.get(key, 0) + 1
    return prevalence


def merge_prevalence(*prevalences: Mapping[int, int]) -> dict[int, int]:
    """Sum multiple class_prevalence() results (e.g. across shards of the
    same split) into one combined count per class."""

    merged: dict[int, int] = {}
    for prevalence in prevalences:
        for key, count in prevalence.items():
            merged[key] = merged.get(key, 0) + count
    return merged


def train_owned_class_weights(
    prevalence: Mapping[int, int], *, maximum_weight: float = DEFAULT_MAXIMUM_WEIGHT
) -> dict[int, float]:
    """Deterministic inverse-frequency class weights (CLASS_WEIGHT_POLICY_
    VERSION) from a TRAIN-split-only `prevalence` mapping. Weight for class
    c is ``total / (num_classes * count[c])``, capped at `maximum_weight`,
    then rescaled so the mean weight across observed classes is exactly
    1.0 -- keeps a class-weighted run's overall loss scale comparable to
    an unweighted run's, so `task_weights` (core-issues3.txt Phase 11.1)
    remains meaningful alongside it rather than being silently multiplied
    by an unrelated factor.

    Raises ValueError on an empty prevalence (nothing to weight) or a
    non-positive count (a caller bug -- class_prevalence never produces
    one)."""

    if not prevalence:
        raise ValueError("cannot compute class weights from an empty prevalence mapping")
    if any(count <= 0 for count in prevalence.values()):
        raise ValueError(f"prevalence contains a non-positive count: {prevalence}")
    total = sum(prevalence.values())
    num_classes = len(prevalence)
    raw = {cls: min(total / (num_classes * count), maximum_weight) for cls, count in prevalence.items()}
    mean = sum(raw.values()) / len(raw)
    return {cls: weight / mean for cls, weight in raw.items()}


def class_weights_tensor(weights: Mapping[int, float], num_classes: int) -> Tensor:
    """Dense `[num_classes]` weight tensor (default 1.0 for any class not
    present in `weights`, e.g. a class with zero TRAIN-split examples --
    it simply gets no reweighting, since there is no train-owned signal to
    derive one from) -- ready for `torch.nn.functional.cross_entropy`'s
    own `weight=` argument via
    hydroswarm.training.losses.compute_multitask_loss's `class_weights`.
    """

    dense = torch.ones(num_classes, dtype=torch.float32)
    for cls, weight in weights.items():
        if not (0 <= cls < num_classes):
            raise ValueError(f"class index {cls} is out of range for num_classes={num_classes}")
        dense[cls] = weight
    return dense
