"""Topology-balanced curriculum sampling (overnight-plan.txt Task 0.7).

A DataLoader sampler that gives every group (topology, source, curriculum
stage, ...) equal total sampling weight regardless of how many examples that
group happens to contain, so "a large topology dominates solely because it
contains more candidate nodes or generated scenarios" (the plan's explicit
failure mode) cannot happen.

Available group-key extractors today are limited to what the learning-v1
schema actually carries: network_id (today's proxy for topology/hydraulic-
regime until Task 1.1 adds a real topology_hash), source_node, and
curriculum stage. Event-cause and sensor-fault grouping are not implemented
here because those targets do not exist until targets_v2 (Task 2.1-2.2);
adding a by_event_cause()/by_sensor_fault() extractor at that point is a
same-shape addition, not a redesign, since GroupBalancedSampler accepts any
group_key callable.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from .data import ScenarioExample

GroupKeyFn = Callable[[ScenarioExample], Hashable]


def by_network(example: ScenarioExample) -> Hashable:
    return ("network", example.network_id)


def by_source_node(example: ScenarioExample) -> Hashable:
    value = example.targets.get("source_node")
    return ("source_node", int(value) if value is not None else None)


def by_curriculum_stage(example: ScenarioExample) -> Hashable:
    return ("stage", example.stage.name)


def composite_key(*key_fns: GroupKeyFn) -> GroupKeyFn:
    """Combine multiple group-key extractors into one composite key, e.g.
    composite_key(by_network, by_curriculum_stage) balances jointly across
    topology and curriculum stage rather than either alone."""

    if not key_fns:
        raise ValueError("composite_key requires at least one key function")

    def _combined(example: ScenarioExample) -> Hashable:
        return tuple(fn(example) for fn in key_fns)

    return _combined


class GroupBalancedSampler(Sampler[int]):
    """Every group receives equal total sampling weight (1/num_groups),
    split evenly among its members (1/(num_groups * group_size) per
    example), so small groups are not drowned out by large ones. Samples
    with replacement, which is the standard approach for group/class-
    balanced training when group sizes are unequal.
    """

    def __init__(
        self,
        examples: Sequence[ScenarioExample],
        *,
        group_key: GroupKeyFn,
        seed: int,
        num_samples: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("cannot build a sampler over an empty dataset")
        self._num_samples = num_samples if num_samples is not None else len(examples)
        self._seed = seed
        self._epoch = 0

        groups: dict[Hashable, list[int]] = {}
        for index, example in enumerate(examples):
            groups.setdefault(group_key(example), []).append(index)
        self._group_sizes: dict[Hashable, int] = {key: len(indices) for key, indices in groups.items()}

        weights = np.zeros(len(examples), dtype=np.float64)
        weight_per_group = 1.0 / len(groups)
        for indices in groups.values():
            weight_per_example = weight_per_group / len(indices)
            for index in indices:
                weights[index] = weight_per_example
        self._weights = weights

    def set_epoch(self, epoch: int) -> None:
        """Call once per training epoch so each epoch draws a distinct but
        deterministically reproducible sample (matches the DistributedSampler
        convention used elsewhere in the PyTorch ecosystem)."""
        self._epoch = epoch

    def group_sizes(self) -> dict[Hashable, int]:
        return dict(self._group_sizes)

    def weight_mass_by_group(self, group_key: GroupKeyFn, examples: Sequence[ScenarioExample]) -> dict[Hashable, float]:
        """Diagnostic: total sampling-probability mass per group, for tests
        and audit reporting. Should be ~equal across groups by construction."""
        mass: dict[Hashable, float] = {}
        for index, example in enumerate(examples):
            key = group_key(example)
            mass[key] = mass.get(key, 0.0) + float(self._weights[index])
        return mass

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self._seed + self._epoch)
        indices = generator.choice(len(self._weights), size=self._num_samples, replace=True, p=self._weights)
        return iter(int(index) for index in indices)
