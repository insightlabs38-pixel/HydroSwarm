"""Governed scenarios, agent trajectories, and deterministic collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CurriculumStage(IntEnum):
    CLEAN = 0
    OPERATIONAL = 1
    DEGRADED = 2
    SHIFT = 3
    ADVERSARIAL = 4


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    step_index: int
    state_hash: str
    action: str
    verifier_decision: str | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("trajectory step index cannot be negative")
        if len(self.state_hash) != 64 or any(character not in "0123456789abcdef" for character in self.state_hash):
            raise ValueError("state_hash must be a lowercase SHA-256 digest")
        if not self.action:
            raise ValueError("trajectory action cannot be empty")


@dataclass(frozen=True, slots=True)
class AgentTrajectory:
    trajectory_id: str
    scenario_id: str
    steps: tuple[TrajectoryStep, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.scenario_id or not self.steps:
            raise ValueError("trajectory IDs and steps are required")
        indices = [step.step_index for step in self.steps]
        if indices != list(range(len(indices))):
            raise ValueError("trajectory steps must be contiguous and zero-based")


@dataclass(frozen=True, slots=True)
class ScenarioExample:
    scenario_id: str
    network_id: str
    split: str
    seed: int
    seed_family: str
    stage: CurriculumStage
    inputs: Mapping[str, Tensor]
    targets: Mapping[str, Tensor]
    trajectory: AgentTrajectory | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.network_id or not self.seed_family:
            raise ValueError("scenario, network, and seed-family IDs are required")
        if self.split not in {"train", "validation", "test", "calibration"}:
            raise ValueError("unsupported dataset split")
        if self.seed < 0:
            raise ValueError("scenario seed cannot be negative")
        if not self.inputs or not self.targets:
            raise ValueError("scenario inputs and targets cannot be empty")
        if any(not isinstance(value, Tensor) for value in (*self.inputs.values(), *self.targets.values())):
            raise TypeError("scenario inputs and targets must be tensors")


class GovernedScenarioDataset(Dataset[ScenarioExample]):
    """Immutable split with manifest hashing and leakage checks."""

    def __init__(self, examples: Sequence[ScenarioExample], *, expected_split: str) -> None:
        self._examples = tuple(examples)
        if not self._examples:
            raise ValueError("governed dataset cannot be empty")
        scenario_ids = [example.scenario_id for example in self._examples]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario IDs must be unique")
        wrong = [example.scenario_id for example in self._examples if example.split != expected_split]
        if wrong:
            raise ValueError(f"examples belong to the wrong split: {wrong}")
        family_keys = [(example.network_id, example.seed_family) for example in self._examples]
        if len(set(family_keys)) != len(family_keys):
            raise ValueError("a seed family may appear only once per governed split")
        self.expected_split = expected_split

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> ScenarioExample:
        return self._examples[index]

    @property
    def manifest_hash(self) -> str:
        entries = [
            {
                "scenario_id": example.scenario_id,
                "network_id": example.network_id,
                "split": example.split,
                "seed": example.seed,
                "seed_family": example.seed_family,
                "stage": example.stage.name,
            }
            for example in self._examples
        ]
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def stages_through(self, stage: CurriculumStage) -> GovernedScenarioDataset:
        selected = [example for example in self._examples if example.stage <= stage]
        return GovernedScenarioDataset(selected, expected_split=self.expected_split)


def validate_split_isolation(*datasets: GovernedScenarioDataset) -> None:
    seen_scenarios: set[str] = set()
    seen_families: set[tuple[str, str]] = set()
    for dataset in datasets:
        for example in dataset:
            if example.scenario_id in seen_scenarios:
                raise ValueError(f"scenario leakage detected: {example.scenario_id}")
            family = (example.network_id, example.seed_family)
            if family in seen_families:
                raise ValueError(f"seed-family leakage detected: {family}")
            seen_scenarios.add(example.scenario_id)
            seen_families.add(family)


@dataclass(frozen=True, slots=True)
class CurriculumSchedule:
    stage_start_epochs: Mapping[CurriculumStage, int]

    @classmethod
    def progressive(cls) -> CurriculumSchedule:
        return cls(
            {
                CurriculumStage.CLEAN: 0,
                CurriculumStage.OPERATIONAL: 1,
                CurriculumStage.DEGRADED: 2,
                CurriculumStage.SHIFT: 3,
                CurriculumStage.ADVERSARIAL: 4,
            }
        )

    def stage_for_epoch(self, epoch: int) -> CurriculumStage:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        available = [stage for stage, start in self.stage_start_epochs.items() if start <= epoch]
        if not available:
            raise ValueError("curriculum must start at or before epoch zero")
        return max(available)


def collate_scenarios(examples: Sequence[ScenarioExample]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    input_keys = set(examples[0].inputs)
    target_keys = set(examples[0].targets)
    if any(set(example.inputs) != input_keys or set(example.targets) != target_keys for example in examples):
        raise ValueError("all examples in a batch require identical tensor keys")
    try:
        inputs = {key: torch.stack([example.inputs[key] for example in examples]) for key in sorted(input_keys)}
        targets = {key: torch.stack([example.targets[key] for example in examples]) for key in sorted(target_keys)}
    except RuntimeError as error:
        raise ValueError("examples must be canonically aligned and padded before collation") from error
    return inputs, targets

