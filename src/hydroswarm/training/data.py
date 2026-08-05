"""Governed scenarios, agent trajectories, and deterministic collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
class TopologyMetadata:
    """Per-example, graph-local topology provenance (overnight-plan.txt Task 1.1).

    Every example owns its own node/edge ordering instead of assuming all
    networks share the same junction IDs and node order. ``source_node`` (and
    any other node-indexed target) is a local index into the *full*
    ``node_ids`` space -- matching how HydroCore's source_node_logits covers
    every node position and source_candidate_mask (also full node_ids-length)
    separately marks which of those positions are eligible -- never a global
    class identity. ``source_candidate_ids`` is the (possibly strict) subset
    of ``node_ids`` that are actually valid candidates, kept as an explicit
    audit field rather than requiring every caller to re-derive it from a
    mask; it is deliberately NOT the index space source_node counts into.
    No learned global node-ID embedding may be introduced on top of this
    metadata.
    """

    topology_hash: str
    network_hash: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[tuple[str, str], ...]
    source_candidate_ids: tuple[str, ...]
    hydraulic_state_hash: str
    signature_library_hash: str
    target_schema_version: str
    feature_schema_version: str

    def __post_init__(self) -> None:
        if not self.topology_hash or not self.network_hash:
            raise ValueError("topology_hash and network_hash are required")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        node_set = set(self.node_ids)
        missing_edge_endpoints = {
            endpoint
            for source, target in self.edge_ids
            for endpoint in (source, target)
            if endpoint not in node_set
        }
        if missing_edge_endpoints:
            raise ValueError(f"edge endpoints missing from node_ids: {sorted(missing_edge_endpoints)}")
        missing_candidates = set(self.source_candidate_ids) - node_set
        if missing_candidates:
            raise ValueError(f"source_candidate_ids missing from node_ids: {sorted(missing_candidates)}")

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)

    def local_index(self, node_id: str) -> int:
        return self.node_ids.index(node_id)

    def source_node_id_for_local_index(self, index: int) -> str:
        """Map a source_node target (a full node_ids-space local index) back
        to its original node ID. Deliberately indexes node_ids, not
        source_candidate_ids, which may be a strict subset."""

        return self.node_ids[index]

    def to_json(self) -> dict[str, Any]:
        return {
            "topology_hash": self.topology_hash,
            "network_hash": self.network_hash,
            "node_ids": list(self.node_ids),
            "edge_ids": [list(pair) for pair in self.edge_ids],
            "source_candidate_ids": list(self.source_candidate_ids),
            "hydraulic_state_hash": self.hydraulic_state_hash,
            "signature_library_hash": self.signature_library_hash,
            "target_schema_version": self.target_schema_version,
            "feature_schema_version": self.feature_schema_version,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TopologyMetadata":
        return cls(
            topology_hash=payload["topology_hash"],
            network_hash=payload["network_hash"],
            node_ids=tuple(payload["node_ids"]),
            edge_ids=tuple(tuple(pair) for pair in payload["edge_ids"]),
            source_candidate_ids=tuple(payload["source_candidate_ids"]),
            hydraulic_state_hash=payload["hydraulic_state_hash"],
            signature_library_hash=payload["signature_library_hash"],
            target_schema_version=payload["target_schema_version"],
            feature_schema_version=payload["feature_schema_version"],
        )


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
    topology: TopologyMetadata | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.network_id or not self.seed_family:
            raise ValueError("scenario, network, and seed-family IDs are required")
        if self.split not in {"train", "validation", "test", "calibration", "development_holdout"}:
            raise ValueError("unsupported dataset split")
        if self.seed < 0:
            raise ValueError("scenario seed cannot be negative")
        if not self.inputs or not self.targets:
            raise ValueError("scenario inputs and targets cannot be empty")
        if any(not isinstance(value, Tensor) for value in (*self.inputs.values(), *self.targets.values())):
            raise TypeError("scenario inputs and targets must be tensors")


def manifest_entry(example: ScenarioExample) -> dict[str, Any]:
    """The governance-relevant fields hashed into manifest_hash. Shared
    between GovernedScenarioDataset and ShardedScenarioDataset so both
    produce identical hashes for identical example sets, and topology-aware
    since Task 1.1: two datasets with the same scenario/seed metadata but
    different topology_hash values must not hash identically."""

    return {
        "scenario_id": example.scenario_id,
        "network_id": example.network_id,
        "split": example.split,
        "seed": example.seed,
        "seed_family": example.seed_family,
        "stage": example.stage.name,
        "topology_hash": example.topology.topology_hash if example.topology else None,
        "network_hash": example.topology.network_hash if example.topology else None,
    }


def resolve_source_node_id(example: ScenarioExample) -> str | None:
    """Map a (graph-local) source_node target back to its original node ID,
    for verification/reporting. Returns None if the example carries no
    topology metadata or no source_node target."""

    if example.topology is None or "source_node" not in example.targets:
        return None
    local_index = int(example.targets["source_node"])
    return example.topology.source_node_id_for_local_index(local_index)


@runtime_checkable
class ScenarioDatasetView(Protocol):
    """Structural interface shared by GovernedScenarioDataset and
    hydroswarm.training.sharded_data.ShardedScenarioDataset (core-issues.txt
    repair item 12).

    Trainer and the Stage 2/3/4 scripts only ever need __len__,
    __getitem__, manifest_hash, and stages_through -- both classes already
    implement all four identically in spirit, but ShardedScenarioDataset
    does so lazily (reading tensors from disk-backed shards on access)
    while GovernedScenarioDataset holds every ScenarioExample resident in
    memory. Typing call sites against this Protocol instead of the
    concrete GovernedScenarioDataset class is what lets a lazy
    ShardedScenarioDataset be passed to Trainer directly -- previously,
    every caller materialized the entire corpus into a Python list purely
    to satisfy this type hint, defeating disk-backed sharding entirely."""

    expected_split: str

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ScenarioExample: ...

    @property
    def manifest_hash(self) -> str: ...

    def stages_through(self, stage: CurriculumStage) -> "ScenarioDatasetView": ...


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
        entries = [manifest_entry(example) for example in self._examples]
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def stages_through(self, stage: CurriculumStage) -> GovernedScenarioDataset:
        selected = [example for example in self._examples if example.stage <= stage]
        return GovernedScenarioDataset(selected, expected_split=self.expected_split)


def load_scenario_examples_jsonl(path: str | Path) -> list[ScenarioExample]:
    """Parse the canonical-tensor JSONL schema shared by data/learning-v1 and
    scripts/train.py::load_dataset into a plain list of ScenarioExample.

    Kept separate from GovernedScenarioDataset construction so callers that
    only need governance metadata and target values (e.g. the label/target
    audit tooling) are not forced to also satisfy split-uniqueness
    invariants meant for training splits.
    """

    examples: list[ScenarioExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(
                    ScenarioExample(
                        scenario_id=record["scenario_id"],
                        network_id=record["network_id"],
                        split=record["split"],
                        seed=int(record["seed"]),
                        seed_family=record["seed_family"],
                        stage=CurriculumStage[record["stage"]],
                        inputs={key: torch.tensor(value) for key, value in record["inputs"].items()},
                        targets={key: torch.tensor(value) for key, value in record["targets"].items()},
                        topology=(
                            TopologyMetadata.from_json(record["topology"])
                            if record.get("topology") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest line {line_number} in {path}: {error}") from error
    return examples


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

