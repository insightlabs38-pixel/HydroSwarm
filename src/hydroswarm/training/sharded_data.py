"""Sharded, lazily-loaded scenario storage (overnight-plan.txt Task 0.4).

The original training path (`scripts/train.py::load_dataset`) parses a JSONL
manifest and materializes every example's tensors into memory up front. That
is fine for the 1,320-example learning-v1 corpus but is a blocker for the
10,000-40,000+ multi-topology corpora planned for learning-v2/v3: tensor
data must be lazily loaded from disk per-example, not eagerly deserialized
at process start.

Storage layout under a shard directory::

    manifest.json   -- schema version, shard list with per-shard sha256 and
                        example counts, index file name + its sha256.
    index.jsonl     -- one line per example: governance metadata (scenario
                        id, network id, split, seed, seed family, curriculum
                        stage) plus its shard file, local offset within that
                        shard, and its tensor key names. This is the only
                        thing read into memory at construction time.
    shard-NNNNN.safetensors
                    -- tensor payloads for a contiguous slice of examples,
                       opened lazily via safetensors' mmap-backed safe_open
                       so no shard is fully deserialized until a specific
                       tensor inside it is requested.

ShardedScenarioDataset exposes the same __len__/__getitem__/manifest_hash/
stages_through surface as GovernedScenarioDataset and yields the same
ScenarioExample objects, so it is a drop-in replacement wherever the
in-memory dataset is used today (Trainer, collate_scenarios, curriculum
filtering, leakage checks).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset

from .data import CurriculumStage, ScenarioExample, TopologyMetadata

SCHEMA_VERSION = 1
DEFAULT_SHARD_SIZE = 256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    scenario_id: str
    network_id: str
    split: str
    seed: int
    seed_family: str
    stage: str
    shard_file: str
    local_index: int
    input_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    topology: TopologyMetadata | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "network_id": self.network_id,
            "split": self.split,
            "seed": self.seed,
            "seed_family": self.seed_family,
            "stage": self.stage,
            "shard_file": self.shard_file,
            "local_index": self.local_index,
            "input_keys": list(self.input_keys),
            "target_keys": list(self.target_keys),
            "topology": self.topology.to_json() if self.topology is not None else None,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "_IndexEntry":
        return cls(
            scenario_id=payload["scenario_id"],
            network_id=payload["network_id"],
            split=payload["split"],
            seed=int(payload["seed"]),
            seed_family=payload["seed_family"],
            stage=payload["stage"],
            shard_file=payload["shard_file"],
            local_index=int(payload["local_index"]),
            input_keys=tuple(payload["input_keys"]),
            target_keys=tuple(payload["target_keys"]),
            topology=(
                TopologyMetadata.from_json(payload["topology"])
                if payload.get("topology") is not None
                else None
            ),
        )


def write_shards(
    examples: Sequence[ScenarioExample],
    output_dir: str | Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    """Write ``examples`` as sharded safetensors + a governed JSONL index.

    This is also the migration path from small in-memory/JSONL fixtures: load
    a Sequence[ScenarioExample] however you like (including the existing
    scripts/train.py::load_dataset JSONL reader) and pass it here.
    """

    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not examples:
        raise ValueError("cannot shard an empty example set")

    index_entries: list[_IndexEntry] = []
    shard_reports: list[dict[str, Any]] = []
    for shard_number, start in enumerate(range(0, len(examples), shard_size)):
        chunk = examples[start : start + shard_size]
        shard_name = f"shard-{shard_number:05d}.safetensors"
        shard_path = output_dir / shard_name
        tensors: dict[str, torch.Tensor] = {}
        for local_index, example in enumerate(chunk):
            for key, value in example.inputs.items():
                tensors[f"{local_index}/input/{key}"] = value
            for key, value in example.targets.items():
                tensors[f"{local_index}/target/{key}"] = value
            index_entries.append(
                _IndexEntry(
                    scenario_id=example.scenario_id,
                    network_id=example.network_id,
                    split=example.split,
                    seed=example.seed,
                    seed_family=example.seed_family,
                    stage=example.stage.name,
                    shard_file=shard_name,
                    local_index=local_index,
                    input_keys=tuple(sorted(example.inputs)),
                    target_keys=tuple(sorted(example.targets)),
                    topology=example.topology,
                )
            )
        save_file(tensors, str(shard_path))
        shard_reports.append(
            {
                "file": shard_name,
                "sha256": _sha256_file(shard_path),
                "example_count": len(chunk),
            }
        )

    index_path = output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for entry in index_entries:
            stream.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "shard_size": shard_size,
        "total_examples": len(examples),
        "shards": shard_reports,
        "index_file": "index.jsonl",
        "index_sha256": _sha256_file(index_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class ShardedScenarioDataset(Dataset[ScenarioExample]):
    """Lazily loads examples from disk-backed shards.

    Safe to hand to a DataLoader with num_workers > 0 under either the
    "fork" or "spawn" multiprocessing start method: shard file handles are
    opened lazily on first access and are excluded from
    __getstate__/__setstate__, so each worker process opens (and mmaps) only
    the shards it actually touches, and never inherits a parent process's
    open handles across a spawn-style re-pickle.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        *,
        expected_split: str,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        manifest_path = self.shard_dir / "manifest.json"
        index_path = self.shard_dir / "index.jsonl"
        if not manifest_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"{self.shard_dir} is not a valid sharded dataset (missing manifest.json/index.jsonl)"
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        measured_index_hash = _sha256_file(index_path)
        if measured_index_hash != self.manifest["index_sha256"]:
            raise ValueError(
                "index.jsonl checksum does not match manifest.json; corpus is corrupted or stale"
            )
        for shard in self.manifest["shards"]:
            shard_path = self.shard_dir / shard["file"]
            if not shard_path.exists():
                raise FileNotFoundError(f"missing shard file referenced by manifest: {shard_path}")

        entries: list[_IndexEntry] = []
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    entries.append(_IndexEntry.from_json(json.loads(line)))
        self._entries: tuple[_IndexEntry, ...] = tuple(entries)

        wrong = [entry.scenario_id for entry in self._entries if entry.split != expected_split]
        if wrong:
            raise ValueError(f"examples belong to the wrong split: {wrong}")
        scenario_ids = [entry.scenario_id for entry in self._entries]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario IDs must be unique")
        family_keys = [(entry.network_id, entry.seed_family) for entry in self._entries]
        if len(set(family_keys)) != len(family_keys):
            raise ValueError("a seed family may appear only once per governed split")

        self.expected_split = expected_split
        self._indices: tuple[int, ...] = (
            tuple(indices) if indices is not None else tuple(range(len(self._entries)))
        )
        self._shard_handles: dict[str, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_handles"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._indices)

    def _handle_for(self, shard_file: str) -> Any:
        handle = self._shard_handles.get(shard_file)
        if handle is None:
            handle = safe_open(str(self.shard_dir / shard_file), framework="pt", device="cpu")
            self._shard_handles[shard_file] = handle
        return handle

    def __getitem__(self, position: int) -> ScenarioExample:
        entry = self._entries[self._indices[position]]
        handle = self._handle_for(entry.shard_file)
        inputs = {key: handle.get_tensor(f"{entry.local_index}/input/{key}") for key in entry.input_keys}
        targets = {
            key: handle.get_tensor(f"{entry.local_index}/target/{key}") for key in entry.target_keys
        }
        return ScenarioExample(
            scenario_id=entry.scenario_id,
            network_id=entry.network_id,
            split=entry.split,
            seed=entry.seed,
            seed_family=entry.seed_family,
            stage=CurriculumStage[entry.stage],
            inputs=inputs,
            targets=targets,
            topology=entry.topology,
        )

    @property
    def manifest_hash(self) -> str:
        entries = [
            {
                "scenario_id": self._entries[position].scenario_id,
                "network_id": self._entries[position].network_id,
                "split": self._entries[position].split,
                "seed": self._entries[position].seed,
                "seed_family": self._entries[position].seed_family,
                "stage": self._entries[position].stage,
                "topology_hash": (
                    self._entries[position].topology.topology_hash
                    if self._entries[position].topology is not None
                    else None
                ),
                "network_hash": (
                    self._entries[position].topology.network_hash
                    if self._entries[position].topology is not None
                    else None
                ),
            }
            for position in self._indices
        ]
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def stages_through(self, stage: CurriculumStage) -> "ShardedScenarioDataset":
        selected = [
            position
            for position in self._indices
            if CurriculumStage[self._entries[position].stage] <= stage
        ]
        clone = ShardedScenarioDataset.__new__(ShardedScenarioDataset)
        clone.shard_dir = self.shard_dir
        clone.manifest = self.manifest
        clone._entries = self._entries
        clone.expected_split = self.expected_split
        clone._indices = tuple(selected)
        clone._shard_handles = {}
        return clone
