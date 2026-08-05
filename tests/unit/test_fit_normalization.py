from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_normalization  # noqa: E402

import torch  # noqa: E402

from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats  # noqa: E402
from hydroswarm.training import write_shards  # noqa: E402
from hydroswarm.training.data import CurriculumStage, ScenarioExample  # noqa: E402


def _write_manifest(path: Path, *, split: str, count: int) -> None:
    lines = []
    for index in range(count):
        record = {
            "scenario_id": f"{split}-{index}",
            "network_id": "net-a",
            "split": split,
            "seed": index,
            "seed_family": f"family-{split}-{index}",
            "stage": "CLEAN",
            "inputs": {
                "node_features": [[float(index)] * len(DEFAULT_FEATURE_SCHEMA.node_features) for _ in range(3)],
                "edge_features": [[float(index)] * len(DEFAULT_FEATURE_SCHEMA.edge_features) for _ in range(2)],
            },
            "targets": {"source_node": 0},
        }
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fit_normalization_rejects_non_train_split(tmp_path) -> None:
    manifest = tmp_path / "test.jsonl"
    _write_manifest(manifest, split="test", count=3)
    with pytest.raises(SystemExit, match="non-train splits"):
        fit_normalization.main(
            [
                "--train-manifest",
                str(manifest),
                "--node-output",
                str(tmp_path / "node.json"),
                "--edge-output",
                str(tmp_path / "edge.json"),
            ]
        )


def test_fit_normalization_cli_rejects_non_train_split_flag() -> None:
    with pytest.raises(SystemExit):
        fit_normalization.build_parser().parse_args(
            ["--train-manifest", "x.jsonl", "--split", "test", "--node-output", "n.json", "--edge-output", "e.json"]
        )


def test_fit_normalization_writes_governed_artifacts(tmp_path) -> None:
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, split="train", count=4)
    node_output = tmp_path / "node-normalization.json"
    edge_output = tmp_path / "edge-normalization.json"

    exit_code = fit_normalization.main(
        [
            "--train-manifest",
            str(manifest),
            "--node-output",
            str(node_output),
            "--edge-output",
            str(edge_output),
        ]
    )
    assert exit_code == 0
    assert node_output.exists()
    assert node_output.with_suffix(".json.sha256").exists()
    assert edge_output.exists()

    node_stats = NormalizationStats.load(node_output)
    assert node_stats.feature_names == DEFAULT_FEATURE_SCHEMA.node_features


def _sharded_examples(count: int, *, split: str, node_count: int) -> list[ScenarioExample]:
    examples = []
    for index in range(count):
        examples.append(
            ScenarioExample(
                scenario_id=f"{split}-{index}",
                network_id="net-a",
                split=split,
                seed=index,
                seed_family=f"family-{split}-{index}",
                stage=CurriculumStage.CLEAN,
                inputs={
                    "node_features": torch.full(
                        (node_count, len(DEFAULT_FEATURE_SCHEMA.node_features)), float(index)
                    ),
                    "edge_features": torch.full(
                        (node_count - 1, len(DEFAULT_FEATURE_SCHEMA.edge_features)), float(index)
                    ),
                },
                targets={"source_node": torch.tensor(0)},
            )
        )
    return examples


def test_fit_normalization_requires_exactly_one_source(tmp_path) -> None:
    with pytest.raises(SystemExit):
        fit_normalization.build_parser().parse_args(
            ["--node-output", "n.json", "--edge-output", "e.json"]
        )
    with pytest.raises(SystemExit):
        fit_normalization.build_parser().parse_args(
            [
                "--train-manifest", "x.jsonl", "--train-shards", str(tmp_path),
                "--node-output", "n.json", "--edge-output", "e.json",
            ]
        )


def test_fit_normalization_from_sharded_corpus_verifies_checksums_and_fits(tmp_path) -> None:
    write_shards(_sharded_examples(4, split="train", node_count=3), tmp_path / "shards", shard_size=2)
    node_output = tmp_path / "node-normalization.json"
    edge_output = tmp_path / "edge-normalization.json"

    exit_code = fit_normalization.main(
        [
            "--train-shards", str(tmp_path / "shards"),
            "--node-output", str(node_output),
            "--edge-output", str(edge_output),
        ]
    )
    assert exit_code == 0
    node_stats = NormalizationStats.load(node_output)
    assert node_stats.feature_names == DEFAULT_FEATURE_SCHEMA.node_features
    edge_stats = NormalizationStats.load(edge_output)
    assert edge_stats.feature_names == DEFAULT_FEATURE_SCHEMA.edge_features


def test_fit_normalization_from_sharded_corpus_rejects_corrupted_shard(tmp_path) -> None:
    write_shards(_sharded_examples(3, split="train", node_count=3), tmp_path / "shards", shard_size=5)
    shard_path = tmp_path / "shards" / "shard-00000.safetensors"
    corrupted = bytearray(shard_path.read_bytes())
    corrupted[-1] ^= 0xFF
    shard_path.write_bytes(bytes(corrupted))

    with pytest.raises(ValueError, match="checksum"):
        fit_normalization.main(
            [
                "--train-shards", str(tmp_path / "shards"),
                "--node-output", str(tmp_path / "node.json"),
                "--edge-output", str(tmp_path / "edge.json"),
            ]
        )


def test_fit_normalization_from_sharded_corpus_rejects_non_train_split(tmp_path) -> None:
    write_shards(_sharded_examples(2, split="validation", node_count=3), tmp_path / "shards", shard_size=5)
    with pytest.raises(ValueError, match="wrong split"):
        fit_normalization.main(
            [
                "--train-shards", str(tmp_path / "shards"),
                "--node-output", str(tmp_path / "node.json"),
                "--edge-output", str(tmp_path / "edge.json"),
            ]
        )
