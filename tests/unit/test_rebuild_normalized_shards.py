from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import rebuild_normalized_shards  # noqa: E402

from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats  # noqa: E402
from hydroswarm.training import ShardedScenarioDataset, write_shards  # noqa: E402
from hydroswarm.training.data import CurriculumStage, ScenarioExample  # noqa: E402

_NODE_WIDTH = len(DEFAULT_FEATURE_SCHEMA.node_features)
_EDGE_WIDTH = len(DEFAULT_FEATURE_SCHEMA.edge_features)


def _examples(count: int, *, split: str, offset: float = 0.0) -> list[ScenarioExample]:
    examples = []
    for index in range(count):
        value = float(index) + offset
        examples.append(
            ScenarioExample(
                scenario_id=f"{split}-{index}",
                network_id="net-a",
                split=split,
                seed=index,
                seed_family=f"family-{split}-{index}",
                stage=CurriculumStage.CLEAN,
                inputs={
                    "node_features": torch.full((3, _NODE_WIDTH), value),
                    "edge_features": torch.full((2, _EDGE_WIDTH), value),
                    "node_mask": torch.ones(3, dtype=torch.bool),
                },
                targets={"source_node": torch.tensor(index % 3)},
            )
        )
    return examples


def _fit_stats(train_examples: list[ScenarioExample]) -> tuple[NormalizationStats, NormalizationStats]:
    import numpy as np

    node_stats = NormalizationStats.fit(
        np.concatenate([e.inputs["node_features"].numpy() for e in train_examples], axis=0),
        DEFAULT_FEATURE_SCHEMA.node_features,
    )
    edge_stats = NormalizationStats.fit(
        np.concatenate([e.inputs["edge_features"].numpy() for e in train_examples], axis=0),
        DEFAULT_FEATURE_SCHEMA.edge_features,
    )
    return node_stats, edge_stats


def _build_corpus(root: Path) -> None:
    train = _examples(6, split="train")
    validation = _examples(3, split="validation", offset=100.0)
    ood = _examples(2, split="development_holdout", offset=200.0)
    write_shards(train, root / "tensors" / "train", shard_size=2)
    write_shards(validation, root / "tensors" / "validation", shard_size=2)
    write_shards(ood, root / "tensors" / "ood-severe_missingness", shard_size=2)


def test_rebuild_applies_normalization_and_preserves_governance_metadata(tmp_path) -> None:
    corpus_dir = tmp_path / "cycle-b2"
    _build_corpus(corpus_dir)

    train_dataset = ShardedScenarioDataset(corpus_dir / "tensors" / "train", expected_split="train")
    train_examples = [train_dataset[i] for i in range(len(train_dataset))]
    node_stats, edge_stats = _fit_stats(train_examples)
    node_path, edge_path = tmp_path / "node.json", tmp_path / "edge.json"
    node_stats.save(node_path)
    edge_stats.save(edge_path)

    before = {
        name: ShardedScenarioDataset(
            corpus_dir / "tensors" / name,
            expected_split="development_holdout" if name.startswith("ood-") else name,
        ).manifest_hash
        for name in ("train", "validation", "ood-severe_missingness")
    }

    exit_code = rebuild_normalized_shards.main(
        [
            "--corpus-dir", str(corpus_dir),
            "--node-normalization", str(node_path),
            "--edge-normalization", str(edge_path),
        ]
    )
    assert exit_code == 0

    for name, expected_hash in before.items():
        expected_split = "development_holdout" if name.startswith("ood-") else name
        # raw tensors/ must survive untouched (needed later to independently
        # recompute normalization stats for the train-only-ownership gate).
        raw_dataset = ShardedScenarioDataset(corpus_dir / "tensors" / name, expected_split=expected_split)
        assert raw_dataset.manifest_hash == expected_hash
        raw_dataset.verify_shard_checksums()

        normalized_dataset = ShardedScenarioDataset(
            corpus_dir / "tensors-normalized" / name, expected_split=expected_split
        )
        assert normalized_dataset.manifest_hash == expected_hash  # governance metadata untouched
        normalized_dataset.verify_shard_checksums()

    raw_train = ShardedScenarioDataset(corpus_dir / "tensors" / "train", expected_split="train")
    rebuilt_train = ShardedScenarioDataset(corpus_dir / "tensors-normalized" / "train", expected_split="train")
    raw_first, normalized_first = raw_train[0], rebuilt_train[0]
    assert not torch.allclose(normalized_first.inputs["node_features"], raw_first.inputs["node_features"])
    assert torch.equal(normalized_first.inputs["node_mask"], raw_first.inputs["node_mask"])  # non-feature untouched
    assert torch.equal(normalized_first.targets["source_node"], raw_first.targets["source_node"])

    report = (corpus_dir / "normalization-rebuild-report.json").read_text(encoding="utf-8")
    assert "node_normalization_sha256" in report
    assert "ood-severe_missingness" in report
    assert "tensors-normalized" in report


def test_rebuild_fails_closed_on_corrupted_shard(tmp_path) -> None:
    corpus_dir = tmp_path / "cycle-b2"
    _build_corpus(corpus_dir)
    train_dataset = ShardedScenarioDataset(corpus_dir / "tensors" / "train", expected_split="train")
    train_examples = [train_dataset[i] for i in range(len(train_dataset))]
    node_stats, edge_stats = _fit_stats(train_examples)
    node_path, edge_path = tmp_path / "node.json", tmp_path / "edge.json"
    node_stats.save(node_path)
    edge_stats.save(edge_path)

    # Windows refuses to overwrite a file that is still memory-mapped
    # elsewhere (POSIX tolerates this); train_dataset's own lazy-loaded
    # safetensors handle for shard-00000 is still open at this point.
    train_dataset.close()

    shard_path = corpus_dir / "tensors" / "train" / "shard-00000.safetensors"
    corrupted = bytearray(shard_path.read_bytes())
    corrupted[-1] ^= 0xFF
    shard_path.write_bytes(bytes(corrupted))

    with pytest.raises(ValueError, match="checksum"):
        rebuild_normalized_shards.main(
            [
                "--corpus-dir", str(corpus_dir),
                "--node-normalization", str(node_path),
                "--edge-normalization", str(edge_path),
            ]
        )


def test_rebuild_refuses_to_target_the_raw_tensors_directory(tmp_path) -> None:
    corpus_dir = tmp_path / "cycle-b2"
    _build_corpus(corpus_dir)
    node_stats, edge_stats = _fit_stats(_examples(3, split="train"))
    node_path, edge_path = tmp_path / "node.json", tmp_path / "edge.json"
    node_stats.save(node_path)
    edge_stats.save(edge_path)

    with pytest.raises(SystemExit, match="raw corpus must survive"):
        rebuild_normalized_shards.main(
            [
                "--corpus-dir", str(corpus_dir),
                "--node-normalization", str(node_path),
                "--edge-normalization", str(edge_path),
                "--output-tensors-dirname", "tensors",
            ]
        )


def test_rebuild_requires_a_train_split(tmp_path) -> None:
    corpus_dir = tmp_path / "cycle-b2"
    write_shards(_examples(3, split="validation"), corpus_dir / "tensors" / "validation", shard_size=2)
    node_path, edge_path = tmp_path / "node.json", tmp_path / "edge.json"
    node_stats, edge_stats = _fit_stats(_examples(3, split="train"))
    node_stats.save(node_path)
    edge_stats.save(edge_path)

    with pytest.raises(SystemExit, match="no train split"):
        rebuild_normalized_shards.main(
            [
                "--corpus-dir", str(corpus_dir),
                "--node-normalization", str(node_path),
                "--edge-normalization", str(edge_path),
            ]
        )
