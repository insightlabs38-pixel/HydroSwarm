"""core-issues2.txt Phase 7/8 bridge: merging trajectory targets into tensor shards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_trajectory_corpus  # noqa: E402
import merge_trajectory_targets  # noqa: E402

from test_run_corpus_gates import _build_mini_corpus  # noqa: E402

from hydroswarm.training.sharded_data import ShardedScenarioDataset  # noqa: E402


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("mini-corpus-for-merge")
    _build_mini_corpus(output)
    return output


@pytest.fixture(scope="module")
def trajectory_jsonl(mini_corpus, tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("trajectories-for-merge")
    exit_code = generate_trajectory_corpus.main(
        [
            "--corpus-dir", str(mini_corpus),
            "--output", str(output),
            "--split", "train",
            "--signature-cache-dir", str(output / "sigcache"),
        ]
    )
    assert exit_code == 0
    return output / "train.jsonl"


#: 822 real WNTR/EPANET verifications (audited call count) -- see
#: pyproject.toml's full_simulation marker docstring.
@pytest.mark.full_simulation
def test_every_example_is_matched_and_enriched(mini_corpus, trajectory_jsonl, tmp_path) -> None:
    output = tmp_path / "merged"
    result = merge_trajectory_targets.merge(
        mini_corpus / "tensors" / "train", trajectory_jsonl, output, expected_split="train"
    )
    assert result["examples_unchanged"] == 0
    assert result["examples_enriched"] == result["examples_total"] > 0


def test_merged_shards_contain_both_old_and_new_targets(mini_corpus, trajectory_jsonl, tmp_path) -> None:
    output = tmp_path / "merged"
    merge_trajectory_targets.merge(
        mini_corpus / "tensors" / "train", trajectory_jsonl, output, expected_split="train"
    )
    dataset = ShardedScenarioDataset(output, expected_split="train")
    assert len(dataset) > 0
    for position in range(len(dataset)):
        example = dataset[position]
        # original sentinel-category targets survive
        assert "source_node" in example.targets
        assert "sensor_fault" in example.targets
        # new Phase 4-6 targets are attached
        assert "ood_class" in example.targets
        assert "next_step" in example.targets
        assert "sensor_reconstruction" in example.targets
        assert "travel_time" in example.targets


def test_merged_shards_are_a_valid_reloadable_dataset(mini_corpus, trajectory_jsonl, tmp_path) -> None:
    output = tmp_path / "merged"
    merge_trajectory_targets.merge(
        mini_corpus / "tensors" / "train", trajectory_jsonl, output, expected_split="train"
    )
    dataset = ShardedScenarioDataset(output, expected_split="train")
    dataset.verify_shard_checksums()  # must not raise


def test_source_shard_directory_is_never_modified(mini_corpus, trajectory_jsonl, tmp_path) -> None:
    source = mini_corpus / "tensors" / "train"
    before = sorted(p.name for p in source.iterdir())
    before_bytes = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}

    output = tmp_path / "merged"
    merge_trajectory_targets.merge(source, trajectory_jsonl, output, expected_split="train")

    after = sorted(p.name for p in source.iterdir())
    assert before == after
    for name, content in before_bytes.items():
        assert (source / name).read_bytes() == content
