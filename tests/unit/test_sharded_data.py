from __future__ import annotations

import copy
import json
import pickle

import pytest
import torch

from hydroswarm.training import ShardedScenarioDataset, write_shards
from hydroswarm.training.data import (
    CurriculumStage,
    GovernedScenarioDataset,
    ScenarioExample,
    TopologyMetadata,
    resolve_source_node_id,
    validate_split_isolation,
)


def _examples(count: int, *, split: str = "train", variable_shapes: bool = False):
    examples = []
    for index in range(count):
        nodes = 3 + index if variable_shapes else 3
        examples.append(
            ScenarioExample(
                scenario_id=f"s-{split}-{index}",
                network_id="NetA" if index % 2 == 0 else "NetB",
                split=split,
                seed=index,
                seed_family=f"family-{split}-{index}",
                stage=CurriculumStage(index % len(CurriculumStage)),
                inputs={
                    "node_features": torch.arange(nodes * 2, dtype=torch.float32).reshape(nodes, 2),
                    "node_mask": torch.ones(nodes, dtype=torch.bool),
                },
                targets={"source_node": torch.tensor(index % max(nodes, 1))},
            )
        )
    return examples


def test_write_shards_then_lazy_dataset_round_trips_examples(tmp_path) -> None:
    examples = _examples(5)
    manifest = write_shards(examples, tmp_path / "shards", shard_size=2)
    assert manifest["total_examples"] == 5
    assert len(manifest["shards"]) == 3  # ceil(5/2)

    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    assert len(dataset) == 5
    loaded = {example.scenario_id: example for example in (dataset[i] for i in range(len(dataset)))}
    for original in examples:
        recovered = loaded[original.scenario_id]
        assert torch.equal(recovered.inputs["node_features"], original.inputs["node_features"])
        assert torch.equal(recovered.targets["source_node"], original.targets["source_node"])
        assert recovered.stage == original.stage


def test_dataset_construction_does_not_eagerly_read_any_shard_tensor_data(tmp_path) -> None:
    examples = _examples(4)
    write_shards(examples, tmp_path / "shards", shard_size=2)

    # Corrupt one shard's tensor payload after the manifest/index were written
    # correctly, but before ever calling __getitem__. If construction eagerly
    # deserialized shard contents, this would already raise here.
    shard_path = tmp_path / "shards" / "shard-00001.safetensors"
    with shard_path.open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"\xff\xff\xff\xff")

    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    assert len(dataset) == 4  # metadata-only construction succeeded despite corrupt tensor bytes


def test_index_checksum_mismatch_is_rejected(tmp_path) -> None:
    examples = _examples(3)
    write_shards(examples, tmp_path / "shards", shard_size=5)
    index_path = tmp_path / "shards" / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")  # tamper without touching manifest.json

    with pytest.raises(ValueError, match="checksum"):
        ShardedScenarioDataset(tmp_path / "shards", expected_split="train")


def test_missing_shard_file_is_rejected(tmp_path) -> None:
    examples = _examples(3)
    write_shards(examples, tmp_path / "shards", shard_size=5)
    (tmp_path / "shards" / "shard-00000.safetensors").unlink()

    with pytest.raises(FileNotFoundError):
        ShardedScenarioDataset(tmp_path / "shards", expected_split="train")


def test_verify_shard_checksums_rejects_a_corrupted_shard_that_still_exists(tmp_path) -> None:
    # core-issues.txt repair item 12: a shard that is present but corrupted
    # (truncated, bit-flipped, wrong bytes entirely) must not silently be
    # trusted -- only its existence was ever checked before this fix, even
    # though write_shards has always recorded each shard's real sha256.
    # verify_shard_checksums is a separate, explicit call (not folded into
    # __init__, which must stay metadata-only -- see
    # test_dataset_construction_does_not_eagerly_read_any_shard_tensor_data)
    # that a caller runs once before training.
    examples = _examples(3)
    write_shards(examples, tmp_path / "shards", shard_size=5)
    shard_path = tmp_path / "shards" / "shard-00000.safetensors"
    corrupted = bytearray(shard_path.read_bytes())
    corrupted[-1] ^= 0xFF
    shard_path.write_bytes(bytes(corrupted))

    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    with pytest.raises(ValueError, match="checksum"):
        dataset.verify_shard_checksums()


def test_verify_shard_checksums_accepts_an_intact_dataset(tmp_path) -> None:
    examples = _examples(3)
    write_shards(examples, tmp_path / "shards", shard_size=5)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    dataset.verify_shard_checksums()  # must not raise


def test_wrong_split_examples_are_rejected(tmp_path) -> None:
    examples = _examples(2, split="train")
    write_shards(examples, tmp_path / "shards", shard_size=5)
    with pytest.raises(ValueError, match="wrong split"):
        ShardedScenarioDataset(tmp_path / "shards", expected_split="validation")


def test_random_access_is_order_independent(tmp_path) -> None:
    examples = _examples(6)
    write_shards(examples, tmp_path / "shards", shard_size=3)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")

    for position in (5, 0, 3, 1, 4, 2):
        example = dataset[position]
        assert example.scenario_id == examples[position].scenario_id


def test_iteration_order_is_deterministic_across_dataset_instances(tmp_path) -> None:
    examples = _examples(7)
    write_shards(examples, tmp_path / "shards", shard_size=3)

    first = [example.scenario_id for example in ShardedScenarioDataset(tmp_path / "shards", expected_split="train")]
    second = [example.scenario_id for example in ShardedScenarioDataset(tmp_path / "shards", expected_split="train")]
    assert first == second == [example.scenario_id for example in examples]


def test_variable_graph_sizes_are_preserved_per_example(tmp_path) -> None:
    examples = _examples(4, variable_shapes=True)
    write_shards(examples, tmp_path / "shards", shard_size=2)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    for index, original in enumerate(examples):
        recovered = dataset[index]
        assert recovered.inputs["node_features"].shape == original.inputs["node_features"].shape


def test_manifest_hash_matches_in_memory_dataset_for_same_examples(tmp_path) -> None:
    examples = _examples(5)
    write_shards(examples, tmp_path / "shards", shard_size=2)
    sharded = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    in_memory = GovernedScenarioDataset(examples, expected_split="train")
    assert sharded.manifest_hash == in_memory.manifest_hash


def test_stages_through_filters_lazily_without_reloading_all_tensors(tmp_path) -> None:
    examples = _examples(len(CurriculumStage) * 2)
    write_shards(examples, tmp_path / "shards", shard_size=3)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")

    filtered = dataset.stages_through(CurriculumStage.DEGRADED)
    assert len(filtered) < len(dataset)
    for index in range(len(filtered)):
        assert filtered[index].stage <= CurriculumStage.DEGRADED


def test_leakage_checks_reject_duplicate_scenario_ids_across_splits(tmp_path) -> None:
    train_examples = _examples(3, split="train")
    validation_examples = [
        ScenarioExample(
            scenario_id=train_examples[0].scenario_id,  # deliberate collision
            network_id="NetA",
            split="validation",
            seed=99,
            seed_family="leaked-family",
            stage=CurriculumStage.CLEAN,
            inputs={"node_features": torch.zeros(3, 2)},
            targets={"source_node": torch.tensor(0)},
        )
    ]
    write_shards(train_examples, tmp_path / "train-shards", shard_size=5)
    write_shards(validation_examples, tmp_path / "val-shards", shard_size=5)
    train_dataset = ShardedScenarioDataset(tmp_path / "train-shards", expected_split="train")
    validation_dataset = ShardedScenarioDataset(tmp_path / "val-shards", expected_split="validation")

    with pytest.raises(ValueError, match="scenario leakage"):
        validate_split_isolation(train_dataset, validation_dataset)


def test_dataset_is_picklable_without_open_shard_handles_for_worker_processes(tmp_path) -> None:
    examples = _examples(4)
    write_shards(examples, tmp_path / "shards", shard_size=2)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")

    # Touch every example so the parent process opens (and caches) shard handles.
    for index in range(len(dataset)):
        dataset[index]
    assert dataset._shard_handles  # sanity: handles were actually opened

    pickled = pickle.loads(pickle.dumps(dataset))
    assert pickled._shard_handles == {}
    # The unpickled copy must still be able to lazily reopen shards on demand.
    for index in range(len(pickled)):
        assert pickled[index].scenario_id == examples[index].scenario_id


def test_deepcopy_also_drops_open_shard_handles(tmp_path) -> None:
    examples = _examples(2)
    write_shards(examples, tmp_path / "shards", shard_size=2)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    dataset[0]
    duplicate = copy.deepcopy(dataset)
    assert duplicate._shard_handles == {}


def test_dataloader_with_multiple_workers_reads_every_example_exactly_once(tmp_path) -> None:
    from torch.utils.data import DataLoader

    examples = _examples(12)
    write_shards(examples, tmp_path / "shards", shard_size=4)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")

    loader = DataLoader(dataset, batch_size=None, num_workers=2, shuffle=False)
    seen = [example.scenario_id for example in loader]
    assert sorted(seen) == sorted(example.scenario_id for example in examples)
    assert len(seen) == len(examples)


def test_write_shards_rejects_empty_input(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_shards([], tmp_path / "shards")


def test_manifest_and_index_are_valid_json(tmp_path) -> None:
    examples = _examples(3)
    write_shards(examples, tmp_path / "shards", shard_size=2)
    manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    for line in (tmp_path / "shards" / "index.jsonl").read_text().splitlines():
        json.loads(line)


def test_topology_metadata_round_trips_through_sharded_storage(tmp_path) -> None:
    topology = TopologyMetadata(
        topology_hash="topo-shard",
        network_hash="net-shard",
        node_ids=("J1", "J2", "J3"),
        edge_ids=(("J1", "J2"), ("J2", "J3")),
        source_candidate_ids=("J1", "J2", "J3"),
        hydraulic_state_hash="state-shard",
        signature_library_hash="sig-shard",
        target_schema_version="targets_v1",
        feature_schema_version="hydroswarm-features-v2",
    )
    example = ScenarioExample(
        scenario_id="topo-example",
        network_id="net-shard",
        split="train",
        seed=0,
        seed_family="fam-topo",
        stage=CurriculumStage.CLEAN,
        inputs={"node_features": torch.zeros(3, 2)},
        targets={"source_node": torch.tensor(2)},
        topology=topology,
    )
    write_shards([example], tmp_path / "shards", shard_size=5)
    dataset = ShardedScenarioDataset(tmp_path / "shards", expected_split="train")
    recovered = dataset[0]
    assert recovered.topology == topology
    assert resolve_source_node_id(recovered) == "J3"
