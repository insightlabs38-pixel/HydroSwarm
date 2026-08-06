"""End-to-end tests for scripts/run_corpus_gates.py against a real (small)
multi-topology corpus, built with the exact same generator functions
generate_cycle_b_corpus.py uses -- not synthetic fixtures -- so a bug in the
gate script's assumptions about real topology hashes, real replay data, or
real signature libraries is caught here rather than only against the full
Cycle B corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_corpus_gates  # noqa: E402
from generate_cycle_b_corpus import TRAIN_TOPOLOGIES, _degradation_probabilities, _stage_for_index  # noqa: E402

import numpy as np  # noqa: E402

from hydroswarm.data.scenarios import (  # noqa: E402
    DatasetSplit,
    EventType,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, fit_signature_library, scenario_to_example  # noqa: E402
from hydroswarm.training.label_audit import audit_corpus  # noqa: E402
from hydroswarm.training.sharded_data import write_shards  # noqa: E402

_FAMILIES = TRAIN_TOPOLOGIES[:2]  # (golden-reference, branched-loop): cheap, real, distinct node counts
_SPLIT_COUNTS = ((DatasetSplit.TRAIN, 8), (DatasetSplit.VALIDATION, 2), (DatasetSplit.CALIBRATION, 2))


def _build_mini_corpus(output: Path) -> None:
    writer = ScenarioDatasetWriter(output / "scenarios")
    examples_by_split: dict[DatasetSplit, list] = {split: [] for split, _ in _SPLIT_COUNTS}

    for family_index, (family, loader) in enumerate(_FAMILIES):
        network = loader()
        junctions = tuple(sorted(network.junction_name_list))
        generator = WNTRScenarioGenerator()
        seed_base = 500_000 + family_index * 1_000_000
        by_split_for_family: dict[DatasetSplit, list] = {split: [] for split, _ in _SPLIT_COUNTS}
        for split_index, (split, count) in enumerate(_SPLIT_COUNTS):
            for index in range(count):
                stage = _stage_for_index(index)
                scenario, randomized_network = generator.generate_with_network(
                    network,
                    ScenarioGenerationConfig(
                        seed=seed_base + split_index * 100_000 + index * 100,
                        network_id=family,
                        network_family=family,
                        split=split,
                        stage=stage,
                        event_type=EventType.CONTAMINATION,
                        source_node=junctions[index % len(junctions)],
                        sensor_count=min(3, len(junctions)),
                        pipe_outage_probability=0.0,
                        **_degradation_probabilities(stage),
                    ),
                )
                writer.write(scenario)
                by_split_for_family[split].append((scenario, randomized_network))

        train_scenarios = [scenario for scenario, _network in by_split_for_family[DatasetSplit.TRAIN]]
        library = fit_signature_library(train_scenarios, junctions)
        for split, pairs in by_split_for_family.items():
            for scenario, randomized_network in pairs:
                context = build_feature_context(randomized_network)
                example = scenario_to_example(scenario, randomized_network, library, feature_context=context)
                examples_by_split[split].append(example)

    for split, _count in _SPLIT_COUNTS:
        write_shards(examples_by_split[split], output / "tensors" / split.value, shard_size=3)

    audit = audit_corpus(
        {split.value: examples for split, examples in examples_by_split.items()},
        decision_splits=("train", "validation", "calibration"),
    )
    (output / "label-audit.json").write_text(__import__("json").dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    train_examples = examples_by_split[DatasetSplit.TRAIN]
    node_stats = NormalizationStats.fit(
        np.concatenate([e.inputs["node_features"].numpy() for e in train_examples], axis=0),
        DEFAULT_FEATURE_SCHEMA.node_features,
    )
    edge_stats = NormalizationStats.fit(
        np.concatenate([e.inputs["edge_features"].numpy() for e in train_examples], axis=0),
        DEFAULT_FEATURE_SCHEMA.edge_features,
    )
    (output / "normalization").mkdir(parents=True, exist_ok=True)
    node_stats.save(output / "normalization" / "node-normalization.json")
    edge_stats.save(output / "normalization" / "edge-normalization.json")


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("mini-corpus")
    _build_mini_corpus(output)
    return output


def test_all_gates_pass_on_a_correctly_generated_real_corpus(mini_corpus) -> None:
    exit_code = run_corpus_gates.main(["--corpus-dir", str(mini_corpus), "--replay-sample-per-split", "2"])
    assert exit_code == 0
    report = __import__("json").loads((mini_corpus / "corpus-gates-report.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "passed"
    for name, result in report["gates"].items():
        assert result["status"] == "passed", (name, result)


def test_shard_checksum_gate_fails_closed_on_corruption(mini_corpus, tmp_path) -> None:
    import shutil

    corrupt = tmp_path / "corrupt-corpus"
    shutil.copytree(mini_corpus, corrupt)
    shard_path = corrupt / "tensors" / "train" / "shard-00000.safetensors"
    corrupted = bytearray(shard_path.read_bytes())
    corrupted[-1] ^= 0xFF
    shard_path.write_bytes(bytes(corrupted))

    exit_code = run_corpus_gates.main(["--corpus-dir", str(corrupt)])
    assert exit_code == 1
    report = __import__("json").loads((corrupt / "corpus-gates-report.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "failed"
    assert report["gates"]["shard_checksums"]["status"] in {"failed", "error"}


def test_topology_provenance_gate_fails_closed_on_tampered_hash(mini_corpus, tmp_path) -> None:
    import dataclasses
    import shutil

    from hydroswarm.training import ShardedScenarioDataset
    from hydroswarm.training.sharded_data import write_shards as _write_shards

    tampered = tmp_path / "tampered-corpus"
    shutil.copytree(mini_corpus, tampered)
    train_dir = tampered / "tensors" / "train"
    dataset = ShardedScenarioDataset(train_dir, expected_split="train")
    examples = [dataset[i] for i in range(len(dataset))]
    bad_topology = dataclasses.replace(examples[0].topology, topology_hash="0" * 64)
    examples[0] = dataclasses.replace(examples[0], topology=bad_topology)
    _write_shards(examples, train_dir, shard_size=3)

    exit_code = run_corpus_gates.main(["--corpus-dir", str(tampered)])
    assert exit_code == 1
    report = __import__("json").loads((tampered / "corpus-gates-report.json").read_text(encoding="utf-8"))
    assert report["gates"]["topology_provenance"]["status"] in {"failed", "error"}


def test_normalization_ownership_gate_fails_closed_on_wrong_artifact(mini_corpus, tmp_path) -> None:
    import shutil

    wrong = tmp_path / "wrong-normalization-corpus"
    shutil.copytree(mini_corpus, wrong)
    fabricated = NormalizationStats.fit(
        np.ones((5, len(DEFAULT_FEATURE_SCHEMA.node_features)), dtype=np.float32) * 999.0,
        DEFAULT_FEATURE_SCHEMA.node_features,
    )
    fabricated.save(wrong / "normalization" / "node-normalization.json")

    exit_code = run_corpus_gates.main(["--corpus-dir", str(wrong)])
    assert exit_code == 1
    report = __import__("json").loads((wrong / "corpus-gates-report.json").read_text(encoding="utf-8"))
    assert report["gates"]["normalization_ownership"]["status"] == "failed"


def test_deterministic_replay_gate_tolerates_hash_only_drift(mini_corpus, tmp_path) -> None:
    """artifact_sha256 hashes raw float bytes, which is not portable across
    CPU architectures (IEEE-754 leaves signed-zero results from ops like
    np.maximum(0.0, x) implementation-defined). Corrupting only the recorded
    hash -- leaving the actual arrays untouched -- must not fail the gate;
    the semantic array-equality check is the real determinism criterion.
    Discovered replaying an x86-generated corpus on an aarch64 host."""
    import json
    import shutil

    drifted = tmp_path / "hash-drift-corpus"
    shutil.copytree(mini_corpus, drifted)
    manifest_path = drifted / "scenarios" / "manifests" / "train.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["artifact_sha256"] = "0" * 64  # simulate cross-architecture hash drift, arrays untouched
    lines[0] = json.dumps(first)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exit_code = run_corpus_gates.main(["--corpus-dir", str(drifted), "--replay-sample-per-split", "20"])
    assert exit_code == 0
    report = json.loads((drifted / "corpus-gates-report.json").read_text(encoding="utf-8"))
    assert report["gates"]["deterministic_replay"]["status"] == "passed"
    assert first["scenario_id"] in report["gates"]["deterministic_replay"]["artifact_sha256_drift_tolerated"]


def test_deterministic_replay_gate_fails_closed_on_tampered_artifact(mini_corpus, tmp_path) -> None:
    import json
    import shutil

    tampered = tmp_path / "tampered-replay-corpus"
    shutil.copytree(mini_corpus, tampered)
    manifest_path = tampered / "scenarios" / "manifests" / "train.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    artifact_path = tampered / "scenarios" / "train" / f"{first['scenario_id']}.npz"
    with np.load(artifact_path) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    payload["observed_concentration"] = payload["observed_concentration"] + 1.0
    np.savez_compressed(artifact_path, **payload)

    exit_code = run_corpus_gates.main(["--corpus-dir", str(tampered), "--replay-sample-per-split", "20"])
    assert exit_code == 1
    report = json.loads((tampered / "corpus-gates-report.json").read_text(encoding="utf-8"))
    assert report["gates"]["deterministic_replay"]["status"] == "failed"
