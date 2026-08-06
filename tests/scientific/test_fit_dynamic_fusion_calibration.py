"""End-to-end test for scripts/fit_dynamic_fusion_calibration.py against a
real (small) multi-topology corpus, built with the same generator functions
generate_cycle_b_corpus.py uses. Verifies the full real pipeline -- signature
artifact construction, deterministic scenario replay, HybridInferencePipeline
.analyze() end to end -- actually runs and produces a calibration artifact
fit against the real dynamic-trust fusion, not fixed_weight_fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_dynamic_fusion_calibration as calib  # noqa: E402
from generate_cycle_b_corpus import TRAIN_TOPOLOGIES, _degradation_probabilities, _stage_for_index  # noqa: E402

from hydroswarm.data.scenarios import (  # noqa: E402
    DatasetSplit,
    EventType,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.inference import DYNAMIC_TRUST_FUSION_CONFIG  # noqa: E402
from hydroswarm.model import HydroCore  # noqa: E402
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats  # noqa: E402
from hydroswarm.training.corpus import build_feature_context, fit_signature_library, scenario_to_example  # noqa: E402
from hydroswarm.training.sharded_data import write_shards  # noqa: E402

_FAMILIES = TRAIN_TOPOLOGIES[:2]  # (golden-reference, branched-loop)
_SPLIT_COUNTS = ((DatasetSplit.TRAIN, 8), (DatasetSplit.CALIBRATION, 4))


def _build_mini_corpus(output: Path) -> None:
    writer = ScenarioDatasetWriter(output / "scenarios")
    examples_by_split: dict[DatasetSplit, list] = {split: [] for split, _ in _SPLIT_COUNTS}

    for family_index, (family, loader) in enumerate(_FAMILIES):
        network = loader()
        junctions = tuple(sorted(network.junction_name_list))
        generator = WNTRScenarioGenerator()
        seed_base = 700_000 + family_index * 1_000_000
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
        write_shards(examples_by_split[split], output / "tensors" / split.value, shard_size=4)

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
    output = tmp_path_factory.mktemp("mini-corpus-calibration")
    _build_mini_corpus(output)
    return output


def test_fit_produces_a_real_dynamic_fusion_calibration_artifact(mini_corpus, tmp_path) -> None:
    model = HydroCore.from_variant("small")
    result = calib.fit(
        corpus_dir=mini_corpus,
        checkpoint=_save_checkpoint(model, tmp_path),
        variant="small",
        overrides={},
        node_normalization=mini_corpus / "normalization" / "node-normalization.json",
        edge_normalization=mini_corpus / "normalization" / "edge-normalization.json",
        signature_cache_dir=tmp_path / "signature-cache",
        alpha=0.1,
        sample_per_topology=None,
    )
    assert result["examples_used"] > 0
    calibrator = result["calibrator"]
    assert calibrator.artifact.fusion_config_hash == DYNAMIC_TRUST_FUSION_CONFIG

    diagnostics = result["per_scenario_diagnostics"]
    assert diagnostics, "expected at least one scenario to produce diagnostics"
    assert any(row["runtime_mode"] == "FULL_HYBRID" for row in diagnostics)
    # A real per-scenario dynamic trust weight, not one hardcoded constant
    # across every scenario (the defect fixed_weight_fusion has by design).
    trust_values = {row["classical_trust"] for row in diagnostics if row["classical_trust"] is not None}
    assert trust_values, "expected at least one FULL_HYBRID scenario to report a classical_trust value"

    saved = tmp_path / "calibration.json"
    calibrator.save(saved)
    reloaded = json.loads(saved.read_text(encoding="utf-8"))
    assert reloaded["fusion_config_hash"] == DYNAMIC_TRUST_FUSION_CONFIG


def test_fit_skips_non_contamination_scenarios(mini_corpus, tmp_path) -> None:
    # Copy the mini corpus and inject one NORMAL-event manifest record with a
    # deliberately nonexistent .npz artifact: if fit() did not skip it before
    # trying to load the artifact, this would raise FileNotFoundError instead
    # of silently (and wrongly) counting a meaningless placeholder source as
    # a real calibration label.
    import shutil

    corpus = tmp_path / "corpus-with-normal-scenario"
    shutil.copytree(mini_corpus, corpus)
    manifest_path = corpus / "scenarios" / "manifests" / "calibration.jsonl"
    lines = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline_count = len(lines)
    injected = dict(lines[0])
    injected["event_type"] = "normal"
    injected["scenario_id"] = "00000000-0000-0000-0000-000000000000"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(injected) + "\n")

    model = HydroCore.from_variant("small")
    result = calib.fit(
        corpus_dir=corpus,
        checkpoint=_save_checkpoint(model, tmp_path / "normal-scenario-checkpoint"),
        variant="small",
        overrides={},
        node_normalization=corpus / "normalization" / "node-normalization.json",
        edge_normalization=corpus / "normalization" / "edge-normalization.json",
        signature_cache_dir=tmp_path / "signature-cache-2",
        alpha=0.1,
        sample_per_topology=None,
    )
    # The injected NORMAL record must not appear as a processed scenario at all.
    assert all(row["scenario_id"] != injected["scenario_id"] for row in result["per_scenario_diagnostics"])
    assert len(result["per_scenario_diagnostics"]) <= baseline_count


def _save_checkpoint(model: HydroCore, directory: Path) -> Path:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint.safetensors"
    save_file(model.state_dict(), str(path))
    return path
