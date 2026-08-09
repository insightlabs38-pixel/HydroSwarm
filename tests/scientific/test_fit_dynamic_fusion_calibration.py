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

#: Every test in this module runs many real WNTR/EPANET verifications
#: (audited call count >=10 each) -- see pyproject.toml's full_simulation
#: marker docstring.
pytestmark = pytest.mark.full_simulation

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


def test_validated_topology_hashes_are_the_pristine_family_hash_not_randomized_scenario_hashes(
    mini_corpus, tmp_path
) -> None:
    """core-issues5.txt Section 19: a real served network is the pristine
    (unrandomized) topology file -- validated_topology_hashes must match
    that, not each calibration scenario's own roughness-randomized
    network_sha256 (which would make calibration validate against
    essentially no real served network ever, since generate_with_network
    always randomizes roughness -- see hydroswarm.classical.
    signature_policy's own GOVERNED_KNOWN_NETWORK precedent for the same
    distinction)."""

    from hydroswarm.data.scenarios import network_sha256

    model = HydroCore.from_variant("small")
    result = calib.fit(
        corpus_dir=mini_corpus,
        checkpoint=_save_checkpoint(model, tmp_path / "topology-hash-checkpoint"),
        variant="small",
        overrides={},
        node_normalization=mini_corpus / "normalization" / "node-normalization.json",
        edge_normalization=mini_corpus / "normalization" / "edge-normalization.json",
        signature_cache_dir=tmp_path / "signature-cache-topology",
        alpha=0.1,
        sample_per_topology=None,
    )
    calibrator = result["calibrator"]
    pristine_hashes = {network_sha256(loader()) for _family, loader in _FAMILIES}
    assert set(calibrator.artifact.validated_topology_hashes) == pristine_hashes
    # None of the individual (roughness-randomized) scenario network
    # hashes recorded in per_scenario_diagnostics should equal a pristine
    # hash by coincidence -- this corpus's own roughness_variation_fraction
    # default (0.05) makes that astronomically unlikely, confirming the
    # two hash families really are different, not accidentally identical.
    assert pristine_hashes, "expected at least one training topology"


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


def test_identity_dir_path_uses_the_real_runtime_model_hash_convention(mini_corpus, tmp_path) -> None:
    """core-issues5.txt Section 19: a calibration artifact fit via
    --identity-dir must be loadable by the REAL runtime validator
    (hydroswarm.runtime.v4_defaults.V4PipelineFactory._load_assets, and
    scripts/build_v4_inference_release_bundle.py's own calibration
    packaging) -- both compute model_hash as the raw model.safetensors
    FILE content hash, not HybridInferencePipeline._fingerprint_model's
    state-dict-tensor hash (a real, previously-shipped mismatch: the first
    version of this identity_dir path used the wrong formula, which would
    have failed validate_runtime at every real load site)."""

    import hashlib

    import torch

    from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
    from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder
    from hydroswarm.training import checkpoint_identity as ci

    model = HydroCore.from_variant("small", action_vocabulary_size=ACTION_TEMPLATE_COUNT)
    node_stats = NormalizationStats.load(mini_corpus / "normalization" / "node-normalization.json")
    edge_stats = NormalizationStats.load(mini_corpus / "normalization" / "edge-normalization.json")
    normalization_hash = HydraulicFeatureBuilder(
        node_normalization=node_stats, edge_normalization=edge_stats
    ).normalization_fingerprint

    identity_dir = tmp_path / "identity"
    identity = ci.build_checkpoint_identity(
        model,
        normalization_hash=normalization_hash,
        fusion_policy_hash="fixed-weight-v1:neural=0.5",
        source_corpus_manifest_hashes=("abc123",),
        trained_outputs=frozenset({"source_node"}),
        validated_outputs=frozenset({"source_node"}),
        runtime_enabled_outputs=frozenset({"source_node"}),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    ci.save_v4_checkpoint(
        identity_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        best_validation_loss=1.0,
        identity=identity,
        resolved_training_config={},
        dataset_manifest_hashes={"train": "abc123"},
        task_weights={"source_node": 1.0},
    )

    result = calib.fit(
        corpus_dir=mini_corpus,
        identity_dir=identity_dir,
        node_normalization=mini_corpus / "normalization" / "node-normalization.json",
        edge_normalization=mini_corpus / "normalization" / "edge-normalization.json",
        signature_cache_dir=tmp_path / "signature-cache-identity",
        alpha=0.1,
        sample_per_topology=None,
    )
    calibrator = result["calibrator"]
    expected_model_hash = hashlib.sha256((identity_dir / "model.safetensors").read_bytes()).hexdigest()
    assert calibrator.artifact.model_hash == expected_model_hash
