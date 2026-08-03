"""Evaluate a fixed neural model on an independent EPANET topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import wntr

from hydroswarm.calibration import SplitConformalCalibrator
from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.inference import OODDetector, OODReference
from hydroswarm.model import HydroCore
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training import GovernedScenarioDataset
from hydroswarm.training.corpus import (
    build_feature_context,
    fit_signature_library,
    scenario_to_example,
)

from evaluate_learning import _classification_metrics, _fuse, _model_report, _predict


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=70)
    args = parser.parse_args()

    network = wntr.network.WaterNetworkModel(str(args.network))
    source_nodes = tuple(sorted(network.junction_name_list))
    if len(source_nodes) < 2:
        raise SystemExit("independent topology requires multiple junctions")
    generator = WNTRScenarioGenerator()
    signature_scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=810_000 + index * 100,
                network_id="branched-loop-signature",
                network_family="branched-loop",
                split=DatasetSplit.TRAIN,
                stage=CurriculumStage.CLEAN,
                source_node=source,
                sensor_count=len(source_nodes),
                pipe_outage_probability=0.0,
            ),
        )
        for index, source in enumerate(source_nodes)
    ]
    library = fit_signature_library(signature_scenarios, source_nodes)
    stages = tuple(CurriculumStage)
    evaluation_scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=910_000 + index * 100,
                network_id="branched-loop-transfer",
                network_family="branched-loop",
                split=DatasetSplit.TEST,
                stage=stages[index % len(stages)],
                source_node=source_nodes[index % len(source_nodes)],
                sensor_count=min(5, len(source_nodes)),
                pipe_outage_probability=0.0,
                missing_probability=0.08,
                frozen_probability=0.05,
                communication_outage_probability=0.05,
            ),
        )
        for index in range(args.examples)
    ]
    context = build_feature_context(network)
    examples = [
        scenario_to_example(scenario, network, library, feature_context=context)
        for scenario in evaluation_scenarios
    ]
    dataset = GovernedScenarioDataset(examples, expected_split="test")
    model = HydroCore.from_variant("medium")
    model.load_state_dict(load_file(args.model, device="cpu"), strict=True)
    predictions, timing = _predict(model, dataset, batch_size=8)
    labels = predictions["label_source"].astype(int)
    classical = predictions["classical"]
    neural = predictions["source"]
    hybrid = _fuse(classical, neural)

    trained_topology_hash = network_sha256(build_wntr_network())
    transfer_topology_hash = network_sha256(network)
    detector = OODDetector(
        OODReference(
            minimum_nodes=6,
            maximum_nodes=6,
            validated_network_hashes=(trained_topology_hash,),
        )
    )
    ood_level = detector.topology_level(
        node_count=len(network.node_name_list), network_hash=transfer_topology_hash
    )
    novelty = detector.topology_novelty(
        node_count=len(network.node_name_list), network_hash=transfer_topology_hash
    )
    calibrator = SplitConformalCalibrator.load(args.calibration)
    candidate_sets = [
        calibrator.candidate_set(
            hybrid[index],
            condition=examples[index].stage.name,
            network_id=examples[index].network_id,
            ood_level=ood_level.value,
        )
        for index in range(len(examples))
    ]
    coverage = float(np.mean([
        int(labels[index]) in selected for index, selected in enumerate(candidate_sets)
    ]))
    report = {
        "schema_version": 1,
        "experiment": "independent_topology_transfer",
        "neural_training_topologies": 1,
        "neural_finetuning_on_transfer_topology": False,
        "classical_signature_templates_fit_on_transfer_topology": True,
        "examples": len(dataset),
        "bootstrap_samples": 2_000,
        "network": {
            "path": str(args.network),
            "sha256": file_hash(args.network),
            "trained_topology_hash": trained_topology_hash,
            "transfer_topology_hash": transfer_topology_hash,
            "topology_hashes_differ": trained_topology_hash != transfer_topology_hash,
            "junctions": len(source_nodes),
            "links": len(network.link_name_list),
        },
        "results": {
            "classical": _classification_metrics(classical, labels),
            "hydrocore_m_neural": _model_report(predictions),
            "hydrocore_m_hybrid": _classification_metrics(hybrid, labels),
        },
        "ood": {
            "state": ood_level.value,
            "network_novelty": novelty,
            "planning_suppressed": ood_level.value != "NORMAL",
        },
        "conformal": {
            "coverage": coverage,
            "mean_candidate_set_size": float(np.mean([len(item) for item in candidate_sets])),
            "calibration_origin": "current-topology calibration split only",
        },
        "timing": timing,
        "limitations": [
            "This is a small synthetic transfer experiment, not field validation.",
            "The classical signature library uses simulator templates from the new topology; neural weights do not.",
            "CAUTION suppresses planning even when localization is correct.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
