"""Build a governed multi-regime WNTR corpus and HydroCore tensor manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    GeneratedScenario,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    SplitPlanner,
    WNTRScenarioGenerator,
)
from hydroswarm.simulation.network import NetworkDefinition, build_wntr_network
from hydroswarm.training.corpus import (
    fit_signature_library,
    scenario_to_example,
    signature_metadata,
    write_tensor_manifest,
)


TRAINING_FAMILIES = (
    NetworkDefinition(name="reference-a", reservoir_head_m=132.0, tank_initial_level_m=8.0),
    NetworkDefinition(name="low-head-b", reservoir_head_m=126.0, tank_initial_level_m=6.0),
    NetworkDefinition(name="high-storage-c", reservoir_head_m=136.0, tank_initial_level_m=11.0),
    NetworkDefinition(name="low-storage-d", reservoir_head_m=130.0, tank_initial_level_m=4.0),
)
HELD_OUT_FAMILY = NetworkDefinition(
    name="held-out-high-head", reservoir_head_m=142.0, tank_initial_level_m=12.0
)
STAGES = tuple(CurriculumStage)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate_split(
    *,
    split: DatasetSplit,
    count: int,
    seed_base: int,
    writer: ScenarioDatasetWriter,
) -> list[tuple[GeneratedScenario, Any]]:
    held_out = (HELD_OUT_FAMILY.name,)
    generator = WNTRScenarioGenerator(SplitPlanner(held_out_network_families=held_out))
    definitions = (HELD_OUT_FAMILY,) if split == DatasetSplit.TEST else TRAINING_FAMILIES
    generated: list[tuple[GeneratedScenario, Any]] = []
    junctions = tuple(sorted(build_wntr_network().junction_name_list))
    for index in range(count):
        definition = definitions[index % len(definitions)]
        network = build_wntr_network(definition)
        seed = seed_base + index * 100
        stage = STAGES[index % len(STAGES)]
        degraded = stage in {
            CurriculumStage.DEGRADED,
            CurriculumStage.SHIFT,
            CurriculumStage.ADVERSARIAL,
        }
        shifted = stage in {CurriculumStage.SHIFT, CurriculumStage.ADVERSARIAL}
        scenario = generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=seed,
                network_id=definition.name,
                network_family=definition.name,
                split=split,
                stage=stage,
                source_node=junctions[index % len(junctions)],
                sensor_count=3,
                pipe_outage_probability=0.0,
                missing_probability=0.08 if stage != CurriculumStage.CLEAN else 0.0,
                frozen_probability=0.06 if degraded else 0.01,
                communication_outage_probability=0.06 if degraded else 0.01,
                unit_mismatch_probability=0.02 if shifted else 0.0,
            ),
        )
        writer.write(scenario)
        generated.append((scenario, network))
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/learning-v1"))
    parser.add_argument("--train-count", type=int, default=320)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=41_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts = {
        DatasetSplit.TRAIN: args.train_count,
        DatasetSplit.VALIDATION: args.validation_count,
        DatasetSplit.CALIBRATION: args.calibration_count,
        DatasetSplit.TEST: args.test_count,
    }
    if any(value < 4 for value in counts.values()):
        raise SystemExit("every split requires at least four source-balanced scenarios")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty corpus directory: {args.output}")
    started = time.perf_counter()
    writer = ScenarioDatasetWriter(args.output / "scenarios")
    splits: dict[DatasetSplit, list[tuple[GeneratedScenario, Any]]] = {}
    for offset, (split, count) in enumerate(counts.items()):
        splits[split] = _generate_split(
            split=split,
            count=count,
            seed_base=args.seed + offset * 1_000_000,
            writer=writer,
        )
    training_scenarios = [item[0] for item in splits[DatasetSplit.TRAIN]]
    node_ids = tuple(sorted(splits[DatasetSplit.TRAIN][0][1].junction_name_list))
    library = fit_signature_library(training_scenarios, node_ids)
    signature_path = args.output / "signatures" / "classical-signatures.json"
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(
        json.dumps(signature_metadata(library), indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_hashes: dict[str, str] = {}
    all_examples = []
    for split, records in splits.items():
        examples = [scenario_to_example(scenario, network, library) for scenario, network in records]
        metadata = {
            example.scenario_id: {
                "network_family": scenario.manifest.network_family,
                "network_sha256": scenario.manifest.network_sha256,
                "artifact_sha256": scenario.manifest.artifact_sha256,
                "replay_sha256": scenario.manifest.replay_sha256,
                "simulator_version": scenario.manifest.simulator_version,
                "generator_version": scenario.manifest.generator_version,
                "source_node": scenario.manifest.incident.source_nodes[0],
            }
            for example, (scenario, _) in zip(examples, records, strict=True)
        }
        manifest_path = args.output / "tensors" / f"{split.value}.jsonl"
        manifest_hashes[split.value] = write_tensor_manifest(
            manifest_path, examples, metadata=metadata
        )
        all_examples.extend(examples)
    stage_counts = Counter(example.stage.name for example in all_examples)
    source_counts = Counter(
        str(example.targets["source_node"].item()) for example in all_examples
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_seconds": time.perf_counter() - started,
        "scenario_count": len(all_examples),
        "split_counts": {split.value: count for split, count in counts.items()},
        "network_families": [item.name for item in TRAINING_FAMILIES] + [HELD_OUT_FAMILY.name],
        "held_out_network_family": HELD_OUT_FAMILY.name,
        "topology_families": 1,
        "hydraulic_regimes": len(TRAINING_FAMILIES) + 1,
        "curriculum_stage_counts": dict(sorted(stage_counts.items())),
        "source_index_counts": dict(sorted(source_counts.items())),
        "quarantined_scenarios": 0,
        "replay_validation_rate": 1.0,
        "storage_bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
        "signature_library_sha256": library.manifest_hash,
        "signature_file_sha256": _sha256(signature_path),
        "tensor_manifest_sha256": manifest_hashes,
        "limitations": [
            "The five named families are hydraulic regimes of one reference topology, not five independent utility topologies.",
            "Held-out results therefore measure hydraulic-family shift, not topology-family transfer or field performance.",
        ],
    }
    report_path = args.output / "dataset-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
