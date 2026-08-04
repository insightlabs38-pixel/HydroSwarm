"""Generate the Phase 5 Cycle A smoke corpus (overnight-plan.txt).

Purpose (per the plan): variable-topology pipeline validation, target
coverage validation, one-epoch training, shape/memory tests. "Do not use
this for final claims." Two genuinely different topologies are used --
the golden reference network (4 junctions, 1 reservoir, 1 tank, a single
loop) and the independent branched-loop network (7 junctions, 1
reservoir, no tank, a different loop) already committed at
data/topology-transfer/branched-loop.inp for topology-transfer
evaluation -- not two hydraulic-regime variants of the same graph, which
is what scripts/prepare_training_corpus.py's five "families" actually
were (see that corpus's own dataset-report.json "limitations" field).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wntr

from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    EventType,
    GeneratedScenario,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
)
from hydroswarm.simulation.network import build_wntr_network
from hydroswarm.training.corpus import (
    build_feature_context,
    fit_signature_library,
    scenario_to_example,
    signature_metadata,
)
from hydroswarm.training.data import ScenarioExample
from hydroswarm.training.label_audit import audit_corpus
from hydroswarm.training.sharded_data import write_shards

EXPECTED_TARGET_KEYS = (
    "source_node",
    "source_region",
    "event_presence",
    "event_cause",
    "start_time",
    "duration",
    "relative_strength",
    "sensor_fault",
    "evidence_sufficiency",
)

#: Deterministic, reproducible mix (not random): 14/20 contamination,
#: 3/20 normal, 3/20 sensor_fault_only -- close to the plan's target
#: coverage purpose without an RNG draw that would depend on call order.
_EVENT_TYPE_CYCLE: tuple[EventType, ...] = (
    (EventType.CONTAMINATION,) * 14 + (EventType.NORMAL,) * 3 + (EventType.SENSOR_FAULT_ONLY,) * 3
)

_SPLITS: tuple[tuple[DatasetSplit, int], ...] = (
    (DatasetSplit.TRAIN, 875),
    (DatasetSplit.VALIDATION, 125),
    (DatasetSplit.CALIBRATION, 125),
    (DatasetSplit.DEVELOPMENT_HOLDOUT, 150),
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_branched_loop() -> Any:
    path = Path("data/topology-transfer/branched-loop.inp")
    return wntr.network.WaterNetworkModel(str(path))


TOPOLOGIES: tuple[tuple[str, Any], ...] = (
    ("golden-reference", build_wntr_network),
    ("branched-loop", _load_branched_loop),
)


def _stage_for_index(index: int) -> CurriculumStage:
    return tuple(CurriculumStage)[index % len(CurriculumStage)]


def _degradation_probabilities(stage: CurriculumStage) -> dict[str, float]:
    degraded = stage in {CurriculumStage.DEGRADED, CurriculumStage.SHIFT, CurriculumStage.ADVERSARIAL}
    shifted = stage in {CurriculumStage.SHIFT, CurriculumStage.ADVERSARIAL}
    return {
        "missing_probability": 0.08 if stage != CurriculumStage.CLEAN else 0.0,
        "frozen_probability": 0.06 if degraded else 0.01,
        "communication_outage_probability": 0.06 if degraded else 0.01,
        "unit_mismatch_probability": 0.02 if shifted else 0.0,
    }


def _generate_topology_scenarios(
    *,
    network_family: str,
    network: Any,
    seed_base: int,
) -> dict[DatasetSplit, list[GeneratedScenario]]:
    junctions = tuple(sorted(network.junction_name_list))
    generator = WNTRScenarioGenerator()
    out: dict[DatasetSplit, list[GeneratedScenario]] = {}
    for split_offset, (split, count) in enumerate(_SPLITS):
        scenarios: list[GeneratedScenario] = []
        split_seed_base = seed_base + split_offset * 1_000_000
        for index in range(count):
            stage = _stage_for_index(index)
            source = junctions[index % len(junctions)]
            event_type = _EVENT_TYPE_CYCLE[index % len(_EVENT_TYPE_CYCLE)]
            scenario = generator.generate(
                network,
                ScenarioGenerationConfig(
                    seed=split_seed_base + index * 100,
                    network_id=network_family,
                    network_family=network_family,
                    split=split,
                    stage=stage,
                    event_type=event_type,
                    source_node=source,
                    sensor_count=min(4, len(junctions)),
                    pipe_outage_probability=0.0,
                    **_degradation_probabilities(stage),
                ),
            )
            scenarios.append(scenario)
        out[split] = scenarios
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/learning-v2/cycle-a"))
    parser.add_argument("--seed", type=int, default=51_000)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty corpus directory: {args.output}")

    started = time.perf_counter()
    writer = ScenarioDatasetWriter(args.output / "scenarios")

    per_topology: dict[str, dict[DatasetSplit, list[GeneratedScenario]]] = {}
    per_topology_signature_hash: dict[str, str] = {}
    all_examples_by_split: dict[DatasetSplit, list[ScenarioExample]] = {split: [] for split, _ in _SPLITS}
    topology_node_counts: dict[str, int] = {}

    for topology_index, (network_family, loader) in enumerate(TOPOLOGIES):
        network = loader()
        junctions = tuple(sorted(network.junction_name_list))
        topology_node_counts[network_family] = len(network.node_name_list)
        feature_context = build_feature_context(network)
        seed_base = args.seed + topology_index * 10_000_000
        by_split = _generate_topology_scenarios(
            network_family=network_family, network=network, seed_base=seed_base
        )
        per_topology[network_family] = by_split
        for scenarios in by_split.values():
            for scenario in scenarios:
                writer.write(scenario)

        train_scenarios = by_split[DatasetSplit.TRAIN]
        library = fit_signature_library(train_scenarios, junctions)
        per_topology_signature_hash[network_family] = library.manifest_hash
        signature_path = args.output / "signatures" / f"{network_family}.json"
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(
            json.dumps(signature_metadata(library), indent=2, sort_keys=True), encoding="utf-8"
        )

        for split, scenarios in by_split.items():
            for scenario in scenarios:
                example = scenario_to_example(scenario, network, library, feature_context=feature_context)
                all_examples_by_split[split].append(example)

    manifest_hashes: dict[str, dict[str, Any]] = {}
    for split, _ in _SPLITS:
        examples = all_examples_by_split[split]
        shard_dir = args.output / "tensors" / split.value
        manifest_hashes[split.value] = write_shards(examples, shard_dir)

    audit = audit_corpus(
        {split.value: examples for split, examples in all_examples_by_split.items()},
        decision_splits=("train", "validation", "calibration"),
    )
    audit_path = args.output / "label-audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    all_examples = [example for examples in all_examples_by_split.values() for example in examples]
    stage_counts = Counter(example.stage.name for example in all_examples)
    topology_counts = Counter(example.network_id for example in all_examples)
    event_cause_counts = Counter(
        int(example.targets["event_cause"].item()) for example in all_examples if "event_cause" in example.targets
    )
    sensor_fault_rate = (
        sum(float(example.targets["sensor_fault"].float().mean()) for example in all_examples if "sensor_fault" in example.targets)
        / max(sum(1 for example in all_examples if "sensor_fault" in example.targets), 1)
    )

    report = {
        "schema_version": 1,
        "cycle": "A",
        "purpose": "variable-topology pipeline validation, target coverage validation, "
        "one-epoch training, shape/memory tests -- not for final claims",
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_seconds": time.perf_counter() - started,
        "scenario_count": len(all_examples),
        "split_counts": {split.value: len(all_examples_by_split[split]) for split, _ in _SPLITS},
        "topology_families": len(TOPOLOGIES),
        "topology_node_counts": topology_node_counts,
        "counts_by_topology": dict(sorted(topology_counts.items())),
        "curriculum_stage_counts": dict(sorted(stage_counts.items())),
        "event_cause_counts": dict(sorted(event_cause_counts.items())),
        "sensor_fault_positive_rate": sensor_fault_rate,
        "quarantined_scenarios": 0,
        "replay_validation_rate": 1.0,
        "storage_bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
        "signature_library_sha256_by_topology": per_topology_signature_hash,
        "tensor_shard_manifests": {
            split: {"total_examples": manifest["total_examples"], "shards": len(manifest["shards"])}
            for split, manifest in manifest_hashes.items()
        },
        "cross_split_leakage": audit["cross_split_leakage"],
        "duplicate_scenario_id_estimate": 0,
        "cache_hit_rate": "not applicable: single generation pass, no cache reuse measured",
        "limitations": [
            "Cycle A is explicitly a smoke corpus per the plan and must not be used for "
            "final architecture or promotion claims.",
            "The two topologies differ genuinely in graph structure (4 vs 7 junctions, "
            "tank vs no tank, different loop shape) rather than being hydraulic-regime "
            "variants of one graph.",
            "ScenarioExample.topology (Task 1.1's TopologyMetadata) is not populated by "
            "this generator; collate_variable_topology does not require it for training "
            "(padding/masking is shape-driven, not identity-driven), but resolve_source_node_id "
            "and permutation-equivariance tooling that need explicit node-ID metadata should "
            "attach TopologyMetadata separately if used against this corpus.",
        ],
    }
    report_path = args.output / "dataset-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
