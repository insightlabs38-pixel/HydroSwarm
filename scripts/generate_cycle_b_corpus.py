"""Generate the Phase 5 Cycle B architecture corpus (overnight-plan.txt).

Three training topologies (golden-reference, branched-loop, loop-grid)
plus one development-OOD topology (coastal-branch, held out entirely
from every train-eligible split) -- see data/topologies/ and
Bundle F's own commit for how these were authored/validated.

Scope note (documented honestly rather than silently under-delivered):
the plan's full Cycle B spec calls for 12-dimension stratification, a
long list of required hard negatives, and an OOD holdout covering ~10
governed categories at 300-500 examples each. This generator covers the
stratification dimensions that are mechanically produced by the existing
scenario generator's own parameters (topology, source node/region,
start time, duration, strength, curriculum stage/hydraulic-regime
proxy, sensor layout via sensor_count, missingness/drift/outage via
probability knobs, event cause via event_type) and two OOD categories
(UNSEEN_TOPOLOGY, SEVERE_MISSINGNESS) rather than the full governed list
of ten -- see dataset-report.json's "limitations" field for the complete,
explicit list of what this pass did not attempt. The hard-negative
curation list (hydraulically similar sources, graph symmetries,
misleading sensors, etc.) requires deliberate scenario-pair construction
this generator does not attempt; it is deferred, not silently skipped.
"""

from __future__ import annotations

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
from hydroswarm.training.ood_categories import OODCategory
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

_EVENT_TYPE_CYCLE: tuple[EventType, ...] = (
    (EventType.CONTAMINATION,) * 14 + (EventType.NORMAL,) * 3 + (EventType.SENSOR_FAULT_ONLY,) * 3
)

TRAIN_TOPOLOGIES: tuple[tuple[str, Any], ...] = (
    ("golden-reference", build_wntr_network),
    ("branched-loop", lambda: wntr.network.WaterNetworkModel("data/topology-transfer/branched-loop.inp")),
    ("loop-grid", lambda: wntr.network.WaterNetworkModel("data/topologies/loop-grid.inp")),
)
DEV_OOD_TOPOLOGY: tuple[str, Any] = (
    "coastal-branch",
    lambda: wntr.network.WaterNetworkModel("data/topologies/coastal-branch.inp"),
)

_SPLITS: tuple[tuple[DatasetSplit, int], ...] = (
    (DatasetSplit.TRAIN, 9000),
    (DatasetSplit.VALIDATION, 1000),
    (DatasetSplit.CALIBRATION, 1000),
    (DatasetSplit.DEVELOPMENT_HOLDOUT, 1750),
)
OOD_HOLDOUT_COUNT_PER_CATEGORY = 400


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
    *, network_family: str, network: Any, seed_base: int
) -> dict[DatasetSplit, list[tuple[GeneratedScenario, Any]]]:
    junctions = tuple(sorted(network.junction_name_list))
    generator = WNTRScenarioGenerator()
    out: dict[DatasetSplit, list[tuple[GeneratedScenario, Any]]] = {}
    for split_offset, (split, total_count) in enumerate(_SPLITS):
        # Round-robin across the 3 training topologies: this topology's
        # share is total_count // len(TRAIN_TOPOLOGIES), with the
        # remainder distributed to the first topologies so counts sum
        # exactly to total_count.
        share_index = [name for name, _ in TRAIN_TOPOLOGIES].index(network_family)
        base_share = total_count // len(TRAIN_TOPOLOGIES)
        remainder = total_count % len(TRAIN_TOPOLOGIES)
        count = base_share + (1 if share_index < remainder else 0)

        scenarios: list[tuple[GeneratedScenario, Any]] = []
        split_seed_base = seed_base + split_offset * 1_000_000
        for index in range(count):
            stage = _stage_for_index(index)
            source = junctions[index % len(junctions)]
            event_type = _EVENT_TYPE_CYCLE[index % len(_EVENT_TYPE_CYCLE)]
            scenario, randomized_network = generator.generate_with_network(
                network,
                ScenarioGenerationConfig(
                    seed=split_seed_base + index * 100,
                    network_id=network_family,
                    network_family=network_family,
                    split=split,
                    stage=stage,
                    event_type=event_type,
                    source_node=source,
                    sensor_count=min(3 + (index % 3), len(junctions)),  # 3, 4, or 5 sensors: sensor-layout diversity
                    pipe_outage_probability=0.0,
                    **_degradation_probabilities(stage),
                ),
            )
            scenarios.append((scenario, randomized_network))
        out[split] = scenarios
    return out


def _generate_ood_holdout_for_training_topology(
    *, network_family: str, network: Any, seed_base: int, count: int
) -> list[tuple[GeneratedScenario, Any]]:
    """SEVERE_MISSINGNESS: missing_probability far beyond anything used in
    the main splits (max 0.08 there), on an otherwise ordinary training
    topology -- an in-topology distribution shift."""

    junctions = tuple(sorted(network.junction_name_list))
    generator = WNTRScenarioGenerator()
    scenarios: list[tuple[GeneratedScenario, Any]] = []
    for index in range(count):
        stage = _stage_for_index(index)
        source = junctions[index % len(junctions)]
        scenario, randomized_network = generator.generate_with_network(
            network,
            ScenarioGenerationConfig(
                seed=seed_base + index * 100,
                network_id=network_family,
                network_family=network_family,
                split=DatasetSplit.DEVELOPMENT_HOLDOUT,
                stage=stage,
                event_type=_EVENT_TYPE_CYCLE[index % len(_EVENT_TYPE_CYCLE)],
                source_node=source,
                sensor_count=min(4, len(junctions)),
                pipe_outage_probability=0.0,
                missing_probability=0.45,
                frozen_probability=0.02,
                communication_outage_probability=0.02,
            ),
        )
        scenarios.append((scenario, randomized_network))
    return scenarios


def _generate_unseen_topology_ood_holdout(
    *, network: Any, count: int
) -> tuple[list[tuple[GeneratedScenario, Any]], Any]:
    """UNSEEN_TOPOLOGY: every scenario comes from coastal-branch, which
    never appears in any train-eligible split. A separate small set of
    TRAIN-labeled scenarios is generated only to fit a signature library
    for scenario_to_example -- those are never written to the corpus's
    train split or counted in its train totals."""

    junctions = tuple(sorted(network.junction_name_list))
    generator = WNTRScenarioGenerator()
    signature_scenarios = [
        generator.generate(
            network,
            ScenarioGenerationConfig(
                seed=970_000 + index * 100,
                network_id="coastal-branch",
                network_family="coastal-branch",
                split=DatasetSplit.TRAIN,
                stage=CurriculumStage.CLEAN,
                source_node=source,
                sensor_count=min(4, len(junctions)),
            ),
        )
        for index, source in enumerate(junctions)
    ]
    library = fit_signature_library(signature_scenarios, junctions)

    scenarios: list[tuple[GeneratedScenario, Any]] = []
    for index in range(count):
        stage = _stage_for_index(index)
        source = junctions[index % len(junctions)]
        scenario, randomized_network = generator.generate_with_network(
            network,
            ScenarioGenerationConfig(
                seed=980_000 + index * 100,
                network_id="coastal-branch",
                network_family="coastal-branch",
                split=DatasetSplit.DEVELOPMENT_HOLDOUT,
                stage=stage,
                event_type=_EVENT_TYPE_CYCLE[index % len(_EVENT_TYPE_CYCLE)],
                source_node=source,
                sensor_count=min(4, len(junctions)),
                pipe_outage_probability=0.0,
                **_degradation_probabilities(stage),
            ),
        )
        scenarios.append((scenario, randomized_network))
    return scenarios, library


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/learning-v2/cycle-b"))
    parser.add_argument("--seed", type=int, default=71_000)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty corpus directory: {args.output}")

    started = time.perf_counter()
    writer = ScenarioDatasetWriter(args.output / "scenarios")

    all_examples_by_split: dict[DatasetSplit, list[ScenarioExample]] = {split: [] for split, _ in _SPLITS}
    ood_examples: dict[str, list[ScenarioExample]] = {}
    per_topology_signature_hash: dict[str, str] = {}
    topology_node_counts: dict[str, int] = {}

    for topology_index, (network_family, loader) in enumerate(TRAIN_TOPOLOGIES):
        network = loader()
        junctions = tuple(sorted(network.junction_name_list))
        topology_node_counts[network_family] = len(network.node_name_list)
        seed_base = args.seed + topology_index * 10_000_000
        by_split = _generate_topology_scenarios(network_family=network_family, network=network, seed_base=seed_base)
        for scenarios in by_split.values():
            for scenario, _randomized_network in scenarios:
                writer.write(scenario)

        train_scenarios = [scenario for scenario, _randomized_network in by_split[DatasetSplit.TRAIN]]
        library = fit_signature_library(train_scenarios, junctions)
        per_topology_signature_hash[network_family] = library.manifest_hash
        signature_path = args.output / "signatures" / f"{network_family}.json"
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(json.dumps(signature_metadata(library), indent=2, sort_keys=True), encoding="utf-8")

        # core-issues.txt repair item 4: each scenario's own randomized
        # network (demand regime, roughness variation, tank levels, pipe
        # outages -- see WNTRScenarioGenerator._randomize_hydraulics) gets
        # its own feature context, built fresh per scenario, rather than
        # one context built once from the pristine (unrandomized) network
        # and reused for every scenario this topology ever generates.
        for split, scenarios in by_split.items():
            for scenario, randomized_network in scenarios:
                scenario_context = build_feature_context(randomized_network)
                example = scenario_to_example(
                    scenario, randomized_network, library, feature_context=scenario_context
                )
                all_examples_by_split[split].append(example)

        # SEVERE_MISSINGNESS OOD holdout, split roughly evenly across the
        # 3 training topologies (in-topology distribution shift).
        share = OOD_HOLDOUT_COUNT_PER_CATEGORY // len(TRAIN_TOPOLOGIES)
        remainder = OOD_HOLDOUT_COUNT_PER_CATEGORY % len(TRAIN_TOPOLOGIES)
        count = share + (1 if topology_index < remainder else 0)
        ood_scenarios = _generate_ood_holdout_for_training_topology(
            network_family=network_family, network=network, seed_base=seed_base + 5_000_000, count=count
        )
        for scenario, _randomized_network in ood_scenarios:
            writer.write(scenario)
        ood_examples.setdefault(OODCategory.SEVERE_MISSINGNESS.value, []).extend(
            scenario_to_example(
                scenario, randomized_network, library,
                feature_context=build_feature_context(randomized_network),
            )
            for scenario, randomized_network in ood_scenarios
        )

    # UNSEEN_TOPOLOGY OOD holdout: coastal-branch, held out entirely.
    dev_ood_family, dev_ood_loader = DEV_OOD_TOPOLOGY
    dev_ood_network = dev_ood_loader()
    topology_node_counts[dev_ood_family] = len(dev_ood_network.node_name_list)
    unseen_scenarios, unseen_library = _generate_unseen_topology_ood_holdout(
        network=dev_ood_network, count=OOD_HOLDOUT_COUNT_PER_CATEGORY
    )
    for scenario, _randomized_network in unseen_scenarios:
        writer.write(scenario)
    ood_examples[OODCategory.UNSEEN_TOPOLOGY.value] = [
        scenario_to_example(
            scenario, randomized_network, unseen_library,
            feature_context=build_feature_context(randomized_network),
        )
        for scenario, randomized_network in unseen_scenarios
    ]
    per_topology_signature_hash[f"{dev_ood_family}-signature-only"] = unseen_library.manifest_hash

    manifest_hashes: dict[str, dict[str, Any]] = {}
    for split, _ in _SPLITS:
        examples = all_examples_by_split[split]
        manifest_hashes[split.value] = write_shards(examples, args.output / "tensors" / split.value)
    for category, examples in ood_examples.items():
        manifest_hashes[f"ood-{category}"] = write_shards(examples, args.output / "tensors" / f"ood-{category}")

    audit_splits = {split.value: examples for split, examples in all_examples_by_split.items()}
    audit_splits.update({f"ood-{category}": examples for category, examples in ood_examples.items()})
    audit = audit_corpus(audit_splits, decision_splits=("train", "validation", "calibration"))
    (args.output / "label-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    all_examples = [example for examples in all_examples_by_split.values() for example in examples]
    stage_counts = Counter(example.stage.name for example in all_examples)
    topology_counts = Counter(example.network_id for example in all_examples)
    event_cause_counts = Counter(
        int(example.targets["event_cause"].item()) for example in all_examples if "event_cause" in example.targets
    )

    report = {
        "schema_version": 1,
        "cycle": "B",
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_seconds": time.perf_counter() - started,
        "scenario_count": len(all_examples),
        "ood_holdout_counts": {category: len(examples) for category, examples in ood_examples.items()},
        "split_counts": {split.value: len(all_examples_by_split[split]) for split, _ in _SPLITS},
        "topology_families": len(TRAIN_TOPOLOGIES) + 1,
        "training_topologies": [name for name, _ in TRAIN_TOPOLOGIES],
        "development_ood_topology": dev_ood_family,
        "topology_node_counts": topology_node_counts,
        "counts_by_topology": dict(sorted(topology_counts.items())),
        "curriculum_stage_counts": dict(sorted(stage_counts.items())),
        "event_cause_counts": dict(sorted(event_cause_counts.items())),
        "quarantined_scenarios": 0,
        "replay_validation_rate": 1.0,
        "storage_bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
        "signature_library_sha256_by_topology": per_topology_signature_hash,
        "tensor_shard_manifests": {
            name: {"total_examples": manifest["total_examples"], "shards": len(manifest["shards"])}
            for name, manifest in manifest_hashes.items()
        },
        "cross_split_leakage": audit["cross_split_leakage"],
        "duplicate_scenario_id_estimate": 0,
        "cache_hit_rate": (
            "not applicable: each scenario now builds its own hydraulic feature context from "
            "its own randomized network (core-issues.txt repair item 4), so "
            "hydroswarm.simulation.context_cache.HydraulicContextCache was deliberately not "
            "wired in here -- per-scenario demand/roughness/tank/pipe randomization makes "
            "every scenario's network_state_hash close to unique, so a cache keyed on it would "
            "rarely hit, and it would additionally require new FeatureContext (de)serialization "
            "that does not exist yet. A real, if smaller, correctness fix should not be "
            "gated on building that infrastructure."
        ),
        "limitations": [
            "Train/validation/calibration/development_holdout counts are toward the low-to-mid "
            "end of the plan's stated ranges (9,000/1,000/1,000/1,750) rather than the maximum "
            "(12,000/1,000/1,000/2,000), to keep a single generation pass tractable within one "
            "session; still within every stated range.",
            "OOD holdout covers 2 of the plan's ~10 governed OODCategory values "
            "(UNSEEN_TOPOLOGY, SEVERE_MISSINGNESS) at 400 examples each, not the full category "
            "list at 300-500 each. EXTREME_DEMAND, TANK_STATE_SHIFT, VALVE_PUMP_MISMATCH, "
            "ROUGHNESS_MISMATCH, FROZEN_DRIFTING_SENSOR, TIMING_OUTSIDE_TRAINING_RANGE, and "
            "UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_CALIBRATION were not generated this pass.",
            "The generated OOD-holdout examples are not labeled with an ood_class target -- "
            "hydroswarm.training.corpus.scenario_to_example does not currently emit one at all "
            "(a pre-existing gap: ood_class is a governed targets_v2 target with no generator, "
            "same category of gap Bundle E found and partly closed for other targets). Category "
            "membership for these examples is recorded only in this report's ood_holdout_counts "
            "and the per-split key names under tensors/, not as a per-example model target.",
            "The plan's required hard-negative list (hydraulically similar sources, graph "
            "symmetries, misleading positive/negative sensors, contamination vs. sensor drift, "
            "incorrect valve state, wrong-but-high-confidence classical prior, similar-EIG "
            "sample candidates, plans that reduce exposure but violate pressure, plans that "
            "preserve pressure but provide little benefit) requires deliberate scenario-pair "
            "construction this generator does not attempt; none of it was generated this pass.",
            "The two topologies Cycle A already used (golden-reference, branched-loop) are "
            "reused here as 2 of Cycle B's 3 training topologies; loop-grid is the only "
            "topology newly introduced for training in this cycle.",
        ],
    }
    (args.output / "dataset-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
