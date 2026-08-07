"""core-issues3.txt Phase 10 item 4 / core-issues2.txt Phase 6.3: generate a
real, balanced supported-category OOD extension corpus.

data/learning-v2/cycle-b2 already has real generated OOD holdout data for
2 of the 6 non-NONE supported categories (`tensors/ood-SEVERE_MISSINGNESS`,
`tensors/ood-UNSEEN_TOPOLOGY` -- see scripts/generate_cycle_b_corpus.py).
The remaining 4 (EXTREME_DEMAND, TANK_STATE_SHIFT, ROUGHNESS_MISMATCH,
FROZEN_DRIFTING_SENSOR) have a governed, already-tested triggering recipe
(`hydroswarm.training.ood_labels.OOD_TRIGGERING_CONFIG_OVERRIDES`,
verified by `tests/unit/test_ood_labels.py`'s
`test_every_recipe_override_reliably_triggers_its_category` across 3 seeds
each) but no generated corpus behind it yet -- this script is that corpus.

Governed constraints this script honors (core-issues3.txt Phase 2 items
5/R, restriction #3):

- Never mutates data/learning-v2/cycle-b2 -- writes only to a new,
  separately-versioned data/learning-v2/cycle-b2-ood-extension/ directory,
  matching cycle-b2's own scenarios/tensors/tensors-normalized/signatures
  layout so existing tooling (ShardedScenarioDataset, run_corpus_gates.py-
  style consumers) recognizes it without modification.
- Never fits a NEW signature artifact from these OOD scenarios. Each
  training topology's signature library is REFIT from that same
  topology's real cycle-b2 TRAIN-split scenarios (deterministic --
  fit_signature_library's own "deterministic rebuild" guarantee, already
  tested in tests/scientific/test_signature_registry.py) and its hash is
  verified against cycle-b2's own recorded signatures/<family>.json before
  use, so this script is provably using the identical artifact, not a
  drifted or newly-fit one.
- Every generated scenario is verified for real: classify_ood_category()
  must actually return the intended category, not merely assumed from the
  override recipe (which is itself tested only at small synthetic scale).
  Fails closed (raises) if any scenario's override does not reliably
  trigger its intended category -- a recipe that only "usually" works is
  not a governed recipe.
- Split is DEVELOPMENT_HOLDOUT throughout (never train/validation/
  calibration) -- these are diagnostic-only OOD scenarios, matching
  SEVERE_MISSINGNESS/UNSEEN_TOPOLOGY's own established convention.

Raw (unnormalized) tensors only -- run scripts/rebuild_normalized_shards.py
afterward, pointed at this corpus with cycle-b2's own node/edge
normalization artifacts, to produce the tensors-normalized/ counterpart
(exactly the same two-step convention cycle-b2 itself uses).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_ROOT))

from generate_cycle_b_corpus import TRAIN_TOPOLOGIES  # noqa: E402

from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioDatasetWriter,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.training.corpus import build_feature_context, fit_signature_library, scenario_to_example  # noqa: E402
from hydroswarm.training.data import ScenarioExample  # noqa: E402
from hydroswarm.training.ood_labels import OOD_TRIGGERING_CONFIG_OVERRIDES, classify_ood_category, ood_class_target  # noqa: E402
from hydroswarm.training.sharded_data import write_shards  # noqa: E402

#: Matches generate_cycle_b_corpus.py's OOD_HOLDOUT_COUNT_PER_CATEGORY --
#: "balanced" means every supported category gets the same real count.
COUNT_PER_CATEGORY = 400

_EVENT_TYPE_CYCLE = (EventType.CONTAMINATION, EventType.CONTAMINATION, EventType.SENSOR_FAULT_ONLY)

#: Real defect found while dry-running this script, not by inspection:
#: hydroswarm.data.scenarios.WNTRScenarioGenerator.generate_with_network
#: unconditionally zeroes missing_probability whenever
#: config.stage == CurriculumStage.CLEAN ("missing_probability = 0.0 if
#: config.stage == CurriculumStage.CLEAN else config.missing_probability"),
#: regardless of what value the caller's ScenarioGenerationConfig actually
#: requested -- generate_cycle_b_corpus.py's own _stage_for_index cycles
#: through every CurriculumStage including CLEAN, so 1/5 of a
#: missing_probability-based OOD-triggering batch would silently fail to
#: trigger (confirmed: 3/6 SEVERE_MISSINGNESS scenarios in a real small-
#: scale dry run of this script came back classified NONE, all at
#: stage=CLEAN). CLEAN is also the most "in-distribution" stage by
#: definition, making it a poor choice for OOD-extension generation
#: regardless of which specific override field a category uses -- this
#: script therefore never selects it, cycling only the 4 non-CLEAN stages.
_OOD_STAGE_CYCLE = tuple(stage for stage in CurriculumStage if stage != CurriculumStage.CLEAN)


def _stage_for_index(index: int) -> CurriculumStage:
    return _OOD_STAGE_CYCLE[index % len(_OOD_STAGE_CYCLE)]


def _load_train_scenarios_for_family(cycle_b2_root: Path, network_family: str) -> list[Any]:
    from hydroswarm.data.scenarios import load_generated_scenarios

    scenarios = load_generated_scenarios(cycle_b2_root / "scenarios", DatasetSplit.TRAIN)
    return [scenario for scenario in scenarios if scenario.manifest.network_family == network_family]


def _refit_and_verify_signature_library(cycle_b2_root: Path, network_family: str, junctions: tuple[str, ...]) -> Any:
    train_scenarios = _load_train_scenarios_for_family(cycle_b2_root, network_family)
    if not train_scenarios:
        raise ValueError(f"no cycle-b2 TRAIN scenarios found for network_family={network_family!r}")
    library = fit_signature_library(train_scenarios, junctions)
    recorded_path = cycle_b2_root / "signatures" / f"{network_family}.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    if library.manifest_hash != recorded["sha256"]:
        raise ValueError(
            f"refit signature library for {network_family!r} (hash {library.manifest_hash}) does not match "
            f"cycle-b2's own recorded artifact (hash {recorded['sha256']}) -- refusing to generate OOD "
            "scenarios against a drifted signature library"
        )
    return library


class _TopologyContext:
    """Everything about one training topology that is genuinely shared
    across every OOD category -- computed once in main() rather than
    redundantly refit (network load + signature library refit) once per
    category. Signature-library refitting in particular is not free (a
    real pass over every cycle-b2 TRAIN scenario for that topology)."""

    __slots__ = ("network_family", "network", "junctions", "topology_hash", "library")

    def __init__(self, network_family: str, network: Any, cycle_b2_root: Path) -> None:
        self.network_family = network_family
        self.network = network
        self.junctions = tuple(sorted(network.junction_name_list))
        self.topology_hash = network_sha256(network)
        self.library = _refit_and_verify_signature_library(cycle_b2_root, network_family, self.junctions)


def generate_category(
    category_name: str,
    overrides: dict[str, Any],
    *,
    topology_contexts: list[_TopologyContext],
    validated_topology_hashes: frozenset[str],
    seed_base: int,
    count: int,
    writer: ScenarioDatasetWriter,
) -> tuple[list[ScenarioExample], dict[str, Any]]:
    examples: list[ScenarioExample] = []
    trigger_failures: list[dict[str, Any]] = []
    per_topology_counts: Counter[str] = Counter()

    share = count // len(topology_contexts)
    remainder = count % len(topology_contexts)
    for topology_index, context in enumerate(topology_contexts):
        network_family, network, junctions, topology_hash, library = (
            context.network_family, context.network, context.junctions, context.topology_hash, context.library,
        )
        generator = WNTRScenarioGenerator()

        this_topology_count = share + (1 if topology_index < remainder else 0)
        for index in range(this_topology_count):
            stage = _stage_for_index(index)
            source = junctions[index % len(junctions)]
            scenario, randomized_network = generator.generate_with_network(
                network,
                ScenarioGenerationConfig(
                    seed=seed_base + topology_index * 1_000_000 + index * 100,
                    network_id=network_family,
                    network_family=network_family,
                    split=DatasetSplit.DEVELOPMENT_HOLDOUT,
                    stage=stage,
                    event_type=_EVENT_TYPE_CYCLE[index % len(_EVENT_TYPE_CYCLE)],
                    source_node=source,
                    sensor_count=min(overrides.get("sensor_count", 4), len(junctions)),
                    pipe_outage_probability=0.0,
                    **{key: value for key, value in overrides.items() if key != "sensor_count"},
                ),
            )
            actual_category = classify_ood_category(
                scenario, topology_hash=topology_hash, validated_topology_hashes=validated_topology_hashes
            )
            if actual_category.value != category_name:
                trigger_failures.append(
                    {
                        "network_family": network_family,
                        "seed": seed_base + topology_index * 1_000_000 + index * 100,
                        "expected": category_name,
                        "actual": actual_category.value,
                    }
                )
                continue
            writer.write(scenario)
            feature_context = build_feature_context(randomized_network)
            example = scenario_to_example(scenario, randomized_network, library, feature_context=feature_context)
            # Real gap found running this at real scale, not by inspection:
            # classify_ood_category was verified against the intended
            # category above but never actually attached to the resulting
            # example -- scenario_to_example alone does not compute
            # ood_class (generate_cycle_b_corpus.py's own OOD holdout path
            # merges it in separately, via ood_class_target, for exactly
            # this reason). Without this, the whole point of a governed
            # OOD-category extension corpus -- carrying a real ood_class
            # label -- would have been silently missing from every example.
            example = dataclasses.replace(example, targets={**example.targets, **ood_class_target(actual_category)})
            examples.append(example)
            per_topology_counts[network_family] += 1

    if trigger_failures:
        raise ValueError(
            f"{len(trigger_failures)} scenario(s) generated for {category_name!r} did not actually classify "
            f"as that category -- the recipe is not reliably triggering it at this scale: {trigger_failures[:5]}"
        )

    return examples, {
        "category": category_name,
        "requested_count": count,
        "generated_count": len(examples),
        "per_topology_counts": dict(per_topology_counts),
        "config_overrides": overrides,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycle-b2-root", type=Path, default=Path("data/learning-v2/cycle-b2"))
    parser.add_argument("--output", type=Path, default=Path("data/learning-v2/cycle-b2-ood-extension"))
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--count-per-category", type=int, default=COUNT_PER_CATEGORY)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=sorted(category.value for category in OOD_TRIGGERING_CONFIG_OVERRIDES),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty corpus directory: {args.output}")

    started = time.perf_counter()
    writer = ScenarioDatasetWriter(args.output / "scenarios")
    overrides_by_name = {category.value: overrides for category, overrides in OOD_TRIGGERING_CONFIG_OVERRIDES.items()}

    # Computed once, shared across every category below -- see
    # _TopologyContext's own docstring for why.
    topology_contexts = [
        _TopologyContext(network_family, loader(), args.cycle_b2_root) for network_family, loader in TRAIN_TOPOLOGIES
    ]
    validated_topology_hashes = frozenset(context.topology_hash for context in topology_contexts)

    manifest_hashes: dict[str, Any] = {}
    category_reports: dict[str, Any] = {}
    for category_index, category_name in enumerate(args.categories):
        overrides = overrides_by_name[category_name]
        examples, report = generate_category(
            category_name,
            overrides,
            topology_contexts=topology_contexts,
            validated_topology_hashes=validated_topology_hashes,
            seed_base=args.seed + category_index * 10_000_000,
            count=args.count_per_category,
            writer=writer,
        )
        manifest_hashes[f"ood-{category_name}"] = write_shards(examples, args.output / "tensors" / f"ood-{category_name}")
        category_reports[category_name] = report
        print(f"{category_name}: {report['generated_count']}/{report['requested_count']} generated and verified")

    report = {
        "schema_version": 1,
        "generation_seconds": time.perf_counter() - started,
        "cycle_b2_root": str(args.cycle_b2_root),
        "categories": category_reports,
        "manifest_hashes": manifest_hashes,
    }
    (args.output / "generation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
