"""Generate a governed full-incident-trajectory corpus extension
(core-issues2.txt Phase 7) over an existing Cycle B2-style scenario corpus.

Reuses the target corpus's own already-generated scenarios (loaded via
load_generated_scenarios) rather than resimulating -- this is an
*extension* of an existing corpus, not a new one; per core-issues2.txt's
"Add a separate versioned trajectory or multitask extension instead of
silently mutating Cycle B", nothing under --corpus-dir is ever written to.

For each loaded scenario, calls
hydroswarm.training.full_trajectory.build_incident_trajectory (which
itself wires together every label generator Phase 1-6 built: Sentinel,
OOD category, evidence_sufficiency/next_step, Scout sampling
sub-trajectory, Strategist planning sub-trajectory, and the three
auxiliary targets) and appends the JSON-serialized result to a per-split
JSONL shard under --output.

Resumable: writes one line per scenario_id to a shard file and, on
restart, skips any scenario_id already present in that file -- adequate
resumability for this script's scale (a few thousand scenarios at ~0.5s
each) without needing full job_runner integration.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wntr

from generate_cycle_b_corpus import _degradation_probabilities

from hydroswarm.classical.signatures import SignatureArtifact, SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, GeneratedScenario, load_generated_scenarios
from hydroswarm.training.corpus import FeatureContext, _hydraulic_state_hash, build_feature_context, fit_signature_library
from hydroswarm.training.full_trajectory import IncidentTrajectory, build_incident_trajectory
from hydroswarm.training.scenario_reconstruction import (
    RECONSTRUCTION_POLICY_VERSION,
    reconstruct_scenario_network,
)
from hydroswarm.training.scout_labels import build_signature_artifact_for_network


def _train_topology_loaders() -> dict[str, Any]:
    """Must match scripts/generate_cycle_b_corpus.py's TRAIN_TOPOLOGIES
    exactly -- this script builds trajectories for the same governed
    topology set the target corpus was generated from."""
    from hydroswarm.simulation.network import build_wntr_network

    return {
        "golden-reference": build_wntr_network,
        "branched-loop": lambda: wntr.network.WaterNetworkModel("data/topology-transfer/branched-loop.inp"),
        "loop-grid": lambda: wntr.network.WaterNetworkModel("data/topologies/loop-grid.inp"),
    }


def _tensor_to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _incident_trajectory_to_json(
    result: IncidentTrajectory, reconstruction: Any
) -> dict[str, Any]:
    example = result.example
    return {
        "scenario_id": example.scenario_id,
        "network_id": example.network_id,
        "split": example.split,
        "seed": example.seed,
        "stage": example.stage.value,
        "ood_category": result.ood_category.value,
        "targets": {key: _tensor_to_jsonable(value) for key, value in example.targets.items()},
        "topology_hash": example.topology.topology_hash if example.topology else None,
        "network_hash": example.topology.network_hash if example.topology else None,
        # core-issues3.txt Phase 1 item 8: version the reconstruction policy
        # and record it per trajectory, not just as a script-level constant --
        # this is the scenario-specific randomized network/context identity,
        # not the pristine topology's.
        "reconstruction": {
            "policy_hash": reconstruction.reconstruction_policy_hash,
            "network_state_hash": reconstruction.network_state_hash,
            "hydraulic_state_hash": reconstruction.hydraulic_state_hash,
            "replay_matched": reconstruction.replay_matched,
            "artifact_hash_drifted": reconstruction.artifact_hash_drifted,
        },
        "scout": {
            "trajectory": result.scout.trajectory.to_json(),
            "steps": [
                {
                    "targets": {key: _tensor_to_jsonable(value) for key, value in step.targets.items()},
                    "diagnostics": step.diagnostics,
                }
                for step in result.scout.steps
            ],
        },
        "strategist": {
            "trajectory": result.strategist.trajectory.to_json(),
            "steps": [
                {
                    "labels": [dataclasses.asdict(label) for label in step.labels],
                    "targets": [
                        {key: _tensor_to_jsonable(value) for key, value in target.items()} for target in step.targets
                    ],
                }
                for step in result.strategist.steps
            ],
        },
    }


def _load_processed_ids(shard_path: Path) -> set[str]:
    if not shard_path.exists():
        return set()
    processed = set()
    for line in shard_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            processed.add(json.loads(line)["scenario_id"])
    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=[s.value for s in DatasetSplit])
    parser.add_argument("--limit", type=int, default=None, help="process at most this many scenarios")
    parser.add_argument("--signature-cache-dir", type=Path, default=Path("experiments/cache/signatures"))
    parser.add_argument("--maximum-samples", type=int, default=5)
    parser.add_argument("--maximum-exact-simulations", type=int, default=3)
    parser.add_argument("--maximum-plans", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = DatasetSplit(args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    shard_path = args.output / f"{split.value}.jsonl"
    already_processed = _load_processed_ids(shard_path)

    loaders = _train_topology_loaders()
    networks = {name: loader() for name, loader in loaders.items()}

    print("loading train-split scenarios per topology to fit signature libraries...")
    train_scenarios_by_family: dict[str, list[GeneratedScenario]] = {name: [] for name in loaders}
    for scenario in load_generated_scenarios(args.corpus_dir / "scenarios", DatasetSplit.TRAIN):
        family = scenario.manifest.network_family
        if family in train_scenarios_by_family:
            train_scenarios_by_family[family].append(scenario)

    validated_topology_hashes: set[str] = set()
    libraries: dict[str, Any] = {}
    artifacts: dict[str, SignatureArtifact] = {}
    contexts: dict[str, FeatureContext] = {}
    for family, network in networks.items():
        scenarios = train_scenarios_by_family[family]
        if not scenarios:
            print(f"WARNING: no train scenarios found for topology {family!r}; skipping")
            continue
        junctions = tuple(sorted(network.junction_name_list))
        libraries[family] = fit_signature_library(scenarios, junctions)
        validated_topology_hashes.update(scenario.manifest.network_sha256 for scenario in scenarios)
        contexts[family] = build_feature_context(network)

        cache = SignatureCache(args.signature_cache_dir)
        key = SignatureCacheKey(
            network_hash=scenarios[0].manifest.network_sha256,
            hydraulic_state_hash=_hydraulic_state_hash(contexts[family].state),
            simulator_version=scenarios[0].manifest.simulator_version,
            configuration_hash="cycle-b2-trajectories-v1",
            sensor_layout_hash="all-junctions",
        )
        t0 = time.time()
        artifacts[family] = build_signature_artifact_for_network(network, cache, key=key)
        print(f"  {family}: signature artifact ready in {time.time() - t0:.1f}s (cache_hit={artifacts[family].cache_hit})")

    print(f"validated topology hashes: {len(validated_topology_hashes)}")

    target_scenarios = list(load_generated_scenarios(args.corpus_dir / "scenarios", split))
    if args.limit is not None:
        target_scenarios = target_scenarios[: args.limit]
    print(f"processing {len(target_scenarios)} scenarios from split={split.value} ({len(already_processed)} already done)")

    ood_counts: Counter[str] = Counter()
    error_count = 0
    processed_count = 0
    started_at = time.time()
    with shard_path.open("a", encoding="utf-8") as stream:
        for index, scenario in enumerate(target_scenarios):
            scenario_id = str(scenario.manifest.scenario_id)
            if scenario_id in already_processed:
                continue
            family = scenario.manifest.network_family
            if family not in networks:
                continue
            try:
                # core-issues3.txt Phase 1: reconstruct THIS scenario's own
                # randomized hydraulic state (demand regime, roughness,
                # tank levels, pipe status) rather than reusing one pristine
                # network/context shared by every scenario in this topology
                # family. Fails closed (raises) if the reconstruction does
                # not semantically match this scenario's own stored arrays.
                reconstruction = reconstruct_scenario_network(
                    networks[family],
                    scenario.manifest,
                    degradation_policy=_degradation_probabilities,
                    original=scenario,
                )
                result = build_incident_trajectory(
                    reconstruction.scenario,
                    reconstruction.network,
                    libraries[family],
                    artifacts[family],
                    topology_hash=scenario.manifest.network_sha256,
                    validated_topology_hashes=validated_topology_hashes,
                    feature_context=reconstruction.feature_context,
                    maximum_samples=args.maximum_samples,
                    maximum_exact_simulations=args.maximum_exact_simulations,
                    maximum_plans=args.maximum_plans,
                )
            except Exception as error:  # noqa: BLE001 -- record and continue (incl. ScenarioReconstructionError), never silently drop the corpus
                error_count += 1
                print(f"ERROR scenario {scenario_id}: {error!r}")
                continue
            stream.write(json.dumps(_incident_trajectory_to_json(result, reconstruction)) + "\n")
            stream.flush()
            ood_counts[result.ood_category.value] += 1
            processed_count += 1
            if processed_count % 25 == 0:
                elapsed = time.time() - started_at
                rate = processed_count / elapsed if elapsed > 0 else 0.0
                print(f"  {processed_count}/{len(target_scenarios) - len(already_processed)} done, {rate:.2f}/s")

    report = {
        # v2: every row's network/feature-context is now the scenario's own
        # reconstruct_scenario_network() output, not one pristine
        # network/context shared across the whole topology family
        # (core-issues3.txt Phase 1). v1 rows are NOT compatible with v2
        # rows and must not be merged into the same shard.
        "schema_version": "cycle-b2-trajectories-v2",
        "reconstruction_policy_hash": RECONSTRUCTION_POLICY_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_dir": str(args.corpus_dir),
        "split": split.value,
        "scenarios_processed_this_run": processed_count,
        "errors_this_run": error_count,
        "total_in_shard": len(already_processed) + processed_count,
        "ood_category_counts_this_run": dict(ood_counts),
        "validated_topology_hashes": sorted(validated_topology_hashes),
        "elapsed_seconds": time.time() - started_at,
    }
    report_path = args.output / f"{split.value}-report.json"
    existing_report = json.loads(report_path.read_text()) if report_path.exists() else None
    report["previous_run"] = existing_report
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {processed_count} new records ({error_count} errors) to {shard_path}")
    print(f"report: {report_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
