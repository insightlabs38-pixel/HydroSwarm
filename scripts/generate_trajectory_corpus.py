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

from hydroswarm.classical.signature_registry import TOPOLOGY_WIDE_REGIME_HASH, SignatureRegistry
from hydroswarm.classical.signatures import SignatureArtifact, SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import DatasetSplit, GeneratedScenario, load_generated_scenarios
from hydroswarm.training.corpus import FeatureContext, _hydraulic_state_hash, build_feature_context, fit_signature_library
from hydroswarm.training.full_trajectory import IncidentTrajectory, build_incident_trajectory
from hydroswarm.training.scenario_reconstruction import (
    RECONSTRUCTION_POLICY_VERSION,
    ScenarioReconstructionError,
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
    result: IncidentTrajectory, reconstruction: Any, weak_verification: bool = False
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
            # True only for development_holdout scenarios whose semantic
            # array-level check was skipped because the corpus's own
            # degradation-formula ambiguity for that split (see the retry
            # comment above) made a spurious failure indistinguishable from
            # a real one. The scenario/network identity (replay_sha256) is
            # still verified either way -- only the degraded-observation
            # outcome check is skipped when this is True.
            "weak_verification": weak_verification,
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
    parser.add_argument("--maximum-plans", type=int, default=9)
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

        # core-issues3.txt Phase 2: index the artifact under the governed
        # registry instead of leaving it a bare, undiscoverable cache entry
        # -- records the bucketing policy (TOPOLOGY_WIDE_REGIME_HASH) and
        # enforces fit_split="train" (SignatureRegistry.register raises for
        # anything else), matching train_scenarios_by_family's own
        # train-only construction above.
        SignatureRegistry(cache).register(
            topology_hash=scenarios[0].manifest.network_sha256,
            hydraulic_regime_hash=TOPOLOGY_WIDE_REGIME_HASH,
            key=key,
            artifact=artifacts[family],
            fit_split="train",
        )

    print(f"validated topology hashes: {len(validated_topology_hashes)}")

    target_scenarios = list(load_generated_scenarios(args.corpus_dir / "scenarios", split))
    if args.limit is not None:
        target_scenarios = target_scenarios[: args.limit]
    print(f"processing {len(target_scenarios)} scenarios from split={split.value} ({len(already_processed)} already done)")

    ood_counts: Counter[str] = Counter()
    error_count = 0
    processed_count = 0
    skipped_unsupported_topology: Counter[str] = Counter()
    started_at = time.time()
    with shard_path.open("a", encoding="utf-8") as stream:
        for index, scenario in enumerate(target_scenarios):
            scenario_id = str(scenario.manifest.scenario_id)
            if scenario_id in already_processed:
                continue
            family = scenario.manifest.network_family
            if family not in networks:
                # core-issues3.txt Phase 1 item J: a governed corpus must
                # report and resolve every omission, not silently continue.
                # This branch is currently reached by development_holdout's
                # coastal-branch (unseen-topology) scenarios: this script
                # only loads the three TRAIN_TOPOLOGIES networks, and
                # Phase 2 item 5 / item R forbid fitting a topology-specific
                # signature library from development-holdout incidents, so
                # there is no governed artifact to build a trajectory with
                # for that topology yet. Recorded below, not hidden.
                skipped_unsupported_topology[family] += 1
                continue
            try:
                # core-issues3.txt Phase 1: reconstruct THIS scenario's own
                # randomized hydraulic state (demand regime, roughness,
                # tank levels, pipe status) rather than reusing one pristine
                # network/context shared by every scenario in this topology
                # family. Fails closed (raises) if the reconstruction does
                # not semantically match this scenario's own stored arrays.
                #
                # development_holdout mixes plain-curriculum scenarios
                # (_degradation_probabilities(stage), verifiable) with two
                # OOD-holdout helpers generate_cycle_b_corpus.py's own
                # _generate_ood_holdout_for_training_topology fits with a
                # DIFFERENT, hardcoded degradation formula
                # (missing_probability=0.45 etc.) -- run_corpus_gates.py's
                # deterministic_replay gate already documents this exact
                # split as excluded because "which formula was used ...
                # cannot be distinguished from the manifest alone." Retry
                # without the array-level check (replay_sha256's seed/
                # source/network/stage/split identity match still holds --
                # only the degradation-outcome verification is skipped) for
                # this split only, rather than spuriously failing ~1/4 of it.
                weak_verification = False
                try:
                    reconstruction = reconstruct_scenario_network(
                        networks[family],
                        scenario.manifest,
                        degradation_policy=_degradation_probabilities,
                        original=scenario,
                    )
                except ScenarioReconstructionError:
                    if split is not DatasetSplit.DEVELOPMENT_HOLDOUT:
                        raise
                    reconstruction = reconstruct_scenario_network(
                        networks[family],
                        scenario.manifest,
                        degradation_policy=_degradation_probabilities,
                        original=None,
                    )
                    weak_verification = True
                # Always build the trajectory from the corpus's own stored,
                # already-ground-truth scenario -- never from
                # reconstruction.scenario, which is a regenerated copy
                # useful only for the verification check above. This also
                # sidesteps the development_holdout degradation-formula
                # ambiguity entirely for the actual targets: only the
                # verification step (irrelevant to what data is used) is
                # affected by it, never the data itself.
                result = build_incident_trajectory(
                    scenario,
                    reconstruction.network,
                    libraries[family],
                    artifacts[family],
                    topology_hash=scenario.manifest.network_sha256,
                    validated_topology_hashes=validated_topology_hashes,
                    feature_context=reconstruction.feature_context,
                    maximum_samples=args.maximum_samples,
                    maximum_exact_simulations=args.maximum_exact_simulations,
                    maximum_plans=args.maximum_plans,
                    reconstruction=reconstruction,
                )
            except Exception as error:  # noqa: BLE001 -- record and continue (incl. ScenarioReconstructionError), never silently drop the corpus
                error_count += 1
                print(f"ERROR scenario {scenario_id}: {error!r}")
                continue
            stream.write(json.dumps(_incident_trajectory_to_json(result, reconstruction, weak_verification)) + "\n")
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
        "signature_artifact_policy": TOPOLOGY_WIDE_REGIME_HASH,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_dir": str(args.corpus_dir),
        "split": split.value,
        "scenarios_processed_this_run": processed_count,
        "errors_this_run": error_count,
        "total_in_shard": len(already_processed) + processed_count,
        "ood_category_counts_this_run": dict(ood_counts),
        "validated_topology_hashes": sorted(validated_topology_hashes),
        # core-issues3.txt Phase 1 item J: report every omission explicitly.
        # Nonzero here means this split contains topology families this
        # script has no governed signature artifact for (currently
        # development_holdout's coastal-branch/unseen-topology scenarios --
        # see Phase 2/5's "do not fit a topology-specific artifact from
        # development-holdout incidents" restriction). Not resolved by this
        # script; resolving it requires the governed unseen-topology
        # Scout/Strategist-unavailable fallback path Phase 2 item 5
        # describes, tracked separately.
        "skipped_unsupported_topology_this_run": dict(skipped_unsupported_topology),
        "elapsed_seconds": time.time() - started_at,
    }
    report_path = args.output / f"{split.value}-report.json"
    existing_report = json.loads(report_path.read_text()) if report_path.exists() else None
    report["previous_run"] = existing_report
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {processed_count} new records ({error_count} errors) to {shard_path}")
    if skipped_unsupported_topology:
        print(f"WARNING: skipped unsupported-topology scenarios: {dict(skipped_unsupported_topology)}")
    print(f"report: {report_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
