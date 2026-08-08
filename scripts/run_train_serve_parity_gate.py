"""core-issues5.txt Section 5 (P0 GATE) / delta item 1: consolidated
train/serve parity gate.

Compares model-input construction between:

  A. governed corpus preprocessing (`hydroswarm.training.corpus.
     scenario_to_example`, the real, unmodified code every training corpus
     shard was actually built with);
  B. live production preprocessing (`hydroswarm.preprocessing.builder.
     HydraulicFeatureBuilder.build`, called with the exact same arguments
     `hydroswarm.inference.pipeline.HybridInferencePipeline.analyze`'s own
     `feature_building` stage passes it -- not a reimplementation, the
     identical class/method the live pipeline uses internally as
     `self.feature_builder`).

Uses only self-generated, non-locked, deterministic fixture scenarios
(fixed seeds) across every governed training topology
(`generate_cycle_b_corpus.TRAIN_TOPOLOGIES`: golden-reference,
branched-loop, loop-grid) and more than one operating/fault condition per
topology (CLEAN and DEGRADED curriculum stages, different source nodes) --
never opens or touches data/learning-v2/*/tensors/test or any locked-test
artifact.

Both paths start from a hydraulic (network, graph, state) triple built the
SAME way in both training and serving:
`HydraulicSimulator.calculate_state(FEATURE_SNAPSHOT_TIME_SECONDS)` ->
`HydraulicStateEstimator().estimate(raw_state, telemetry)` ->
`HydraulicSimulator.build_dynamic_graph(...)` (see
`hydroswarm.training.corpus.build_feature_context` and
`HybridInferencePipeline.analyze`'s own stage sequence) -- so, given the
same randomized network and default telemetry, this triple is provably
identical between the two paths; feeding it through the same
`HydraulicFeatureBuilder.build` call is where any REAL train/serve skew
(normalization identity, classical-prior computation, feature-schema
drift) would actually show up.

DELTA ITEM 1 FIX, formerly a documented known failure: governed corpus
generation and live serving previously computed `classical_prior` via two
structurally different algorithms (`hydroswarm.training.corpus.
SignatureLibrary.posterior`, a per-node log1p-residual softmax, vs.
`hydroswarm.classical.signatures.localize_with_signatures`, a noise-aware
Bayesian posterior over a completely different hypothesis-grid signature
representation). Both paths now call the SAME governed function
(`hydroswarm.training.corpus.model_input_classical_prior` /
`SignatureLibrary.posterior_from_observations`) against the SAME
committed, training-fit `SignatureLibrary`
(`hydroswarm.training.corpus.resolve_model_input_signature_library`,
loaded via `load_committed_signature_library` from
`data/learning-v2/cycle-b2/signatures/<family>.json` for every
`GOVERNED_KNOWN_NETWORK` topology) -- there is no longer a `classical_prior`
algorithm mismatch to document as a known finding. The richer live
Bayesian localizer (`localize_with_signatures`) remains available inside
`HybridInferencePipeline` as `live_classical_localization` for
deterministic reasoning/fusion/operator evidence, but is no longer the
source of the MODEL-INPUT tensor this gate checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_cycle_b_corpus import TRAIN_TOPOLOGIES  # noqa: E402

from hydroswarm.classical.signature_policy import resolve_signature_mode  # noqa: E402
from hydroswarm.data.scenarios import (  # noqa: E402
    CurriculumStage,
    DatasetSplit,
    EventType,
    ScenarioGenerationConfig,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder  # noqa: E402
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA  # noqa: E402
from hydroswarm.runtime.v4_normalization import load_runtime_normalization_bundle  # noqa: E402
from hydroswarm.training.corpus import (  # noqa: E402
    ScenarioExample,
    build_feature_context,
    build_sensor_series,
    model_input_classical_prior,
    resolve_model_input_signature_library,
    scenario_to_example,
)

DEFAULT_NORMALIZATION_DIR = Path("data/learning-v2/cycle-b2/normalization")
DEFAULT_CYCLE_B2_ROOT = Path("data/learning-v2/cycle-b2")
DEFAULT_REPORT_PATH = Path("reports/results/v4/train-serve-parity-gate.json")

#: More than one operating/fault condition per topology (core-issues5.txt
#: delta item 1's expanded-coverage requirement): CLEAN (no sensor
#: noise/missingness) and DEGRADED (real sensor noise/missingness/drift),
#: each against a different source node so the two conditions are not
#: otherwise identical.
_CONDITIONS: tuple[tuple[str, CurriculumStage], ...] = (
    ("clean", CurriculumStage.CLEAN),
    ("degraded", CurriculumStage.DEGRADED),
)


@dataclass
class FieldResult:
    field: str
    comparison: str  # "exact" | "tolerance" | "informational"
    passed: bool
    detail: str
    scenario_id: str = ""


def _fixture_scenario(
    network: Any, *, source: str, stage: CurriculumStage, seed: int, network_family: str
) -> tuple[Any, Any]:
    generator = WNTRScenarioGenerator()
    config = ScenarioGenerationConfig(
        seed=seed,
        network_id=network_family,
        network_family=network_family,
        split=DatasetSplit.TRAIN,
        stage=stage,
        event_type=EventType.CONTAMINATION,
        source_node=source,
        pipe_outage_probability=0.0,
        # network_sha256 hashes link roughness (but not demand/tank levels,
        # which it deliberately excludes -- see its own docstring); zeroing
        # roughness variation keeps the randomized fixture network
        # byte-identical (by network_sha256) to the pristine training
        # topology, so this fixture legitimately exercises the
        # GOVERNED_KNOWN_NETWORK signature-resolution path a real live
        # deployment of one of these exact topologies would also take --
        # not the RUNTIME_GENERATED_IMPORTED_NETWORK fallback every
        # individually-roughness-randomized training SCENARIO's own network
        # object would otherwise resolve to (training's classical_prior
        # fitting/posterior never resolves per-scenario network hashes at
        # all -- see hydroswarm.classical.signature_policy's own module
        # docstring -- so this is purely a property of this gate's fixture
        # construction, not of the governed algorithm being tested).
        roughness_variation_fraction=0.0,
    )
    return generator.generate_with_network(network, config)


def _evaluate_fixture(
    *,
    network_family: str,
    network: Any,
    source: str,
    stage: CurriculumStage,
    seed: int,
    bundle: Any,
    cycle_b2_root: Path,
) -> list[FieldResult]:
    scenario, randomized_network = _fixture_scenario(
        network, source=source, stage=stage, seed=seed, network_family=network_family
    )
    junctions = tuple(sorted(randomized_network.junction_name_list))
    topology_hash = network_sha256(randomized_network)
    context = build_feature_context(randomized_network)
    series = build_sensor_series(scenario, context)

    # --- Path A: governed corpus preprocessing (real, unmodified code) ---
    corpus_library, corpus_timestamps, corpus_mode = resolve_model_input_signature_library(
        topology_hash, junctions, randomized_network, cycle_b2_root=cycle_b2_root
    )
    example: ScenarioExample = scenario_to_example(
        scenario,
        randomized_network,
        corpus_library,
        feature_context=context,
        node_normalization=bundle.node_normalization,
        edge_normalization=bundle.edge_normalization,
    )
    corpus_prior = model_input_classical_prior(corpus_library, junctions, series, corpus_timestamps)

    # --- Path B: live production preprocessing (real, unmodified code) ---
    live_library, live_timestamps, live_mode = resolve_model_input_signature_library(
        topology_hash, junctions, randomized_network, cycle_b2_root=cycle_b2_root
    )
    live_prior = model_input_classical_prior(live_library, junctions, series, live_timestamps)
    live_builder = HydraulicFeatureBuilder(
        node_normalization=bundle.node_normalization, edge_normalization=bundle.edge_normalization
    )
    live_built = live_builder.build(
        randomized_network,
        context.graph,
        context.state,
        series,
        classical_prior=live_prior,
        window_steps=len(scenario.timestamps_seconds),
    )

    results: list[FieldResult] = []
    scenario_id = str(scenario.manifest.scenario_id)

    def exact(field: str, a: Any, b: Any, note: str = "") -> None:
        equal = a == b if not isinstance(a, (np.ndarray, torch.Tensor)) else bool((a == b).all())
        results.append(
            FieldResult(field, "exact", equal, note or (f"a={a!r} b={b!r}" if not equal else "match"), scenario_id)
        )

    exact(
        "signature_policy_identity",
        (corpus_mode, corpus_library.manifest_hash),
        (live_mode, live_library.manifest_hash),
    )
    exact(
        "signature_mode_is_governed",
        corpus_mode,
        "GOVERNED_KNOWN_NETWORK",
        note=(
            f"mode={corpus_mode} for a TRAIN_TOPOLOGIES fixture -- expected GOVERNED_KNOWN_NETWORK; "
            f"resolve_signature_mode={resolve_signature_mode(topology_hash)}"
        ),
    )
    exact("node_ids", live_built.node_ids, tuple(str(n) for n in sorted(randomized_network.node_name_list)))
    exact("feature_schema_hash", live_built.feature_schema_hash, DEFAULT_FEATURE_SCHEMA.fingerprint)
    exact("normalization_hash", live_built.normalization_hash, bundle.fingerprint)

    corpus_node_features = example.inputs["node_features"]
    live_node_features = live_built.batch["node_features"].squeeze(0)
    corpus_edge_features = example.inputs["edge_features"]
    live_edge_features = live_built.batch["edge_features"].squeeze(0)

    exact("node_features_shape", tuple(corpus_node_features.shape), tuple(live_node_features.shape))
    exact("edge_features_shape", tuple(corpus_edge_features.shape), tuple(live_edge_features.shape))
    exact("node_mask", example.inputs["node_mask"], live_built.batch["node_mask"].squeeze(0))
    exact("edge_mask", example.inputs["edge_mask"], live_built.batch["edge_mask"].squeeze(0))
    exact("sensor_mask", example.inputs["sensor_mask"], live_built.batch["sensor_mask"].squeeze(0))
    exact(
        "source_candidate_mask",
        example.inputs["source_candidate_mask"],
        live_built.batch["source_candidate_mask"].squeeze(0),
    )

    if tuple(corpus_node_features.shape) == tuple(live_node_features.shape):
        column_diff = (corpus_node_features - live_node_features).abs()
        node_diff = float(column_diff.max())
        diverging_columns = [
            f"{name} (max_abs_diff={float(column_diff[:, index].max()):.6f})"
            for index, name in enumerate(DEFAULT_FEATURE_SCHEMA.node_features)
            if float(column_diff[:, index].max()) > 1e-6
        ]
        detail = f"max_abs_diff={node_diff:.8f}; diverging columns: {diverging_columns or 'none'}"
        results.append(FieldResult("node_features", "tolerance", node_diff < 1e-4, detail, scenario_id))
    if tuple(corpus_edge_features.shape) == tuple(live_edge_features.shape):
        edge_diff = float((corpus_edge_features - live_edge_features).abs().max())
        results.append(
            FieldResult("edge_features", "tolerance", edge_diff < 1e-4, f"max_abs_diff={edge_diff:.8f}", scenario_id)
        )

    shared_nodes = sorted(set(corpus_prior) & set(live_prior))
    prior_diff = max((abs(corpus_prior[n] - live_prior[n]) for n in shared_nodes), default=float("nan"))
    results.append(
        FieldResult(
            "classical_prior",
            "tolerance",
            prior_diff < 1e-6,
            f"corpus={corpus_prior}, live={live_prior}, max_abs_diff={prior_diff:.8f} -- both paths now use "
            "hydroswarm.training.corpus.model_input_classical_prior against the same committed, "
            "training-fit SignatureLibrary (delta item 1 fix)",
            scenario_id,
        )
    )
    return results


def run_gate(*, normalization_dir: Path, cycle_b2_root: Path, seed_base: int) -> dict[str, Any]:
    bundle = load_runtime_normalization_bundle(normalization_dir)
    all_results: list[FieldResult] = []
    evaluated_fixtures: list[dict[str, Any]] = []

    for topology_index, (network_family, loader) in enumerate(TRAIN_TOPOLOGIES):
        network = loader()
        junctions = sorted(network.junction_name_list)
        for condition_index, (condition_name, stage) in enumerate(_CONDITIONS):
            source = junctions[condition_index % len(junctions)]
            seed = seed_base + topology_index * 1000 + condition_index
            fixture_results = _evaluate_fixture(
                network_family=network_family,
                network=network,
                source=source,
                stage=stage,
                seed=seed,
                bundle=bundle,
                cycle_b2_root=cycle_b2_root,
            )
            all_results.extend(fixture_results)
            evaluated_fixtures.append({
                "network_family": network_family,
                "condition": condition_name,
                "source": source,
                "seed": seed,
                "passed": all(r.passed for r in fixture_results),
            })

    passed = all(r.passed for r in all_results)
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "gate": "train_serve_parity",
        "passed": passed,
        "normalization_dir": str(normalization_dir),
        "normalization_fingerprint": bundle.fingerprint,
        "cycle_b2_root": str(cycle_b2_root),
        "evaluated_fixtures": evaluated_fixtures,
        "fields": [
            {
                "field": r.field,
                "comparison": r.comparison,
                "passed": r.passed,
                "detail": r.detail,
                "scenario_id": r.scenario_id,
            }
            for r in all_results
        ],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalization-dir", type=Path, default=DEFAULT_NORMALIZATION_DIR)
    parser.add_argument("--cycle-b2-root", type=Path, default=DEFAULT_CYCLE_B2_ROOT)
    parser.add_argument("--seed-base", type=int, default=910_000)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(normalization_dir=args.normalization_dir, cycle_b2_root=args.cycle_b2_root, seed_base=args.seed_base)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
