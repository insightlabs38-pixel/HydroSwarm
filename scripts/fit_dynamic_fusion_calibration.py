"""Fit conformal calibration on the exact deployed dynamic hybrid predictor
(core-issues.txt Phase 3 item 18), not the fixed_weight_fusion approximation
Stage 3/4 use for their own internal per-seed comparison.

Why this exists (see reports/results/v3/phase3-handoff.md's "Calibration fix
design notes" for the full writeup): a stored ScenarioExample's tensors do
not preserve the live HydraulicStateEstimator/classical-localizer state
fuse_source_probabilities' dynamic per-incident trust weighting
(TrustFeatures) needs, so Stage 3/4 calibrate against fixed_weight_fusion, a
documented static-weight approximation instead. The correct fix is not a
smarter formula over stored tensors -- it is re-running the real classical/
estimation pipeline per calibration scenario:

1. Build (once per training topology, cached) a real SignatureArtifact via
   SignatureBuilder, using the exact bins production actually uses
   (src/hydroswarm/runtime/defaults.py), not the training corpus's broader
   generation grid -- matching what will actually run in deployment is the
   point.
2. For each calibration-split scenario, regenerate its exact scenario +
   randomized network deterministically from its recorded seed (same
   mechanism as run_corpus_gates.py's deterministic_replay gate), rebuild
   its real SensorSeries (hydroswarm.training.corpus.build_sensor_series,
   the same function scenario_to_example itself uses), and run the real,
   already-trained-checkpoint-wired HybridInferencePipeline.analyze() end to
   end -- the actual production entry point, not a hand-assembled
   approximation of it.
3. Extract the fused belief, map it into topology.node_ids order, and fit
   SplitConformalCalibrator on those real fused vectors with
   fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG (not
   fixed_weight_fusion_config(...)).

Scope note, stated honestly rather than silently assumed away: corpus
generation's own hydraulic-state estimate is always built from an *empty*
OperationalTelemetry (see hydroswarm.training.corpus.build_feature_context),
so this script's estimated.normalized_uncertainty is a fixed baseline for
every scenario -- there is no separate, telemetry-based state correction
signal in this corpus to recover. Every other TrustFeatures input
(healthy_sensor_fraction, missing_rate, neural/classical entropy, OOD score,
and the classical/neural disagreement) is genuinely computed fresh per
scenario from the real re-run classical localizer and the trained model's
own output. This is still categorically more faithful than
fixed_weight_fusion's single hardcoded neural_weight, and it is what
src/hydroswarm/runtime/defaults.py's production wiring actually computes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from safetensors.torch import load_file

from hydroswarm.calibration.conformal import CalibrationExample, SplitConformalCalibrator
from hydroswarm.classical import (
    SignatureArtifact,
    SignatureBuilder,
    SignatureCache,
    SignatureCacheKey,
)
from hydroswarm.data.scenarios import (
    CurriculumStage,
    DatasetSplit,
    EventType,
    GeneratedScenario,
    IncidentTruth,
    ScenarioGenerationConfig,
    ScenarioManifest,
    WNTRScenarioGenerator,
)
from hydroswarm.inference import DYNAMIC_TRUST_FUSION_CONFIG, HybridInferencePipeline
from hydroswarm.inference.pipeline import RUNTIME_TASKS
from hydroswarm.model import HydroCore
from hydroswarm.preprocessing import HydraulicFeatureBuilder
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats
from hydroswarm.simulation.wrapper import HydraulicSimulator
from hydroswarm.training.corpus import build_feature_context, build_sensor_series

# Importing generate_cycle_b_corpus's own topology loaders and per-stage
# degradation-probability formula (rather than duplicating them) is what
# makes deterministic scenario replay here actually deterministic -- see
# run_corpus_gates.py's deterministic_replay gate, which does the same.
from generate_cycle_b_corpus import DEV_OOD_TOPOLOGY, TRAIN_TOPOLOGIES, _degradation_probabilities

# Production wiring's exact signature-artifact configuration
# (src/hydroswarm/runtime/defaults.py) -- deliberately not the training
# corpus's broader generation grid; matching what will actually run in
# deployment is the point of this script.
START_TIME_BINS = (0, 60)
DURATION_BINS = (30, 60)
STRENGTH_BINS = (0.5, 1.0)
DEMAND_REGIMES = ("nominal",)
SAMPLE_TIMES_SECONDS = tuple(range(0, 6 * 3_600 + 1, 3_600))


def _topology_loaders() -> dict[str, Any]:
    loaders = {name: loader for name, loader in TRAIN_TOPOLOGIES}
    dev_name, dev_loader = DEV_OOD_TOPOLOGY
    loaders[dev_name] = dev_loader
    return loaders


def _record_to_manifest(record: dict[str, Any]) -> ScenarioManifest:
    incident_record = dict(record["incident"])
    incident_record["source_nodes"] = tuple(incident_record["source_nodes"])
    return ScenarioManifest(
        scenario_id=UUID(record["scenario_id"]),
        split=DatasetSplit(record["split"]),
        stage=CurriculumStage(record["stage"]),
        network_id=record["network_id"],
        network_family=record["network_family"],
        network_sha256=record["network_sha256"],
        seed=record["seed"],
        seed_family=record["seed_family"],
        simulator_version=record["simulator_version"],
        generator_version=record["generator_version"],
        incident=IncidentTruth(**incident_record),
        sensor_nodes=tuple(record["sensor_nodes"]),
        model_mismatch=record["model_mismatch"],
        artifact_sha256=record["artifact_sha256"],
        replay_sha256=record["replay_sha256"],
        created_with_python=record["created_with_python"],
        event_type=record.get("event_type", "contamination"),
    )


def build_signature_artifact(network_family: str, network: Any, *, cache_dir: Path) -> SignatureArtifact:
    """One real SignatureArtifact per training topology, cached to disk so a
    full calibration run only pays for the hypothesis-grid simulation once
    per topology (see SignatureBuilder.build_or_load)."""

    simulator = HydraulicSimulator(network)
    junctions = tuple(sorted(network.junction_name_list))
    key = SignatureCacheKey(
        network_hash=simulator.state_hash(),
        hydraulic_state_hash=simulator.state_hash(),
        simulator_version=simulator.simulator_version,
        configuration_hash=f"dynamic-fusion-calibration-v1:{network_family}",
        sensor_layout_hash="|".join(junctions),
    )
    return SignatureBuilder(simulator, SignatureCache(cache_dir)).build_or_load(
        key=key,
        source_nodes=junctions,
        start_time_bins=START_TIME_BINS,
        duration_bins=DURATION_BINS,
        strength_bins=STRENGTH_BINS,
        demand_regimes=DEMAND_REGIMES,
        sensor_nodes=junctions,
        sample_times_seconds=SAMPLE_TIMES_SECONDS,
    )


def load_model(checkpoint: Path, *, variant: str, overrides: dict[str, Any]) -> HydroCore:
    model = HydroCore.from_variant(variant, **overrides)
    state_dict = load_file(str(checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint does not match model architecture: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def fit(
    *,
    corpus_dir: Path,
    checkpoint: Path,
    variant: str,
    overrides: dict[str, Any],
    node_normalization: Path,
    edge_normalization: Path,
    signature_cache_dir: Path,
    alpha: float,
    sample_per_topology: int | None,
) -> dict[str, Any]:
    model = load_model(checkpoint, variant=variant, overrides=overrides)
    model_hash = HybridInferencePipeline._fingerprint_model(model)

    node_stats = NormalizationStats.load(node_normalization)
    edge_stats = NormalizationStats.load(edge_normalization)
    feature_builder = HydraulicFeatureBuilder(node_normalization=node_stats, edge_normalization=edge_stats)

    loaders = _topology_loaders()
    signature_cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_by_family: dict[str, SignatureArtifact] = {}

    scenarios_root = corpus_dir / "scenarios"
    manifest_path = scenarios_root / "manifests" / "calibration.jsonl"
    lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_per_topology is not None:
        by_family: dict[str, list[str]] = {}
        for line in lines:
            family = json.loads(line)["network_family"]
            by_family.setdefault(family, []).append(line)
        lines = [line for group in by_family.values() for line in group[:sample_per_topology]]

    examples: list[CalibrationExample] = []
    per_scenario_diagnostics: list[dict[str, Any]] = []
    topology_hashes_used: set[str] = set()
    for line in lines:
        record = json.loads(line)
        manifest = _record_to_manifest(record)
        if manifest.event_type != EventType.CONTAMINATION.value:
            # NORMAL/SENSOR_FAULT_ONLY scenarios still call simulate_incident
            # with a real (but near-zero-strength, uninjected-in-effect)
            # source node -- manifest.incident.source_nodes[0] is not a real
            # source to localize for these (mirrors how Stage 3's
            # _predict_rows skips masked source_node placeholders); fitting
            # calibration against them as if it were would train the
            # conformal quantile on meaningless labels.
            continue
        family = manifest.network_family
        if family not in loaders:
            raise ValueError(f"no known pristine topology loader for network_family {family!r}")
        pristine = loaders[family]()
        if family not in artifacts_by_family:
            artifacts_by_family[family] = build_signature_artifact(family, pristine, cache_dir=signature_cache_dir)

        generator = WNTRScenarioGenerator()
        artifact_path = scenarios_root / manifest.split.value / f"{manifest.scenario_id}.npz"
        with np.load(artifact_path) as arrays:
            scenario = GeneratedScenario(
                manifest=manifest,
                timestamps_seconds=arrays["timestamps_seconds"],
                truth_concentration=arrays["truth_concentration"],
                observed_concentration=arrays["observed_concentration"],
                observation_mask=arrays["observation_mask"],
                frozen_mask=arrays["frozen_mask"],
                communication_outage_mask=arrays["communication_outage_mask"],
                timestamp_jitter_seconds=arrays["timestamp_jitter_seconds"],
                sensor_nodes=manifest.sensor_nodes,
                flow_reversal_mask=arrays["flow_reversal_mask"],
                drift_mask=arrays["drift_mask"],
                unit_mismatch_mask=arrays["unit_mismatch_mask"],
            )
        # Rebuild the exact randomized network this scenario was simulated
        # against, deterministically from its recorded seed/config (same
        # mechanism as run_corpus_gates.py's deterministic_replay gate) --
        # the raw .npz artifact stores telemetry arrays, not the model.
        config = ScenarioGenerationConfig(
            seed=manifest.seed,
            network_id=manifest.network_id,
            network_family=family,
            split=manifest.split,
            stage=manifest.stage,
            event_type=EventType(manifest.event_type),
            source_node=manifest.incident.source_nodes[0],
            sensor_count=len(manifest.sensor_nodes),
            pipe_outage_probability=0.0,
            **_degradation_probabilities(manifest.stage),
        )
        _replayed_scenario, randomized_network = generator.generate_with_network(pristine, config)

        context = build_feature_context(randomized_network)
        series = build_sensor_series(scenario, context)

        pipeline = HybridInferencePipeline(
            simulator=HydraulicSimulator(randomized_network),
            signature_artifact=artifacts_by_family[family],
            model=model,
            model_hash=model_hash,
            calibration_artifact=None,
            feature_builder=feature_builder,
            trained_tasks=frozenset({"sentinel"}) & RUNTIME_TASKS,
            fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
        )
        result = pipeline.analyze(manifest.scenario_id, randomized_network, series)

        node_order = tuple(sorted(randomized_network.node_name_list))
        fused_vector = np.asarray([result.fused_belief.get(node, 0.0) for node in node_order], dtype=np.float64)
        if not np.isfinite(fused_vector).all() or fused_vector.sum() <= 0:
            continue  # CLASSICAL_SAFE fallback or degenerate belief: not usable for calibration
        true_source = manifest.incident.source_nodes[0]
        if true_source not in node_order:
            continue
        true_index = node_order.index(true_source)

        examples.append(
            CalibrationExample(
                probabilities=tuple(float(value) for value in fused_vector),
                true_index=true_index,
                condition=manifest.stage.name,
                network_id=manifest.network_id,
            )
        )
        topology_hashes_used.add(manifest.network_sha256)
        classical_top1 = max(result.classical_belief, key=result.classical_belief.get)
        neural_top1 = (
            max(result.neural_belief, key=result.neural_belief.get) if result.neural_belief else None
        )
        fused_top1 = max(result.fused_belief, key=result.fused_belief.get)
        per_scenario_diagnostics.append(
            {
                "scenario_id": str(manifest.scenario_id),
                "network_family": family,
                "runtime_mode": result.runtime_mode.value,
                "classical_trust": (
                    result.fusion_diagnostics.classical_trust if result.fusion_diagnostics else None
                ),
                "disagreement_js": (
                    result.fusion_diagnostics.disagreement_js if result.fusion_diagnostics else None
                ),
                "true_source": true_source,
                "classical_top1_correct": classical_top1 == true_source,
                "neural_top1_correct": (neural_top1 == true_source) if neural_top1 is not None else None,
                "fused_top1_correct": fused_top1 == true_source,
            }
        )

    if not examples:
        raise ValueError("no usable calibration examples were produced (see per-scenario diagnostics)")

    calibrator = SplitConformalCalibrator.fit(
        examples,
        alpha=alpha,
        model_hash=model_hash,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        dataset_manifest_hash="calibration-split:" + manifest_path.name,
        normalization_hash=feature_builder.normalization_fingerprint,
        fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
        topology_hashes=tuple(sorted(topology_hashes_used)),
    )
    return {
        "calibrator": calibrator,
        "examples_used": len(examples),
        "examples_skipped": len(lines) - len(examples),
        "per_scenario_diagnostics": per_scenario_diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="small")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--node-normalization", type=Path, required=True)
    parser.add_argument("--edge-normalization", type=Path, required=True)
    parser.add_argument("--signature-cache-dir", type=Path, default=Path("experiments/cache/signatures"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--sample-per-topology",
        type=int,
        default=None,
        help="cap calibration scenarios per topology (default: use the full calibration split)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-artifact-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = fit(
        corpus_dir=args.corpus_dir,
        checkpoint=args.checkpoint,
        variant=args.variant,
        overrides=json.loads(args.overrides_json),
        node_normalization=args.node_normalization,
        edge_normalization=args.edge_normalization,
        signature_cache_dir=args.signature_cache_dir,
        alpha=args.alpha,
        sample_per_topology=args.sample_per_topology,
    )
    calibrator: SplitConformalCalibrator = result.pop("calibrator")
    calibrator.save(args.calibration_artifact_output)
    report = {
        "schema_version": 1,
        "fusion_config": DYNAMIC_TRUST_FUSION_CONFIG,
        "checkpoint": str(args.checkpoint),
        "calibration_artifact": str(args.calibration_artifact_output),
        "calibration_artifact_hash": calibrator.artifact.artifact_hash,
        "alpha": args.alpha,
        "examples_used": result["examples_used"],
        "examples_skipped": result["examples_skipped"],
        "coverage": calibrator.artifact.report.coverage,
        "mean_set_size": calibrator.artifact.report.mean_set_size,
        "expected_calibration_error": calibrator.artifact.report.expected_calibration_error,
        "per_scenario_diagnostics": result["per_scenario_diagnostics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_scenario_diagnostics"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
