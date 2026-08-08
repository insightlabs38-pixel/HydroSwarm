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
   SignatureBuilder, using a hypothesis grid wide enough to actually
   represent this corpus's incident parameters (start/duration/strength
   ranges matching ScenarioGenerationConfig's own defaults) -- NOT
   src/hydroswarm/runtime/defaults.py's narrower production grid. That
   narrower grid was tried first and is a real, separately-flagged finding
   in its own right (see START_TIME_BINS's docstring below): it cannot
   represent this corpus's actual incidents and measurably degrades even
   the classical localizer alone (65% vs. 95% top-1 in a direct 20-scenario
   A/B), which is a live-inference concern for a human to weigh, not
   something to silently work around by picking whichever bins happen to
   produce a better number here.
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
import hashlib
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
    network_sha256,
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

# A second real finding from this script (Phase 3 item 18), distinct from
# the hydraulic-snapshot-time defect fixed in hydroswarm/inference/pipeline.py:
# src/hydroswarm/runtime/defaults.py's production wiring builds its
# SignatureArtifact with a narrow hypothesis grid --
# start_time_bins=(0, 60), duration_bins=(30, 60), strength_bins=(0.5, 1.0)
# -- that does not span this corpus's actual incident-parameter ranges
# (ScenarioGenerationConfig's defaults: start up to 240 min, duration up to
# 120 min, strength up to 2.0). Verified empirically: on 20 real
# golden-reference calibration scenarios, the classical localizer's own
# top-1 accuracy was 13/20 (65%) with the narrow production bins and 19/20
# (95%) with bins spanning the corpus's actual generation grid -- a
# hypothesis grid that cannot represent the true incident cannot localize
# it correctly regardless of the model. This is very likely a live
# accuracy gap in production today, not only a corpus-comparison artifact
# -- flagged prominently in reports/results/v3/phase3-handoff.md as its own
# follow-up, deliberately NOT changed in src/hydroswarm/runtime/defaults.py
# by this script (that is a live-inference latency/config tradeoff decision
# for a human to make, not a mechanical bug fix). This script uses the
# wider, corpus-matching grid below so ITS OWN calibration fit reflects the
# trained model's and real dynamic fusion's genuine quality, not an
# artifact of a hypothesis grid that structurally cannot answer correctly.
START_TIME_BINS = (0, 60, 120, 240)
DURATION_BINS = (30, 60, 120)
STRENGTH_BINS = (0.5, 1.0, 2.0)
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
    checkpoint: Path | None = None,
    variant: str = "small",
    overrides: dict[str, Any] | None = None,
    identity_dir: Path | None = None,
    node_normalization: Path,
    edge_normalization: Path,
    signature_cache_dir: Path,
    alpha: float,
    sample_per_topology: int | None,
) -> dict[str, Any]:
    # core-issues5.txt Section 19: fit against the exact frozen V4 serving
    # path -- the real governed checkpoint identity (architecture,
    # trained/validated/runtime_enabled_outputs), not a hand-specified
    # variant/overrides dict that could silently drift from what the
    # checkpoint was actually built/governed as. identity_dir takes
    # priority when supplied; checkpoint/variant/overrides remain for the
    # pre-v4 caller/tests this script has always supported.
    runtime_enabled_outputs: frozenset[str] | None = None
    if identity_dir is not None:
        from hydroswarm.training.checkpoint_identity import load_v4_checkpoint

        model, identity, _trainer_state = load_v4_checkpoint(identity_dir)
        model.eval()
        runtime_enabled_outputs = identity.runtime_enabled_outputs
        # core-issues5.txt Section 19: model_hash must match what the REAL
        # runtime loader actually validates a calibration artifact against
        # -- hydroswarm.runtime.v4_defaults.V4PipelineFactory._load_assets
        # (and, downstream, scripts/build_v4_inference_release_bundle.py's
        # own calibration packaging) both compute model_hash as the raw
        # model.safetensors FILE content hash, NOT
        # HybridInferencePipeline._fingerprint_model's state-dict-tensor
        # hash (a different, incompatible formula used only by this
        # script's own legacy checkpoint/variant/overrides path below). A
        # calibration artifact fit with the wrong formula would fail
        # validate_runtime at every real load site.
        model_hash = hashlib.sha256((identity_dir / "model.safetensors").read_bytes()).hexdigest()
    else:
        assert checkpoint is not None
        model = load_model(checkpoint, variant=variant, overrides=overrides or {})
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
        # Rebuild the exact randomized network this scenario was simulated
        # against, deterministically from its recorded seed/config (same
        # mechanism as run_corpus_gates.py's deterministic_replay gate).
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
        replayed_scenario, randomized_network = generator.generate_with_network(pristine, config)
        artifact_path = scenarios_root / manifest.split.value / f"{manifest.scenario_id}.npz"
        # core-issues5.txt Section 19: prefer the committed, disk-verified
        # raw scenario artifact when present (byte-identical telemetry
        # arrays); fall back to the deterministic in-process reconstruction
        # above when it is not -- e.g. this sandbox's raw scenario .npz
        # archives are gitignored/ephemeral (see hydroswarm_checkpoint_
        # persistence memory record / Phase 17's own established handling
        # of this exact condition) and are not materialized here. Both
        # paths use the identical seeded WNTRScenarioGenerator call; the
        # fallback is real physics (a real EPANET simulation just run),
        # not fabricated data -- only unverified against a persisted
        # baseline that does not exist in this environment. Counted
        # separately in the report so this is never silently indistinguishable
        # from a disk-verified run.
        if artifact_path.exists():
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
            reconstructed_from_replay = False
        else:
            scenario = replayed_scenario
            reconstructed_from_replay = True

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
            # core-issues5.txt Section 19: None (legacy behavior) unless
            # identity_dir governs this run, in which case the real
            # checkpoint identity's own runtime_enabled_outputs is
            # authoritative here exactly as it is in live V4 serving
            # (hydroswarm.runtime.v4_defaults.V4PipelineFactory) -- source_
            # node itself is gated by it (core-issues5.txt delta item 2),
            # so calibration must be fit under the same gating live serving
            # will actually apply, not a permissive default.
            runtime_enabled_outputs=runtime_enabled_outputs,
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
        # core-issues5.txt Section 19: record the PRISTINE per-family
        # network hash, not manifest.network_sha256 (that scenario's own
        # roughness-randomized network) -- a real served network is the
        # unrandomized pristine topology (matching hydroswarm.classical.
        # signature_policy's own GOVERNED_KNOWN_NETWORK precedent: topology
        # governance identity is the pristine family hash, since training/
        # calibration scenarios randomize per-scenario roughness but a live
        # operator serves the base topology file). Recording the
        # per-scenario randomized hash instead would make
        # validated_topology_hashes match essentially no real served
        # network ever, silently making calibration unusable in practice.
        topology_hashes_used.add(network_sha256(pristine))
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
                "reconstructed_from_replay": reconstructed_from_replay,
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
        "examples_reconstructed_from_replay": sum(
            1 for row in per_scenario_diagnostics if row["reconstructed_from_replay"]
        ),
        "per_scenario_diagnostics": per_scenario_diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant", default="small")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument(
        "--identity-dir",
        type=Path,
        default=None,
        help=(
            "core-issues5.txt Section 19: a real v4 checkpoint directory "
            "(hydroswarm.training.checkpoint_identity.save_v4_checkpoint's own output) -- when "
            "given, the model and runtime_enabled_outputs are loaded from the identity itself "
            "instead of --checkpoint/--variant/--overrides-json"
        ),
    )
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
    if args.identity_dir is None and args.checkpoint is None:
        raise SystemExit("one of --identity-dir or --checkpoint is required")
    result = fit(
        corpus_dir=args.corpus_dir,
        checkpoint=args.checkpoint,
        variant=args.variant,
        overrides=json.loads(args.overrides_json),
        identity_dir=args.identity_dir,
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
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "identity_dir": str(args.identity_dir) if args.identity_dir else None,
        "calibration_artifact": str(args.calibration_artifact_output),
        "calibration_artifact_hash": calibrator.artifact.artifact_hash,
        "alpha": args.alpha,
        "examples_used": result["examples_used"],
        "examples_skipped": result["examples_skipped"],
        "examples_reconstructed_from_replay": result["examples_reconstructed_from_replay"],
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
