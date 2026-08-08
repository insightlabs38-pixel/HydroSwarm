"""Compose a v4-architecture pipeline from a checkpoint_identity-governed
checkpoint directory (core-issues3.txt Phase 15).

This is deliberately separate from `DefaultPipelineFactory`
(`hydroswarm.runtime.defaults`), which still serves the currently promoted
`models/hydrocore-s-learning-v1.safetensors` checkpoint and MUST NOT be
touched by this pre-architecture-freeze pass (restriction: do not overwrite
`data/learning-v1`/current checkpoints/historical results). No v4
architecture has passed Phase 19 selection or the locked test yet, so
nothing in this module is wired into the live production entry point --
it exists so the granular output-governance machinery
(`hydroswarm.training.output_governance`/`checkpoint_identity`) has a real,
testable runtime consumer ahead of that selection, per Phase 15's own
"update the runtime loader to reconstruct architecture v4 from checkpoint
metadata" and "replace broad trained_tasks behavior with granular output
gating" requirements.

Fails closed (`fallback_reason` set, `trained_assets_ready=False`,
`self.identity`/`self.model` stay `None`) on: missing checkpoint directory,
mismatched/corrupt `checkpoint_identity.json`, invalid calibration, or any
other asset-loading error -- mirrors `DefaultPipelineFactory`'s own
try/except discipline exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from hydroswarm.calibration import SplitConformalCalibrator
from hydroswarm.classical import SignatureBuilder, SignatureCache, SignatureCacheKey
from hydroswarm.inference import DYNAMIC_TRUST_FUSION_CONFIG, HybridInferencePipeline
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder
from hydroswarm.runtime.v4_normalization import (
    NO_NORMALIZATION_SENTINEL,
    NormalizationBundleError,
    load_runtime_normalization_bundle,
)
from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.simulation.wrapper import wntr
from hydroswarm.training.checkpoint_identity import (
    CheckpointIdentity,
    CheckpointIdentityError,
    NotAV4CheckpointError,
    load_v4_checkpoint,
)

#: core-issues5.txt Section 3: the real, committed, train-split-fit
#: normalization artifact Stage F's joint-v4 corpus was actually built
#: from (data/learning-v2/cycle-b2/tensors-normalized, merged into
#: cycle-b2-joint-v4 by scripts/build_stage_f_joint_corpus.py) -- NOT a
#: models/-prefixed path, because unlike the V3 baseline (which genuinely
#: trained without governed normalization -- see DefaultPipelineFactory's
#: own comment), this is the real corpus-owned artifact the currently
#: identified v4 checkpoint depends on. A future promoted V4 release
#: bundle (core-issues5.txt Section 12) should copy this artifact
#: alongside the model rather than reference the corpus tree directly;
#: this constant is this pass's correct, truthful default until that
#: packaging step exists.
DEFAULT_V4_NORMALIZATION_DIR = Path("data/learning-v2/cycle-b2/normalization")

#: Matches DefaultPipelineFactory's own established convention: only
#: "sentinel" is ever passed, since Scout/Strategist/OOD have not passed
#: Phase 14 promotion for any v4 checkpoint built so far -- see
#: reports/results/v4/phase14-promotion-gates.md. This is intentionally a
#: constant, not derived from the identity's runtime_enabled_outputs: the
#: granular set governs the NEW v4-only advisory fields
#: (event_presence/event_cause/next_step/evidence_sufficiency/etc, via
#: HybridInferencePipeline's own runtime_enabled_outputs parameter);
#: trained_tasks is the older, coarser Scout/Strategist/OOD role switch
#: that predates Phase 9 and still fully suppresses those roles' raw
#: outputs (sample_node/plan_value/ood_logits) regardless of what a v4
#: identity's finer-grained sets say.
V4_TRAINED_TASKS: frozenset[str] = frozenset({"sentinel"})


class V4PipelineFactory:
    """Lazily load a v4 checkpoint directory and build a per-network
    HybridInferencePipeline, granularly gated by the checkpoint's own
    declared `runtime_enabled_outputs`."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        project_root: str | Path | None = None,
        normalization_dir: str | Path | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.project_root = (
            Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
        )
        self.calibration_path: Path | None = None
        self.signature_cache = self.project_root / "data" / "generated" / "signatures"
        # core-issues5.txt Section 3: where to load the governed train-owned
        # node/edge normalization artifact from, keyed off
        # identity.normalization_hash at load time (NO_NORMALIZATION_SENTINEL
        # skips this entirely -- see _load_assets).
        self.normalization_dir = (
            Path(normalization_dir) if normalization_dir is not None
            else self.project_root / DEFAULT_V4_NORMALIZATION_DIR
        )
        self._model: Any | None = None
        self._identity: CheckpointIdentity | None = None
        self._calibrator: SplitConformalCalibrator | None = None
        self._feature_builder: HydraulicFeatureBuilder | None = None
        self._model_hash: str | None = None
        self._load_attempted = False
        self.fallback_reason: str | None = None

    @property
    def identity(self) -> CheckpointIdentity | None:
        self._load_assets()
        return self._identity

    @property
    def trained_assets_ready(self) -> bool:
        self._load_assets()
        return self._model is not None and self._identity is not None

    def _load_assets(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            model, identity, _trainer_state = load_v4_checkpoint(self.checkpoint_dir)
            model_weights_path = self.checkpoint_dir / "model.safetensors"
            model_hash = hashlib.sha256(model_weights_path.read_bytes()).hexdigest()

            # core-issues5.txt Section 3 (P0 blocker): a checkpoint that
            # declares normalized training input (normalization_hash is not
            # the sentinel) MUST have its exact train-owned normalization
            # artifact loaded, hash-verified against the checkpoint identity,
            # and injected into the live feature-building path -- never
            # silently served with an unnormalized HydraulicFeatureBuilder.
            # Any failure here (missing/stale/corrupted/schema-incompatible
            # artifact, or a fingerprint mismatch) falls into the same
            # except-block below as every other asset-loading failure: fail
            # closed, no neural branch, classical-safe remains available.
            if identity.normalization_hash == NO_NORMALIZATION_SENTINEL:
                feature_builder = HydraulicFeatureBuilder()
            else:
                bundle = load_runtime_normalization_bundle(self.normalization_dir)
                if bundle.fingerprint != identity.normalization_hash:
                    raise NormalizationBundleError(
                        f"runtime normalization bundle fingerprint ({bundle.fingerprint}) at "
                        f"{self.normalization_dir} does not match checkpoint identity's "
                        f"normalization_hash ({identity.normalization_hash})"
                    )
                feature_builder = bundle.feature_builder()

            calibrator: SplitConformalCalibrator | None = None
            if self.calibration_path is not None and self.calibration_path.exists():
                calibrator = SplitConformalCalibrator.load(self.calibration_path)
                calibrator.artifact.validate_runtime(
                    model_hash=model_hash,
                    feature_schema_hash=identity.feature_schema_hash,
                    normalization_hash=identity.normalization_hash,
                )
            # Phase 2 item 5 / Phase 6.1: no calibration artifact means
            # calibration is invalid -- HybridInferencePipeline is handed
            # calibration_artifact=None exactly like DefaultPipelineFactory
            # does for an unsupported topology, which its own downstream
            # logic already treats as "planning must not proceed on a
            # calibrated candidate set" (CLASSICAL_SAFE / uncalibrated
            # mode). No separate fail-closed branch is needed here: the
            # existing pipeline already fails closed on this exact
            # condition.

            self._model = model
            self._identity = identity
            self._calibrator = calibrator
            self._feature_builder = feature_builder
            self._model_hash = model_hash
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            NotAV4CheckpointError,
            CheckpointIdentityError,
            NormalizationBundleError,
        ) as error:
            self._model = None
            self._identity = None
            self._calibrator = None
            self._feature_builder = None
            self._model_hash = None
            self.fallback_reason = f"v4_trained_assets_unavailable:{type(error).__name__}"

    def __call__(self, _network_record: Any, network_path: str | Path) -> HybridInferencePipeline:
        self._load_assets()
        if wntr is None:
            raise RuntimeError("WNTR is unavailable")
        network = wntr.network.WaterNetworkModel(str(network_path))
        simulator = HydraulicSimulator(network)
        source_nodes = tuple(map(str, network.junction_name_list))
        if not source_nodes:
            raise ValueError("network has no junction source candidates")
        sensor_nodes = source_nodes
        sample_times = tuple(range(0, 6 * 3_600 + 1, 3_600))
        key = SignatureCacheKey(
            network_hash=simulator.state_hash(),
            hydraulic_state_hash=simulator.state_hash(),
            simulator_version=simulator.simulator_version,
            configuration_hash="runtime-signatures-v4",
            sensor_layout_hash=hashlib.sha256("|".join(sensor_nodes).encode()).hexdigest(),
        )
        artifact = SignatureBuilder(simulator, SignatureCache(self.signature_cache)).build_or_load(
            key=key,
            source_nodes=source_nodes,
            start_time_bins=(0, 60),
            duration_bins=(30, 60),
            strength_bins=(0.5, 1.0),
            demand_regimes=("nominal",),
            sensor_nodes=sensor_nodes,
            sample_times_seconds=sample_times,
        )
        calibration = self._calibrator.artifact if self._calibrator is not None else None
        return HybridInferencePipeline(
            simulator=simulator,
            signature_artifact=artifact,
            model=self._model,
            model_hash=self._model_hash,
            calibration_artifact=calibration,
            # core-issues5.txt Section 3: inject the same governed,
            # hash-verified train-owned feature builder _load_assets
            # resolved above -- None only when the model itself is None
            # (trained_assets_ready is False), in which case
            # HybridInferencePipeline falls back to its own default
            # unnormalized builder, which is moot since model=None already
            # forces CLASSICAL_SAFE for every analyze() call regardless.
            feature_builder=self._feature_builder,
            trained_tasks=V4_TRAINED_TASKS,
            runtime_enabled_outputs=self._identity.runtime_enabled_outputs if self._identity is not None else frozenset(),
            fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
        )
