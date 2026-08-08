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
from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.simulation.wrapper import wntr
from hydroswarm.training.checkpoint_identity import (
    CheckpointIdentity,
    CheckpointIdentityError,
    NotAV4CheckpointError,
    load_v4_checkpoint,
)

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

    def __init__(self, checkpoint_dir: str | Path, *, project_root: str | Path | None = None) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.project_root = (
            Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
        )
        self.calibration_path: Path | None = None
        self.signature_cache = self.project_root / "data" / "generated" / "signatures"
        self._model: Any | None = None
        self._identity: CheckpointIdentity | None = None
        self._calibrator: SplitConformalCalibrator | None = None
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
            self._model_hash = model_hash
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            NotAV4CheckpointError,
            CheckpointIdentityError,
        ) as error:
            self._model = None
            self._identity = None
            self._calibrator = None
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
            trained_tasks=V4_TRAINED_TASKS,
            runtime_enabled_outputs=self._identity.runtime_enabled_outputs if self._identity is not None else frozenset(),
            fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
        )
