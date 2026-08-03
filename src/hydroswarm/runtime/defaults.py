"""Compose the production pipeline from local, checksummed scientific assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from hydroswarm.calibration import SplitConformalCalibrator
from hydroswarm.classical import SignatureBuilder, SignatureCache, SignatureCacheKey
from hydroswarm.inference import HybridInferencePipeline
from hydroswarm.model import HydroCore
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.simulation.wrapper import wntr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DefaultPipelineFactory:
    """Lazily load trained assets and build a per-network exact simulator pipeline."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self.model_path = self.project_root / "models" / "hydrocore-s-learning-v1.safetensors"
        self.metadata_path = self.model_path.with_suffix(".metadata.json")
        self.calibration_path = (
            self.project_root / "reports" / "results" / "hydrocore-s-calibration.json"
        )
        self.signature_cache = self.project_root / "data" / "generated" / "signatures"
        self._model: HydroCore | None = None
        self._calibrator: SplitConformalCalibrator | None = None
        self._load_attempted = False
        self.fallback_reason: str | None = None

    def _load_assets(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            checkpoint_hash = _sha256(self.model_path)
            if metadata["sha256"] != checkpoint_hash:
                raise ValueError("checkpoint checksum does not match metadata")
            if metadata["feature_schema_sha256"] != DEFAULT_FEATURE_SCHEMA.fingerprint:
                raise ValueError("checkpoint feature schema is incompatible")
            model = HydroCore.from_variant("small")
            model.load_state_dict(load_file(self.model_path, device="cpu"), strict=True)
            model.eval()
            calibrator = SplitConformalCalibrator.load(self.calibration_path)
            calibrator.artifact.validate_runtime(
                model_hash=checkpoint_hash,
                feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
            )
            self._model = model
            self._calibrator = calibrator
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            self._model = None
            self._calibrator = None
            self.fallback_reason = f"trained_assets_unavailable:{type(error).__name__}"

    @property
    def trained_assets_ready(self) -> bool:
        self._load_assets()
        return self._model is not None and self._calibrator is not None

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
            configuration_hash="runtime-signatures-v2",
            sensor_layout_hash=hashlib.sha256(
                "|".join(sensor_nodes).encode()
            ).hexdigest(),
        )
        artifact = SignatureBuilder(
            simulator, SignatureCache(self.signature_cache)
        ).build_or_load(
            key=key,
            source_nodes=source_nodes,
            start_time_bins=(0, 60),
            duration_bins=(30, 60),
            strength_bins=(0.5, 1.0),
            demand_regimes=("nominal",),
            sensor_nodes=sensor_nodes,
            sample_times_seconds=sample_times,
        )
        calibration = (
            self._calibrator.artifact
            if self._calibrator is not None and len(source_nodes) == 4
            else None
        )
        return HybridInferencePipeline(
            simulator=simulator,
            signature_artifact=artifact,
            model=self._model,
            model_hash=_sha256(self.model_path) if self._model is not None else None,
            calibration_artifact=calibration,
        )
