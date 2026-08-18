"""Immutable M10.5 HydroCore-v5 release loader.

The loader is deliberately independent of v4: a bad v5 bundle leaves the
neural branch unavailable and therefore uses the pre-existing classical-safe
pipeline path; it never selects the historical v4 release as a fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from hydroswarm.calibration import SplitConformalCalibrator
from hydroswarm.classical import GOVERNED_TRAINING_SIGNATURE_POLICY, SignatureBuilder, SignatureCache, SignatureCacheKey, resolve_signature_mode
from hydroswarm.data.scenarios import network_sha256
from hydroswarm.inference import DYNAMIC_TRUST_FUSION_CONFIG, HybridInferencePipeline, OODDetector, OODReference
from hydroswarm.model import HydroCore
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.runtime.paths import resolve_data_dir
from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.simulation.wrapper import wntr

V5_TRAINED_TASKS = frozenset({"sentinel"})
V5_RUNTIME_ENABLED_OUTPUTS = frozenset({"event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"})


class V5InferenceBundleError(Exception):
    """A v5 release identity failed validation; callers must fail closed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V5PipelineFactory:
    """Load only the immutable selected-seed v5 release bundle."""

    def __init__(self, bundle_dir: str | Path, *, project_root: str | Path | None = None) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
        self.signature_cache = resolve_data_dir(self.project_root) / "signatures"
        self._load_attempted = False
        self._model: HydroCore | None = None
        self._model_hash: str | None = None
        self._calibrator: SplitConformalCalibrator | None = None
        self.fallback_reason: str | None = None
        self.signature_mode = None
        self.signature_policy_hash: str | None = None

    @property
    def model_hash(self) -> str | None:
        self._load_assets()
        return self._model_hash

    @property
    def trained_assets_ready(self) -> bool:
        self._load_assets()
        return self._model is not None and self._calibrator is not None

    def _load_assets(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            manifest_path = self.bundle_dir / "runtime_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("release_schema_version") != "hydroswarm-v5-release-v1":
                raise V5InferenceBundleError("unsupported v5 release schema")
            if set(manifest.get("runtime_enabled_outputs", ())) != V5_RUNTIME_ENABLED_OUTPUTS:
                raise V5InferenceBundleError("runtime output governance differs from frozen allowlist")
            if set(manifest.get("trained_tasks", ())) != V5_TRAINED_TASKS:
                raise V5InferenceBundleError("trained tasks differ from frozen sentinel-only governance")
            if manifest.get("feature_schema_hash") != DEFAULT_FEATURE_SCHEMA.fingerprint:
                raise V5InferenceBundleError("feature schema identity mismatch")
            if manifest.get("fusion_config_hash") != DYNAMIC_TRUST_FUSION_CONFIG:
                raise V5InferenceBundleError("fusion configuration identity mismatch")
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != {"model.safetensors", "calibration.json", "calibration.json.sha256"}:
                raise V5InferenceBundleError("malformed v5 release file manifest")
            for name, expected in files.items():
                if _sha256(self.bundle_dir / name) != expected:
                    raise V5InferenceBundleError(f"{name} checksum mismatch")
            model_hash = _sha256(self.bundle_dir / "model.safetensors")
            if model_hash != manifest.get("model_sha256"):
                raise V5InferenceBundleError("checkpoint SHA mismatch")
            calibrator = SplitConformalCalibrator.load(self.bundle_dir / "calibration.json")
            if calibrator.artifact.artifact_hash != manifest.get("calibration_artifact_hash"):
                raise V5InferenceBundleError("calibration artifact hash mismatch")
            calibrator.artifact.validate_runtime(
                model_hash=model_hash, feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
                fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG,
            )
            config = manifest.get("model_config")
            if not isinstance(config, dict):
                raise V5InferenceBundleError("missing model configuration")
            model = HydroCore.from_variant("small", use_adapters=False, **config)
            model.load_state_dict(load_file(self.bundle_dir / "model.safetensors", device="cpu"), strict=True)
            model.eval()
            self._model, self._model_hash, self._calibrator = model, model_hash, calibrator
        except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError, V5InferenceBundleError) as error:
            self._model = self._model_hash = self._calibrator = None  # type: ignore[assignment]
            self.fallback_reason = f"v5_trained_assets_unavailable:{type(error).__name__}"

    def __call__(self, _network_record: Any, network_path: str | Path) -> HybridInferencePipeline:
        self._load_assets()
        if wntr is None:
            raise RuntimeError("WNTR is unavailable")
        network = wntr.network.WaterNetworkModel(str(network_path))
        simulator = HydraulicSimulator(network)
        source_nodes = tuple(map(str, network.junction_name_list))
        if not source_nodes:
            raise ValueError("network has no junction source candidates")
        policy = GOVERNED_TRAINING_SIGNATURE_POLICY
        self.signature_mode = resolve_signature_mode(network_sha256(network))
        self.signature_policy_hash = policy.policy_hash
        key = SignatureCacheKey(network_hash=simulator.state_hash(), hydraulic_state_hash=simulator.state_hash(), simulator_version=simulator.simulator_version, configuration_hash=policy.policy_hash, sensor_layout_hash=hashlib.sha256("|".join(source_nodes).encode()).hexdigest())
        signature = SignatureBuilder(simulator, SignatureCache(self.signature_cache)).build_or_load(
            key=key, source_nodes=source_nodes, start_time_bins=policy.start_time_bins,
            duration_bins=policy.duration_bins, strength_bins=policy.strength_bins,
            demand_regimes=policy.demand_regimes, sensor_nodes=source_nodes,
            sample_times_seconds=policy.sample_times_seconds,
        )
        calibration = self._calibrator.artifact if self._calibrator else None
        return HybridInferencePipeline(simulator=simulator, signature_artifact=signature, model=self._model,
            model_hash=self._model_hash, calibration_artifact=calibration,
            ood_detector=OODDetector(OODReference(validated_network_hashes=calibration.validated_topology_hashes if calibration else ())),
            feature_builder=HydraulicFeatureBuilder(), trained_tasks=V5_TRAINED_TASKS,
            runtime_enabled_outputs=V5_RUNTIME_ENABLED_OUTPUTS, fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG)
