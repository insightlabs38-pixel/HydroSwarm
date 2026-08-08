"""Load a self-contained V4 inference release bundle directly (core-issues5.txt
delta item 3, P0 blocker).

`scripts/build_v4_inference_release_bundle.py` (core-issues5.txt Section 12)
already produces a lean, inference-only bundle directory: `model.safetensors`,
`checkpoint_identity.json`, `runtime_manifest.json`, `output_governance.json`,
`node-normalization.json`, `edge-normalization.json`,
`signature-policy-manifest.json`, `calibration-status.json` (or, once Section
19 fits one, a real calibration artifact), `artifact-manifest.json`, and
`SHA256SUMS` -- deliberately NO `optimizer_state.pt`/`rng_state.pt`/
`trainer_state.json`, since this is not a resumable training checkpoint.

Before this module, the only loader was
`hydroswarm.training.checkpoint_identity.load_v4_checkpoint`, which is built
for the OTHER (resumable training) checkpoint format `save_v4_checkpoint`
produces:

- it unconditionally reads `trainer_state.json`, which does not exist in an
  inference bundle -- a real `FileNotFoundError`, not a graceful fallback;
- it reads `artifact_manifest.json` (underscore) expecting the checkpoint's
  own tamper-check schema (`{"checkpoint_identity_fingerprint": ...}`), a
  DIFFERENT file from the release bundle's `artifact-manifest.json` (hyphen,
  a general file-hash inventory -- see `build_v4_inference_release_bundle.py`).

So a bundle built by `scripts/build_v4_inference_release_bundle.py` could not
actually be loaded by anything -- this module is the fix: a loader built for
the bundle's own real shape, validating every file the bundle actually
contains, that a `V4PipelineFactory` can construct a real runtime from
without any dependency on the original training corpus or checkpoint
directory.

Normalization sidecar resolution (core-issues5.txt delta item 3): rather than
requiring `node-normalization.json.sha256`/`edge-normalization.json.sha256`
sidecars (`hydroswarm.runtime.v4_normalization.load_runtime_normalization_bundle`'s
convention, built for a normalization artifact living directly under
governed corpus data), this loader validates every bundle file's content
against the bundle's own `SHA256SUMS` (cross-checked against
`artifact-manifest.json`) -- one hash-integrity mechanism per bundle, not two
independently-maintained ones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file

from hydroswarm.calibration import CalibrationArtifact, SplitConformalCalibrator
from hydroswarm.model import HydroCore
from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats
from hydroswarm.training.checkpoint_identity import (
    CheckpointIdentity,
    CheckpointIdentityError,
    build_model_from_identity,
    validate_checkpoint_identity,
)

__all__ = [
    "InferenceBundleError",
    "REQUIRED_BUNDLE_FILES",
    "V4InferenceBundle",
    "load_v4_inference_bundle",
]

#: Section 12's minimum contents list, restricted to the files a loader
#: needs present to construct a real runtime (calibration is handled
#: separately below: either `calibration.json` or `calibration-status.json`
#: must exist, not both required).
REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "model.safetensors",
    "checkpoint_identity.json",
    "runtime_manifest.json",
    "output_governance.json",
    "node-normalization.json",
    "edge-normalization.json",
    "signature-policy-manifest.json",
    "artifact-manifest.json",
    "SHA256SUMS",
)

CALIBRATION_ARTIFACT_FILENAME = "calibration.json"
CALIBRATION_STATUS_FILENAME = "calibration-status.json"


class InferenceBundleError(Exception):
    """Raised when a V4 inference release bundle is missing required files,
    hash-mismatched, or internally inconsistent. Callers must fail closed:
    no neural branch on an unverified bundle, classical-safe operation
    remains available."""


@dataclass(frozen=True, slots=True)
class V4InferenceBundle:
    model: HydroCore
    identity: CheckpointIdentity
    model_hash: str
    node_normalization: NormalizationStats
    edge_normalization: NormalizationStats
    normalization_fingerprint: str
    signature_policy_hash: str
    calibration: CalibrationArtifact | None
    calibration_status: str
    runtime_manifest: Mapping[str, Any]
    bundle_dir: Path

    def feature_builder(self, **kwargs: object) -> HydraulicFeatureBuilder:
        return HydraulicFeatureBuilder(
            node_normalization=self.node_normalization,
            edge_normalization=self.edge_normalization,
            **kwargs,  # type: ignore[arg-type]
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sha256sums(bundle_dir: Path) -> dict[str, str]:
    """Parse `SHA256SUMS` and verify every file it references against real
    content on disk. Fails closed on any missing file or mismatch -- this
    is the bundle's ONE hash-integrity mechanism (see module docstring)."""

    sums_path = bundle_dir / "SHA256SUMS"
    recorded: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, name = line.partition("  ")
        if not sha or not name:
            raise InferenceBundleError(f"malformed SHA256SUMS line: {line!r}")
        recorded[name] = sha
    for name, expected in recorded.items():
        path = bundle_dir / name
        if not path.exists():
            raise InferenceBundleError(f"SHA256SUMS references a missing file: {name}")
        actual = _sha256_file(path)
        if actual != expected:
            raise InferenceBundleError(
                f"{name}: content does not match SHA256SUMS (recorded={expected}, actual={actual}) -- "
                "the bundle is corrupted or was modified after being built"
            )
    return recorded


def load_v4_inference_bundle(bundle_dir: str | Path) -> V4InferenceBundle:
    """Load and fully validate a V4 inference release bundle, with NO
    dependency on the original training corpus or checkpoint directory.

    Validates (core-issues5.txt delta item 3's required list):

    - every required file is present;
    - `SHA256SUMS` matches real file content for every referenced file;
    - `artifact-manifest.json`'s recorded hashes match `SHA256SUMS`;
    - `checkpoint_identity.json` is well-formed and self-consistent
      (`validate_checkpoint_identity`);
    - `runtime_manifest.json`'s `checkpoint_identity_fingerprint` and
      `model_sha256` match the real identity/model file;
    - `output_governance.json` matches `checkpoint_identity.json` verbatim
      (no silent re-decision of governance at load time);
    - bundled node/edge normalization reproduces
      `checkpoint_identity.json`'s own `normalization_hash`;
    - `signature-policy-manifest.json`'s `policy_hash` matches
      `runtime_manifest.json`;
    - calibration, when present as a real artifact
      (`calibration.json`), validates against the model/feature/
      normalization identity; when absent, `calibration-status.json` must
      explicitly record why (only `NOT_YET_FIT` is recognized today).

    Raises `InferenceBundleError` (never a bare `KeyError`/`FileNotFoundError`)
    on any failure -- callers should treat this exactly like any other
    asset-loading failure: fail closed, classical-safe remains available.
    """

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise InferenceBundleError(f"{bundle_dir} is not a directory")
    for name in REQUIRED_BUNDLE_FILES:
        if not (bundle_dir / name).exists():
            raise InferenceBundleError(f"{bundle_dir} is missing required inference-bundle file: {name}")

    try:
        checksums = _verify_sha256sums(bundle_dir)
        artifact_manifest = _read_json(bundle_dir / "artifact-manifest.json")
        if dict(artifact_manifest.get("files", {})) != checksums:
            raise InferenceBundleError(
                f"{bundle_dir}: artifact-manifest.json's recorded file hashes do not match SHA256SUMS"
            )

        raw_identity = _read_json(bundle_dir / "checkpoint_identity.json")
        identity = CheckpointIdentity(
            **{
                key: (
                    frozenset(value) if key.endswith("_outputs")
                    else tuple(value) if isinstance(value, list)
                    else value
                )
                for key, value in raw_identity.items()
            }
        )
        validate_checkpoint_identity(identity)

        runtime_manifest = _read_json(bundle_dir / "runtime_manifest.json")
        if runtime_manifest.get("checkpoint_identity_fingerprint") != identity.fingerprint():
            raise InferenceBundleError(
                f"{bundle_dir}: runtime_manifest.json's checkpoint_identity_fingerprint does not match "
                "checkpoint_identity.json -- one file was modified independently of the other"
            )

        model_hash = checksums.get("model.safetensors")
        if model_hash is None or runtime_manifest.get("model_sha256") != model_hash:
            raise InferenceBundleError(
                f"{bundle_dir}: runtime_manifest.json's model_sha256 does not match model.safetensors content"
            )

        model = build_model_from_identity(identity)
        state_dict = load_file(bundle_dir / "model.safetensors", device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        governance = _read_json(bundle_dir / "output_governance.json")
        for field_name, recorded_set in (
            ("trained_outputs", identity.trained_outputs),
            ("validated_outputs", identity.validated_outputs),
            ("runtime_enabled_outputs", identity.runtime_enabled_outputs),
            ("training_only_outputs", identity.training_only_outputs),
        ):
            if set(governance.get(field_name, [])) != recorded_set:
                raise InferenceBundleError(
                    f"{bundle_dir}: output_governance.json's {field_name} does not match "
                    "checkpoint_identity.json -- the bundle is internally inconsistent"
                )

        node_stats = NormalizationStats.load(bundle_dir / "node-normalization.json")
        edge_stats = NormalizationStats.load(bundle_dir / "edge-normalization.json")
        for stats, label in ((node_stats, "node"), (edge_stats, "edge")):
            if stats.schema_version != DEFAULT_FEATURE_SCHEMA.version:
                raise InferenceBundleError(
                    f"{bundle_dir}: {label}-normalization.json schema_version {stats.schema_version!r} is "
                    f"incompatible with the current feature schema {DEFAULT_FEATURE_SCHEMA.version!r}"
                )
        normalization_fingerprint = HydraulicFeatureBuilder(
            node_normalization=node_stats, edge_normalization=edge_stats
        ).normalization_fingerprint
        if normalization_fingerprint != identity.normalization_hash:
            raise InferenceBundleError(
                f"{bundle_dir}: bundled normalization fingerprint ({normalization_fingerprint}) does not "
                f"match checkpoint identity's normalization_hash ({identity.normalization_hash})"
            )

        signature_policy = _read_json(bundle_dir / "signature-policy-manifest.json")
        signature_policy_hash = signature_policy.get("policy_hash")
        if not signature_policy_hash or runtime_manifest.get("signature_policy_hash") != signature_policy_hash:
            raise InferenceBundleError(
                f"{bundle_dir}: signature-policy-manifest.json's policy_hash does not match runtime_manifest.json"
            )

        calibration: CalibrationArtifact | None = None
        calibration_status = "MISSING"
        calibration_artifact_path = bundle_dir / CALIBRATION_ARTIFACT_FILENAME
        calibration_status_path = bundle_dir / CALIBRATION_STATUS_FILENAME
        if calibration_artifact_path.exists():
            calibration = SplitConformalCalibrator.load(calibration_artifact_path).artifact
            calibration.validate_runtime(
                model_hash=model_hash,
                feature_schema_hash=identity.feature_schema_hash,
                normalization_hash=identity.normalization_hash,
            )
            calibration_status = "FITTED"
        elif calibration_status_path.exists():
            status = _read_json(calibration_status_path)
            calibration_status = str(status.get("status", "MISSING"))
            if calibration_status != "NOT_YET_FIT":
                raise InferenceBundleError(
                    f"{bundle_dir}: unrecognized calibration-status.json status: {calibration_status!r}"
                )
        else:
            raise InferenceBundleError(
                f"{bundle_dir}: has neither {CALIBRATION_ARTIFACT_FILENAME} nor {CALIBRATION_STATUS_FILENAME} "
                "-- calibration state must be explicit, never silently absent"
            )
    except (OSError, KeyError, TypeError, ValueError, CheckpointIdentityError) as error:
        # InferenceBundleError itself is not caught here (it is not a
        # member of this tuple), so every explicit raise above propagates
        # with its own specific message; this clause only wraps genuinely
        # unexpected failures (corrupt JSON, missing dict keys, a
        # CheckpointIdentity schema/policy mismatch) into the same
        # fail-closed exception type callers only need to handle once.
        raise InferenceBundleError(f"{bundle_dir}: failed to load inference bundle: {error}") from error

    return V4InferenceBundle(
        model=model,
        identity=identity,
        model_hash=model_hash,
        node_normalization=node_stats,
        edge_normalization=edge_stats,
        normalization_fingerprint=normalization_fingerprint,
        signature_policy_hash=signature_policy_hash,
        calibration=calibration,
        calibration_status=calibration_status,
        runtime_manifest=runtime_manifest,
        bundle_dir=bundle_dir,
    )
