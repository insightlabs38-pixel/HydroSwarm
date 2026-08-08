"""core-issues5.txt Section 12 (P0 governance fix): produce a truthful final
V4 inference release bundle.

`hydroswarm.training.checkpoint_identity.save_v4_checkpoint`'s own output
is a real, resumable TRAINING checkpoint directory (model +
optimizer_state.pt + rng_state.pt + trainer_state.json, per Section B).
Reusing that directory as-is for inference packaging would either force a
caller to depend on training-only files it will never use, or -- worse --
invite fabricating fake "resumability" metadata around an already-
resumable checkpoint that was never actually the run's live optimizer
state (see scripts/build_phase15_v4_checkpoint.py's own documented
disclaimer: its optimizer/scheduler state is FRESH, reconstructed
post-hoc, not the real Stage-F run's). This script builds the SEPARATE
artifact Section 12 asks for: an inference-only release bundle containing
exactly the files a live V4PipelineFactory-style loader needs, each with
a real content hash, and nothing that pretends to be resumable.

Deliberately NOT a promotion action: does not repoint DefaultPipelineFactory
or V4PipelineFactory's own defaults, does not create final-selection.json,
does not open the locked test. Phase 19 (architecture selection) and the
locked-test boundary remain untouched, per this pass's own restrictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydroswarm.classical.signature_policy import GOVERNED_TRAINING_SIGNATURE_POLICY
from hydroswarm.runtime.v4_normalization import load_runtime_normalization_bundle
from hydroswarm.simulation.wrapper import PlanEvaluationContext
from hydroswarm.training.artifacts import git_commit_hash
from hydroswarm.training.checkpoint_identity import CheckpointIdentity, load_v4_checkpoint

DEFAULT_NORMALIZATION_DIR = Path("data/learning-v2/cycle-b2/normalization")
#: Read from the dataclass field default rather than duplicating the
#: literal here, so this can never silently drift from the real policy
#: version PlanEvaluationContext/evaluate_plan_consequences actually use.
CONSEQUENCE_POLICY_VERSION = PlanEvaluationContext.__dataclass_fields__["consequence_policy_version"].default


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _output_governance_payload(identity: CheckpointIdentity) -> dict[str, Any]:
    """core-issues5.txt Section 12: "Do not claim an output was trained
    merely because the head exists." trained_outputs/validated_outputs/
    runtime_enabled_outputs/diagnostic_only_outputs/training_only_outputs
    on `identity` were derived from real Phase 13/14 governed evaluation
    evidence (reports/results/v4/phase13-metrics-and-baselines.md,
    reports/results/v4/phase14-promotion-gates.md) at
    scripts/build_phase15_v4_checkpoint.py's own build time -- reused
    verbatim here, not re-decided, so this bundle can never silently drift
    from the governance decision that already happened."""

    return {
        "schema_version": 1,
        "trained_outputs": sorted(identity.trained_outputs),
        "validated_outputs": sorted(identity.validated_outputs),
        "runtime_enabled_outputs": sorted(identity.runtime_enabled_outputs),
        "diagnostic_only_outputs": sorted(identity.diagnostic_only_outputs),
        "training_only_outputs": sorted(identity.training_only_outputs),
        "notes": {
            "sensor_reconstruction": (
                "training_only by construction (masked/corrupted-observation "
                "reconstruction target has no live-inference use)"
                if "sensor_reconstruction" in identity.training_only_outputs
                else (
                    "not trained by the selected run at all (core-issues5.txt delta item 4, Section C): "
                    "auxiliary_heads=False for this model config, so sensor_reconstruction_head was never "
                    "even physically constructed, let alone supervised -- absent from every governance set"
                    if "sensor_reconstruction" not in identity.trained_outputs
                    else "trained but not training_only in this identity -- verify against Phase 14 before trusting"
                )
            ),
            "travel_time": (
                "training_only by construction"
                if "travel_time" in identity.training_only_outputs
                else (
                    "not trained by the selected run at all (core-issues5.txt delta item 4, Section C): "
                    "auxiliary_heads=False for this model config, so travel_time_head was never even "
                    "physically constructed, let alone supervised -- absent from every governance set"
                    if "travel_time" not in identity.trained_outputs
                    else "trained but not training_only in this identity -- verify against Phase 14 before trusting"
                )
            ),
            "ood_category": (
                "excluded from all governance sets: near-chance macro F1 (~1/11 chance), zero real "
                "train-split gradient (Phase 13 finding 2) -- physically present in the architecture, "
                "not promoted"
                if "ood_category" not in identity.trained_outputs
                else "present in trained_outputs in this identity -- verify against Phase 13/14 before trusting"
            ),
        },
    }


def _signature_policy_manifest() -> dict[str, Any]:
    policy = GOVERNED_TRAINING_SIGNATURE_POLICY
    return {
        "schema_version": 1,
        "policy_version": policy.policy_version,
        "policy_hash": policy.policy_hash,
        "source_candidate_definition": policy.source_candidate_definition,
        "start_time_bins": list(policy.start_time_bins),
        "duration_bins": list(policy.duration_bins),
        "strength_bins": list(policy.strength_bins),
        "demand_regimes": list(policy.demand_regimes),
        "sample_times_seconds": list(policy.sample_times_seconds),
        "sensor_layout_policy": policy.sensor_layout_policy,
    }


def build(
    *,
    checkpoint_dir: Path,
    normalization_dir: Path,
    output_dir: Path,
    workdir: Path,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty release bundle directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and fully validate the v4 checkpoint identity (fails closed on
    # any hash/schema mismatch) before packaging anything from it.
    _model, identity, _trainer_state = load_v4_checkpoint(checkpoint_dir)

    # --- model.safetensors (inference weights only) ---------------------
    model_source = checkpoint_dir / "model.safetensors"
    model_dest = output_dir / "model.safetensors"
    shutil.copyfile(model_source, model_dest)
    model_sha256 = _sha256_file(model_dest)

    # --- checkpoint_identity.json (already real; copied verbatim) -------
    identity_dest = output_dir / "checkpoint_identity.json"
    shutil.copyfile(checkpoint_dir / "checkpoint_identity.json", identity_dest)

    # --- governed train-owned normalization artifact ---------------------
    normalization_bundle = load_runtime_normalization_bundle(normalization_dir)
    if normalization_bundle.fingerprint != identity.normalization_hash:
        raise SystemExit(
            f"normalization artifact at {normalization_dir} (fingerprint "
            f"{normalization_bundle.fingerprint}) does not match checkpoint identity's "
            f"normalization_hash ({identity.normalization_hash}) -- refusing to package a "
            "mismatched bundle"
        )
    shutil.copyfile(normalization_dir / "node-normalization.json", output_dir / "node-normalization.json")
    shutil.copyfile(normalization_dir / "edge-normalization.json", output_dir / "edge-normalization.json")

    # --- output governance summary ---------------------------------------
    _write_json(output_dir / "output_governance.json", _output_governance_payload(identity))

    # --- signature policy manifest ----------------------------------------
    _write_json(output_dir / "signature-policy-manifest.json", _signature_policy_manifest())

    # --- calibration -------------------------------------------------------
    # core-issues5.txt Section 19: the final conformal calibration artifact
    # must be refit AFTER the serving path (this bundle) is frozen, against
    # the exact deployed path -- not available yet at bundle-build time for
    # a bundle that has not itself been selected/frozen. Recorded honestly
    # as absent rather than reusing a calibration artifact fit against an
    # earlier/different preprocessing path.
    calibration_status = {
        "schema_version": 1,
        "status": "NOT_YET_FIT",
        "reason": (
            "core-issues5.txt Section 19: final calibration must be refit against the exact frozen "
            "serving path (incident-only/candidate-scoring separation, normalization parity, "
            "signature-policy parity, output governance) AFTER this bundle exists, not before. This "
            "bundle intentionally ships without a calibration artifact rather than reusing one fit "
            "against an earlier or different preprocessing path."
        ),
    }
    _write_json(output_dir / "calibration-status.json", calibration_status)

    # --- runtime manifest (top-level, ties everything together) ----------
    runtime_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_git_sha": git_commit_hash(workdir),
        "architecture_version": identity.architecture_version,
        "variant": identity.variant,
        "model_sha256": model_sha256,
        "checkpoint_identity_fingerprint": identity.fingerprint(),
        "feature_schema_hash": identity.feature_schema_hash,
        "target_schema_hash": identity.target_schema_hash,
        "action_template_schema_hash": identity.action_template_schema_hash,
        "ood_category_schema_hash": identity.ood_category_schema_hash,
        "next_step_schema_hash": identity.next_step_schema_hash,
        "plan_value_policy_version": identity.plan_value_policy_version,
        "consequence_policy_version": CONSEQUENCE_POLICY_VERSION,
        "normalization_hash": identity.normalization_hash,
        "node_normalization_sha256": _sha256_file(output_dir / "node-normalization.json"),
        "edge_normalization_sha256": _sha256_file(output_dir / "edge-normalization.json"),
        "signature_policy_hash": GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash,
        "fusion_policy_hash": identity.fusion_policy_hash,
        "calibration_status": "NOT_YET_FIT",
        "source_corpus_manifest_hashes": list(identity.source_corpus_manifest_hashes),
        "supported_outputs": sorted(identity.trained_outputs | identity.diagnostic_only_outputs),
        "validated_outputs": sorted(identity.validated_outputs),
        "runtime_enabled_outputs": sorted(identity.runtime_enabled_outputs),
        "note": (
            "This is an inference-only release bundle, deliberately separate from a resumable "
            "training checkpoint -- contains no optimizer/scheduler/RNG state. Not wired into any "
            "default runtime factory by this build; Phase 19 selection and the locked test remain "
            "untouched."
        ),
    }
    _write_json(output_dir / "runtime_manifest.json", runtime_manifest)

    # --- artifact manifest + SHA256SUMS (real content hashes) -----------
    bundle_files = sorted(p for p in output_dir.iterdir() if p.is_file())
    artifact_manifest = {
        "schema_version": 1,
        "generated_at": runtime_manifest["generated_at"],
        "files": {p.name: _sha256_file(p) for p in bundle_files},
    }
    _write_json(output_dir / "artifact-manifest.json", artifact_manifest)

    sha256sums_lines = [f"{sha}  {name}" for name, sha in sorted(artifact_manifest["files"].items())]
    (output_dir / "SHA256SUMS").write_text("\n".join(sha256sums_lines) + "\n", encoding="utf-8")

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810"),
    )
    parser.add_argument("--normalization-dir", type=Path, default=DEFAULT_NORMALIZATION_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs/v4-release-bundle/no_adapters-seed20260810"),
    )
    parser.add_argument("--workdir", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = build(
        checkpoint_dir=args.checkpoint_dir,
        normalization_dir=args.normalization_dir,
        output_dir=args.output_dir,
        workdir=args.workdir,
    )
    print(f"wrote v4 inference release bundle to {output_dir}")
    for line in (output_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
