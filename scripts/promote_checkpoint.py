"""Promote safe model weights and provenance without optimizer pickle state."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import load_file

from hydroswarm.model import HydroCore, load_state_dict_with_v2_migration
from hydroswarm.model.core import (
    INCIDENT_POOLING_MODES,
    MESSAGE_DIRECTIONS,
    PRIOR_MODES,
)
from hydroswarm.preprocessing.builder import NO_NORMALIZATION_SENTINEL
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA
from hydroswarm.tasks import validate_tasks


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_set(value: str) -> frozenset[str]:
    return frozenset(token.strip() for token in value.split(",") if token.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--corpus-report", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--training-seconds", type=float, required=True)
    parser.add_argument("--status", choices=("trained", "partial"), required=True)
    parser.add_argument(
        "--use-adapters", action=argparse.BooleanOptionalAction, default=True,
        help="must match how the checkpoint being promoted was actually constructed",
    )
    parser.add_argument("--prior-mode", choices=PRIOR_MODES, default="feature_and_logit")
    parser.add_argument("--incident-pooling", choices=INCIDENT_POOLING_MODES, default="mean")
    parser.add_argument("--message-direction", choices=MESSAGE_DIRECTIONS, default="forward_only")
    parser.add_argument(
        "--event-control-heads", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--auxiliary-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--consequence-prescreening-heads", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--normalization-hash",
        default=NO_NORMALIZATION_SENTINEL,
        help=(
            "governed normalization artifact fingerprint this checkpoint was trained "
            "against (see HydraulicFeatureBuilder.normalization_fingerprint); defaults to "
            "the explicit no-normalization sentinel, matching every checkpoint trained so far."
        ),
    )
    parser.add_argument(
        "--trained-tasks",
        required=True,
        help=(
            "comma-separated runtime tasks genuinely optimized against real labels "
            "for this checkpoint (see hydroswarm.tasks.RUNTIME_TASKS); e.g. 'sentinel'. "
            "core-issues.txt repair item 8: HybridInferencePipeline ignores any task's "
            "neural outputs that is not declared here."
        ),
    )
    parser.add_argument(
        "--validated-tasks",
        required=True,
        help=(
            "comma-separated subset of --trained-tasks that has also passed "
            "audit/calibration validation (e.g. label_audit, corpus gates)."
        ),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help=(
            "ExperimentRegistry JSONL ledger to look up --registry-run-id in. "
            "core-issues.txt repair item 9: when given, the run's own seed, "
            "git_commit, manifest_hashes, and topology_hashes are copied verbatim "
            "into this checkpoint's metadata -- one canonical source, not a second "
            "hand-typed copy that can drift from what the registry actually recorded."
        ),
    )
    parser.add_argument(
        "--registry-run-id",
        default=None,
        help="run_id (in --registry-path) that produced the checkpoint being promoted",
    )
    args = parser.parse_args()
    if bool(args.registry_path) != bool(args.registry_run_id):
        raise ValueError("--registry-path and --registry-run-id must be given together")
    training_provenance: dict[str, object] | None = None
    if args.registry_path is not None:
        from hydroswarm.training.registry import ExperimentRegistry

        run = ExperimentRegistry(args.registry_path).runs().get(args.registry_run_id)
        if run is None:
            raise ValueError(
                f"run_id {args.registry_run_id!r} not found in registry {args.registry_path}"
            )
        training_provenance = {
            "run_id": run["run_id"],
            "seed": run["seed"],
            "git_commit": run["git_commit"],
            "manifest_hashes": run["manifest_hashes"],
            "topology_hashes": run.get("topology_hashes", []),
        }
    trained_tasks = _task_set(args.trained_tasks)
    validated_tasks = _task_set(args.validated_tasks)
    validate_tasks(trained_tasks, label="--trained-tasks")
    validate_tasks(validated_tasks, label="--validated-tasks")
    if not validated_tasks <= trained_tasks:
        raise ValueError(
            f"--validated-tasks {sorted(validated_tasks)} must be a subset of "
            f"--trained-tasks {sorted(trained_tasks)}"
        )
    if args.checkpoint.is_dir():
        source = args.checkpoint / "model.safetensors"
        trainer_state = json.loads(
            (args.checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
    else:
        source = args.checkpoint
        summary = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
        trainer_state = {
            key: summary[key]
            for key in ("best_epoch", "best_validation_loss", "epochs_completed", "global_steps")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, args.output)
    with safe_open(args.output, framework="pt", device="cpu") as artifact:
        tensor_count = len(artifact.keys())
        parameter_count = sum(artifact.get_tensor(key).numel() for key in artifact.keys())
    # core-issues.txt repair item 9: actually construct and load the model
    # the declared flags describe, rather than recording --architecture/
    # --variant as opaque strings disconnected from what the tensors being
    # promoted really are. This also means promotion itself now fails
    # closed if the declared configuration cannot load these tensors,
    # instead of silently copying bytes no one has verified load at all.
    model = HydroCore.from_variant(
        args.variant,
        use_adapters=args.use_adapters,
        prior_mode=args.prior_mode,
        incident_pooling=args.incident_pooling,
        message_direction=args.message_direction,
        event_control_heads=args.event_control_heads,
        auxiliary_heads=args.auxiliary_heads,
        consequence_prescreening_heads=args.consequence_prescreening_heads,
    )
    load_state_dict_with_v2_migration(model, load_file(args.output, device="cpu"))
    metadata = {
        "schema_version": 1,
        "architecture": args.architecture,
        "architecture_config": model.architecture_config(),
        "variant": args.variant,
        "normalization_hash": args.normalization_hash,
        "training_status": args.status,
        "sha256": _hash(args.output),
        "bytes": args.output.stat().st_size,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "corpus_report_sha256": _hash(args.corpus_report),
        "calibration_manifest_sha256": _hash(args.calibration_manifest),
        "feature_schema_version": DEFAULT_FEATURE_SCHEMA.version,
        "feature_schema_sha256": DEFAULT_FEATURE_SCHEMA.fingerprint,
        "training_seconds": args.training_seconds,
        "trainer_state": trainer_state,
        "promoted_at": datetime.now(UTC).isoformat(),
        "optimizer_state_included": False,
        "trained_tasks": sorted(trained_tasks),
        "validated_tasks": sorted(validated_tasks),
        "training_provenance": training_provenance,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
