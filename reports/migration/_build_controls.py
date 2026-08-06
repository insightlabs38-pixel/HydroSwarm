"""One-off migration helper: build models/cycle-b2-controls/<label>/ from
each control's final model-export.safetensors plus run-level metadata."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("/workspace/HydroSwarm")

CONTROLS = {
    "no-adapter-seed20260810": {
        "seed": 20260810,
        "run_dir": ROOT / "experiments/runs/cycle-b2-stage4/no-adapter-S-seed20260810/20260806T044109Z-dfb38628",
        "expected_model_sha256": "fe2bd18b6849d680beae3c4274481797d873f752388ca412ee0a9965b4bb0e3b",
    },
    "no-adapter-seed20260811": {
        "seed": 20260811,
        "run_dir": ROOT / "experiments/runs/cycle-b2-stage4/no-adapter-S-seed20260811/20260806T053230Z-932df6d2",
        "expected_model_sha256": "f9fa30883cfe33c7ac1f272daaed62d9e263f743760fc7717e00e354703f5d2e",
    },
}
REGISTRY = ROOT / "experiments/registry/cycle-b2-stage4.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_opened_record(seed: int) -> dict:
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") == "opened" and rec.get("seed") == seed:
            return rec
    raise ValueError(f"no opened record for seed {seed}")


def main() -> None:
    node_norm_sha = (ROOT / "data/learning-v2/cycle-b2/normalization/node-normalization.json.sha256").read_text().strip().split()[0]
    edge_norm_sha = (ROOT / "data/learning-v2/cycle-b2/normalization/edge-normalization.json.sha256").read_text().strip().split()[0]

    results = {}
    for label, info in CONTROLS.items():
        dest = ROOT / "models/cycle-b2-controls" / label
        dest.mkdir(parents=True, exist_ok=True)

        shutil.copy2(info["run_dir"] / "model-export.safetensors", dest / "model.safetensors")
        for name in ("config.json", "metadata.json", "summary.json"):
            src = info["run_dir"] / name
            if src.exists():
                shutil.copy2(src, dest / name)

        run_level_calibration = info["run_dir"].parent / "calibration.json"
        if run_level_calibration.exists():
            shutil.copy2(run_level_calibration, dest / "stage4-fixed-weight-calibration.json")
            shutil.copy2(
                info["run_dir"].parent / "calibration.json.sha256",
                dest / "stage4-fixed-weight-calibration.json.sha256",
            )

        opened = registry_opened_record(info["seed"])
        model_hash = sha256(dest / "model.safetensors")
        architecture_config = {
            "schema_version": 1,
            "label": label,
            "control": "no-adapter-S",
            "seed": info["seed"],
            "variant": opened["variant"],
            "overrides": opened["resolved_config"]["overrides"],
            "model_sha256": model_hash,
            "feature_schema_hash": opened["feature_schema_hash"],
            "target_schema_hash": opened["target_schema_hash"],
            "topology_hashes": opened["topology_hashes"],
            "manifest_hashes": opened["manifest_hashes"],
            "node_normalization_sha256": node_norm_sha,
            "edge_normalization_sha256": edge_norm_sha,
            "normalization_dir": "data/learning-v2/cycle-b2/normalization",
            "source_git_commit": opened["git_commit"],
            "source_run_id": opened["run_id"],
            "source_checkpoint_path": str((info["run_dir"] / "model-export.safetensors").relative_to(ROOT)),
            "notes": (
                "model.safetensors here is the run's final model-export.safetensors "
                "(verified to match the migration task's expected hash), not "
                "checkpoints/checkpoint-0016/model.safetensors -- per instruction, only the "
                "final export and required metadata are retained for controls, not periodic "
                "checkpoints."
            ),
        }
        (dest / "architecture_config.json").write_text(
            json.dumps(architecture_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        results[label] = {
            "model_sha256": model_hash,
            "model_expected": info["expected_model_sha256"],
            "model_ok": model_hash == info["expected_model_sha256"],
        }

    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(r["model_ok"] for r in results.values()):
        raise SystemExit("MISMATCH DETECTED -- see output above")


if __name__ == "__main__":
    main()
