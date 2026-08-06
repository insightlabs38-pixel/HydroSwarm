"""One-off migration helper: build models/cycle-b2-candidates/<label>/ from
each finalist's checkpoint-0016 directory plus run-level metadata. Not a
product script -- lives under reports/migration/ and is not committed."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/workspace/HydroSwarm")
sys.path.insert(0, str(ROOT / "src"))
from hydroswarm.calibration.conformal import SplitConformalCalibrator  # noqa: E402

CANDIDATES = {
    "E1-seed20260810": {
        "finalist": "E1",
        "seed": 20260810,
        "run_dir": ROOT / "experiments/runs/cycle-b2-stage3/E1-seed20260810/20260806T012354Z-7645b52d",
        "registry": ROOT / "experiments/registry/cycle-b2-stage3-E1.jsonl",
        "overrides": {"prior_mode": "feature_only"},
        "expected_model_sha256": "051cfd94dec4a7ec61e559a1268b66acaada2d6248bda8c976846f9064ef3a23",
        "expected_calibration_sha256": "639384e86ce3c6ad30fb73914b08b8aa302337d77feb3472481e00c6d6cf040d",
    },
    "E1-seed20260811": {
        "finalist": "E1",
        "seed": 20260811,
        "run_dir": ROOT / "experiments/runs/cycle-b2-stage3/E1-seed20260811/20260806T023604Z-282c8c2f",
        "registry": ROOT / "experiments/registry/cycle-b2-stage3-E1.jsonl",
        "overrides": {"prior_mode": "feature_only"},
        "expected_model_sha256": "4ae71f3b31c3e7d4e10667126aad5343d64dad513aa48c626c4e3fa42a5dd63a",
        "expected_calibration_sha256": None,
    },
    "E0-seed20260810": {
        "finalist": "E0",
        "seed": 20260810,
        "run_dir": ROOT / "experiments/runs/cycle-b2-stage3/E0-seed20260810/20260806T012353Z-a8f77fde",
        "registry": ROOT / "experiments/registry/cycle-b2-stage3-E0.jsonl",
        "overrides": {},
        "expected_model_sha256": "04ada898f994c8cd54e12d65a7997256d80e5d6fb4c96a003f56e3492ad43580",
        "expected_calibration_sha256": None,
    },
    "E0-seed20260811": {
        "finalist": "E0",
        "seed": 20260811,
        "run_dir": ROOT / "experiments/runs/cycle-b2-stage3/E0-seed20260811/20260806T023546Z-993cb3e1",
        "registry": ROOT / "experiments/registry/cycle-b2-stage3-E0.jsonl",
        "overrides": {},
        "expected_model_sha256": "548009981c74a1d1c66c28936c1e66d65eca670638329c737101dad3d22a922f",  # placeholder, fixed below
        "expected_calibration_sha256": "548009981c74a1d1c66c28936c1e66d65eca670638329c737101dad3d22a922f",
    },
}
# fix E0-seed20260811's model hash (was accidentally duplicated above)
CANDIDATES["E0-seed20260811"]["expected_model_sha256"] = "c8f6a5e62a09264f653eec90854ca4934581348e05aa3c86cefc65cb5eee65df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_opened_record(registry_path: Path, seed: int) -> dict:
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") == "opened" and rec.get("seed") == seed:
            return rec
    raise ValueError(f"no opened record for seed {seed} in {registry_path}")


def main() -> None:
    node_norm = ROOT / "data/learning-v2/cycle-b2/normalization/node-normalization.json"
    edge_norm = ROOT / "data/learning-v2/cycle-b2/normalization/edge-normalization.json"
    node_norm_sha = json.loads((node_norm.parent / "node-normalization.json.sha256").read_text().split()[0]) \
        if False else (node_norm.parent / "node-normalization.json.sha256").read_text().strip().split()[0]
    edge_norm_sha = (edge_norm.parent / "edge-normalization.json.sha256").read_text().strip().split()[0]

    results = {}
    for label, info in CANDIDATES.items():
        dest = ROOT / "models/cycle-b2-candidates" / label
        dest.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = info["run_dir"] / "checkpoints" / "checkpoint-0016"

        shutil.copy2(checkpoint_dir / "model.safetensors", dest / "model.safetensors")
        shutil.copy2(checkpoint_dir / "optimizer_state.pt", dest / "optimizer_state.pt")
        shutil.copy2(checkpoint_dir / "trainer_state.json", dest / "trainer_state.json")

        for name in ("config.json", "metadata.json", "summary.json"):
            src = info["run_dir"] / name
            if src.exists():
                shutil.copy2(src, dest / name)

        run_level_calibration = info["run_dir"].parent / "calibration.json"
        if run_level_calibration.exists():
            shutil.copy2(run_level_calibration, dest / "stage3-fixed-weight-calibration.json")
            shutil.copy2(
                info["run_dir"].parent / "calibration.json.sha256",
                dest / "stage3-fixed-weight-calibration.json.sha256",
            )

        dyn_cal = checkpoint_dir / "calibration-dynamic-fusion.json"
        if dyn_cal.exists():
            shutil.copy2(dyn_cal, dest / "calibration-dynamic-fusion.json")
            shutil.copy2(checkpoint_dir / "calibration-dynamic-fusion.json.sha256", dest / "calibration-dynamic-fusion.json.sha256")

        opened = registry_opened_record(info["registry"], info["seed"])
        model_hash = sha256(dest / "model.safetensors")
        architecture_config = {
            "schema_version": 1,
            "label": label,
            "finalist": info["finalist"],
            "seed": info["seed"],
            "variant": opened["variant"],
            "overrides": info["overrides"],
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
            "source_checkpoint_path": str(checkpoint_dir.relative_to(ROOT)),
            "notes": (
                "architecture_config synthesized during Arm migration from the experiment "
                "registry's 'opened' record plus checkpoint-0016's own model.safetensors -- "
                "these training runs predate promote_checkpoint.py's architecture_config "
                "sidecar convention, so no such file existed at training time."
            ),
        }
        (dest / "architecture_config.json").write_text(
            json.dumps(architecture_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        model_ok = model_hash == info["expected_model_sha256"]
        cal_hash = None
        cal_ok = True
        if info["expected_calibration_sha256"] is not None:
            cal_path = dest / "calibration-dynamic-fusion.json"
            cal_hash = SplitConformalCalibrator.load(cal_path).artifact.artifact_hash
            cal_ok = cal_hash == info["expected_calibration_sha256"]
        results[label] = {
            "model_sha256": model_hash,
            "model_expected": info["expected_model_sha256"],
            "model_ok": model_ok,
            "calibration_sha256": cal_hash,
            "calibration_expected": info["expected_calibration_sha256"],
            "calibration_ok": cal_ok,
        }

    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(r["model_ok"] and r["calibration_ok"] for r in results.values()):
        raise SystemExit("MISMATCH DETECTED -- see output above")


if __name__ == "__main__":
    main()
