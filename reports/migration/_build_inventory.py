"""One-off migration helper: build reports/migration/arm-migration-inventory.json
from the currently-staged migration paths."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/workspace/HydroSwarm")
SOURCE_SHA = "6fea9f220f1a63bbf10c85fa241e7716d52477df"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lfs_tracked_paths() -> set[str]:
    result = subprocess.run(["git", "lfs", "ls-files", "-n"], cwd=ROOT, check=True, capture_output=True, text=True)
    return set(result.stdout.splitlines())


def classify(path: Path, lfs_paths: set[str]) -> str:
    rel = str(path.relative_to(ROOT))
    return "git-lfs" if rel in lfs_paths else "standard-git"


def entries_for(glob_root: Path, pattern: str, source_run: str, source_seed, required: str) -> list[dict]:
    lfs_paths = lfs_tracked_paths()
    entries = []
    for path in sorted(glob_root.glob(pattern)):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        entries.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tracking": classify(path, lfs_paths),
            "source_run": source_run,
            "source_seed": source_seed,
            "status": required,
        })
    return entries


def main() -> None:
    lfs_paths = lfs_tracked_paths()
    inventory: list[dict] = []

    # 1. Cycle B2 tensor corpus (raw + normalized)
    for tensors_dir, label in (
        (ROOT / "data/learning-v2/cycle-b2/tensors", "cycle-b2-raw-tensors"),
        (ROOT / "data/learning-v2/cycle-b2/tensors-normalized", "cycle-b2-normalized-tensors"),
    ):
        for path in sorted(tensors_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ROOT))
            inventory.append({
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tracking": classify(path, lfs_paths),
                "source_run": "generate_cycle_b_corpus.py (seed 71000) + fit_normalization.py + rebuild_normalized_shards.py",
                "source_seed": 71000,
                "status": "required",
            })

    # 2. Finalist candidates
    candidate_seeds = {
        "E1-seed20260810": ("cycle-b2-stage3-E1", 20260810),
        "E1-seed20260811": ("cycle-b2-stage3-E1", 20260811),
        "E0-seed20260810": ("cycle-b2-stage3-E0", 20260810),
        "E0-seed20260811": ("cycle-b2-stage3-E0", 20260811),
    }
    for label, (run, seed) in candidate_seeds.items():
        inventory.extend(
            entries_for(ROOT / "models/cycle-b2-candidates" / label, "*", run, seed, "required")
        )

    # 3. No-adapter controls
    control_seeds = {
        "no-adapter-seed20260810": ("cycle-b2-stage4", 20260810),
        "no-adapter-seed20260811": ("cycle-b2-stage4", 20260811),
    }
    for label, (run, seed) in control_seeds.items():
        inventory.extend(
            entries_for(ROOT / "models/cycle-b2-controls" / label, "*", run, seed, "required")
        )

    # 4. Normalization artifacts (already committed pre-migration; listed as required references)
    for name in ("node-normalization.json", "node-normalization.json.sha256",
                 "edge-normalization.json", "edge-normalization.json.sha256"):
        path = ROOT / "data/learning-v2/cycle-b2/normalization" / name
        inventory.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tracking": classify(path, lfs_paths),
            "source_run": "fit_normalization.py",
            "source_seed": None,
            "status": "required",
        })

    # 5. Scenario archives + manifest
    for path in sorted((ROOT / "artifacts/migration").glob("*")):
        if not path.is_file():
            continue
        inventory.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tracking": classify(path, lfs_paths),
            "source_run": "generate_cycle_b_corpus.py (seed 71000)",
            "source_seed": 71000,
            "status": "recommended",
        })

    # 6. Intentionally omitted artifacts (documented, not staged)
    omitted = [
        {
            "path": "experiments/runs/cycle-b2-stage2/*",
            "reason": "Stage 2 screening checkpoints -- diagnostic only, superseded by Stage 3 finalist training",
            "status": "omitted",
        },
        {
            "path": "experiments/runs/cycle-b2-stage3/*/*/checkpoints/checkpoint-{0004,0008,0012}",
            "reason": "periodic (non-final) checkpoints; only checkpoint-0016 (final) is preserved per finalist/seed",
            "status": "omitted",
        },
        {
            "path": "experiments/runs/cycle-b2-stage3/*/*/{best-model,model-export}.safetensors",
            "reason": "duplicate exports of checkpoint-0016's own model.safetensors under a different save path; "
                      "the migration task's provided hashes match checkpoint-0016/model.safetensors specifically",
            "status": "omitted",
        },
        {
            "path": "experiments/runs/cycle-b2-stage4/*/*/checkpoints/*",
            "reason": "periodic control checkpoints and their optimizer state; only the final model-export.safetensors "
                      "is preserved per instruction 5 ('do not copy all periodic control checkpoints')",
            "status": "omitted",
        },
        {
            "path": "experiments/cache/signatures/*",
            "reason": "regenerable SignatureArtifact cache (fit_dynamic_fusion_calibration.py); already .gitignored",
            "status": "omitted",
        },
        {
            "path": "data/learning-v2/cycle-b (old) and cycle-a",
            "reason": "preserved historical corpora, out of scope for this migration (Cycle B2 only)",
            "status": "omitted",
        },
        {
            "path": "the locked final test",
            "reason": "never opened, inspected, copied, or archived at any point during this migration, per explicit instruction",
            "status": "omitted",
        },
    ]

    manifest = {
        "schema_version": 1,
        "source_git_sha": SOURCE_SHA,
        "source_branch": "agent/gcp-multitopology-v3",
        "entries": inventory,
        "omitted": omitted,
        "summary": {
            "total_entries": len(inventory),
            "total_bytes": sum(e["size_bytes"] for e in inventory),
            "git_lfs_entries": sum(1 for e in inventory if e["tracking"] == "git-lfs"),
            "git_lfs_bytes": sum(e["size_bytes"] for e in inventory if e["tracking"] == "git-lfs"),
            "standard_git_entries": sum(1 for e in inventory if e["tracking"] == "standard-git"),
            "standard_git_bytes": sum(e["size_bytes"] for e in inventory if e["tracking"] == "standard-git"),
        },
    }
    output_path = ROOT / "reports/migration/arm-migration-inventory.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
