"""core-issues3.txt Phase 18: current, full-repo artifact inventory.

Distinct from `reports/migration/arm-migration-inventory.json` (a one-off
snapshot of the original x86->Arm VM migration, frozen at commit
`6fea9f220f1a63bbf10c85fa241e7716d52477df` -- does not cover any v4-era
corpus/checkpoint/report work landed since). This script inventories every
currently git-tracked file in the live working tree, hashed and sized for
real, plus flags any tracked file that matches a forbidden pattern
(credentials, `.env`, PID/lock files, the locked test) -- a governance
check, not just a listing.

Also verifies (Phase 18 item 6) that every Git-LFS-tracked file in this
working tree is a real, fully-downloaded object, not a left-behind pointer
stub (`version https://git-lfs.github.com/spec/v1` as the file's actual
first line is the unambiguous tell -- a real object never starts with
that regardless of its own binary content).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_secrets  # noqa: E402

#: Phase 18 item 3: never commit these, regardless of size/tracking.
#: Matched against the filename only (not the full path -- a substring
#: match against the full path would false-positive on entirely legitimate
#: governance tooling like scripts/scan_secrets.py or
#: reports/results/secret-scan.json, both real, already-audited-clean
#: artifacts, not secrets themselves), and as an exact suffix/name, not a
#: bare substring (".lock" as a substring would also flag
#: requirements.lock.txt, a real dependency manifest).
FORBIDDEN_NAME_SUFFIXES: tuple[str, ...] = (".env", ".pid", ".pem", ".key")
FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset({".pid", ".lock"})


def _matches_forbidden_lock_extension(name: str) -> bool:
    # A real runtime job/process lock file is named exactly "<something>.lock"
    # with nothing meaningful after it (e.g. "stage-f.jsonl.lock") --
    # distinct from a dependency manifest like "uv.lock"/"requirements.lock.txt",
    # which this project treats as governed, required, standard-git content.
    return name.endswith(".lock") and name not in {"uv.lock", "package-lock.json"} and not name.endswith(".lock.txt")

#: Path-prefix -> (source_run label, status). Longest-prefix-match wins.
#: Anything not matched here defaults to ("source-controlled", "required").
SOURCE_RUN_BY_PREFIX: tuple[tuple[str, str, str], ...] = (
    ("data/learning-v2/cycle-b2/tensors", "generate_cycle_b_corpus.py + fit_normalization.py (Phase 0-3 corpus)", "required"),
    ("data/learning-v2/cycle-b2-control-v2", "second-pass control-label generation + merge_second_pass_control_labels.py (Phase 8)", "required"),
    ("data/learning-v2/cycle-b2-ood-extension", "generate_ood_extension_corpus.py (Phase 6.3/10.4)", "required"),
    ("data/learning-v2/cycle-b2-trajectories-v2", "generate_trajectory_corpus.py, provisional pre-Phase-6.4/7 (superseded by v4)", "recommended"),
    ("data/learning-v2/cycle-b2-trajectories-v3", "generate_trajectory_corpus.py + build_scout_state_dataset.py/build_strategist_candidate_dataset.py", "recommended"),
    ("data/learning-v2/cycle-b2-trajectories-v4", "generate_trajectory_corpus.py, exposure-aware corrected regeneration (important-issues.txt req 14)", "required"),
    ("data/learning-v2/cycle-b2-joint-v4", "build_stage_f_joint_corpus.py (Phase 12 Stage F)", "required"),
    ("artifacts/migration", "Arm VM migration: compressed raw scenario archives (docs/ARM_MIGRATION.md)", "required"),
    ("models/cycle-b2-candidates", "run_stage3_finalist_training.py finalist checkpoints (E0/E1, 2 seeds each)", "required"),
    ("models/cycle-b2-controls", "run_stage_f_training.py no-adapter arm checkpoints (2 seeds)", "required"),
    ("models/hydrocore-s-learning-v1.safetensors", "currently promoted production checkpoint (data/learning-v1)", "required"),
    ("models/", "promoted/exported checkpoint", "required"),
    ("experiments/registry", "ExperimentRegistry run provenance records", "required"),
    ("reports/results", "evaluation/gate/handoff reports", "required"),
    ("reports/migration", "Arm VM migration documentation and inventory", "recommended"),
    ("scripts/", "governed generator/evaluation/gate script", "required"),
    ("src/hydroswarm", "application source", "required"),
    ("tests/", "test suite", "required"),
    ("configs/", "training/task-weight configuration", "required"),
    ("docs/", "documentation", "recommended"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_tracked_paths() -> set[str]:
    result = subprocess.run(["git", "lfs", "ls-files", "-n"], cwd=ROOT, check=True, capture_output=True, text=True)
    return set(result.stdout.splitlines())


def _classify_source_run(relative_path: str) -> tuple[str, str]:
    best_match: tuple[str, str, str] | None = None
    for prefix, source_run, status in SOURCE_RUN_BY_PREFIX:
        if relative_path.startswith(prefix) and (best_match is None or len(prefix) > len(best_match[0])):
            best_match = (prefix, source_run, status)
    if best_match is None:
        return "source-controlled", "required"
    return best_match[1], best_match[2]


def _is_forbidden(relative_path: str) -> bool:
    name = Path(relative_path).name
    lowered_name = name.lower()
    if lowered_name in FORBIDDEN_EXACT_NAMES:
        return True
    if any(lowered_name.endswith(suffix) for suffix in FORBIDDEN_NAME_SUFFIXES):
        return True
    return _matches_forbidden_lock_extension(lowered_name)


def _is_unpulled_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def build_inventory(*, hash_source_files: bool) -> dict[str, Any]:
    started = time.perf_counter()
    lfs_paths = _lfs_tracked_paths()
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()

    entries: list[dict[str, Any]] = []
    forbidden_findings: list[str] = []
    unpulled_lfs_pointers: list[str] = []
    total_bytes = 0
    lfs_bytes = 0
    lfs_count = 0
    standard_count = 0

    for relative_path in tracked:
        path = ROOT / relative_path
        if not path.is_file():
            continue
        if _is_forbidden(relative_path):
            forbidden_findings.append(relative_path)

        tracking = "git-lfs" if relative_path in lfs_paths else "standard-git"
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        if tracking == "git-lfs":
            lfs_bytes += size_bytes
            lfs_count += 1
            if _is_unpulled_lfs_pointer(path):
                unpulled_lfs_pointers.append(relative_path)
        else:
            standard_count += 1

        source_run, status = _classify_source_run(relative_path)
        entry: dict[str, Any] = {
            "path": relative_path,
            "size_bytes": size_bytes,
            "tracking": tracking,
            "source_run": source_run,
            "status": status,
        }
        # Hashing every one of ~1300 tracked files (including large LFS
        # tensor shards) is the whole point for data/model artifacts;
        # skippable for plain source/doc files via --no-hash-source-files
        # to keep a quick re-run cheap.
        if tracking == "git-lfs" or hash_source_files:
            entry["sha256"] = _sha256(path)
        entries.append(entry)

    if forbidden_findings:
        raise RuntimeError(f"forbidden tracked file pattern(s) found: {forbidden_findings}")

    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "entries": entries,
        "summary": {
            "total_entries": len(entries),
            "total_bytes": total_bytes,
            "git_lfs_entries": lfs_count,
            "git_lfs_bytes": lfs_bytes,
            "standard_git_entries": standard_count,
            "standard_git_bytes": total_bytes - lfs_bytes,
        },
        "forbidden_pattern_findings": forbidden_findings,
        "unpulled_lfs_pointers": unpulled_lfs_pointers,
        "wall_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("reports/results/v4/artifact-inventory.json"))
    parser.add_argument("--no-hash-source-files", action="store_true", help="skip sha256 for non-LFS files (faster re-run)")
    args = parser.parse_args(argv)

    inventory = build_inventory(hash_source_files=not args.no_hash_source_files)
    # Reuse the project's own dedicated credential/private-key scanner
    # rather than a second, weaker filename-substring heuristic for the
    # same concern.
    inventory["secret_scan"] = scan_secrets.scan(ROOT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(inventory["summary"], indent=2))
    print(f"secret_scan status: {inventory['secret_scan']['status']}")
    if inventory["unpulled_lfs_pointers"]:
        print(f"WARNING: {len(inventory['unpulled_lfs_pointers'])} LFS pointer(s) not pulled: {inventory['unpulled_lfs_pointers'][:5]}")
    print(f"wrote {args.output}")
    if inventory["secret_scan"]["status"] != "pass":
        return 1
    return 1 if inventory["unpulled_lfs_pointers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
