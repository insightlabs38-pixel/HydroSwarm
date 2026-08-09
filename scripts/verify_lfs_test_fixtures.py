"""CI-only, cross-platform Git-LFS hydration verifier for the real tensor
fixtures the Python test suite reads (backend CI/portability hotfix).

Run in the ``python-quality`` CI job right after ``git lfs pull --include=
...`` and before the real test suite, so an LFS hydration problem produces

    REQUIRED TEST FIXTURE NOT HYDRATED: <path>

immediately, instead of surfacing later as an opaque low-level safetensors
"header too large" parse exception deep inside a test.

Two checks per required directory, both dependency-free (stdlib only, so
this can run before `uv sync` -- see .github/workflows/ci.yml):

1. every ``*.safetensors`` file is real binary tensor content, not still an
   unhydrated Git-LFS pointer stub (a tiny ASCII file whose first line is
   ``version https://git-lfs.github.com/spec/v1``);
2. every shard's real SHA-256 matches its own committed ``manifest.json``
   entry, where a manifest exists for that directory. This does NOT change
   or duplicate the corpus's own expected checksums -- it reads the exact
   same committed manifest.json shard hashes ShardedScenarioDataset.
   verify_shard_checksums() already checks at load time, just earlier and
   with a clearer, CI-specific message.

REQUIRED_TENSOR_DIRS is the exact, audited set of committed LFS-tracked
tensor directories the real (non-locked-test) Python test suite reads --
found by grepping tests/ for every `data/learning-v2/**/*.safetensors`-
backed path actually referenced (see the backend-ci-portability fix commit
for the audit trail). Do not add other corpora "just in case" -- if a new
test starts reading a new committed corpus directory, add it here deliberately.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_TENSOR_DIRS: tuple[str, ...] = (
    "data/learning-v2/cycle-b2/tensors-normalized/validation",
    "data/learning-v2/cycle-b2-control-v2/tensors-normalized/train",
    "data/learning-v2/cycle-b2-control-v2/tensors-normalized/validation",
    "data/learning-v2/cycle-b2-ood-extension/tensors-normalized",
)

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(len(_LFS_POINTER_PREFIX))
    return head == _LFS_POINTER_PREFIX


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hashes(directory: Path) -> dict[str, str]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {shard["file"]: shard["sha256"] for shard in manifest.get("shards", [])}


def _check_directory(directory: Path) -> list[str]:
    if not directory.exists():
        return [f"REQUIRED TEST FIXTURE NOT HYDRATED: {directory} does not exist at all"]

    shard_files = sorted(directory.rglob("*.safetensors"))
    if not shard_files:
        return [f"REQUIRED TEST FIXTURE NOT HYDRATED: no .safetensors files found under {directory}"]

    manifest_hashes = _manifest_hashes(directory)
    errors: list[str] = []
    for shard_file in shard_files:
        if _is_lfs_pointer(shard_file):
            errors.append(
                f"REQUIRED TEST FIXTURE NOT HYDRATED: {shard_file} is still a Git-LFS "
                "pointer file, not real tensor content -- `git lfs pull` did not "
                "hydrate this path"
            )
            continue
        expected = manifest_hashes.get(shard_file.name)
        if expected is not None:
            actual = _sha256(shard_file)
            if actual != expected:
                errors.append(
                    f"COMMITTED TENSOR SHARD CHECKSUM MISMATCH: {shard_file} "
                    f"sha256={actual} but manifest.json declares sha256={expected}"
                )
    return errors


def main() -> int:
    all_errors: list[str] = []
    for relative in REQUIRED_TENSOR_DIRS:
        all_errors.extend(_check_directory(REPO_ROOT / relative))

    if all_errors:
        print("Git-LFS test fixture verification FAILED:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nHint: this usually means `git lfs pull --include=...` did not run, "
            "ran with the wrong --include patterns, or the LFS objects are missing "
            "from the remote. See the Git LFS hydration step in "
            ".github/workflows/ci.yml.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Git-LFS test fixture verification OK -- {len(REQUIRED_TENSOR_DIRS)} "
        "required directories hydrated and checksummed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
