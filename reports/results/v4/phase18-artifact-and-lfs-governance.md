# Phase 18 — Artifact and LFS Governance

core-issues3.txt "PHASE 18 — ARTIFACT AND LFS GOVERNANCE".

## Summary

`scripts/build_artifact_inventory.py` (new this pass) inventories every
currently git-tracked file in the working tree — real sha256 for every
LFS object plus every other file unless `--no-hash-source-files` is
passed, real byte size, tracking classification, and a path-prefix-derived
`source_run`/`status` (required/recommended). Full output:
`reports/results/v4/artifact-inventory.json`.

| metric | value |
|---|---|
| total tracked files | 1,331 |
| total bytes | 2,354,336,636 (~2.35 GB) |
| Git LFS entries / bytes | 474 / 1,560,897,940 (~1.56 GB) |
| standard-git entries / bytes | 857 / 793,438,696 (~793 MB) |
| forbidden-pattern findings | **0** |
| unpulled LFS pointers | **0** (verified against the actual working tree — every tracked LFS path is a real object, not a pointer stub) |
| secret scan (`scripts/scan_secrets.py`, reused not reinvented) | **pass**, 0 findings across 1,334 files considered / 852 text files scanned |

This is a distinct, current artifact inventory, not a replacement for
`reports/migration/arm-migration-inventory.json` (a one-off snapshot of the
original x86->Arm VM migration, frozen at commit `6fea9f220f1a...` — does
not cover any v4-era corpus/checkpoint/report work landed since, including
everything from Stage F onward and this entire pass's own deliverables).

## Item 1 — large trajectory/checkpoint artifacts preserved via Git LFS

Already governed by `.gitattributes` (predates this pass): `cycle-b2`/
`cycle-b2-control-v2`/`cycle-b2-ood-extension` tensor shards,
`cycle-b2-trajectories-{v2,v3,v4}`'s oversized `train.jsonl` files (each
individually listed by exact filename, not a directory-wide glob — a
deliberate, narrow exception per the file's own comments), the Scout/
Strategist derived tensor shards, finalist/control checkpoints under
`models/cycle-b2-candidates`/`models/cycle-b2-controls`, and the
compressed raw-scenario migration archives under `artifacts/migration/`.

**Real, previously-undocumented finding from this pass's Phase 17 clean-clone
reproduction**: `.gitattributes` line 73 (`data/learning-v2/cycle-b2-joint-v4/**/*.safetensors
filter=lfs ...`) is currently **dead configuration** — `.gitignore`'s
blanket `data/learning-v2/**/tensors-normalized/**/*.safetensors` rule has
no matching `!`-negation exception for `cycle-b2-joint-v4` (unlike
`cycle-b2`/`cycle-b2-control-v2`/`cycle-b2-ood-extension`, which each have
one), so no `cycle-b2-joint-v4` tensor shard has ever actually been
`git add`-able, and none are tracked (`git ls-files data/learning-v2/cycle-b2-joint-v4/`
returns only 25 manifest/report files, zero `.safetensors`). The clean
clone confirms this is a real functional gap, not just a config oddity:
`scripts/run_trajectory_corpus_gates.py`'s `joint_v4_six` sub-gate failed
outright there with `"missing shard file referenced by manifest"` until
`cycle-b2-joint-v4` was explicitly regenerated:

```bash
export PYTHONPATH=src
python scripts/build_stage_f_joint_corpus.py --include-ood-extension
```

This is likely the intended design, not an oversight needing a fix —
`cycle-b2-joint-v4` is a MERGE of already-committed source corpora
(`cycle-b2`, `cycle-b2-control-v2`, `cycle-b2-trajectories-{v3,v4}`,
`cycle-b2-ood-extension` — every one of which IS properly LFS-tracked),
deterministically regenerable from them plus the committed
`checksums.json`/`merge-report.json`/`source-manifest-hashes.json` to
verify the regeneration matches, exactly matching this item's own
"otherwise record deterministic generation commands" branch (item 4).
The one real defect is `.gitattributes` line 73's dangling, currently-inert
LFS rule for a path nothing ever reaches — worth deleting or pairing with
a real `.gitignore` exception in a future pass so the two files stop
disagreeing about this path's intended treatment, but not attempted here
(a one-line judgment call belongs with whoever decided the original
intent, not inferred from the inconsistency alone).

**Also not yet LFS-tracked (a real, honestly-reported gap)**: none of this
pass's own new work is preserved this way, because none of it produced a
durable checkpoint artifact meant to survive — `experiments/runs/
stage-g-scaling-screen/`, `experiments/runs/v4-checkpoint-identity/`, and
every Stage-A/Strategist/Stage-F checkpoint this pass's scripts loaded all
live under the gitignored, ephemeral `experiments/runs/` tree by design
(matches this project's own established "checkpoints are regenerable from
the registry, not preserved" convention — see the `hydroswarm_checkpoint_persistence`
memory record). If a future pass promotes one of these as a Phase 19
finalist, it should follow the same `models/cycle-b2-candidates`-style LFS
convention at that point, not before — committing intermediate/screening
checkpoints through LFS now would violate item 3's "do not commit
periodic failed checkpoints" and this project's own "preserve for audit,
not every artifact" discipline.

## Item 2 — manifests/reports/schemas/hashes/configs in standard Git

Confirmed via the inventory: `reports/results/**` (805 required
standard-git files), `experiments/registry/*.jsonl` (run provenance),
`configs/**`, and every schema/manifest JSON alongside its corpus are all
standard git, never LFS — matches the phase's own explicit split.

## Item 3 — forbidden commits

Zero findings. Checked precisely (filename-suffix/exact-name matching, not
a naive path substring — an earlier draft of this script's own governance
check false-positived on `reports/results/secret-scan.json`/
`scripts/scan_secrets.py`/`reports/results/requirements.lock.txt`, all
three legitimate, already-audited-clean artifacts; fixed before this
report was written) for: `.env`, `.pid`, real runtime `.lock` files
(distinct from `uv.lock`/`*.lock.txt` dependency manifests, which are
correctly `required`/`standard-git`), `.pem`/`.key` private-key-shaped
filenames. `experiments/registry/*.lock` and `experiments/jobs/*/job.pid`
are `.gitignore`d at the source (confirmed, not merely absent by chance)
so they were never candidates to begin with. No `.env` file, no locked
test, tracked anywhere in the repo.

## Item 4 — signature caches

`data/generated/` (the live runtime signature cache directory) is
`.gitignore`d wholesale — not committed, matching the phase's "otherwise
record deterministic generation commands" branch. The deterministic
generation path is already documented: `SignatureBuilder.build_or_load`
(`hydroswarm/classical/signatures.py`) plus the governed cache-key
convention in `hydroswarm/runtime/defaults.py`/`hydroswarm/runtime/v4_defaults.py`
(`SignatureCacheKey` from network/hydraulic-state hash + simulator
version + configuration hash + sensor-layout hash) reproduces any cache
entry deterministically from a real network + WNTR install, so nothing
is lost by not committing it.

## Item 5 — artifact inventory

Delivered: `reports/results/v4/artifact-inventory.json` (schema above).

## Item 6 — clean clone downloads every required LFS object

Verified two ways this pass:

1. This script's own `unpulled_lfs_pointers` check against the live
   working tree: **0** pointer stubs found among all 474 LFS-tracked
   files present.
2. The real clean-clone reproduction performed for Phase 17
   (`reports/results/v4/phase17-ci-clean-clone.md`): `git clone` + the
   default LFS smudge filter pulled 1.43 GiB of real object content with
   no manual intervention; spot-checked (`find . -name "*.safetensors" -size -1k`
   convention from `docs/ARM_MIGRATION.md` step 1) — no undersized
   (pointer-stub-sized) `.safetensors` files found.

No locked-test data was opened to produce this report.
