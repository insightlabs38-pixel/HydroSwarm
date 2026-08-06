# core-issues2.txt (Scout/Strategist/OOD/control/auxiliary expansion) — live handoff report

Branch: `agent/gcp-multitopology-v3`. This is a **living document**, updated as each
milestone completes. Continues from `core-issues-repair-report.md` (Phase 1+2 repair,
done) and `phase3-completion-report.md`/`phase3-handoff.md` (Phase 3 rebuild/retrain,
done) and the ARM migration (`reports/migration/arm-migration-completion-report.md`,
done). See `/workspace/core-issues2.txt` for the full specification this implements.

**The locked final test has not been opened. `final-selection.json` does not exist.**
No work has occurred on `main`. `data/learning-v2/cycle-b`/`cycle-b2`, all promoted
checkpoints, and every existing result artifact remain untouched.

## Status summary (updated each milestone)

| Phase | Task | Status |
|---|---|---|
| 0 | Arm VM environment repair (EPANET build, pyarrow dependency, cross-arch FP determinism) | **DONE** — see "Arm environment fixes" below |
| 1 | Define and validate missing targets_v2 heads | **DONE** (schema/audit scope) — see "Phase 1" below |
| 2 | HydroScout supervision | **NOT STARTED** |
| 3 | HydroStrategist supervision | **NOT STARTED** |
| 4 | Learned OOD supervision | **NOT STARTED** |
| 5 | Complete control targets (evidence_sufficiency/next_step) | **NOT STARTED** |
| 6 | Auxiliary objectives | **NOT STARTED** |
| 7 | Full trajectory corpus | **NOT STARTED** |
| 8-10 | Staged training, experiments, promotion gates | **NOT STARTED** |

## Arm environment fixes (prerequisite, not in core-issues2.txt itself)

This session started on a freshly migrated aarch64 GCP VM (prior sessions ran on
x86). Three real environment defects were found and fixed before any core-issues2.txt
work could begin:

1. **wntr has zero linux-arm64 EPANET support** (`wntr/epanet/toolkit.py`'s Linux
   branch is a bare `else` with no arch check, unlike its darwin branch). Fixed by
   `scripts/build_epanet_arm64.sh`, which builds EPANET 2.2 from source
   (`OpenWaterAnalytics/EPANET` tag `v2.2`) and installs it over wntr's hardcoded
   `libepanet/linux-x64/libepanet22.so` path. **Must be re-run after every `uv sync`**
   (a fresh sync reinstalls wntr's wheel and reverts the patch). Commit `f540b54`.
2. **pyarrow was an undeclared dependency** — `scenarios.py` unconditionally writes
   `.parquet`, but pyarrow was never in `pyproject.toml` (only present on the old x86
   VM outside the lockfile). Added `pyarrow>=17` to `pyproject.toml`, relocked. Commit
   `c851daf`.
3. **Cross-architecture signed-zero non-determinism**: replaying the x86-generated
   Cycle B2 corpus's `deterministic_replay` gate on this Arm VM failed for 3 sampled
   scenarios on `artifact_sha256` alone (IEEE-754 leaves `np.maximum(0.0, -0.0)`
   implementation-defined across CPU/SIMD backends). Fixed the generator
   (`_degrade` now normalizes `-0.0` to `+0.0`) and hardened the gate to fall through
   to its existing semantic array-equality check instead of hard-failing on a hash
   mismatch alone. Neither the corpus's stored data nor its recorded hashes were
   altered. Commit `672dce7`.

Post-fix: `python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2`
— **9/9 passed**. Full test suite: 451 passed, 0 failed. `ruff`/`pyright` clean.

Exact continuation commands for a fresh clone on this Arm VM (or another one):
```bash
git clone --branch agent/gcp-multitopology-v3 https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm && git lfs install && git lfs pull
uv sync --frozen --extra dev
./scripts/build_epanet_arm64.sh
export PYTHONPATH=src
for split in train validation calibration development-holdout; do
  tar --use-compress-program=unzstd -xf artifacts/migration/cycle-b2-scenarios-${split}.tar.zst \
    -C data/learning-v2/cycle-b2/scenarios
done
python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2  # expect 9/9 passed
```

## Phase 1: define and validate missing targets_v2 heads

**Audit (item 1)**: a full head/target/loss/runtime-authority mapping was produced
(two parallel research passes — model/heads side and data/corpus side). Key findings:

- Of ~24 semantic `HydroCore` output keys, only 3 (`source_node_logits`,
  `evidence_sufficiency`, `sensor_fault_logits`) are both trained and read at runtime
  today. 4 more (`source_region_logits`, `start_time_logits`, `duration_logits`,
  `relative_strength_logits`) are trained against real corpus labels but never read
  by anything outside `core.py`/`losses.py` — trained-but-orphaned. **Not fixed in
  this pass** (deliberately out of Phase 1's scope — it's product/API wiring, not a
  target-schema gap); flagged here as a cheap, low-risk follow-up for whoever next
  touches `inference/results.py`.
- Three silent loss-key/targets_v2-name mismatches existed:
  `"action"`/`"action_pointer"`/`"ood"` (loss dict keys, matching the model's own
  *output* attribute names) vs. `"action_template"`/`"target_pointer"`/`"ood_class"`
  (targets_v2's governed *target* names). Any correctly generated governed target for
  these three heads would have silently trained nothing. **Fixed**, commit `882761f`
  (also removed 3 dead regression-loss entries for outputs `HydroCore` never
  produces: `residual_prediction`/`pressure_prediction`/`flow_prediction`).
- **Scout/Strategist/OOD label-generation logic already exists**, just isn't wired
  into a corpus builder: `hydroswarm/training/scout_labels.py`'s
  `generate_scout_label()`, `hydroswarm/training/strategist_labels.py`'s
  `generate_strategist_labels()`, and `hydroswarm/training/trajectory_v2.py`'s
  `TrajectoryState`/`FullTrajectory` container (hash-chained, integrity-checked) are
  all implemented and unit-tested in isolation, but no code anywhere constructs a
  real `FullTrajectory` from a scenario. This materially reduces Phase 2/3/7's scope
  from "build from scratch" to "wire an existing driver loop." Full detail in the
  session's research notes (not separately filed; see commit messages and this
  report for the load-bearing findings).
- The classical/deterministic OOD stack (`inference/ood.py`'s `OODDetector`,
  `training/ood_categories.py`'s `OODCategory` taxonomy + fail-closed behavior table,
  `calibration/conformal.py`'s topology/schema validity gates) is mature and already
  governs runtime behavior. Only the learned `ood_class` head's training signal is
  purely absent — architecture and the `"ood" in trained_tasks` runtime gate already
  exist and wait for a real label generator (Phase 4).
- `evidence_sufficiency`'s corpus label (`corpus.py:_evidence_sufficiency`) is
  explicitly documented as only the sensor-health-threshold subset of the full rule;
  the full rule (candidate-set size, posterior entropy, disagreement, OOD state)
  needs a live controller-loop step, i.e. depends on Phase 7's trajectory
  infrastructure existing first. A third, independently-defined "evidence sufficient"
  notion also exists in `agents/sentinel.py`'s FSM fallback and disagrees with both —
  reconciling all three is in scope for Phase 5, not resolved yet.
- `next_step` has no dormant/partial implementation anywhere — a clean from-scratch
  Phase 5 task, derived from `agents/controller.py`'s FSM transition logic.

**Schema validation (item 3)**: `validate_targets_v2` previously only checked
mask/value key consistency. Extended to fail closed on all six required checks
(missing required masks; invalid class ranges; invalid graph-local indices;
incorrect plan dimensions; non-finite regression values; disagreement between target
metadata and topology metadata) via an optional `topology: TopologyMetadata | None`
parameter (backward compatible). Commit `eb3fa71`, 12 new regression tests.

**Item 4** (labels only from deterministic simulation/policy/exact verification, no
manually invented labels): upheld throughout — no new labels were invented in this
phase; the existing label generators found in the audit already satisfy this (Scout
from EIG ranking, Strategist from WNTR verification, OOD categories from governed
distribution-shift generation).

## What's next

Phase 2 (HydroScout supervision) is next: build a trajectory-driver loop that calls
`scout_labels.generate_scout_label()` repeatedly per incident with a growing
`already_sampled` set, packaging each step into a `TrajectoryState` and the sequence
into a `FullTrajectory`, then a corpus-level script to run this over a set of
scenarios (reusing the existing Cycle B2 scenario archives per Phase 2's data spec,
writing output to a new versioned sibling directory rather than mutating
`data/learning-v2/cycle-b2` in place — following the exact precedent
`cycle-b2` itself set relative to `cycle-b`).

## Restrictions honored

No work on `main`. `data/learning-v1`, `data/learning-v2/cycle-b`,
`data/learning-v2/cycle-b2`'s existing contents, all promoted checkpoints, and every
existing result artifact are untouched. Locked test data was not opened;
`final-selection.json` does not exist. No destructive git/filesystem commands were
used. No sudo, no credential exposure. All commits pushed to
`origin/agent/gcp-multitopology-v3`.
