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
| 2 | HydroScout supervision | **DONE** (label-generation/trajectory library) — see "Phase 2" below |
| 3 | HydroStrategist supervision | **DONE** (label-generation/trajectory library) — see "Phase 3" below |
| 4 | Learned OOD supervision | **DONE** (label-generation library) — see "Phase 4" below |
| 5 | Complete control targets (evidence_sufficiency/next_step) | **DONE** (label-generation library) — see "Phase 5" below |
| 6 | Auxiliary objectives | **DONE** (label-generation library) — see "Phase 6" below |
| 7 | Full trajectory corpus | **IN PROGRESS** |
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

## Phase 2: HydroScout supervision (label-generation library, commit `8e59a71`)

`hydroswarm.training.scout_trajectory.build_scout_trajectory()` repeatedly calls
`generate_scout_label()` with a growing `already_sampled` set, packaging each step
into a hash-chained `FullTrajectory` plus a `ScoutTrajectoryStep` carrying a strictly
targets_v2-governed `targets` dict (passes `validate_targets_v2()` as-is) and a
separate `diagnostics` dict for the non-governed extras Phase 2 item 3 also asks for
(per-node information gain, per-candidate accessibility). `scout_labels.ScoutLabel`
gained a `candidates` field (every candidate `rank_sample_locations` evaluated,
previously discarded) to support this without recomputing posterior/signature work.
10 tests, including one that caught a real off-by-one in the `maximum_samples`
termination check before it shipped.

Known, documented simplification: `generate_scout_label` always evaluates the *same*
base observations at every step (`already_sampled` changes which candidates are
excluded, not what evidence is revealed) — the fuller "simulate a genuinely new
observation after each sample" notion belongs to Phase 7, which doesn't exist yet
either. Phase 2's own literal spec is satisfied as written.

**Not yet done** (deferred to Phase 7, see below): an actual corpus-generation script
that runs this over real scenarios and writes a versioned dataset; the Scout benchmark
comparison against random/fixed/classical-EIG baselines (needs a trained Scout head,
which needs Phase 7's corpus + Phase 8 training).

## Phase 3: HydroStrategist supervision (label-generation library, commit `f517294`)

`hydroswarm.training.strategist_trajectory.build_strategist_trajectory()` classifies
an incident's probable source nodes via the same classical localizer the live
pipeline uses (`HybridInferencePipeline._signature_observations`/`localize_with_
signatures`/`_credible_nodes`/`_planning_context`, reused directly rather than
re-derived — deliberately avoiding the same train/serve-skew defect class the repair
pass found in commit `a99cdbc`), generates and exactly WNTR-verifies a bounded plan
set via the already-existing `generate_strategist_labels()`, and packages the result
as a single-step `TrajectoryState` plus one governed target dict per plan label. 7
tests, including independent re-verification that `plan_validity` is read only from
WNTR, never a template's predicted score.

**Real defect found and documented, not fixed** (out of this module's scope):
`hydroswarm.planning.response`'s bounded template generator produces 9 distinct
`action_template` values, but `HydroCore.action_head` defaults to
`action_vocabulary_size=8`. Any future Strategist-enabled training run must pass
`action_vocabulary_size=9` explicitly — the bare default is insufficient and would
misclassify or error on the last template. Not changed unilaterally: a model-
architecture default change affects checkpoint-loading compatibility
(core-issues.txt Task 4.0), and no promoted checkpoint currently trains this head.

**Not yet done** (deferred to Phase 7): multi-round plan revision (`revise_rejected_
plan`) is not wired in — Strategist supervision here is single-decision-point, per
Phase 3's own spec; corpus-generation script and Strategist benchmark, same reasons
as Scout above.

## Phase 4: learned OOD supervision (commit `47b94df`)

`hydroswarm.training.ood_labels.classify_ood_category()` derives a governed
`OODCategory` purely from `GeneratedScenario`/`ScenarioManifest` fields already
recorded at generation time (no config object needs threading through). Reproduces
6 of 11 categories (NONE, UNSEEN_TOPOLOGY, EXTREME_DEMAND, TANK_STATE_SHIFT,
ROUGHNESS_MISMATCH, SEVERE_MISSINGNESS, FROZEN_DRIFTING_SENSOR), each thresholded
with a wide margin above the corpus generator's documented in-distribution ranges.
The other 4 are explicitly not generated, documented in the module docstring
(UNSEEN_SENSOR_LAYOUT/VALVE_PUMP_MISMATCH/TIMING_OUTSIDE_TRAINING_RANGE need
generator capabilities that don't exist yet; UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_
CALIBRATION is a calibration-artifact concern, not a scenario property).

**Real defect found and fixed**: Phase 1's `TARGET_CLASS_COUNTS["ood_class"]` was
wrongly set to 3 (copied from `ood_head`'s width, a distinct 3-way severity concept —
`OODLevel` — not `ood_class`'s own 11-category definition). Would have silently
rejected 8 of 11 real categories as "invalid" the moment this phase generated real
labels. Fixed to `len(OODCategory)`.

## Phase 5: complete control targets (commit `899c7fb`)

`hydroswarm.training.control_labels.classify_evidence_sufficiency()` extends the
existing sensor-health-only rule with posterior entropy and OOD-category calibration
validity — the two signals from the governed definition that are actually computable
without a trained model. **Calibrated candidate-set size and classical-neural
disagreement remain out of reach**: both need a `CalibrationArtifact` fitted against
an already-*trained* Sentinel checkpoint, and evidence_sufficiency is itself a
training target for that same checkpoint — a genuine ordering dependency (resolves
after Stage 1 of Phase 8), not an oversight.

`classify_next_step()` mirrors `agents/controller.py`'s `EVIDENCE_CHECK` state
exactly for 3 of 4 `NextStep` values, and adds `INSPECT_SENSOR` (which has no live
FSM state) derived from `event_cause == SENSOR_FAULT` — flagged as this module's own
reasoned extension, not a literal port.

## Phase 6: auxiliary objectives (commit `0f48a8b`)

All three (`sensor_reconstruction_target`, `future_concentration_target`,
`travel_time_target`) reuse existing computed structures rather than adding new
simulation: sensor reconstruction/future concentration read the scenario's own
already-simulated `truth_concentration` at a reference/future time; travel time
reuses `HydraulicSimulator.build_dynamic_graph`'s `"travel_time_seconds"` edge
weight (the same structural feature already used as a model *input*, reused here as
ground truth for the *target*).

## Phase 7: full trajectory corpus (commits `5fdb9ac`, `e012bea`)

`hydroswarm.training.full_trajectory.build_incident_trajectory()` combines every
label generator Phase 1-6 built (Sentinel via `scenario_to_example`, OOD category,
evidence_sufficiency/next_step, Scout sampling sub-trajectory, Strategist planning
sub-trajectory, all three auxiliary targets) into one governed `IncidentTrajectory`
per scenario. Deliberately not flattened into one `FullTrajectory` hash chain — Scout's
and Strategist's own `TrajectoryState` sequences already validate their own internal
integrity independently; `IncidentTrajectory` is a container over the three pieces.

**Real, previously-undiscovered bug found and fixed while wiring this together**:
Phase 2/3/6's Scout/Strategist/auxiliary label generators were tested (in isolation)
against `sorted(network.junction_name_list)` as their node index space, but the
canonical space every other node-indexed target actually uses (`source_node`,
`sensor_fault`, `TopologyMetadata.node_ids`) is `canonical_node_order(network.
node_name_list)` — ALL nodes (junctions + reservoirs + tanks). The two only coincide
when a network has zero reservoirs/tanks, never true for a real network. Left
uncorrected, `sample_node`/`target_pointer`/`travel_time` indices would have silently
pointed at the wrong node once compared against `source_node_logits`'s shared index
space. Caught by `validate_targets_v2`'s `NODE_ARRAY_TARGETS` disagreement check
(Phase 1's own hardening) the moment Phase 7 combined everything against one real
topology. Fixed by having `build_incident_trajectory` derive `node_ids` from
`example.topology.node_ids` rather than accepting it as a parameter — removes the
possibility of a caller passing the wrong space, not just documents the requirement.

`scripts/generate_trajectory_corpus.py` is the corpus-generation driver: reuses an
existing Cycle B2-style corpus's own already-generated scenarios (never resimulates,
never writes under `--corpus-dir`), fits one `SignatureLibrary`/`SignatureArtifact`
per training topology from the target corpus's own train-split scenarios, and calls
`build_incident_trajectory` once per scenario, appending JSON-serialized results to a
per-split JSONL shard. Resumable (tracks processed `scenario_id`s, skips them on
restart) at the scale this targets. Smoke-tested against the real
`data/learning-v2/cycle-b2` corpus (10 scenarios, zero errors, ~0.5s/scenario
including topology-hash/OOD-category/Scout/Strategist/auxiliary computation).

**Real-scale generation launched** as a background job (`experiments/jobs/
cycle-b2-trajectories-train/`, PID recorded in that directory's `status.json`) over
the full 9,000-scenario train split, writing to `data/learning-v2/cycle-b2-trajectories/
train.jsonl` — at ~0.5s/scenario this is projected to take roughly 75 minutes. Being
polled every 10 minutes. Exact resume command if it needs restarting:
```bash
export PYTHONPATH=src
python scripts/generate_trajectory_corpus.py \
  --corpus-dir data/learning-v2/cycle-b2 \
  --output data/learning-v2/cycle-b2-trajectories \
  --split train
```
(idempotent -- already-processed scenario_ids in `train.jsonl` are skipped automatically).

## What's next

Once the train-split trajectory corpus finishes generating: commit it (`data/
learning-v2/cycle-b2-trajectories/{train.jsonl,train-report.json}` plus the signature
cache under `experiments/cache/signatures/`), then decide whether to also generate the
validation/calibration/development_holdout splits before moving to Phase 8-10 (staged
training against the new heads, bounded experiment matrix, promotion gates). Note that
Phase 8's own staged sequence starts with "train the corrected Sentinel backbone,"
which is already done (Phase 3's E0/E1 finalists) -- the next real step is Stage 2,
"add event and control heads" (`event_control_heads=True`), trained against this new
corpus's `event_presence`/`event_cause`/`evidence_sufficiency`/`next_step` targets.

## Restrictions honored

No work on `main`. `data/learning-v1`, `data/learning-v2/cycle-b`,
`data/learning-v2/cycle-b2`'s existing contents, all promoted checkpoints, and every
existing result artifact are untouched. Locked test data was not opened;
`final-selection.json` does not exist. No destructive git/filesystem commands were
used. No sudo, no credential exposure. All commits pushed to
`origin/agent/gcp-multitopology-v3`.
