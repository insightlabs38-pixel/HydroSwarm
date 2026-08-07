# core-issues3.txt pre-freeze implementation — live handoff report

Branch: `agent/gcp-multitopology-v3`. Living document, updated after each
major phase. See `reports/results/v4/pre-freeze-audit.md` for the Phase 0
audit this pass started from, and `/workspace/core-issues3.txt` for the full
specification.

**The locked final test has not been opened. `final-selection.json` does not
exist.** No work has occurred on `main`. `data/learning-v2/cycle-b2` (existing
contents), all promoted checkpoints, and every existing v3 result artifact
remain untouched.

## Status summary

| Phase | Task | Status |
|---|---|---|
| 0 | Audit current HEAD | **DONE** |
| 1 | Reconstruct exact scenario hydraulic state | **DONE** (code + tests) |
| 2 | Governed signature-artifact policy | **DONE** (documented + wired; approximation error unmeasured, flagged) |
| 3 | Repair Strategist label semantics | **DONE** (code + tests) |
| 4 | Candidate-conditioned Strategist | **DONE** (architecture + tests; not yet wired to real training data) |
| 5 | Closed-loop Scout states | **DONE** (core mechanism + tests; hard-case generation not started) |
| 6 | OOD taxonomy / event-cause | **DONE** (6.1 crash-bug + 6.2/6.4/6.5/6.6 all done + tested; 6.3 scoped to a tested recipe, full-scale corpus deferred to post-Phase-8 — see below) |
| 7 | Auxiliary objectives / regression losses | **IN PROGRESS** (7.1/7.2/7.3/7.4/7.5 done + tested; 7.6/7.7 scoped and deferred, see below) |
| 8 | Second-pass calibrated control targets | **IN PROGRESS** (code + unit tests done; Stage-A checkpoint training running as a background job; second-pass generation not yet run against the real checkpoint) |
| 9-20 | Architecture v4, training, gates, selection | not started |

Corpus regeneration (`data/learning-v2/cycle-b2-trajectories-v2/`) running;
`validation`/`calibration`/`development_holdout` complete with 0 errors,
`train` in progress (~3875/9000 as of this update, ~1.8h remaining at
observed rate) — see its own section below. Note: `train`'s in-progress
run predates Phase 5's Scout improvement (not restarted a third time for
it — see Phase 5's own section).

## Commits this pass (newest last)

1. `5f459e7` audit(pre-freeze): record current architecture and artifact gaps
2. `76ec631` fix(data): reconstruct exact scenario hydraulic contexts
3. `ce82bde` fix(data): report unsupported-topology skips instead of silently continuing
4. `5372075` fix(data): tolerate sub-float32-rounding noise on negligible-strength scenarios
5. `bb7698e` feat(classical): wire the signature registry into trajectory generation
6. `93d5bc3` fix(data): use stored scenario data for trajectories, not regenerated arrays
7. `668092b` docs(handoff): publish pre-freeze implementation handoff after Phase 1-2
8. `bfaf676` fix(strategist): derive exact consequence values and semantic targets
9. `849accf` docs(handoff): update after Phase 3 completion and corpus regeneration restart
10. `7628714` feat(model): add candidate-conditioned strategist architecture v4
11. `3db51d5` docs(handoff): update after Phase 4 completion, scope Phase 5
12. `aa3589d` feat(sampling): add arbitrary-node scenario truth extraction
13. `d56f976` feat(sampling): reveal genuinely new evidence in closed-loop Scout states
14. `1c1ae02` fix(test): stop seeding a scout test from Python's randomized string hash()
15. `3470bab` docs(handoff): update after Phase 5 completion
16. `13f4b09` fix(ood): separate category supervision from severity control
14. `1c1ae02` fix(test): stop seeding a scout test from Python's randomized string hash()

All pushed to `origin/agent/gcp-multitopology-v3`. Working tree clean apart
from the in-progress `data/learning-v2/cycle-b2-trajectories-v2/` and its
`experiments/jobs/cycle-b2-trajectories-v2-*/` job directories (uncommitted
until generation finishes with 0 errors on every split).

## Phase 1: reconstruct exact scenario hydraulic state — DONE

**Root cause confirmed**: `scripts/generate_trajectory_corpus.py` built exactly
one pristine WNTR network + `FeatureContext` per topology family and reused
both for every scenario in that family, discarding each scenario's own
randomized demand regime, roughness perturbation, tank-level variation, and
pipe-outage state. `build_incident_trajectory`, `scenario_to_example`, and
`build_strategist_trajectory` themselves already correctly accept and use a
per-scenario `network`/`feature_context` (fixed in the earlier
`core-issues.txt` repair pass) — the defect was isolated to this one caller,
narrowing the fix's scope considerably.

**Fix**: `hydroswarm/training/scenario_reconstruction.py`'s
`reconstruct_scenario_network()` is now the single canonical replay/
reconstruction function (used by both `run_corpus_gates.py`'s
`deterministic_replay` gate and `generate_trajectory_corpus.py`, per Phase 1
item 3). It replays a stored `ScenarioManifest` against its pristine topology
and returns the exact randomized network, its derived `FeatureContext`, and
identity hashes (topology/network-state/hydraulic-state), failing closed
(`ScenarioReconstructionError`) when the replay doesn't semantically match.

**Three real defects found running this at real scale against
`data/learning-v2/cycle-b2` (13,150 scenarios), not by inspection alone**:

1. A regression in my own first-draft refactor of the gate's replay-
   verification logic: short-circuited the array-level comparison whenever
   the *recorded* manifest hash matched, which doesn't detect tampering that
   only touches the raw `.npz` file. Caught by the existing
   `test_deterministic_replay_gate_fails_closed_on_tampered_artifact` test.
   Fixed: the array comparison is now unconditional whenever an `original`
   scenario is supplied.
2. Sub-float32-rounding noise (~2.7e-19 on ~1.5e-8-magnitude
   `NEGLIGIBLE_STRENGTH_MG_MIN` concentrations) between two independently-
   deterministic reconstructions and the originally-stored corpus array —
   same class of cross-environment nondeterminism as the already-documented
   signed-zero case, just below `np.array_equal`'s resolution instead of at
   it. Fixed with an `atol=1e-6` `np.allclose` fallback (three orders of
   magnitude below `quantization_step`, the smallest physically meaningful
   resolution anywhere else in the system) — verified this does NOT mask a
   real difference (a hand-tampered 0.01 mg/L difference still raises).
3. `development_holdout` mixes plain-curriculum scenarios with two
   OOD-holdout helpers (`generate_cycle_b_corpus.py`'s
   `_generate_ood_holdout_for_training_topology`) that use a materially
   different, hardcoded degradation-probability formula
   (`missing_probability=0.45` vs. the curriculum formula's `<=0.08`) —
   already known and documented: `run_corpus_gates.py`'s own
   `deterministic_replay` gate excludes this exact split because "which
   formula was used ... cannot be distinguished from the manifest alone."
   This caused a genuine ~23% spurious failure rate on `development_holdout`
   (218/950 scenarios in the run that surfaced it). Fixed: retries with the
   array-level check skipped (only for this split, only when the full check
   fails) and records `weak_verification: true` on the affected rows —
   visible, not silently smoothed over. `replay_sha256` (seed/source/network/
   stage/split identity) is still verified either way.

Also found and fixed while wiring: `build_incident_trajectory` was being
called with `reconstruction.scenario` (a freshly regenerated copy, produced
only to support the verification check) instead of the original,
already-ground-truth scenario loaded from disk. There is no reason to prefer
regenerated data over data the corpus already has stored correctly. Now
always builds from the original `scenario`; only `reconstruction.network`/
`.feature_context` (genuinely not stored anywhere else) come from
reconstruction.

Also fixed (Phase 1 item J): unsupported-topology scenarios (currently
`development_holdout`'s coastal-branch/unseen-topology subset — no governed
signature artifact exists for it yet, see Phase 2) were silently `continue`d
past. Now counted and reported in both the per-split `report.json`
(`skipped_unsupported_topology_this_run`) and a printed warning.

**6 + 2 = 8 new regression tests**
(`tests/scientific/test_scenario_reconstruction.py`): different seeds produce
different network/hydraulic-state hashes; reconstruction matches original
semantic replay; fails closed on a manifest that doesn't match the given
topology; travel-time labels change with hydraulic state; reconstructed
network is not the pristine object; a direct proof the old shared-context
shape would have collapsed two scenarios' travel-time labels; negligible-
magnitude float noise does not fail reconstruction; a real-magnitude
difference still fails closed.

**521 tests passing** (was 513 at session start), ruff/pyright clean, 9/9
corpus gates.

**Not done in this pass** (Phase 1 item 6, explicitly deferred, not silently
dropped): extending topology loading to `development_holdout`'s coastal-
branch/unseen-topology scenarios. Phase 2 items 5/R forbid fitting a
topology-specific signature artifact from development-holdout incidents, so
processing these requires the governed "Scout/Strategist unavailable,
abstain/fall back" path Phase 2 item 5 describes, which doesn't exist yet.
Tracked, reported per-run, not resolved.

## Phase 2: governed signature-artifact policy — DONE

**Audit finding**: `hydroswarm/classical/signature_registry.py`'s
`SignatureRegistry` (train-only-fitting guard, deterministic cache-key
digests, fail-closed `require()`) already existed from the earlier
`overnight-plan.txt` Task 1.2 repair pass, with its own full test suite
(`tests/scientific/test_signature_registry.py` — cache-key completeness,
no-cross-state-hit, deterministic rebuild, separate topology hashes) — but
was never actually called anywhere. `generate_trajectory_corpus.py` fit
artifacts straight against `SignatureCache`, bypassing the registry layer.

**Policy documented and named**: `TOPOLOGY_WIDE_REGIME_HASH` — every consumer
today fits exactly one signature artifact per topology from that topology's
full train-split population, regardless of demand regime or other in-corpus
hydraulic variation. This is Phase 2's "Option B: bucketed hydraulic-regime
artifacts" at the coarsest possible bucket boundary (one bucket per
topology, not per regime). Not a behavior change — `register()` only adds
governed bookkeeping on top of the same underlying artifact-fitting calls.

**Remaining work, explicitly flagged**: approximation error against exact
per-scenario state-specific artifacts is unmeasured. Runtime has no
signature-artifact consumption path yet to compare against training's
policy for equality (grepped `src/hydroswarm/runtime/defaults.py` and
`inference/`: zero references) — there is nothing on the runtime side to
wire the registry into yet, so "runtime/training policy equality" (Phase 2
item 7's last bullet) has no runtime half to test against.

## Phase 3: repair Strategist label semantics — DONE

Three real defects fixed in `strategist_labels.py` (full detail in commit
`bfaf676`'s message):

1. **Selection bias (3.1)**: training-label generation only verified plans
   the old heuristic prescreener selected (`prescreen_top_plans`,
   `predicted_validity >= 0.5`, top-3-by-predicted-score) plus `NO_ACTION` —
   using a heuristic to decide which candidates receive labels would have
   made it structurally impossible for a learned prescreener to ever beat
   it. `generate_strategist_labels` now exactly WNTR-verifies the FULL
   bounded candidate set (up to the canonical 9 templates).
2. **Ungoverned plan_value (3.2/3.3)**: was
   `proposal.predicted_value * proposal.predicted_validity` — the old
   heuristic's own unverified score. New
   `hydroswarm.planning.plan_value_policy` module (versioned,
   `PLAN_VALUE_POLICY_VERSION`) derives `plan_value`/`regret` and all five
   previously-defined-but-never-populated consequence-proxy targets
   (`exposure_proxy`, `pressure_risk_proxy`, `service_loss_proxy`,
   `containment_time_proxy`, `plan_regret_proxy`) from exact WNTR
   `ConsequenceMetrics` only. 6 monotonicity tests pass (lower exposure/
   fewer pressure violations/greater service availability/shorter
   containment time cannot reduce `plan_value`; best plan in a pool has
   zero regret; `NO_ACTION` is scored identically to every other
   candidate, never given an automatic free pass).
3. **Broken target-pointer semantics (3.4)**: `target_pointer` was computed
   via `sorted(network.junction_name_list)` at label-generation time — the
   same node-space bug class already found and fixed for Scout/Sentinel
   (junction-only order silently disagrees with the canonical node space
   every other node-indexed target uses), and link targets were silently
   dropped entirely (`positions.get()` returned `None` for link names,
   since `positions` only ever contained junction names).
   `StrategistLabel` now carries semantic identity
   (`primary_target_id`/`primary_target_type: NONE|NODE|LINK`); index
   resolution against the scenario's own canonical `node_ids`/`edge_ids`
   now happens only at tensor-building time
   (`strategist_trajectory._resolve_target_pointer`).

Also (3.5): centralized the canonical 9-template vocabulary into
`hydroswarm.planning.action_templates`; raised `generate_response_plans`'
`maximum_plans` cap from 8 to 9 (it structurally excluded the 9th template —
`ALTERNATE_VALVE_CUT` — whenever the other 8 were eligible, a second,
independent instance of the known 8-vs-9 mismatch).

**Attempted and reverted**: raising `HydroCore.action_vocabulary_size`'s
default from 8 to 9 to match. The promoted checkpoint does not pin this in
its recorded architecture config, so it silently relies on the constructor
default despite never training the action head — raising it broke strict
reload of that checkpoint (`action_head`'s saved weight shape is `[8, ...]`),
caught immediately by the full test suite
(`test_default_pipeline_factory.py`). Reverted with a comment explaining
why; deferred to Phase 9's architecture-v4 contract, where config
completeness is meant to be strictly validated rather than left to a
constructor default. **This is a real, generalizable lesson for the rest of
this pass**: any shared model-constructor default change needs a full test
run before being treated as safe, even when the head in question appears
untrained everywhere the audit checked.

15 new/rewritten tests. 536 tests passing (was 513 at session start),
ruff/pyright clean.

## Trajectory corpus regeneration (data/learning-v2/cycle-b2-trajectories-v2/) — COMPLETE, PROVISIONAL

All 4 splits finished with 0 errors and are now committed:

| split | scenarios | errors | notes |
|---|---|---|---|
| train | 9000/9000 | 0 | |
| validation | 1000/1000 | 0 | |
| calibration | 1000/1000 | 0 | |
| development_holdout | 2150/2150 | 0 | 400 `coastal-branch` (unseen-topology) scenarios skipped -- reported, per Phase 1 item 6/J, not silently dropped |

Total 181 MB, committed as regular Git text (not LFS) -- matches the
established convention for `data/learning-v2/cycle-b2-trajectories/`'s own
JSONL (`.gitattributes`'s comment: "The trajectory JSONL files themselves
stay regular git text"). Only derived tensor shards would need LFS, and
none have been built from this corpus yet (`scripts/merge_trajectory_
targets.py` / a tensors-enriched build was not run this pass).

**This corpus predates three fixes landed during this same pass and is
therefore PROVISIONAL, not ready for real training**, per the established
"don't restart an already-90%-complete run for an enhancement" precedent
from Phase 5:

1. Phase 6.4's `HYDRAULIC_MISMATCH` mislabeling fix (`corpus._event_cause`)
   -- every NORMAL-event SHIFT/ADVERSARIAL-stage scenario in this corpus
   still carries the old, incorrect label.
2. Phase 7.2's Scout `information_gain`/`candidate_reduction` per-node
   target shape (this corpus still has the old scalar shape).
3. Phase 7.3's sensor_reconstruction denoising-only mask, Phase 7.4's
   future_concentration disable, and Phase 7.5's travel_time log1p
   transform (all still reflect pre-fix semantics in this corpus).

A full regeneration incorporating everything from Phases 1-7 together is
needed before this corpus can be used for real Scout/Strategist/auxiliary
training -- deliberately deferred to Phase 10's dataset-versioning work
(which needs its own regeneration anyway, for the sharded Scout/
Strategist/OOD dataset layout Phase 10 specifies) rather than a fourth
partial restart of this JSONL-only artifact.

Was run as 4 resumable background jobs (`experiments/jobs/
cycle-b2-trajectories-v2-{train,validation,calibration,development_holdout}`),
launched via `hydroswarm.training.job_runner`, polled at a 10-minute interval.

**Regenerated from scratch after Phase 3** (second restart): the first
complete run (Phase 1+2 fixes only, `validation`/`calibration` reached 0
errors, `development_holdout` reached 0 errors with 327/2150 rows using the
documented `weak_verification` fallback, `train` was ~18% through) predated
the Phase 3 Strategist fix — every row written under that run had the old,
buggy `plan_value`/target-pointer semantics. Rather than ship a corpus with
inconsistent Strategist-label semantics across rows, all four splits' output
was deleted (confirmed via `git status` that nothing had been committed yet)
and regeneration restarted clean against commit `bfaf676`. In progress as of
this report.

The old `data/learning-v2/cycle-b2-trajectories/` (built with the pristine-
context bug) is left untouched and remains marked provisional/invalid per
restriction #5 — not used for any of this pass's work.

Exact resume commands (idempotent — skips already-processed scenario_ids):

```bash
export PYTHONPATH=src
for split in train validation calibration development_holdout; do
  python scripts/generate_trajectory_corpus.py \
    --corpus-dir data/learning-v2/cycle-b2 \
    --output data/learning-v2/cycle-b2-trajectories-v2 \
    --split "$split"
done
```

Job status: `cat experiments/jobs/cycle-b2-trajectories-v2-<split>/status.json`
(note: `state` is only updated by an explicit `mark_finished()` call — check
`pid` liveness directly with `kill -0 <pid>` for ground truth while a job is
running, since nothing auto-reconciles `status.json` when a plain script
process exits on its own).

## Phase 4: candidate-conditioned Strategist architecture — DONE (architecture only)

`hydroswarm.model.candidate_plan_encoder.CandidatePlanEncoder` (commit
`7628714`) replaces the anonymous learned plan-query representation with one
built from each candidate plan's own template/target/features. Wired into
`HydroCore` behind a new `strategist_mode` flag
(`"anonymous_queries"` default / `"candidate_conditioned"`), following the
existing net-new-parameters-gated-behind-a-flag convention
(`event_control_heads`/`auxiliary_heads`/`consequence_prescreening_heads`):
zero new parameters and zero behavior change in default mode. Added to
`architecture_config()`/`verify_architecture_compatibility()`.

**Real regression found and fixed during this phase, not just the intended
work**: raising `action_vocabulary_size`'s default from 8 to 9 (attempted as
part of Phase 3, reverted) broke the promoted checkpoint's strict reload.
Learned from that: before committing Phase 4, re-ran
`test_default_pipeline_factory.py` specifically (the file that caught it)
in addition to the full suite — both clean.

**Not done in this pass** (explicitly deferred, not silently dropped):
wiring real training data into this path. That requires Phase 10's
Strategist collator (variable candidate count per incident, none of which
exists yet) and the runtime top-K-exact-verification integration Phase 15
describes. This commit delivers and tests the architecture component and
its `HydroCore` integration; nothing trains through it yet.

## Phase 5: closed-loop Scout states — DONE (core mechanism)

**Root capability delivered**: `hydroswarm.training.scenario_reconstruction.
simulate_all_node_truth` (commit `aa3589d`) reruns `simulate_incident`
against the already-reconstructed exact randomized network and the
manifest's own recorded incident parameters — no fresh RNG draws needed —
returning concentration at EVERY node, not just a scenario's originally-
chosen sensor subset. Verified two ways: its values at the original sensor
nodes reproduce the corpus's own stored `truth_concentration` array
exactly (not just within tolerance), and it successfully reaches nodes
outside that subset with finite values.

**Incremental revelation wired in** (commit `d56f976`):
`generate_scout_label` gained an optional `revealed_samples` parameter
(merged into the signature-matching observation grid before computing the
posterior); `build_scout_trajectory` gained an optional `reconstruction`
parameter — when supplied, each step reveals a genuinely new deterministic
measurement (seed = `scenario_id + step_index + node_id`) at the PREVIOUS
step's recommended node and folds it into evidence for every subsequent
step, replacing the old behavior where every step re-ranked the same fixed
base observations. Proven with a real test scenario constructed to have
nonzero baseline posterior entropy (1.0 bits): entropy provably changes
after a genuine reveal. Omitting `reconstruction` reproduces the exact
pre-Phase-5 behavior (backward-compatibility test included) — existing
callers are unaffected.

**Real bug found while testing** (not by inspection): the observation
arrays `_reindex_to_signature_grid` returns can be read-only pandas-backed
views; mutating them for a revealed sample raised `ValueError` the moment
a real (non-synthetic-array) test exercised the path. Fixed by copying
before mutation.

**Incidental fix**: found and fixed an unrelated pre-existing flaky test
(`test_information_gain_is_nonnegative_within_tolerance` seeded itself from
Python's per-process-randomized `hash(str)`) while working in the same file
(commit `1c1ae02`).

**Not done in this pass** (Phase 5 items 5.4, explicitly deferred):
accessibility/hard-case generation (best-EIG-node inaccessible, near-tied
EIG, exhausted budget, severe missingness). The mechanism these cases need
to exercise (real incremental revelation) now exists; generating the cases
themselves is a smaller, separable follow-up. Also not done: wiring
`reconstruction` into the currently in-progress `train` corpus regeneration
(would require a third restart of an already-90%-complete run for an
enhancement, not a correctness fix — deferred to the next full
regeneration).

## Phase 6: OOD taxonomy — PARTIAL (core crash-bug fixed)

**Real, previously-unobserved defect found and fixed** (commit `13f4b09`):
`compute_multitask_loss` mapped the governed `ood_class` target (11
categories) to `HydroCore`'s pre-existing `ood_head` — a 3-logit head for
an entirely different concept (`OODLevel`'s deterministic severity,
computed by `OODDetector`, which remains authoritative and untouched).
Training `ood_class` with a real label index >= 3 (`SEVERE_MISSINGNESS=7`,
`FROZEN_DRIFTING_SENSOR=8`) would raise — not yet observed in any promoted
run only because the prior Stage-1 smoke screening's 200-example
mini-corpus subset apparently never happened to sample one of the
559/13,150 (~4.2%) non-NONE examples in the real corpus.

Added a correctly-sized, separately-gated `ood_category_head` (11 logits)
as a net-new opt-in output, following the same checkpoint-compatibility
convention as every other Phase 4.x/6.x flag this pass established
(gated construction, `architecture_config()` entry,
`verify_architecture_compatibility()` check, parameter-report accounting).
The old 3-logit `ood_head` is completely untouched. Fixed the loss mapping
from `"ood_class": "ood_logits"` to `"ood_class": "ood_category_logits"`.
11 new tests, plus 2 pre-existing `test_training.py` tests updated (they
constructed a synthetic `ood_logits` output intentionally exercising the
old, now-fixed mapping).

**6.2 — supported-category metadata (DONE, in a follow-up sub-pass)**:
added `ood_labels.SUPPORTED_OOD_CATEGORIES`/`UNSUPPORTED_OOD_CATEGORIES`
(7 supported: NONE + UNSEEN_TOPOLOGY/EXTREME_DEMAND/TANK_STATE_SHIFT/
ROUGHNESS_MISMATCH/SEVERE_MISSINGNESS/FROZEN_DRIFTING_SENSOR; 4
unsupported: UNSEEN_SENSOR_LAYOUT/VALVE_PUMP_MISMATCH/
TIMING_OUTSIDE_TRAINING_RANGE/UNSUPPORTED_NETWORK_ELEMENT_OR_INVALID_
CALIBRATION) and `corpus.SUPPORTED_EVENT_CAUSES`/`UNSUPPORTED_EVENT_
CAUSES` (HYDRAULIC_MISMATCH + AMBIGUOUS unsupported). These turn what was
previously only module-docstring prose into queryable constants, each
backed by a real regression test that exercises the actual classifier
across every documented threshold/config combination (not merely asserted
to match the docstring) — the exact metadata Phase 9's checkpoint contract
(`trained_ood_categories`/`validated_ood_categories`) will consume.

**6.3 — balanced OOD extension (PARTIAL, scoped deliberately)**: most of
this item's required measurements (false-normal rate, macro F1,
calibration by category, unsafe non-abstention) are evaluation metrics
against a *trained* classifier — structurally blocked on Phase 8's Stage-A
checkpoint, the same dependency `control_labels.py`'s own docstring
already documents for evidence_sufficiency's remaining signals. What IS
achievable now: `ood_labels.OOD_TRIGGERING_CONFIG_OVERRIDES`, a governed,
versioned recipe of `ScenarioGenerationConfig` overrides that reliably
triggers each of the 5 non-topology reproducible categories — verified
(not merely asserted) against `classify_ood_category` across 3 seeds each
in `test_every_recipe_override_reliably_triggers_its_category`. This is
the reusable building block a real balanced-corpus generation script
needs; turning it into an actual Cycle-B2-scale corpus is deferred —
running one now would compete for CPU with the still-in-progress
`cycle-b2-trajectories-v2` train regeneration, and most of the item's
required metrics can't be computed without Phase 8's checkpoint regardless.

**6.4 — VALVE_PUMP_MISMATCH / HYDRAULIC_MISMATCH (DONE — "remove" branch
taken)**: found a real, *active* defect while implementing this, not by
design review: `corpus._event_cause` was unconditionally labeling every
NORMAL-event scenario generated at `CurriculumStage.SHIFT`/`ADVERSARIAL`
as `EventCause.HYDRAULIC_MISMATCH` — but `scenarios.py`'s
`model_mismatch["valve_telemetry_incorrect"]` is purely
`stage in {SHIFT, ADVERSARIAL}`, a curriculum-stage *label* with **no
corresponding simulated valve/pump/topology perturbation behind it**
(confirmed by inspection of `_randomize_hydraulics` and
`generate_with_network`: the flag is written into the manifest and never
otherwise consulted). Every such scenario was a genuinely quiet,
internally-consistent network mislabeled as a hydraulic mismatch — the
currently-running `cycle-b2-trajectories-v2` train regeneration has been
generating these bad labels throughout its run (see corpus-regeneration
note below). Fixed by removing `HYDRAULIC_MISMATCH` from `_event_cause`'s
derivation entirely (the "remove" branch of Phase 6.4's explicit
implement-or-remove choice) rather than fabricate a real perturbation
under time pressure. `OODCategory.VALVE_PUMP_MISMATCH` was already
unreachable (never generated) prior to this pass — now formally recorded
in `UNSUPPORTED_OOD_CATEGORIES`.

**6.5 — AMBIGUOUS event cause (DONE, resolved as "mark unsupported")**:
already never generated (unchanged); now formally recorded in
`UNSUPPORTED_EVENT_CAUSES` and covered by
`test_event_cause_never_assigns_an_unsupported_class`, which exercises
`_event_cause` across every event_type × curriculum_stage combination and
asserts the result is always in `SUPPORTED_EVENT_CAUSES`.

**6.6 — `category != NONE -> OUTSIDE_VALIDATED_RANGE` collapse (DONE —
verified, no code change needed)**: `full_trajectory.py`'s
`classify_next_step(ood_level_outside_validated_range=category !=
OODCategory.NONE, ...)` looked like the exact anti-pattern this item
warns against, but is actually *consistent* with the governed
`OOD_CATEGORY_BEHAVIOR` table: every currently-defined non-NONE category
already has `planning_permitted=False, calibration_valid=False` (no
CAUTION-only/partial-severity category exists yet). Added
`test_every_non_none_category_currently_suppresses_planning_and_
invalidates_calibration` to make this invariant an explicit, checked fact
rather than an implicit one — if a future category is added with
`planning_permitted=True`, this test fails and flags every
`category != NONE` shortcut in caller code for review, rather than one
silently becoming wrong.

**Consequence for the in-progress corpus regeneration**: the 6.4 fix
changes `event_cause` label semantics for NORMAL-event SHIFT/ADVERSARIAL-
stage scenarios. `cycle-b2-trajectories-v2`'s `train` split (still running
as of this update) was generated entirely under the pre-fix code and
therefore contains the HYDRAULIC_MISMATCH mislabeling bug throughout —
consistent with the established precedent (Phase 5's Scout-reveal
improvement was likewise not restarted a third time into an
already-90%-complete run). This corpus remains useful for validating the
Phase 1-3/5 pipeline mechanics themselves, but is now also provisional
with respect to Phase 6.4's fix, on top of Phase 7's auxiliary-target
fixes — a full regeneration incorporating everything from Phases 1-7
together is needed before this corpus can be used for real training, and
belongs with Phase 10's dataset-versioning work rather than a fourth
partial restart.

## Phase 7: auxiliary objectives / regression losses — IN PROGRESS

**7.1 — `masked_regression()` helper (DONE)**: `losses.py`'s old
`_masked_mse` checked only `torch.isfinite(target)`, silently ignoring
every regression target's `f"{task}_mask"` companion. Since every masked
regression target's generator (plan_value, the five consequence proxies,
sensor_reconstruction, future_concentration, travel_time) writes an
explicit `0.0` *finite* placeholder at masked positions specifically so
absence is recorded in the mask rather than the value, this trained the
model directly against those placeholders — confirmed by inspection, not
guesswork (Phase 7.1's exact description). New `masked_regression()`
combines the mask with the finite check and additionally raises on a
prediction/target shape mismatch instead of silently broadcasting
(directly closes 7.2 below). Wired into every entry in
`compute_multitask_loss`'s `regressions` dict.

**7.2 — Scout EIG per-node alignment (DONE)**: `information_gain`/
`candidate_reduction` were previously scalars (only the selected
sample_node's value), which would have silently broadcast-mismatched
against HydroCore's real per-node `expected_information_gain` output
(`[B, N]`) the moment `masked_regression`'s new shape check ran — not yet
triggered in production only because no caller currently merges Scout
targets into a real training batch (Phase 10's Scout collator doesn't
exist yet). Converted to per-node arrays (`shape [node_count]`), populated
at every candidate `rank_sample_locations` actually scored (accessible,
ranked), masked elsewhere. Added `ScoutLabel.total_candidate_count` so the
per-node reduction-fraction normalization isn't duplicated logic.
`targets_v2.NODE_ARRAY_TARGETS` now includes both. 2 new tests, incl. one
that runs the real per-node target through `masked_regression` against a
same-shaped fake prediction (not just an isolated shape assertion).

**7.3 — Sensor reconstruction denoising-only (DONE)**: the target
previously unmasked *every* sensor node's truth value, regardless of
whether that position's own input reading was degraded. Since
`corpus.build_sensor_series` feeds the model
`scenario.observed_concentration` directly (which equals
`truth_concentration` exactly at every healthy position), this mostly
supervised trivial identity copying (item I). Now masked to only
genuinely missing/frozen/communication-outage positions at the reference
instant. Rewrote the two existing tests (which asserted "every sensor is
unmasked", now false) into a healthy-scenario test (nothing masked in) and
a forced-degradation test (everything masked in, correct truth value).

**7.4 — Future-concentration leakage (DONE — disabled, not silently
shipped)**: confirmed by inspection **and a passing regression test**
that `corpus.scenario_to_example` builds temporal/quality input tensors
via `HydraulicFeatureBuilder.build(..., window_steps=len(scenario.
timestamps_seconds))` — i.e. the model's own visible input already spans
the *entire* simulated window (~24h per `simulation/network.py`'s
`options.time.duration`), not a bounded lookback from a "current" instant.
`future_concentration_target`'s target instant (2h in, by default) is
therefore already directly present as a raw input feature — exact
leakage, not approximate (item H). A real fix needs a genuinely
cutoff-aware `ScenarioExample` variant (Phase 5's "explicit observation
cutoff" applied to the base Sentinel example, not only Scout) — materially
larger than a Phase 7 loss-masking fix, so out of scope here. Rather than
ship it broken or half-fix it under-tested, `future_concentration_target`
now always returns an all-masked placeholder (fail-closed — masked
positions contribute nothing to the loss either way), with a test
(`test_future_concentration_disable_is_justified_by_real_input_window_
leakage`) that builds a real `ScenarioExample` and proves the target
timestamp is literally present in `example.inputs["timestamps"]`, so the
justification for disabling it is checked, not just asserted in a
docstring.

**7.5 — Travel-time log1p transform (DONE)**: raw seconds (0 to tens of
thousands) would dominate a multitask MSE next to the mostly-[0,1]-scaled
other regression targets. Added `auxiliary_labels.TRAVEL_TIME_TRANSFORM =
"log1p"`, applied at generation time; inverse (`expm1`) documented for
physical-unit reporting once an evaluation script exists. New test
verifies the stored value against an independently recomputed
`np.log1p(raw_seconds)`.

**7.6 — Normalize heterogeneous regression targets (mostly already true,
confirmed not re-derived)**: the five Strategist consequence-proxy targets
were already train-owned-scale-normalized by Phase 3's
`plan_value_policy.py` (each component divided by a fixed, documented,
train-owned scale — not fit from data). Travel-time is now covered by 7.5.
Not further touched: sensor_reconstruction/future_concentration's mg/L
scale (future_concentration is disabled per 7.4 regardless).

**7.7 — Binary logits (deferred to Phase 9, not done)**: `evidence_head`
ends in `nn.Sigmoid()` (trained with `F.binary_cross_entropy` against a
probability, correctly, just not the v4-preferred logits+
`BCEWithLogitsLoss` convention); `event_presence_head` is already a raw
`RoleHead` (logits + `BCEWithLogitsLoss`, already correct);
`should_continue_sampling` has no model head/output at all yet (no loss
entry either — Phase 5.5/Phase 9 Scout-head work, not a Phase 7 bug).
Changing `evidence_head`'s activation in place would silently break strict
reload of the promoted checkpoint (established lesson from Phase 3/4's
`action_vocabulary_size` regression) — a new logits-based head belongs
behind a checkpoint-compatibility flag with Phase 9's architecture-v4
contract, not as an in-place Phase 7 edit.

**Not done in this pass**: `should_continue_sampling`/`candidate_reduction`
still have no model head (Phase 9 Scout-head-configuration territory);
sensor_reconstruction/future_concentration mg/L-scale normalization
(future_concentration is disabled anyway).

## Phase 8: second-pass calibrated control targets — IN PROGRESS

**Stage-A checkpoint training (steps 1-3) — running as a background job.**
Reuses the already-tested `scripts/run_stage3_finalist_training.py`
(train → select via validation → fit conformal calibration on the fused
hybrid predictor, exactly Phase 8 steps 1-3) against `data/learning-v2/
cycle-b2/tensors-normalized` with the previously-best config (`E1`,
`prior_mode=feature_only`), 2 seeds. No prior checkpoint files survived on
this environment to reuse (only their result JSON metadata did — see
`reports/results/v3/cycle-b2-stage3-E1.json`), so this is a genuine fresh
training run, not a re-analysis of stale artifacts.

Launched via `hydroswarm.training.job_runner.launch()`:

```
run dir: experiments/jobs/v4-stage-a-sentinel
command: python -u scripts/run_stage3_finalist_training.py \
  --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
  --finalists E1 \
  --run-root experiments/runs/v4-stage-a-sentinel \
  --registry experiments/registry/v4-stage-a-sentinel.jsonl \
  --output reports/results/v4/stage-a-sentinel-training.json
env: OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
```

Resume command is identical (this script has no `--resume-from`; a
re-invocation starts a fresh timestamped run directory, not a checkpoint
resume — noted honestly, not claimed as true interrupt/resume support).
Per-epoch progress: `experiments/runs/v4-stage-a-sentinel/E1-seed<seed>/
<timestamp>/metrics.jsonl`. Expected wall time ~70-75 min/seed (matching
the historical `cycle-b2-stage3-E1` run's 4235-4329s), ~2.5h total for both
seeds + calibration/eval. Polled at 10-minute intervals via a persistent
Monitor.

**Known data-quality caveat, carried into this checkpoint**: `data/
learning-v2/cycle-b2` predates this session's Phase 6.4 fix and contains
633/12750 (~5%) examples with the `HYDRAULIC_MISMATCH` mislabeling bug
(see Phase 6 section above) in their `event_cause` target. `cycle-b2` is a
protected, immutable artifact per this spec's restriction #3 and must not
be regenerated to fix this. The resulting Stage-A checkpoint's
`event_cause` head will have learned from ~5% mislabeled examples for that
one class -- a real, bounded, now-documented limitation, not a blocker for
Phase 8's purpose (the checkpoint's `source_node`/`source_region`/profile
heads, which second-pass control labels actually depend on, are unaffected
by this specific label class).

**Second-pass label generation (steps 4, 5, 7) — code + unit tests done,
not yet run against the real checkpoint.**
New module `hydroswarm.training.second_pass_control_labels`:

- `classify_evidence_sufficiency_second_pass()`: extends `control_labels.
  classify_evidence_sufficiency`'s sensor-health/entropy/OOD-validity rule
  with the two signals that need a trained checkpoint + calibration
  artifact -- a narrow, non-empty calibrated candidate set (bounded by
  `DEFAULT_MAXIMUM_CANDIDATE_SET_SIZE=3`, matching `inference.pipeline`'s
  own `maximum_planning_candidates` default) and low classical-neural
  disagreement (Jensen-Shannon divergence, reusing `inference.fusion.
  jensen_shannon_divergence` and `DEFAULT_DISAGREEMENT_THRESHOLD=0.5`,
  matching `uncertainty_control`'s own threshold). Pure function, 7 unit
  tests, no model required.
- `generate_second_pass_control_labels()`: batched, lazy (one batch
  materialized at a time, matching `_predict_rows`'s established
  discipline), runs a **frozen** model forward (raises `ValueError` if
  `model.training` is `True` -- the most detectable version of item 7's
  circular-self-label-leakage concern), fuses with the classical prior via
  the same `fixed_weight_fusion` weighting Stage 3's calibration fitting
  uses, queries the calibrator's real `candidate_set()`, and yields one
  `SecondPassControlLabel` per example carrying `teacher_checkpoint_hash`.
  4 tests against a real "small"-variant `HydroCore` and real
  `ScenarioExample`s (not synthetic-shape fixtures) -- including a test
  that an unvalidated topology forces `calibration_valid=False`,
  `calibrated_candidate_set_size=0`, and `next_step=ABSTAIN`.

**Deliberate design choice, documented in the module docstring**: unlike
`inference.pipeline`'s live `evidence_sufficient` decision (`calibrated and
0 < len(conformal_nodes) <= maximum_planning_candidates and
model_evidence`), the second-pass label does NOT fold the model's own
`evidence_sufficiency` head output into itself -- doing so would be
exactly the circularity item 7 warns against (a checkpoint's own current
prediction feeding its own next training label for the same target). The
live pipeline can safely use it because that is an operational decision,
not something fed back into further training.

**Item 8 (INSPECT_SENSOR) — resolved, with a real correction made mid-pass**:
initially concluded "no live INSPECT_SENSOR-equivalent exists anywhere,"
based only on `agents.controller`'s FSM (`FSMState` has no matching state
-- true). Before committing, re-checked more broadly and found this was
wrong: `hydroswarm.inference.fusion.uncertainty_control()` (called live
from `inference/pipeline.py`) already has an authoritative
`ControlAction.INSPECT_SENSORS`, triggered by `disagreement_js >= 0.5` --
a genuinely different signal than `targets_v2.NextStep.INSPECT_SENSOR`'s
`event_cause == SENSOR_FAULT` derivation. Corrected the docstring/tests
before this was committed anywhere, rather than shipping the overclaim:
`control_labels.NEXT_STEP_RUNTIME_ENABLED` excludes `INSPECT_SENSOR`
specifically because the *agent-FSM controller* has no matching state,
not because no inspect-sensor concept exists in the codebase at all. Two
independently-triggered "inspect the sensor" signals now exist,
agreeing only in name -- flagged as real design work for Phase 9's
architecture-v4 contract to reconcile (or deliberately keep separate),
not resolved unilaterally here. 3 tests, including one that exercises the
real `uncertainty_control()` call and asserts it returns
`ControlAction.INSPECT_SENSORS` for a high-disagreement input.

**Not done in this pass (steps 6, 9 — blocked on the Stage-A checkpoint,
not skipped)**: actually running `generate_second_pass_control_labels`
against the real trained checkpoint + fitted calibrator; training control
heads from the resulting labels (freeze-backbone-first, then optional
low-LR joint fine-tune); policy-agreement/unsafe-action tests against real
second-pass labels (as opposed to the synthetic-model unit tests already
written). Will run once the background Stage-A job completes.

## Restrictions honored

No work on `main`. `data/learning-v2/cycle-b2`'s existing contents, all
promoted checkpoints, and every existing v3 result artifact are untouched.
Locked test not opened; `final-selection.json` does not exist. No destructive
git/filesystem commands used. No sudo, no credential exposure. All commits
pushed to `origin/agent/gcp-multitopology-v3`.

## Next steps

1. Let `train`'s regeneration finish (~3875/9000 as of this update,
   ~0.78 scenarios/s, ~1.8h remaining), verify `errors_this_run: 0`, then
   commit the new corpus's JSONL/manifests/reports (JSONL only — no Git
   LFS needed for this artifact).
2. Phase 6 follow-up (items 6.2-6.6): supported-category metadata,
   balanced OOD training extension, `VALVE_PUMP_MISMATCH`/
   `HYDRAULIC_MISMATCH` real-perturbation implementation (or removal from
   supervision), `AMBIGUOUS` event-cause generation.
3. Phase 5 follow-up (optional, smaller): accessibility/hard-case
   generation now that incremental revelation works; consider a future
   regeneration pass with `reconstruction` wired into Scout for `train`.
4. Phase 7: masked-regression helper honoring target masks (currently the
   generic MSE path ignores mask companions for several targets); Scout
   EIG per-node alignment; denoising-only sensor reconstruction; future-
   concentration cutoff; travel-time transform.
5. Phase 9: architecture-v4 contract — `strategist_mode`, `ood_category_head`,
   and every other Phase-4.x/6.x flag this pass added are already
   individually recorded in `architecture_config()`, but Phase 9 wants a
   single strictly-validated contract (trained/validated/runtime-enabled
   output sets, not role-level gating) rather than the current per-flag
   compatibility checks.
6. A recurring pattern worth carrying forward: every phase this pass
   touched surfaced at least one real, previously-unobserved defect only
   visible once exercised at real scale or with a genuinely adversarial
   test case (not by code inspection alone) — Phase 1's three sub-bugs,
   Phase 3's checkpoint-compatibility regression, Phase 5's read-only-array
   bug, Phase 6's latent crash. Continue prioritizing real-scale/real-data
   testing over unit tests with synthetic arrays wherever practical.
