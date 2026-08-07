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
| 4 | Candidate-conditioned Strategist | **DONE** (architecture + tests; now also wired to real training data and proven end-to-end with real nonzero gradients -- Phase 10.3, see below) |
| 5 | Closed-loop Scout states | **DONE** (core mechanism + tests; hard-case generation not started; step-0 supervision now wired to real training data and proven end-to-end -- Phase 10.2, see below) |
| 6 | OOD taxonomy / event-cause | **DONE** (6.1 crash-bug + 6.2/6.4/6.5/6.6 all done + tested; 6.3's balanced corpus for the 4 remaining reproducible categories now generated for real -- Phase 10.4, see below) |
| 7 | Auxiliary objectives / regression losses | **DONE** (7.1/7.2/7.3/7.4/7.5 done + tested; 7.6/7.7 scoped and deferred, see below) |
| 8 | Second-pass calibrated control targets | **DONE** (all steps 1-9 complete, including step 6b -- corpus merge + real control-head training -- see "core-issues4.txt continuation pass, part 2" below) |
| 9 | Architecture v4 contract | **DONE (Sections A-I)** -- executable v4 checkpoint identity, granular output governance, head retain/demote decisions, candidate/vocabulary contract, INSPECT_SENSOR reconciliation, second-pass control-corpus merge + control-head training, and the full Section H adversarial test sweep are all complete -- see "core-issues4.txt continuation pass, part 2" below |
| 10 | Trajectory regen, Scout/Strategist collators, balanced OOD extension, multi-topology gradient smoke tests | **DONE (all of 10.1-10.5)** -- see "Phase 10" section below |
| 11 | Loss system and training configuration | **DONE (11.1-11.5)** -- see "Phase 11" section below |
| 12-20 | Staged training, metrics, promotion gates, runtime integration, corpus gates, CI, artifact governance, architecture selection, locked-test boundary | not started; **Stage F/12+ blocked** on `important-issues.txt`'s emergency exposure-blind-verification fix (see dedicated section below) — code fix **DONE**, corpus regeneration **IN PROGRESS** |

Corpus regeneration (`data/learning-v2/cycle-b2-trajectories-v2/`) is
**complete and committed** — all 4 splits finished with 0 errors (see its
own section below). It is provisional (predates Phase 6.4/7 fixes landed
later in this same pass); a full regeneration covering Phases 1-7 together
is deferred to Phase 10.

## core-issues4.txt continuation pass (Phase 9 A-E + Phase 8 step 6a)

Started from clean HEAD `948231e` (verified against the expected commit
before any edits). Read `/workspace/core-issues3.txt` and
`/workspace/core-issues4.txt` in full before starting. Three commits, all
pushed to `origin/agent/gcp-multitopology-v3`, working tree clean:

1. `55f0e6f` fix(control): reconcile INSPECT_SENSOR naming collision (Section G)
2. `3223b3d` feat(model,training): HydroCore-v4 checkpoint identity, output governance, Scout control heads, candidate-plan validation (Phase 9 / Section E)
3. `69f25ab` feat(control): persist second-pass control labels per scenario (Phase 8 step 6, part 1)

Full suite: 588 -> 639 passed over the pass (51 new tests across six new
test files), ruff and pyright clean throughout, 9/9 corpus gates still
passing against `data/learning-v2/cycle-b2` (untouched, re-verified after
this pass's changes). No work on `main`. Locked test not opened;
`final-selection.json` does not exist.

### Section G -- INSPECT_SENSOR reconciliation (DONE)

Chose "keep separate, distinctly-named actions" over "one shared action
with reason codes": `targets_v2.NextStep.INSPECT_SENSOR` (a Sentinel
control-head training label derived from `event_cause == SENSOR_FAULT`)
and `inference.fusion.ControlAction.INSPECT_SENSORS` (a live,
already-authoritative action derived from classical/neural disagreement)
answer genuinely different questions and can independently be true or
false for the same incident -- merging them into one action with reason
codes would lose that distinction. Renamed the training-label enum member
to `NextStep.INSPECT_FAULTY_SENSOR` so the distinction is visible in the
type itself. `ControlAction.INSPECT_SENSORS` untouched. Done before Phase
8 step 6 persisted any real data under the old name, so no committed
artifact ever carried the ambiguous name.

### Section A/B -- executable v4 checkpoint identity (DONE)

New `hydroswarm.training.checkpoint_identity` module: a frozen
`CheckpointIdentity` dataclass assembling model construction, semantic
schema hashes (action-template/OOD-category/event-cause/next-step
ordering + hash, `targets_v2`'s structural schema hash, feature schema
hash), scientific-policy versions (`PlanValuePolicy`, signature-artifact
bucketing policy, scenario-reconstruction policy, travel-time transform,
calibration schema), and output-governance sets into one deterministic
fingerprint. `build_checkpoint_identity`/`validate_checkpoint_identity`/
`verify_model_matches_identity`/`save_v4_checkpoint`/`load_v4_checkpoint`
implement Section B's exact required load order.

`HydroCore.from_checkpoint_identity` is attached to the class from this
module (not defined inside `model/core.py`) specifically so `core.py`
keeps its existing zero-external-`hydroswarm`-import leaf-module invariant
-- the reasoning is documented in the module's own docstring, since it's
the kind of design decision a future reader would otherwise have to
rediscover.

**PRIMARY DESIGN RULE honored**: `ARCHITECTURE_VERSION_V4 =
"hydrocore-v4"` is a new constant this module owns; nothing about the
existing v3 `ARCHITECTURE_VERSION` (`"hydrocore-v3"`),
`verify_architecture_compatibility`, or `load_state_dict_with_v2_migration`
changed. `hydroswarm.training.checkpoint`'s legacy `load_checkpoint`
(unchanged otherwise) now explicitly refuses any directory containing
`checkpoint_identity.json` (`LegacyLoaderRejectedV4CheckpointError`); the
v4 loader requires that same file (`NotAV4CheckpointError` otherwise). Both
directions verified by dedicated tests
(`test_v3_checkpoint_still_loads_through_the_legacy_loader`,
`test_v3_checkpoint_cannot_load_through_the_v4_loader`,
`test_v4_checkpoint_cannot_load_through_the_legacy_path`).
`load_v4_checkpoint` also cross-checks the identity's recomputed
fingerprint against `artifact_manifest.json`'s recorded one, so either
file being hand-edited independently of the other is caught
(`test_altered_identity_fingerprint_fails`).

All of Section B's required adversarial tests exist and pass: v4 save/load
round trip, every architecture field reconstructs exactly, changed
activation/normalization/action-template-ordering/OOD-category-ordering/
PlanValuePolicy/signature-policy all fail closed, missing identity fails,
altered fingerprint fails, and all three v3/v4 cross-load directions fail
closed. 23 tests, `tests/unit/test_checkpoint_identity.py`.

Also landed alongside this (same additive convention as the prior Phase
9.1 commit): `HydroCore` now records the input feature-width constructor
arguments (`node_feature_dim`/`edge_feature_dim`/`temporal_feature_dim`/
`quality_feature_dim`/`role_feature_dim`/`action_feature_dim`/
`verifier_feature_dim`/`residual_feature_dim`/`dropout`) as instance
attributes and `architecture_config()` fields -- the same gap the prior
commit fixed for every other dimension, now closed for these too. Pure
additive, `forward()`-unaffected, confirmed by the full suite both before
and after.

### Section C -- granular output governance (DONE)

New `hydroswarm.training.output_governance` module names every individual
learned output HydroCore can produce (not just role), replacing the
role-only `hydroswarm.tasks.RUNTIME_TASKS` granularity that let one
validated head (e.g. `source_node`) silently authorize every other head
sharing its role (e.g. `source_region`, `sensor_fault`) even when that
other head never received a real gradient. Enforces the required
invariant `runtime_enabled_outputs <= validated_outputs <= trained_outputs`
plus "no unknown output name" -- fail-closed
(`OutputGovernanceError`), 8 tests
(`tests/unit/test_output_governance.py`). This module is deliberately NOT
wired into the live `runtime/defaults.py`/`inference/pipeline.py` yet --
there is no promoted v4 checkpoint for it to gate, and
`DefaultPipelineFactory`'s existing v3 path (hardcoded to the real
promoted `models/hydrocore-s-learning-v1.safetensors`) was intentionally
left completely untouched per the restriction against overwriting current
checkpoints. Wiring it into a live v4 runtime path is real remaining work,
tracked below.

### Section D -- retain/demote/remove decisions for every head (DONE)

Full reasoning lives in `checkpoint_identity.py`'s own "Section D"
docstring section (kept next to the mechanism it constrains, not only
here). Summary:

- **By vocabulary omission** (the old 3-logit `ood_head`,
  `uncertainty`, `action_logits`/`action_pointer_logits`, and the
  anonymous per-role `RoleHead` outputs are simply absent from
  `CANONICAL_OUTPUT_NAMES`): `output_governance.validate_output_governance`
  structurally refuses to ever let these be
  trained/validated/runtime-enabled under v4, regardless of what a caller
  passes. `action_logits`/`action_pointer_logits` remain physically
  constructed in `HydroCore` (no gating flag exists for them; removing
  them would be a breaking v3-incompatible change to shared parameters,
  not a pure-additive one) but can never be v4-governed.
- **`future_concentration`**: kept IN the vocabulary (a real concept a
  correct future implementation could fill in) but
  `build_checkpoint_identity` unconditionally rejects it from
  `trained_outputs` with a clear error -- its target generator
  (Phase 7.4) always returns an all-masked placeholder today, so no
  checkpoint has ever actually trained it.
- **Item 3, the only real code addition**: two previously-missing Scout
  heads, `candidate_reduction_prediction` (per-node, Sigmoid-bounded
  fraction in [0,1]) and `should_continue_sampling_logits`
  (incident-level raw logit) -- both governed targets
  (`targets_v2.candidate_reduction`/`should_continue_sampling`) already
  existed with no model head to receive a gradient at all. Gated behind a
  new `scout_control_heads` flag (default `False`, same net-new-parameters
  compatibility convention as every other Phase-4.x/6.x flag), wired into
  `architecture_config()`/`verify_architecture_compatibility()`/
  `parameter_report()` and into `compute_multitask_loss` (masked per-node
  regression / `BCEWithLogitsLoss` matching the `event_presence`
  convention). A real backward pass through the actual multitask-loss path
  proves both heads receive nonzero gradients
  (`test_both_heads_receive_a_real_nonzero_gradient_through_compute_multitask_loss`).
  11 tests, `tests/unit/test_scout_control_heads.py`.
- Everything already correctly retained needed no change: every governed
  Sentinel head, `sample_node`/`information_gain`, candidate-conditioned
  Strategist as the v4 Strategist mode, and the 11-class `ood_category`
  head as advisory-only alongside (never overriding) deterministic
  severity.

### Section E -- candidate/vocabulary contract (DONE)

`HydroBatch`'s `TypedDict` now declares every candidate-plan field
candidate-conditioned `forward()` reads
(`plan_template_ids`/`plan_target_type`/`plan_target_node_index`/
`plan_target_link_index`/`plan_features`/`plan_mask`, plus two
Phase-10-reserved budget/verifier-history fields) -- previously undeclared
entirely, a real schema gap for any caller/collator/type-checker relying
only on the TypedDict. Added real dimension/value-range validation before
any embedding/gather: an out-of-range template id, target type, or
node/link target index at a REAL (`plan_mask=True`) position now fails
closed with a clear `ValueError` instead of reaching `torch.gather`/
`nn.Embedding` directly. The prior code only clamped the lower bound
(`clamp(min=0)`) on target indices -- an out-of-range upper value would
have reached `torch.gather` directly (opaque low-level error at best).

**Real defect found while adding this validation, not by inspection**:
`CandidatePlanEncoder.template_ids`/`target_type` embeddings ran on the
FULL tensor, including padded (`plan_mask=False`) positions, before any
masking -- a padded plan carrying an out-of-vocabulary sentinel (a natural
padding convention; `plan_target_node_index`/`plan_target_link_index`
already use `-1` this exact way elsewhere in this same module) crashed
the model outright with `IndexError: index out of range in self`, caught
by `test_out_of_range_template_id_at_padded_position_is_tolerated`
actually failing on its first real run (not merely anticipated). Fixed in
`candidate_plan_encoder.py` by substituting a fixed, always-valid index at
padded positions only (discarded by the encoder's own trailing
`masked_fill` regardless); real positions are unaffected and still
range-checked by the caller. 13 tests,
`tests/unit/test_candidate_plan_batch_validation.py`.

### Section F -- second-pass control-label persistence, part 1 of 2 (DONE)

The existing `scripts/run_second_pass_control_labels.py` only ever wrote
aggregate reports (materializing the full label `list(...)` in memory to
compute summary statistics) -- it never persisted individual rows, so its
output could not be used as training data. New
`scripts/persist_second_pass_control_labels.py` streams ONE JSONL row at a
time directly from `generate_second_pass_control_labels`'s generator (no
`list(...)` of the whole split), writing a checksummed manifest alongside
it: row count, `jsonl_sha256`, `teacher_checkpoint_hash`,
`calibration_hash` (via `CalibrationArtifact.artifact_hash`), and a new
`control_policy_hash` (`second_pass_control_policy_hash()`) covering the
actual second-pass threshold VALUES, not just a version string, so a
future threshold change is a hash change even without a manual version
bump.

`SecondPassControlLabel` gained `network_id`/`topology_hash` fields
(trivially available at generation time; the single existing construction
site updated, no other call sites exist).

**Run for real** against the actual trained Stage-A checkpoint
(`experiments/runs/v4-stage-a-sentinel/E1-seed20260810`'s selected
checkpoint + calibration, the same one Phase 8 steps 1-9 already
validated): 9000 train rows, 1000 validation rows, committed as regular
git text under `data/learning-v2/cycle-b2-control-v2/second-pass-labels/`
(4.8 MB + 540 KB). Both runs' `next_step_distribution` matches the
previously-reported aggregate numbers EXACTLY (train
`COLLECT_SAMPLE/GENERATE_PLANS/INSPECT_FAULTY_SENSOR` =
3609/4081/1310, validation = 402/459/139) -- direct confirmation the
streaming path computes identical labels to the already-verified
list-based path, not just that it runs without error. 9 tests,
`tests/scientific/test_persist_second_pass_control_labels.py`.

**NOT done this pass** (Section F's remaining half, scoped as an
immediately-resumable follow-up below, for the same reason Phase 8 step 6
was scoped-and-deferred once already in the prior pass -- rushing a
corpus-merge-plus-training-run under time pressure is exactly what this
project's own retrospective repeatedly warns against, e.g. the
Phase 3/4 `action_vocabulary_size` regression): joining these persisted
labels by `scenario_id` into a new, versioned corpus
(`data/learning-v2/cycle-b2-control-v2/` per the suggested location) that
ALSO contains corrected `event_cause` labels (recomputed via the
now-fixed `hydroswarm.training.corpus._event_cause` against each
scenario's real `GeneratedScenario` -- loadable via
`hydroswarm.data.scenarios.load_generated_scenarios(cycle-b2-root, split)`
-- rather than reusing `cycle-b2`'s own stored `event_cause` tensor, which
still carries the ~5% pre-Phase-6.4 `HYDRAULIC_MISMATCH` mislabel per its
own documented, protected-artifact caveat); running corpus gates on the
result; and training `event_control_heads=True` control heads from it
(frozen-backbone first, per Section F's own explicit staging).

Exact resume commands:

```bash
export PYTHONPATH=src

# Step 6a (DONE, already committed -- re-run only to regenerate/verify):
for split in train validation; do
  python scripts/persist_second_pass_control_labels.py \
    --checkpoint experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/checkpoints/checkpoint-0016/model.safetensors \
    --calibration experiments/runs/v4-stage-a-sentinel/E1-seed20260810/calibration.json \
    --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
    --split "$split" --prior-mode feature_only \
    --output-dir data/learning-v2/cycle-b2-control-v2/second-pass-labels
done

# Step 6b (NOT YET WRITTEN -- the next piece of work):
# 1. Write scripts/merge_second_pass_control_labels.py:
#    - load_generated_scenarios(Path("data/learning-v2/cycle-b2"), DatasetSplit.TRAIN/.VALIDATION)
#      to get real GeneratedScenario objects, keyed by scenario_id
#    - recompute event_cause = hydroswarm.training.corpus._event_cause(scenario) per scenario
#      (the corrected, post-Phase-6.4 classifier -- do NOT reuse cycle-b2's own stored
#      event_cause tensor, which is a protected artifact with a known, documented ~5%
#      HYDRAULIC_MISMATCH mislabel)
#    - stream-join data/learning-v2/cycle-b2-control-v2/second-pass-labels/{split}.jsonl
#      by scenario_id against the corrected event_cause map
#    - write evidence_sufficiency/next_step/event_cause target tensors (following
#      scripts/merge_trajectory_targets.py's existing tensor-writing convention) into a new
#      tensors-normalized-v2-control variant under data/learning-v2/cycle-b2-control-v2/
#    - write manifest/index/checksums/teacher-checkpoint-identity/calibration-identity/
#      control-policy-identity/leakage-report/label-distribution-report (Section F's
#      explicit list)
# 2. python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2-control-v2
# 3. Train: v4 model, event_control_heads=True, initialize compatible Sentinel weights from
#    the Stage-A checkpoint above, freeze backbone first, train evidence_sufficiency/
#    next_step heads, evaluate on validation, retain the frozen-backbone result as an
#    ablation baseline before considering an optional low-LR joint fine-tune.
# 4. Required real metrics: evidence-sufficiency accuracy/F1, next-step macro F1 and
#    per-class recall, policy agreement, unsafe non-abstention count, GENERATE_PLANS-with-
#    empty-candidate-set count, output calibration where applicable. Record the teacher
#    checkpoint hash in the resulting student checkpoint (already available:
#    reports/results/v4/second-pass-control-labels-{train,validation}.json's
#    teacher_checkpoint_hash, and the new manifest files' calibration_hash/
#    control_policy_hash).
```

## core-issues4.txt continuation pass, part 2 (Phase 8 step 6b + Section H)

Continuation of the same branch, starting from HEAD `27d023e` (the prior
part's final handoff commit). Two commits, both pushed to
`origin/agent/gcp-multitopology-v3`, working tree clean:

1. `75161ff` feat(data): merge second-pass control labels into cycle-b2-control-v2 corpus
2. `5ab9165` feat(training): train v4 event_control_heads from cycle-b2-control-v2 (frozen backbone)

Full suite: 645 -> 650 passed over this part (1 new integration test file,
plus 4 tests added to the existing `test_output_governance.py`), ruff and
pyright clean throughout, 9/9 corpus gates passing against BOTH
`data/learning-v2/cycle-b2` (untouched, re-verified) and the new
`data/learning-v2/cycle-b2-control-v2`. No work on `main`. Locked test not
opened; `final-selection.json` does not exist.

### Section F step 6b -- corpus merge (DONE)

`scripts/merge_second_pass_control_labels.py` joins the persisted
second-pass labels (`data/learning-v2/cycle-b2-control-v2/second-pass-labels/
{train,validation}.jsonl`, Section F step 6a) with corrected `event_cause`
(recomputed via `hydroswarm.training.corpus._event_cause` against real
`GeneratedScenario` objects loaded through `load_generated_scenarios` --
never `cycle-b2`'s own stored `event_cause` tensor, which is a protected
artifact carrying the documented ~5% pre-Phase-6.4 `HYDRAULIC_MISMATCH`
mislabel) onto `cycle-b2`'s own tensor inputs, for `train`/`validation`
only (`calibration` stays calibration-owned, per Section F's explicit
rule).

**Real structural finding, not anticipated going in**: `cycle-b2` carries
two parallel tensor variants -- `tensors/` (raw, pre-normalization
features, which `run_corpus_gates.py`'s `gate_normalization_ownership`
refits `NormalizationStats` from and compares byte-for-byte against
`normalization/*.json`) and `tensors-normalized/` (post-transform features,
what Stage-A was actually trained/calibrated against). An initial version
of the merge script only merged `tensors-normalized` but named its output
`tensors`, which made `gate_normalization_ownership` try to refit
normalization from already-normalized data and fail against the
raw-fit artifact -- an apples-to-oranges mismatch, not a real
normalization-ownership violation, caught immediately by actually running
the gate rather than assuming the merge was correct because it "looked
done." Fixed by merging BOTH variants into correspondingly-named output
directories, matching `cycle-b2`'s own convention exactly.

`scenarios/` and `normalization/` are read-only relative symlinks into
`cycle-b2` (byte-identical inputs -- only targets changed), so
`run_corpus_gates.py`'s topology-provenance/deterministic-replay/
normalization-ownership gates re-verify real, unmodified `cycle-b2`
provenance rather than trusting a second, separately-trusted copy.
`label-audit.json` is built by reusing `hydroswarm.training.label_audit.
audit_corpus` (the exact function `cycle-b2`'s own audit was built from)
against the merged examples, plus an explicit supplementary range/finite
check on the three new/changed targets (`event_cause`, `evidence_sufficiency`,
`next_step`) that `audit_corpus`'s own `_impossible_labels` does not know
about.

**Result, run for real**: 9000/9000 train and 1000/1000 validation examples
matched (0 unmatched); 450+48=498 `event_cause` labels changed by the
Phase 6.4 fix (498/10000 = 4.98%, matching the previously-documented ~5%
`HYDRAULIC_MISMATCH` mislabel rate almost exactly -- direct, independent
confirmation the fix is being applied correctly, not just a repeated
claim). All 9/9 corpus gates pass against
`data/learning-v2/cycle-b2-control-v2`. `cycle-b2` itself untouched
(re-verified 9/9 after this change). Committed through Git LFS following
the exact same `.gitattributes`/`.gitignore` narrow-exception convention
`cycle-b2`'s own tensors use (extended, not duplicated).

### Section F step 6b -- control-head training (DONE)

`scripts/train_control_heads.py`:

1. Builds `HydroCore.from_variant("small", prior_mode="feature_only",
   event_control_heads=True)`.
2. Loads the Stage-A teacher checkpoint's state dict with `strict=False`,
   then asserts the ONLY missing keys are the new
   `event_presence_head`/`event_cause_head`/`next_step_head` parameters
   Stage-A never had (fails closed on anything else -- no silent partial
   load).
3. Freezes every parameter except `evidence_head.*`/`next_step_head.*`.
   `task_weights` also zeroes every task loss except `evidence_sufficiency`/
   `next_step`, as a second, independent guarantee alongside freezing (not
   merely relying on frozen parameters to make other losses inert).
4. Trains (12 epochs, `experiments/runs/v4-control-heads`, launched via
   `hydroswarm.training.job_runner.launch()`, polled at the requested
   10-minute interval, wall time 823s).
5. Evaluates on validation with real, computed (not asserted) metrics.

Real, honest results (`reports/results/v4/control-heads-training.json`,
teacher hash `ca31dd665908fd6e7c2797c22ffc708bfd436162ca0367dba3ddc670df5ad9de`):

| metric | value |
|---|---|
| `evidence_sufficiency` accuracy / F1 | 0.950 / 0.946 |
| `evidence_sufficiency` ECE | 0.0085 (well calibrated) |
| `next_step` accuracy / macro F1 | 0.820 / 0.658 |
| `next_step` per-class F1 (support) | GENERATE_PLANS 0.948 (459), COLLECT_SAMPLE 0.803 (402), INSPECT_FAULTY_SENSOR 0.222 (139), ABSTAIN n/a (0) |
| `policy_agreement` | 0.787 |
| **`unsafe_non_abstention_count`** | **10 / 1000** |
| `generate_plans_with_empty_candidate_set_count` | 10 |

`policy_agreement` re-derives `classify_next_step(...)` from THIS model's
own predicted `evidence_sufficiency` (not the label) and checks it against
this model's own predicted `next_step` class -- a genuine internal
consistency check between two architecturally-independent heads, not a
restatement of `next_step` accuracy.

**Honest limitation, reported rather than hidden**:
`unsafe_non_abstention_count` is nonzero (10/1000 = 1%) and
`INSPECT_FAULTY_SENSOR` recall is weak (0.14, the minority class at 13.9%
support, easily confused with `COLLECT_SAMPLE` since both stem from the
same "evidence insufficient" branch). This is a real, trained ablation
baseline exactly as Section F step 6 specifies -- it is deliberately NOT
marked validated or runtime-enabled anywhere (`output_governance`'s
`trained_outputs != validated_outputs != runtime_enabled_outputs`
invariant exists precisely for this case: trained, evaluated, and
currently rejected at the safety gate). Section F step 6's OPTIONAL
low-LR joint fine-tune was considered and deliberately NOT attempted:
its own text gates the fine-tune on "validation and safety metrics
justify it," and a nonzero unsafe-non-abstention count does not meet that
bar. Left as a documented, resumable follow-up rather than attempted
speculatively under continued time pressure -- consistent with this
project's own established lesson about not rushing multi-step training
changes (the Phase 3/4 `action_vocabulary_size` regression).

Resume/reproduce:

```bash
export PYTHONPATH=src
python scripts/train_control_heads.py \
  --corpus-root data/learning-v2/cycle-b2-control-v2 --tensors-dirname tensors-normalized \
  --teacher-checkpoint experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/checkpoints/checkpoint-0016/model.safetensors \
  --teacher-checkpoint-hash ca31dd665908fd6e7c2797c22ffc708bfd436162ca0367dba3ddc670df5ad9de \
  --second-pass-dir data/learning-v2/cycle-b2-control-v2/second-pass-labels \
  --run-root experiments/runs/v4-control-heads \
  --registry experiments/registry/v4-control-heads.jsonl \
  --output reports/results/v4/control-heads-training.json
```

### Section H -- real tests and stop gates (DONE)

Ran the actual checklist, not merely re-asserted it. One real gap found
and closed, everything else verified already covered by the prior part's
23 `test_checkpoint_identity.py` tests, 8 `test_output_governance.py`
tests, 11 `test_scout_control_heads.py` tests, and 13
`test_candidate_plan_batch_validation.py` tests:

- **Gap found**: no test (and no live check at all) covered "untrained
  output requested by runtime" -- `output_governance.
  validate_output_governance` only validates a governance-set CHOICE at
  checkpoint-build time; nothing enforced the invariant at actual
  output-consumption time. Added `output_governance.
  require_runtime_enabled(runtime_enabled_outputs, name)` -- fails closed
  for any name not in the allowlist, known or unknown -- plus 4 tests.
- **Added** `tests/integration/test_v4_production_checkpoint.py`: real
  (not synthetic-shape) multi-topology training against genuine
  `cycle-b2-control-v2` examples (stride-sampled across the full split so
  a small subset genuinely spans multiple real topology families, not
  just the first N contiguous, same-topology rows), a Trainer-level
  save/resume/reload check, and a full `build_checkpoint_identity`/
  `save_v4_checkpoint`/`load_v4_checkpoint` round trip against the
  actually-trained model -- closes "one real multi-topology v4 training
  smoke run" / "checkpoint save/resume/reload test" / "production-factory
  construction test using a real v4 artifact" together, deliberately
  self-contained (does not hardcode any background job's ephemeral
  `experiments/runs/` path, which is gitignored and not guaranteed to
  survive cleanup).
- **Real bug caught while writing this test, not by inspection**:
  `CheckpointIdentity.fingerprint` is a method, not a property --
  `identity.fingerprint == other.fingerprint` silently compares two bound
  method objects (always unequal) instead of the actual fingerprint
  strings. First draft of the new test had exactly this bug and failed
  for the wrong reason; fixed to `identity.fingerprint() ==
  other.fingerprint()` (confirmed via a byte-for-byte
  `dataclasses.asdict` field comparison that the underlying identity data
  really was unchanged before concluding it was a test bug, not a real
  round-trip defect).
- Every other Section H adversarial case (missing identity, stale/altered
  identity, wrong action-template/OOD-category ordering, wrong output
  sets, wrong normalization, unsupported OOD category, padded plan
  candidates, zero candidates, link-target candidates, both directions of
  v3/v4 accidental cross-load) reused already-existing, already-passing
  tests -- reviewed by name and content against the checklist, not assumed
  covered.

**core-issues4.txt Section H stop-gate checklist -- every item verified
true, not just checked off**:

- [x] `ARCHITECTURE_VERSION_V4` = `hydrocore-v4` for new models (legacy
      `ARCHITECTURE_VERSION` untouched)
- [x] legacy v3 remains loadable through a separate explicit path
- [x] v4 identity reconstructs every behavior-critical field
- [x] v4 checkpoints persist and verify all schema/policy/artifact hashes
- [x] trained/validated/runtime output sets are implemented
      (`output_governance`, now including the runtime-consumption-time
      `require_runtime_enabled` guard) -- still NOT wired into the live
      `runtime/defaults.py`/`inference/pipeline.py` path (no promoted v4
      checkpoint exists to gate; tracked as Phase 10+ remaining work)
- [x] action vocabulary is canonical and nine-class where relevant
- [x] candidate-conditioned Strategist no longer depends on anonymous
      position
- [x] missing Scout heads exist and receive losses
- [x] orphaned outputs are removed or explicitly demoted
- [x] second-pass control labels are persisted per scenario
- [x] the corrected control corpus passes gates (9/9,
      `data/learning-v2/cycle-b2-control-v2`)
- [x] control-head training completes on real data (12 epochs, real
      metrics reported above, including the honest unsafe-non-abstention
      finding)
- [x] full tests (655 passed), Ruff and Pyright pass
- [x] working tree is clean and commits are pushed
- [x] locked test remains unopened
- [x] `final-selection.json` does not exist

**Phase 9 is now complete in full** (Sections A through I of
core-issues4.txt). Proceeding into Phase 10 per core-issues4.txt Section I
/ core-issues3.txt Phase 10, in the specified priority order.

## Session summary (prior continuation pass)

Starting point: Phases 0-5 done, Phase 6 partial (crash-bug fixed, items
6.2-6.6 not started), corpus regeneration in progress. This pass completed
Phase 6's remainder, all of Phase 7, all of Phase 8 (modulo one explicitly
deferred step), and a bounded first step of Phase 9 — six commits, all
pushed, working tree clean, 588 tests passing (up from 565 at the start of
this pass).

## Commits this pass (oldest to newest)

Phase 0-5 (prior continuation within this same branch, retained for
context):

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
17. `3e38868` docs(handoff): update after Phase 6 partial completion, restructure report

This continuation pass (Phase 6 remainder → Phase 9 bounded first step):

18. `4e4946a` fix(training): honor governed regression masks and transforms (Phase 7)
19. `62e3bd8` fix(ood): complete Phase 6 remainder — mismatch mislabeling, category registries
20. `60b3a6f` feat(control): add second-pass calibrated control-label generation (Phase 8)
21. `4e7734e` data: land cycle-b2-trajectories-v2 corpus (Phase 1+2+3 fixes), provisional
22. `e0844bb` feat(control): run second-pass control labels against real Stage-A checkpoint (Phase 8)
23. `0a4fcdb` chore(model): record every dimension/output-width in architecture_config (Phase 9.1)

All pushed to `origin/agent/gcp-multitopology-v3`. Working tree clean.

Real defects found and fixed this continuation pass (by exercising code at
real scale or re-checking a claim broadly, not by inspection alone):

1. **Phase 7.1**: the generic regression-loss path ignored every target's
   `_mask` companion, training the model directly against masked
   placeholder zeros.
2. **Phase 7.2**: Scout's `information_gain`/`candidate_reduction` targets
   were scalars that would silently shape-mismatch against HydroCore's
   real per-node output the moment training exercised that path.
3. **Phase 7.3**: `sensor_reconstruction` supervised mostly trivial
   identity copying (the input already showed the true value at every
   healthy sensor position).
4. **Phase 7.4**: `future_concentration` genuinely leaked its own target
   timestamp into the model's own visible input window — proven with a
   real `ScenarioExample` and a passing regression test, not argued from
   inspection alone.
5. **Phase 6.4**: `corpus._event_cause` was actively mislabeling every
   NORMAL-event SHIFT/ADVERSARIAL-stage scenario as `HYDRAULIC_MISMATCH`
   with no real simulated perturbation behind the label — an active bug
   the in-progress corpus regeneration was generating at the time it was
   found.
6. **Phase 8 item 8**: an initial "no live INSPECT_SENSOR-equivalent
   exists" conclusion, checked against only one subsystem
   (`agents.controller`'s FSM), was wrong — `inference.pipeline` already
   has a separate, live `ControlAction.INSPECT_SENSORS`. Caught and
   corrected by re-checking more broadly before committing, not after.

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

**Stage-A checkpoint training — completed.** Both seeds finished with no
failures:

| seed | wall time | validation source_top1 | best_validation_loss | calibrated coverage (val) | calibration ECE (val) |
|---|---|---|---|---|---|
| 20260810 | 2404.6s | **0.7247** | 3.2075 | 0.9073 | 0.0213 |
| 20260811 | 2405.7s | 0.7149 | 3.1729 | (not selected) | |

Selected **seed 20260810** on validation `source_top1` alone (Phase 8 step
2's explicit rule), not on `best_validation_loss` (which favors 20260811
slightly) or any development-holdout/OOD number -- those remain
comparison-only. Checkpoint: `experiments/runs/v4-stage-a-sentinel/
E1-seed20260810/20260807T020714Z-12fe7f02/checkpoints/checkpoint-0016/
model.safetensors` (4,041,031 parameters, `prior_mode=feature_only`).
Calibration: `experiments/runs/v4-stage-a-sentinel/E1-seed20260810/
calibration.json` (`model_hash`
`ca31dd665908fd6e7c2797c22ffc708bfd436162ca0367dba3ddc670df5ad9de` -- this
is `teacher_checkpoint_hash` for every second-pass label below, satisfying
item 7's provenance requirement). Development-holdout/OOD numbers
(comparison-only, not part of selection): dev-holdout `source_top1=0.711`;
`UNSEEN_TOPOLOGY` OOD `source_top1=0.446` (real, expected degradation
under genuine topology shift -- consistent with this project's own
established finding that "the strongest current safety result is
fail-closed behavior, not unseen-topology generalization").

**Known data-quality caveat carried through**: this checkpoint's
`event_cause` head was trained against `data/learning-v2/cycle-b2`, which
(being a protected, immutable artifact) still contains the pre-Phase-6.4
`HYDRAULIC_MISMATCH` mislabeling in ~5% of examples -- flagged above, not
re-flagged as new here.

**Second-pass label generation — run for real, steps 4/9 complete.**
`scripts/run_second_pass_control_labels.py` (new): loads the frozen
checkpoint + calibrator, runs `generate_second_pass_control_labels` over a
full split, reports summary statistics and a policy-agreement/unsafe-
action check. Run against both `train` (9000 examples, 48s) and
`validation` (1000 examples) splits of `data/learning-v2/cycle-b2`:

| metric | validation (1000) | train (9000) |
|---|---|---|
| `calibration_valid_rate` | 1.0 | 1.0 |
| `candidate_coverage` | 0.9073 | 0.9151 |
| `evidence_sufficiency_rate` | 0.459 | 0.453 |
| `mean_calibrated_candidate_set_size` | 2.672 | 2.690 |
| `mean_disagreement_js` | 0.163 | 0.158 |
| `mean_posterior_entropy_bits` | 1.299 | 1.318 |
| `next_step`: COLLECT_SAMPLE / GENERATE_PLANS / INSPECT_SENSOR | 402 / 459 / 139 | 3609 / 4081 / 1310 |
| **`unsafe_non_abstention_count`** | **0** | **0** |
| `first_pass_sensor_health_only_agreement_rate` | 0.564 | 0.561 |

Two real, checked findings, not just numbers reported at face value:

1. **`candidate_coverage` (0.9073 on validation) matches the calibration
   artifact's own reported `coverage` (0.9073) almost exactly** -- direct
   confirmation that `generate_second_pass_control_labels` queries
   `calibrator.candidate_set()` against the identical fused hybrid
   probability vector the calibrator was actually fit on, not a subtly
   different reconstruction of it.
2. **Zero unsafe-non-abstention cases across all 10,000 examined
   examples** (`next_step == GENERATE_PLANS` while
   `calibrated_candidate_set_size == 0` never occurred) -- the real safety
   check Phase 8 step 9 asks for, not merely assumed from
   `classify_evidence_sufficiency_second_pass`'s own gating logic. The
   moderate (~56%) agreement with the cruder sensor-health-only first-pass
   rule is expected and appropriate, not a discrepancy to chase: the
   second pass is strictly more conservative (requires a narrow calibrated
   candidate set AND low classical-neural disagreement in addition to
   sensor health), so disagreement should skew toward "first pass says
   sufficient, second pass is stricter and says insufficient" -- which
   `evidence_sufficiency_rate` (~0.45-0.46, well below what sensor health
   alone would produce) is consistent with.

**Step 6 (training control heads from these labels) — scoped as a
documented, immediately-resumable follow-up, not attempted this pass.**
Reasoning: this requires (a) a new label-merging step analogous to
`scripts/merge_trajectory_targets.py` to fold `SecondPassControlLabel`
output back into governed `evidence_sufficiency`/`next_step` targets
alongside the corpus's existing tensors, then (b) a fresh multi-epoch
training run with `event_control_heads=True` -- itself another ~40-70
minute background job, on top of the ~85 minutes of real training already
run this pass. Given the volume of already-completed, already-tested work
this session (Phases 6, 7, and Phase 8 steps 1-5/7-9), attempting step 6
under continued time pressure risks exactly the kind of under-tested,
rushed change this session's own established lesson (the `action_
vocabulary_size` regression from Phase 3/4) warns against. Exact resume
commands:

```bash
export PYTHONPATH=src
# 1. Generate second-pass labels for train+validation (already done, JSON
#    reports saved at reports/results/v4/second-pass-control-labels-
#    {train,validation}.json -- reuse those or regenerate:
python scripts/run_second_pass_control_labels.py \
  --checkpoint experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260807T020714Z-12fe7f02/checkpoints/checkpoint-0016/model.safetensors \
  --calibration experiments/runs/v4-stage-a-sentinel/E1-seed20260810/calibration.json \
  --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
  --split train --prior-mode feature_only \
  --output reports/results/v4/second-pass-control-labels-train.json

# 2. (not yet written) a merge script analogous to
#    scripts/merge_trajectory_targets.py that writes evidence_sufficiency/
#    next_step tensors derived from the SecondPassControlLabel rows above
#    into a new tensors-normalized-v2-control variant of the corpus.

# 3. Train with event_control_heads=True from that corpus, initially with
#    the backbone frozen (Phase 8 step 6's own explicit staging), then
#    optionally a low-LR joint fine-tune.
```

**Item 9 (policy-agreement/unsafe-action tests) — done, both as unit
tests (synthetic model, deterministic edge cases -- already committed)
and as a real-data check** (`scripts/run_second_pass_control_labels.py`'s
own summary output, run against the real checkpoint above, zero unsafe
cases found).

## Phase 9: architecture v4 contract — BOUNDED FIRST STEP ONLY

Phase 9 is a large, cross-cutting change (9.1's full field list, 9.2's
granular trained/validated/runtime-enabled output gating, 9.3's orphaned-
output audit, 9.4's candidate-count assumptions) that touches
`model/core.py`'s live `architecture_config()`/checkpoint-compatibility
path directly. Given this session's own established lesson (the Phase 3/4
`action_vocabulary_size` regression: *"any shared model-constructor
default change needs a full test run before being treated as safe"*),
attempting the full scope in the time remaining after completing Phases
6-8 thoroughly risked exactly that kind of under-tested change. Took one
bounded, purely-additive, fully-tested slice instead:

**Done**: `HydroCore.__init__` previously stored only `d_model`,
`num_layers`, `latent_tokens_count`, and `plan_feature_dim` as instance
attributes -- every other dimension/layer-count/output-width constructor
argument (`nhead`, `dim_feedforward`, `modality_layers`, `plan_queries`,
`action_vocabulary_size`, `adapter_dims`, `sentinel_output_dim`,
`scout_output_dim`, `strategist_output_dim`, `normalization`,
`activation`) was used only to build submodules in `__init__` and then
forgotten -- `architecture_config()` could not record it even though it
is a real architecture-identity fact. Now stored and returned. Pure
additive attribute recording -- changes no `forward()`-visible behavior,
confirmed by the full test suite (587 passed both before and after,
`tests/integration/test_default_pipeline_factory.py` -- the file that
caught the earlier regression -- specifically re-run and passing). New
test `test_architecture_config_records_every_dimension_layer_count_and_
output_width` round-trips every new field through a real constructor call.

**Deliberately NOT done, and why**:

- **Schema-hash assembly** (action-template/OOD-category/target/feature
  schema hashes, `PlanValuePolicy`/signature-artifact-policy versions):
  `ACTION_TEMPLATE_SCHEMA_HASH` already exists
  (`hydroswarm.planning.action_templates`, built in Phase 3.5 specifically
  "so a future change... cannot silently drift" and explicitly naming
  "checkpoint metadata" as a consumer), but `model/core.py` currently has
  **zero external `hydroswarm`-package imports** -- it is a clean leaf
  module. Importing from `planning` or `training` into it to pull these
  hashes in would invert that layering. Assembling `architecture_config()`
  plus these externally-owned hashes into one complete checkpoint identity
  belongs in whichever layer already orchestrates checkpoint export (a
  future dedicated checkpoint-identity module), not inside
  `model/core.py` itself.
- **9.2's granular trained/validated/runtime-enabled output gating**: a
  new governance concept requiring design work across every head, the
  loss/eval code, and the checkpoint save/load path -- not an incremental
  addition to the existing per-flag `verify_architecture_compatibility()`
  checks.
- **9.3's orphaned-output audit** and **9.4's candidate-count/fixed-query
  cleanup**: require deciding which existing outputs to keep, demote, or
  remove -- product/architecture decisions, not mechanical additions.
- **`verify_architecture_compatibility()`'s enforcement logic**: untouched.
  Confirmed by reading it that `metadata.get(key)`-based checks are
  additive-safe by construction (an old checkpoint's metadata simply won't
  have the new keys, so no new check fires against it) -- but no NEW
  checks were added for the newly-recorded fields either, since most of
  them (dimensions) already have their mismatch caught by
  `load_state_dict(strict=True)`'s own shape check, and adding redundant
  explicit checks was judged not worth the additional surface area in a
  bounded pass.
- **`ARCHITECTURE_VERSION` bump to `hydrocore-v4`**: not done -- bumping
  it before the rest of the v4 contract exists would only create
  compatibility friction without the payoff Phase 9 is meant to deliver.

## Phase 10 (core-issues4.txt Section I / core-issues3.txt Phase 10) -- ALL 5 ITEMS DONE

Started immediately after Phase 9 completed, in the exact priority order
Section I specifies. Six commits, all pushed to
`origin/agent/gcp-multitopology-v3`:

1. `dc2741f` feat(training): build real Scout-state and Strategist-candidate datasets (Phase 10.2/10.3, `validation` split)
2. `b140b0c` feat(data): generate balanced supported-category OOD extension corpus (Phase 10.4)
3. `54ae4e9` test(v4): real multi-topology gradient smoke test for every retained output (Phase 10.5)
4. `7f8a11f` feat(data): complete Phase 10.1 trajectory corpus regeneration (`train` split, plus re-running Phase 10.2/10.3 datasets against it)

### Phase 10.1 -- regenerate the final non-provisional trajectory corpus (DONE)

`data/learning-v2/cycle-b2-trajectories-v2` (committed in the prior pass)
was explicitly provisional: it predated Phase 6.4's `HYDRAULIC_MISMATCH`
mislabel fix, Phase 7.2's Scout per-node target shape, Phase 7.3's
sensor-reconstruction masking, Phase 7.4's `future_concentration` disable,
and Phase 7.5's travel-time transform. Re-running
`scripts/generate_trajectory_corpus.py` against the current HEAD (which
already contains all of those fixes in the underlying label-generation
code -- confirmed by inspection: `build_incident_trajectory` already
threads `reconstruction` into `build_scout_trajectory`, Phase 5's
incremental-revelation mechanism) produced a genuinely corrected corpus
with no additional code changes needed for the regeneration itself.

Launched as 4 independent resumable background jobs (`hydroswarm.training.
job_runner`, `experiments/jobs/cycle-b2-trajectories-v3-{split}`), polled
at the requested 10-minute interval. **All 4 splits now complete and
committed, 0 errors throughout**:

| split | scenarios | errors | wall time |
|---|---:|---:|---:|
| validation | 1000/1000 | 0 | ~19 min |
| calibration | 1000/1000 | 0 | ~19 min |
| development_holdout | 2150/2150 (400 coastal-branch skipped, unsupported topology -- same documented behavior as the prior pass) | 0 | ~44 min |
| train | 9000/9000 | 0 | ~3h16m |

`train.jsonl` (126 MB) needed the same >100MB Git LFS filename exception
`cycle-b2-trajectories-v2/train.jsonl` already established (added by exact
filename, not a directory-wide glob, matching that precedent exactly).

`cycle-b2-trajectories-v3` fully supersedes the provisional `-v2` corpus
as the governed trajectory corpus going forward. `-v2` is left untouched
(not deleted), consistent with this project's "preserve for audit" policy.

Reproduce (idempotent -- skips already-processed scenario_ids):

```bash
export PYTHONPATH=src
for split in validation calibration development_holdout train; do
  python scripts/generate_trajectory_corpus.py \
    --corpus-dir data/learning-v2/cycle-b2 \
    --output data/learning-v2/cycle-b2-trajectories-v3 \
    --split "$split"
done
```

### Phase 10.2 -- sharded Scout-state datasets and collators (DONE, full `train`+`validation` scale)

`scripts/build_scout_state_dataset.py`: merges each trajectory's Scout
step 0 (the initial, unconditioned decision) onto `cycle-b2`'s base
tensors. **Deliberate scoping decision, not an oversight**: `HydroCore`
has no `already_sampled`-equivalent input anywhere in `HydroBatch` to
condition later Scout steps on, so training against Phase 5's full
multi-step trajectory would hand the model contradictory targets for an
identical input representation across steps of the same scenario -- the
same class of gap `future_concentration_target`'s Phase 7.4 disable
already established a precedent for. Examples with zero Scout steps
(Phase 2 item 4: "no useful candidate exists") are masked, not dropped,
preserving 1:1 correspondence with the base corpus.

No new collator was needed for Scout specifically -- but building this
surfaced two real, previously-undiscovered defects in the *existing*
collator (`variable_collate.py`) and its `permutation.py` counterpart,
both instances of the same "two independently-maintained lists silently
drift apart" defect class this project has now hit three times (after the
8-vs-9 action-template count and the `ood_head`-vs-`ood_category_head`
loss mapping):

1. `variable_collate.NODE_INDEXED_TARGET_KEYS` (a hand-maintained tuple)
   was missing `information_gain`/`candidate_reduction` entirely (Phase
   7.2's per-node conversion) -- collating a real batch with these targets
   raised. Now derived from `targets_v2.NODE_ARRAY_TARGETS` programmatically.
2. `permutation.py` had its own separate, even narrower copy
   (`("sensor_fault",)` only) for equivariance-testing node permutation,
   plus an incomplete `NODE_INDEX_TARGET_KEYS` (missing `sample_node`).
   Both now derived from `targets_v2` directly.

Verified for real against `cycle-b2-trajectories-v3/validation.jsonl`
(1000/1000 examples matched, genuine per-node shapes confirmed) with a
real forward+backward pass across all 3 topology families: every Scout
head (`sample_node_head`, `information_gain_head`,
`candidate_reduction_head`, `should_continue_sampling_head`) receives a
real nonzero gradient. `tests/integration/test_scout_state_dataset.py`
(self-contained, hand-built fixtures, does not depend on the background
job's output).

**Re-run against the completed `train` split** once Phase 10.1 finished:
9000/9000 matched, 0 masked-placeholder fallbacks. A real forward+backward
smoke pass across a stride-sampled, all-3-topology subset of the full
train-scale dataset confirmed it remains finite and trainable at full
scale (`data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized/
{train,validation}`, both committed).

Reproduce:

```bash
export PYTHONPATH=src
for split in train validation; do
  python scripts/build_scout_state_dataset.py \
    --tensor-shard-dir data/learning-v2/cycle-b2/tensors-normalized/"$split" \
    --trajectory-jsonl data/learning-v2/cycle-b2-trajectories-v3/"$split".jsonl \
    --output data/learning-v2/cycle-b2-trajectories-v3/scout-tensors-normalized/"$split" \
    --split "$split"
done
```

### Phase 10.3 -- sharded Strategist-candidate datasets and collators (DONE, full `train`+`validation` scale)

`scripts/build_strategist_candidate_dataset.py`: `build_strategist_trajectory`
is single-step by design (no Scout-style multi-step ambiguity), so every
scenario's one real Strategist step is used directly. Builds BOTH the
candidate-conditioned architecture's real INPUT contract
(`plan_template_ids`/`plan_target_type`/`plan_target_node_index`/
`plan_target_link_index`/`plan_features`/`plan_mask`, from each candidate
label's semantic identity plus its already-resolved `target_pointer`) and
its governed TARGETS (`plan_validity`/`plan_value`/all 5 consequence
proxies, copied directly from the exact-WNTR-derived per-candidate
targets).

`plan_features` scoping decision, documented in the script's own
docstring: `CandidatePlanEncoder` describes richer pre-verification
proposal features (the old heuristic score, cost estimate, action count,
source-region overlap, etc.) that are not present anywhere in the
persisted trajectory JSONL today -- computing them for real requires the
original network/incident context at *generation* time, not something
derivable from the already-serialized labels. Rather than fabricate
plausible-looking numbers (forbidden by this project's own labeling
restrictions), `plan_features` uses only genuinely available, leakage-free,
purely structural facts (6 dims, matching `HydroCore.PLAN_FEATURE_DIM`'s
default width). The richer feature set is real, scoped follow-up work.

This *is* a genuinely new collator: `variable_collate.py` gained
candidate-dimension (P) padding for both the `plan_*` INPUT keys (a new
`_pad_plan_indexed_inputs` helper) and the governed per-plan TARGET keys
(reusing `targets_v2.PLAN_DIMENSION_TARGETS` rather than a fifth
hand-maintained list). Verified with a genuinely variable candidate count
per example (not just the real corpus's always-9), not merely the
fixed-P case.

Verified for real against `cycle-b2-trajectories-v3/validation.jsonl`
(1000/1000 examples matched, canonical 9 candidates/example) with a real
forward+backward pass across all 3 topology families: `CandidatePlanEncoder`,
`plan_value_head`, `plan_validity_head`, and all 5 consequence-proxy heads
receive real nonzero gradients -- **the first time the candidate-conditioned
Strategist architecture has trained on real data in this project's
history** (previously "architecture + tests; not yet wired to real
training data" per the prior handoff). `tests/integration/
test_strategist_candidate_dataset.py` (self-contained fixtures).

**Re-run against the completed `train` split** once Phase 10.1 finished:
9000/9000 matched, canonical 9 candidates/example throughout. A real
forward+backward smoke pass across a stride-sampled, all-3-topology subset
confirmed the full train-scale dataset remains finite and trainable
(`data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized/
{train,validation}`, both committed).

Reproduce:

```bash
export PYTHONPATH=src
for split in train validation; do
  python scripts/build_strategist_candidate_dataset.py \
    --tensor-shard-dir data/learning-v2/cycle-b2/tensors-normalized/"$split" \
    --trajectory-jsonl data/learning-v2/cycle-b2-trajectories-v3/"$split".jsonl \
    --output data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized/"$split" \
    --split "$split"
done
```

### Phase 10.4 -- balanced supported-category OOD extension corpus (DONE)

`cycle-b2` already had real generated data for 2 of the 6 non-NONE
supported OOD categories (`SEVERE_MISSINGNESS`, `UNSEEN_TOPOLOGY`).
`scripts/generate_ood_extension_corpus.py` fills the remaining 4
(`EXTREME_DEMAND`, `TANK_STATE_SHIFT`, `ROUGHNESS_MISMATCH`,
`FROZEN_DRIFTING_SENSOR`), using the already-tested
`OOD_TRIGGERING_CONFIG_OVERRIDES` recipe at real scale (400/category,
matching the existing `SEVERE_MISSINGNESS` holdout's own count) -- written
to a new `data/learning-v2/cycle-b2-ood-extension/`, never touching
`cycle-b2` itself. Each topology's signature library is refit from that
topology's real `cycle-b2` TRAIN scenarios and hash-verified against
`cycle-b2`'s own recorded artifact before use (never fit from holdout
data, never a silently-drifted copy).

Every one of the 1600 generated scenarios is verified for real via
`classify_ood_category()` before being kept -- the script fails closed if
any scenario's override does not actually trigger its intended category.
All 1600 passed.

**Two real defects found running this at real scale, not by inspection**:

1. `WNTRScenarioGenerator.generate_with_network` unconditionally zeroes
   `missing_probability` whenever `stage == CurriculumStage.CLEAN`,
   silently discarding whatever the caller's config actually requested. A
   real small-scale dry run caught 3/6 `SEVERE_MISSINGNESS` scenarios
   silently classifying as `NONE` because of exactly this (the *existing*,
   protected `cycle-b2` `SEVERE_MISSINGNESS` holdout likely carries the
   same latent ~1/5 gap, since `generate_cycle_b_corpus.py`'s own
   `_stage_for_index` cycles through every stage including `CLEAN` -- not
   touched here, flagged for awareness). Fixed by never selecting `CLEAN`
   for any OOD-extension scenario.
2. `generate_category` verified each scenario's category but never
   actually attached `ood_class_target(...)` to the resulting example --
   the entire point of this corpus would have been silently missing. The
   first (buggy) full-scale run, already complete, was killed and its
   untracked, never-committed output discarded before this fix; the corpus
   actually committed is the corrected regeneration.

`tests/scientific/test_generate_ood_extension_corpus.py` runs the real
recipe for all 5 governed categories at small real scale against the
actual `cycle-b2` corpus, asserting `ood_class` is both present and
correct (the regression test for defect 2) and that `CLEAN` is never
selected (defect 1). Both raw and `tensors-normalized/` variants
committed (the latter via direct reuse of `rebuild_normalized_shards.py`'s
`rebuild_split` function, since that script's `main()` hard-requires a
`train` split this corpus deliberately does not have).

### Phase 10.5 -- real multi-topology gradient smoke tests for every retained v4 output (DONE)

`tests/integration/test_full_output_gradient_smoke.py`, two tests:

1. Merges `cycle-b2-trajectories-v3`'s completed `validation` split onto
   `cycle-b2`'s base tensors (`scripts/merge_trajectory_targets.py`) and
   runs one real forward+backward with `event_control_heads` +
   `auxiliary_heads` + `ood_category_head` + `scout_control_heads` all
   enabled *together* (a genuine joint-multitask configuration, not one
   flag at a time) across all 3 real topologies: every governed
   Sentinel/control/auxiliary/OOD head receives a real nonzero gradient.
2. Uses `cycle-b2-ood-extension`'s real data (Phase 10.4) to exercise
   `ood_category_head` against 4 genuinely distinct real non-NONE classes
   -- `cycle-b2`'s own data is almost entirely `NONE`-class, so a batch
   drawn only from it would not meaningfully exercise the head's non-NONE
   logits.

Scout and Strategist are covered by their own Phase 10.2/10.3 tests, not
duplicated here. Together with Phase 10.2/10.3's own tests, every retained
v4 output (per `checkpoint_identity.py`'s Section D decisions) now has at
least one real, committed, passing test proving it receives a nonzero
gradient from real multi-topology data.

## Phase 11 (core-issues3.txt "PHASE 11 -- LOSS SYSTEM AND TRAINING CONFIGURATION") -- ALL 5 ITEMS DONE

Overnight autonomous continuation, started from clean HEAD `a187add` (the
prior pass's final Phase 10 handoff commit, verified against the expected
commit before any edits). Read `overnight-plan.txt`,
`core-issues.txt`/`core-issues2.txt`/`core-issues4.txt` (context) and
`/workspace/core-issues3.txt` (task spec) in full before starting. Nothing
was left mid-task from the prior pass -- Phase 10 was fully complete, so
this pass began Phase 11 fresh, in order (11.1 -> 11.4 -> 11.5 -> 11.3 ->
11.2; 11.3/11.2 were reordered after 11.1 to land the checkpoint/gradient
plumbing first, since 11.2's class-weights wiring is a small addition to
the same `compute_multitask_loss` signature 11.1 already touched). Five
commits, all pushed to `origin/agent/gcp-multitopology-v3`, working tree
clean after each:

1. `33eb229` feat(training): explicit task weights and per-task loss diagnostics (11.1)
2. `d647348` feat(training): gradient-conflict diagnostics and per-task validation history (11.4)
3. `d498376` feat(training): v4 checkpoints preserve RNG state, source Git SHA, task weights (11.5)
4. `a24564f` feat(training): reusable scale-safety preflight check (11.3)
5. `344ec21` feat(training): class-imbalance reporting and train-owned class weights (11.2)

No work on `main`. `data/learning-v2/cycle-b2` and every other protected
corpus/checkpoint/v3 result artifact untouched (this pass only READ real
corpus data, for 11.2's real class-prevalence report -- no corpus write).
Locked test not opened; `final-selection.json` does not exist.

### 11.1 -- explicit task weights for every retained target (DONE)

`configs/training.yaml`'s `task_weights` previously covered 13 of the ~27
tasks `compute_multitask_loss` can produce, silently relying on its hidden
per-call default (1.0, or 0.1 for `AUXILIARY_TASKS`) for the rest --
exactly what core-issues3.txt Phase 11.1 says not to do for "the final
experiments." New `hydroswarm.training.losses.ALL_TASK_NAMES` (the union of
every task key `compute_multitask_loss` can produce, kept next to the
function itself so a future new task is a one-place addition, not a fourth
instance of this project's "two lists drift apart" defect class) plus
`validate_task_weights_complete()`/`IncompleteTaskWeightsError` fail closed
on a `task_weights` mapping missing an explicit entry.
`configs/training.yaml` now declares all ~27 explicit weights, grouped and
commented by role/reasoning; a new test
(`test_configs_training_yaml_declares_every_retained_task_weight_explicitly`)
proves the actual production config -- not just a synthetic fixture --
passes. `TrainingConfig.from_yaml` gained an opt-in
`require_complete_task_weights` flag for a training entry point to enforce
this at config-load time.

`MultiTaskLoss` now also carries, per task: `valid_counts` (post-mask/
finite target count), `weights` (the value actually applied, post-override/
default), and `weighted` (`weight * mean unweighted loss`) -- Trainer's
per-batch `metrics.jsonl` entries log all three alongside the existing
`task_losses`/`task_gradient_norms`. 8 new tests,
`tests/unit/test_losses.py`.

**Deliberately NOT retroactively touched**: the historical Stage 2/3/4
screening scripts (`scripts/run_stage2_architecture_screening.py`,
`run_stage3_finalist_training.py`, `run_stage4_controls_training.py`)
construct `TrainingConfig(...)` directly with their own inline kwargs
(mostly relying on the same hidden default this item fixes) rather than
loading `configs/training.yaml` -- these already produced preserved,
provisional Stage 3/4 results per this project's own restriction #1, and
retroactively changing their task-weight behavior would change what a
future re-run of them produces without being asked to. The
completeness-validated `configs/training.yaml` is the intended input for
Phase 12's still-unstarted staged training, not a rewrite of already-run
historical scripts.

### 11.4 -- gradient-interference diagnostics (DONE)

New `hydroswarm.training.losses.task_gradient_conflict`: pairwise cosine
similarity between each `PRIMARY_TASKS` member's gradient and every other
present task's gradient (primary-vs-all, not all-vs-all -- a full T x T
sweep over ~27 possible tasks would cost O(T^2) extra
`torch.autograd.grad` calls on top of the T calls `task_gradient_norms`
already costs, for pairs nobody is making a promotion decision about).
Wired into `Trainer` behind a new opt-in `gradient_conflict_logging` flag
(default `False`, same sparse `gradnorm_log_every_n_batches` interval as
`task_gradient_norms`) -- a guard test
(`test_pcgrad_and_gradient_conflict_logging_default_off`) locks both this
flag and `pcgrad_enabled` to their required-off-by-default state,
per-core-issues3.txt Phase 11.4's explicit "Do not enable PCGrad or a
complex automatic weighting scheme by default."

`Trainer._validate` previously discarded every per-task validation loss,
keeping only the scalar mean -- exactly the information needed to see
*which* task's validation performance moved (a "primary-task regression")
across epochs or configs. Now returns and persists the per-task breakdown
to a new accumulating `validation_history.jsonl` (one entry per epoch;
`epoch_summary.json` only ever keeps the latest epoch, since `atomic_json`
overwrites the same path every call). `RunArtifacts` gained a generic
`append_jsonl`, with `append_metric` now a thin wrapper over it (identical
`metrics.jsonl` behavior, unchanged, confirmed by the existing
`test_gradnorm_logging_only_runs_on_the_configured_batch_interval` test
still passing unmodified). 10 new tests across
`tests/unit/test_losses.py` and `tests/scientific/test_training_smoke.py`.

### 11.5 -- checkpoint completeness: RNG, source Git SHA, task weights (DONE)

Audited `save_v4_checkpoint`/`load_v4_checkpoint`
(`hydroswarm.training.checkpoint_identity`) against core-issues3.txt Phase
11.5's exact required-preservation list. Architecture config, data
manifests, transform hashes, and trained/validated/runtime output metadata
were already covered by the existing `identity`/`resolved_training_config`/
`dataset_manifest_hashes` parameters; RNG state, source Git SHA, and task
weights were not.

**Real, concrete gap, not a paperwork gap**: `Trainer`'s DataLoader
generator is deterministically re-derived from `config.seed + epoch`
(independent of any accumulated global RNG state), so resuming already
reproduces the exact per-epoch data order correctly -- but `nn.Dropout`
draws from the global torch RNG, which was only ever reset to
`config.seed` once at `Trainer.__init__`. Resuming from a checkpoint would
silently restart dropout's randomness from the fresh-run state rather than
continuing from wherever the pre-resume run had actually advanced it to --
a real (if narrow) resume-determinism defect for any model with nonzero
dropout, not merely a missing "nice to have" field.

`save_v4_checkpoint` now captures Python/NumPy/torch RNG state to a new
`rng_state.pt`; `load_v4_checkpoint` gained an opt-in `restore_rng` flag
(default `False` -- an inference/inspection load should not silently
mutate process-global RNG state as a side effect; a training-resume caller
passes `True`), verified by a real round-trip test proving three
independent RNG streams' subsequent draws match a pre-advance control
exactly after restore. Also now requires an explicit `task_weights`
mapping (recorded in `trainer_state.json` exactly as given -- a checkpoint
preserves whatever weights a run actually used, complete or not; full
completeness against `ALL_TASK_NAMES` stays a training-entry-point-time
concern per 11.1, not re-enforced a second time here) and a `workdir`
parameter used to record `source_git_sha`
(`hydroswarm.training.artifacts.git_commit_hash`, newly made public and
reused here rather than a second independently-drifting copy of the same
subprocess call). 9 new/updated tests,
`tests/unit/test_checkpoint_identity.py`; the real trained-model
integration test (`test_v4_production_checkpoint.py`) updated to pass
`task_weights`/`workdir` through its actual `save_v4_checkpoint` call and
still passes end to end.

**Deliberately NOT done**: rewiring `Trainer`'s default (v3) checkpoint
path to use `save_v4_checkpoint`. No promoted v4 checkpoint exists yet to
make that switch meaningful (Phase 15's "wire `output_governance`/
checkpoint-identity into a live v4 runtime path" is still open, as already
flagged in the prior pass's handoff), and `hydroswarm.training.checkpoint`
's legacy `save_checkpoint`/`load_checkpoint` explicitly "stays exactly
as-is for existing v3 checkpoints" per its own docstring and
core-issues4.txt's PRIMARY DESIGN RULE -- left untouched.

### 11.3 -- scale-safety preflight check (DONE)

New `hydroswarm.training.scale_safety.run_scale_safety_check`: one real
forward pass, asserting every governed property in one place -- every
present task reaches `compute_multitask_loss`, every caller-declared
`required_tasks` has positive `valid_counts`, every task with positive
`valid_counts` gets a real finite nonzero gradient, every task with zero
`valid_counts` gets *exactly* zero gradient (the concrete "padded/masked
positions contribute zero" check -- proves `masked_regression`'s
`prediction.sum() * 0.0` fallback is still gradient-inert, not merely
loss-value-inert), and no NaN/Inf anywhere. "No accidental broadcasting" is
already enforced structurally by `masked_regression`'s own shape check.

`required_tasks` is deliberately scoped per call rather than "every task
`compute_multitask_loss` knows about" -- a structurally-disabled target
like `future_concentration` (Phase 7.4: always all-masked) would make an
unscoped version of this check impossible to ever pass on a real
full-multitask batch.

Generalizes the ad-hoc version of this same check already duplicated in
`scripts/run_event_control_smoke_screening.py`'s `_gradient_check` and
`scripts/run_architecture_smoke_jobs.py` -- existing scripts deliberately
left untouched (same historical-preservation reasoning as 11.1's Stage
2-4 scripts note above); this is the one place a new training entry point
(Phase 12) should call instead of writing a fourth copy.

**Real bug caught while testing, not by inspection**: an earlier draft
called `result.total.backward()` before `task_gradient_norms`'s own
per-task `torch.autograd.grad(retain_graph=True)` calls --
`backward()` without `retain_graph=True` frees the graph, so the second
task's gradient computation raised "Trying to backward through the graph a
second time." Fixed by relying entirely on `task_gradient_norms`'s own
graph-preserving calls (no separate `.backward()` needed for a preflight
check that does not itself take an optimizer step). 5 new tests,
`tests/unit/test_scale_safety.py`.

### 11.2 -- class-imbalance reporting and train-owned class weights (DONE)

New `hydroswarm.training.class_balance` module: `class_prevalence()`
counts label occurrences (mask/`ignore_index`-aware, uniform over a
scalar-per-example class index or a per-position array like
`plan_validity`), `merge_prevalence()` combines shard-level counts, and
`train_owned_class_weights()` derives deterministic, versioned
(`CLASS_WEIGHT_POLICY_VERSION`), capped inverse-frequency weights from
TRAIN-split prevalence only -- the same train-only-fitting discipline this
project already applies to normalization/signature artifacts, never
validation/calibration/development-holdout.

`compute_multitask_loss` gained an optional `class_weights` argument
(distinct from `task_weights`: reweights *within* one classification
task's loss by class, not the task's overall contribution), threaded to
`F.cross_entropy`'s own `weight=`. Applying it changes a task's loss value
but never its `valid_counts` -- "evaluate unweighted real-distribution
metrics separately" is therefore a property callers get for free by
construction, confirmed by a dedicated test
(`test_class_weights_change_the_loss_but_not_the_valid_count`).

**Real report, run against real committed corpora** (not merely a
demonstration of the mechanism): `scripts/report_class_prevalence.py`,
output committed at `reports/results/v4/class-prevalence.json`:

| task | corpus | split coverage |
|---|---|---|
| `event_cause` | `data/learning-v2/cycle-b2` | train/validation/calibration/development_holdout |
| `next_step` | `data/learning-v2/cycle-b2-control-v2` | train/validation |
| `plan_validity` | `data/learning-v2/cycle-b2-trajectories-v3/strategist-tensors-normalized` | train/validation |

Real train-split prevalence: `event_cause` {NONE=6300, class1=1350,
class2=450, class4=900} out of 9000 (class 3 never appears in ANY split --
consistent with, and independent confirmation of, the already-documented
Phase 6.4 `HYDRAULIC_MISMATCH`-removal / Phase 6.5 `AMBIGUOUS`-unsupported
decisions); `next_step` {COLLECT_SAMPLE=3609, GENERATE_PLANS=4081,
INSPECT_FAULTY_SENSOR=1310} out of 9000, zero `ABSTAIN` examples --
independently matches `control-heads-training.json`'s already-reported
"ABSTAIN n/a (0)" finding via a completely different code path (direct
corpus counting here vs. a trained model's predictions there);
`plan_validity` {valid=59522, invalid=21478} out of 81000 (9000
examples x 9 candidate plans each). Train-owned weights derived from
each of these (e.g. `event_cause` class 2's rarity -> weight 2.1x vs.
class 0's 0.15x) are recorded in the same JSON, ready for a future Phase
12 training run to opt into via `class_weights=` -- not applied to any
training run in this pass (11.2's own text: "use... when justified"; no
staged-training run exists yet to judge that against).

**Deliberately not covered this pass**: `ood_class`'s real non-NONE
prevalence (`data/learning-v2/cycle-b2-ood-extension`) -- that corpus's
per-category shard directories (`ood-EXTREME_DEMAND` etc.) carry an
internal split-consistency check in `ShardedScenarioDataset` that this
script's straightforward "one conventional split name per corpus" loading
pattern doesn't satisfy; real, scoped follow-up work, not silently
skipped (flagged in the script's own module docstring). 12 new tests,
`tests/unit/test_class_balance.py`.

### Full suite, Ruff, Pyright

`ruff check src/ scripts/ tests/` and `pyright src/` both clean
(0 errors/warnings) after every commit in this pass, re-verified against
the final HEAD. Full `pytest` suite: **697 passed, 0 failed** (533s), up
from **661 passed** on the unmodified pre-Phase-11 code (a real baseline
run captured at the start of this pass, before any edits, not merely
quoted from the prior handoff). A second, targeted run of
`tests/unit/`+`tests/scientific/`+the three v4/Scout/Strategist
integration files independently reported 663 passed, 0 failed -- both
runs agree (a full-suite run necessarily includes more files than the
targeted subset, hence the different totals; neither reported a failure).

## Phase 12 Stage B (core-issues3.txt "PHASE 12 -- STAGED TRAINING AND ABLATIONS", Stage B: flat control/OOD/auxiliary) -- DONE

Continuation within the same overnight autonomous run, immediately after
Phase 11's completion (`0b6b456`). Phase 12 begins the staged-training
curriculum core-issues3.txt Phase 12 specifies (Stage A -- Sentinel
foundation, already effectively satisfied by the existing Stage-A
checkpoint from Phase 8; Stage B is the first genuinely new real training
work in this phase). Two commits, both pushed to
`origin/agent/gcp-multitopology-v3`:

1. `8ebb682` feat(training): Stage B control/OOD/auxiliary ablation script and real run (core-issues3.txt Phase 12 Stage B)

No work on `main`. Every protected artifact untouched. `data/learning-v2/
cycle-b2` and `cycle-b2-trajectories-v3` were only READ (via
`scripts/merge_trajectory_targets.py`, which "never mutates the source
shard directory" per its own docstring) to build a derived, gitignored
merge under `experiments/runs/v4-stage-b-control-ood-auxiliary/
merged-corpus/`. Locked test not opened; `final-selection.json` does not
exist.

### Real finding #1, caught before wasting compute: `cycle-b2-control-v2` is the wrong corpus for this ablation

The first design draft planned to reuse `data/learning-v2/
cycle-b2-control-v2` (Phase 8/Section F's second-pass control corpus) for
Stage B, since it already carries corrected `event_cause` and calibrated
`evidence_sufficiency`/`next_step`. A smoke-test run before launching the
real job caught that this corpus has **no `ood_class`, `sensor_reconstruction`,
or `travel_time` targets at all** -- confirmed by directly inspecting a real
loaded example's `targets.keys()`, not assumed. Training against it would
have made the entire ablation vacuous: the "no auxiliary" and "all
auxiliary" arms would have been computationally IDENTICAL, since those
loss terms would never reach `compute_multitask_loss` regardless of
`task_weights` (the targets are simply absent from every batch).

Fixed by using `scripts/merge_trajectory_targets.py` (Phase 10.5's own
mechanism) to merge `data/learning-v2/cycle-b2`'s base tensors with
`cycle-b2-trajectories-v3`'s per-example flat targets -- verified for real
to carry `ood_class`/`sensor_reconstruction`/`travel_time`/`future_concentration`
alongside `event_presence`/`event_cause`/`evidence_sufficiency`/`next_step`.
This also turns out to be the LITERALLY correct corpus for Stage B's own
wording ("**preliminary** control targets"): the trajectory corpus's
`evidence_sufficiency`/`next_step` are the FIRST-pass (sensor-health-only-
derived) labels, genuinely preliminary relative to Stage C's already-
completed second-pass calibrated ones in `cycle-b2-control-v2` -- not
merely a fallback choice made necessary by the missing targets, but the
better-matching corpus on both counts. Merged for real: train 9000/9000
matched, validation 1000/1000 matched (0 unmatched in either split),
written to `experiments/runs/v4-stage-b-control-ood-auxiliary/
merged-corpus/{train,validation}` (gitignored, reproducible on demand from
two already-committed inputs, not a new governed corpus needing its own
manifest/gates infrastructure just for an ablation run).

### Ablation design

One consistent architecture across all 4 arms (`event_control_heads=True`,
`ood_category_head=True`, `auxiliary_heads=True` -- every new head
physically exists in every arm's model), ablated via `task_weights` alone
(`sensor_reconstruction`/`travel_time` zeroed or not per arm) --
guarantees "identical manifests, seeds, budgets, and policies" (Phase
12's own requirement) trivially, and avoids a fourth, subtly-different
`verify_architecture_compatibility` identity per arm. Base task weights
loaded from `configs/training.yaml` via `TrainingConfig.from_yaml(
require_complete_task_weights=True)` (Phase 11.1) -- every arm's
`task_weights` differs from every other ONLY in the two ablated entries,
directly exercising Phase 11.1's new completeness-validated config in real
Phase 12 work. Every arm initializes from the same Stage-A Sentinel
teacher checkpoint (`experiments/runs/v4-stage-a-sentinel/
E1-seed20260810/.../checkpoint-0016`) via `strict=False` load +
fail-closed verification that the ONLY missing keys are the 7 brand-new
Stage-B head prefixes (verified against the actual checkpoint's real
parameter names, not assumed from the training overrides that produced
it) -- unlike `train_control_heads.py`'s frozen-backbone ablation
baseline, the backbone is left UNFROZEN here: Stage B's new heads all
initialize from scratch and need real gradient flow through the shared
backbone to mean anything as a joint-multitask integration test.

8 epochs/arm, `maximum_runtime_seconds=2400`, `learning_rate=1e-4`
(lower than `train_control_heads.py`'s `3e-4` -- unfrozen backbone,
avoid disturbing Stage-A's already-good Sentinel weights too fast),
`batch_size=16`, single seed (`20260807`, matching every other v4 training
script in this repo). Single-seed, 8-epoch runs are a real, honest
scientific limitation of this pass (see "Limitations" below), not
concealed.

### Real results (real training, all 4 arms, 4332.3s / 72.2 min total)

| arm | source_node_acc | event_cause_acc | sensor_reconstruction_mse | travel_time_mse |
|---|---|---|---|---|
| `no_aux` (baseline) | 0.512 | 0.831 | 83693 | 45.62 |
| `aux_sensor_reconstruction_only` | 0.515 | 0.832 | 83554 | 50.93 |
| `aux_travel_time_only` | 0.511 | 0.829 | 83638 | **5.11** |
| `all_aux` | 0.510 | 0.829 | 83555 | **5.28** |

`sensor_fault_accuracy` was 1.0 in every arm (this corpus's real
`sensor_fault` prevalence is heavily one-sided -- not a useful
discriminating metric for this particular ablation, reported for
completeness rather than treated as a signal).

### Real finding #2: `travel_time` is a clean win; `sensor_reconstruction` is not, but does not hurt either

**`travel_time`**: enabling it drops `travel_time_mse` from ~46-51 down to
~5.1-5.3 -- a genuine ~9x improvement -- with primary-task metrics
essentially unchanged (source_node_accuracy within 0.2pp, event_cause
within 0.3pp of baseline in every arm). A clean, unambiguous retain case.

**`sensor_reconstruction`**: `sensor_reconstruction_mse` does NOT improve
when its loss is enabled (83554-83638 across all arms, indistinguishable
from the `no_aux` baseline's 83693 within noise) -- 8 epochs at this
learning rate did not teach the head anything useful. Primary-task metrics
are ALSO not measurably hurt (same <0.5pp range as `travel_time`'s arms).
Digging into WHY the auxiliary made no progress surfaced a real,
concrete, previously-unquantified problem: `sensor_reconstruction`'s
target is raw, unnormalized mg/L concentration (unlike `travel_time`,
which Phase 7.5 already log1p-transforms) --

- its raw per-batch loss spikes as high as **50.6 million** in outlier
  batches (a batch containing an example near a high-concentration
  source), driving `gradient_norm` as high as **59,377** against a
  typical ~5-7 for every other batch;
- `gradient_clip_norm=1.0` keeps training numerically stable throughout
  (zero NaN/Inf across any arm's full `metrics.jsonl`, confirmed by
  scanning every row, not sampling) -- but clipping preserves gradient
  DIRECTION while only rescaling magnitude, so on an outlier batch the
  entire parameter update is effectively 100% driven by
  `sensor_reconstruction`'s noisy signal, drowning out every other task
  that batch;
- more concretely damaging: `aux_sensor_reconstruction_only`'s
  `best_validation_loss` (16328.13, `all_aux`'s is 16329.25) is almost
  entirely `sensor_reconstruction`'s own unweighted validation loss
  (~65,300) times its 0.25 weight (~16,326) -- meaning `Trainer`'s
  epoch/checkpoint SELECTION criterion for these two arms is effectively
  noise-dominated by which epoch happened to have the least-extreme rare
  high-concentration validation example, not by genuine overall multitask
  quality. (`no_aux`/`aux_travel_time_only`'s `best_validation_loss`
  values, 3.10/3.62, are sane by comparison -- direct confirmation this is
  specific to `sensor_reconstruction`'s scale, not a systemic issue.)

**Verdict** (script-computed, `>2pp primary-task-accuracy-drop` threshold,
matching Phase 12's "materially degrade" language): all three
auxiliary-enabled arms verdict `retain` -- no arm shows a primary-task
degradation past the threshold. This is accurate as stated, but the fuller
picture is: `travel_time` is a genuine, clean win; `sensor_reconstruction`
is retain-by-the-letter-of-the-threshold but is not currently learning
anything useful and its unnormalized scale makes any run that includes it
unreliable for automatic checkpoint selection. **Concrete recommendation,
not attempted this pass**: apply a `log1p`-style transform to
`sensor_reconstruction`'s target (mirroring Phase 7.5's own precedent for
`travel_time`) before relying on it in any promotion-quality run's
`best_validation_loss` selection.

### Limitations, reported honestly

- Single seed, 8 epochs, one corpus, one learning rate per arm -- a real
  first ablation pass, not a statistically robust multi-seed comparison.
  Phase 12 does not require more at this stage; a future Stage F joint
  fine-tuning pass should re-examine this with more seeds if
  `sensor_reconstruction`'s scale is fixed and it becomes a real
  promotion candidate.
- `ood_class`/`next_step` were present in every arm's `task_weights` and
  training loss (confirmed via `metrics.jsonl`'s `task_losses`/
  `task_valid_counts` -- both reached real, positive-valid-count
  supervision every batch) but were not separately evaluated in this
  script's `evaluate()` (which only covers `source_node`/`event_cause`/
  `sensor_fault`/the two auxiliaries) -- real, scoped follow-up, not
  silently dropped: OOD-category and control-head evaluation already has
  dedicated, more thorough treatment in Phase 8/Section F's own scripts
  and was not the point of THIS ablation (which is specifically about the
  two validated auxiliaries per Phase 12 Stage B's own text).

Reproduce:

```bash
export PYTHONPATH=src
# 1. Build the merged corpus (fast, deterministic join, ~20s total):
python scripts/merge_trajectory_targets.py \
  --tensor-shard-dir data/learning-v2/cycle-b2/tensors-normalized/train \
  --trajectory-jsonl data/learning-v2/cycle-b2-trajectories-v3/train.jsonl \
  --output experiments/runs/v4-stage-b-control-ood-auxiliary/merged-corpus/train \
  --split train
python scripts/merge_trajectory_targets.py \
  --tensor-shard-dir data/learning-v2/cycle-b2/tensors-normalized/validation \
  --trajectory-jsonl data/learning-v2/cycle-b2-trajectories-v3/validation.jsonl \
  --output experiments/runs/v4-stage-b-control-ood-auxiliary/merged-corpus/validation \
  --split validation

# 2. Run all 4 ablation arms (~72 min total on this environment):
python scripts/run_stage_b_control_ood_auxiliary.py
# (--arms <name> [<name> ...] to run/resume a subset; --train-limit/
# --validation-limit/--max-epochs/--maximum-runtime-seconds for a fast
# smoke test only -- never for a real run)
```

### Stage C cross-check (no new work required -- already satisfied)

core-issues3.txt Phase 12 Stage C's text is exactly: "Fit exact hybrid
calibration and regenerate complete control labels as described above.
Train control heads." Cross-checked against already-completed work rather
than re-run from scratch: Phase 8 steps 1-9 (`v4-stage-a-sentinel`'s
calibration.json, fit on the exact deployed hybrid predictor) + Section F
step 6a/6b (`persist_second_pass_control_labels.py`/
`merge_second_pass_control_labels.py`, "regenerate complete control
labels") + `train_control_heads.py` ("train control heads", real metrics
in `reports/results/v4/control-heads-training.json`) already satisfy this
stage's text verbatim, completed in a prior pass of this same branch
(commits `5ab9165`/`75161ff`, see the "core-issues4.txt continuation pass,
part 2" section above). Recorded here explicitly so Stage C is not
mistaken for unstarted work in a future pass's status check.

### Full suite, Ruff, Pyright

`ruff check scripts/run_stage_b_control_ood_auxiliary.py` and `pyright`
on it both clean. Full `pytest` suite re-run after this pass: **697
passed, 0 failed** (548s) -- identical count to the post-Phase-11 baseline
(this pass added one new script, no new/changed library code or tests, so
an unchanged count is the expected, correct outcome, not a gap).

## Phase 12 Stage D (core-issues3.txt "PHASE 12", Stage D: Scout) -- DONE

Continuation within the same overnight autonomous run, immediately after
the Phase 12 Stage B/C work above. Three commits, all pushed to
`origin/agent/gcp-multitopology-v3`:

1. `7035114` refactor(scout): promote reindex_to_signature_grid to public
2. `6c20921` refactor(scout): promote reveal_sample_measurement to public; add Scout-head training script
3. `4168d60` feat(training): train Scout heads for real; Stage D Scout-policy comparison script

No work on `main`. Every protected artifact untouched. Locked test not
opened; `final-selection.json` does not exist.

### Prerequisite: train Scout's heads for the first time (real run)

`sample_node_head`/`information_gain_head` are unconditional base heads
already present in the Stage-A Sentinel checkpoint, but Stage-A trained
against `data/learning-v2/cycle-b2`, which carries no Scout targets at all
(confirmed by inspecting its real target keys) -- those heads never
received a real gradient there, and `candidate_reduction_head`/
`should_continue_sampling_head` (gated behind `scout_control_heads=True`)
did not exist in that checkpoint at all. New `scripts/train_scout_heads.py`
(frozen backbone, matching `train_control_heads.py`'s established pattern)
trains all 4 Scout heads on Phase 10.2's real
`cycle-b2-trajectories-v3/scout-tensors-normalized` dataset (~30% of
examples carry a real step-0 recommendation; the rest are correctly masked
"no useful candidate" examples).

**Real result** (619.6s, `experiments/runs/v4-scout-heads/
20260807T164149Z-adf93d12/checkpoints/checkpoint-0010`):

| metric | value |
|---|---|
| `sample_node` top-1 accuracy | **0.564** (well above chance given typical 4-8 candidates/scenario) |
| `should_continue_sampling` accuracy | 0.925 |
| `information_gain` MSE | 2.772 |
| `candidate_reduction` MSE | 0.021 |

First real supervision either head has ever received in this project's
history.

### Two preparatory API promotions (small, tested, no behavior change)

`hydroswarm.training.scout_labels._reindex_to_signature_grid` ->
`reindex_to_signature_grid` and `hydroswarm.training.scout_trajectory.
_reveal_sample_measurement` -> `reveal_sample_measurement`: both needed a
second real caller (Stage D's own pluggable multi-step trajectory loop,
which cannot use `build_scout_trajectory` directly since that function
always uses `generate_scout_label`'s own classical-EIG choice for which
node to reveal next). Pure renames, reused rather than duplicated so both
callers stay provably consistent; full existing test suite re-verified
passing after each.

Considered and reverted mid-design: initially added a
`posterior_probabilities` field to `ScoutLabel` for the same purpose, but
`hydroswarm.classical.signatures.localize_with_signatures` already does
the hypothesis-to-node marginalization and cumulative-mass candidate-set
construction this needs (found by searching for an existing function
before writing a new one) -- the `ScoutLabel` field would have been unused
surface area duplicating what that function already provides more
completely.

### Comparison design: two honestly-scoped modes, not one overclaimed mode

`scripts/run_stage_d_scout_policy_comparison.py` implements Stage D's
exact required 5-policy set (random, fixed_order, classical_eig,
learned_scout, classical_plus_residual), but as **two genuinely different
comparisons**, not one:

**A real architectural limitation this design works within, not around**:
HydroCore's Scout input has no `already_sampled`/revealed-evidence
conditioning (Phase 10.2's own documented scoping decision, this same
report's Phase 10.2 section) -- the trained model can only make ONE
well-supported decision, from the scenario's ORIGINAL evidence. Re-running
the same model on the same input at step 2+ would deterministically
recommend the same node again, which is not a real "keep sampling" policy.

1. **Step-0 comparison, all 5 policies**: each picks one node from the
   scenario's original evidence; the node is revealed once (via
   `simulate_all_node_truth`); realized entropy reduction and agreement
   with classical EIG's own top pick are measured. Fair for every policy.
2. **Multi-step operational trajectory comparison, `random`/`fixed_order`/
   `classical_eig` ONLY**: runs up to `--maximum-samples` steps, revealing
   genuinely new evidence each step, tracking `localize_with_signatures`'
   source posterior/candidate set at every step. `learned_scout`/
   `classical_plus_residual` are explicitly EXCLUDED here and the report
   itself records why (`exclusion_reason` field) -- not a silent omission.

### Two real bugs found while smoke-testing (not by inspection)

1. **`policy_classical_eig` conflated two separate questions.** First draft
   returned `label.sample_node_id`, which already folds in
   `generate_scout_label`'s own SEPARATE stop-threshold decision ("should
   we sample at all", gated by `minimum_information_gain_bits`) -- so it
   answered a different question than every other policy here ("given we
   ARE taking one more sample, which node?"). A 15-scenario smoke test
   showed `classical_eig` with `no_recommendation_count=3/3` while every
   other policy had `recommendation_count=3/3` -- an immediate, visible
   asymmetry that caught it before any real run. Fixed to pick the top of
   the already-EIG-sorted candidate list directly (`rank_sample_locations`
   already sorts by `expected_information_gain_bits`), independent of the
   stop threshold. The `agrees_with_classical_eig` reference metric had the
   identical bug (compared against the same conflated value) and was fixed
   the same way.
2. **Curriculum-ordered scenario prefix biased the smoke test.** The first
   15 validation scenarios (a curriculum-ordered corpus's own prefix) are
   almost entirely already-resolved, near-zero-entropy CLEAN-stage cases --
   every policy showed `mean_entropy_reduction_bits ≈ 0.0` before this fix,
   which was uninformative rather than a real null result. Fixed to stride
   across the full split (matching `test_v4_production_checkpoint.py`'s own
   established convention), after which the same 15-scenario smoke test
   immediately showed real, varied, non-degenerate numbers.

### Real results (300 scenarios, stride-sampled across the full validation split, 217.9s)

**Step-0 comparison** (all 5 policies, one sample each):

| policy | mean realized entropy reduction (bits) | agreement with classical EIG |
|---|---|---|
| `classical_eig` | -0.210 | 1.000 (reference) |
| `classical_plus_residual` | -0.242 | 0.830 |
| `learned_scout` | -0.219 | **0.567** |
| `random` | +0.007 | 0.180 |
| `fixed_order` | +0.015 | 0.123 |

**Multi-step operational trajectory** (`random`/`fixed_order`/
`classical_eig` only, budget 3 samples):

| policy | resolved within 1 | resolved within 2 | resolved within 3 | never resolved (of 300) |
|---|---|---|---|---|
| `classical_eig` | 0.637 | 0.683 | **0.697** | **91** |
| `fixed_order` | 0.580 | 0.603 | 0.610 | 117 |
| `random` | 0.580 | 0.593 | 0.610 | 117 |

Two genuinely different pictures, both real, reported as-is rather than
reconciled into one artificially tidy story:

- **On the operational metric that matters most for a promotion decision**
  (does the incident actually get resolved within a real sampling
  budget), classical EIG clearly beats random/fixed-order: 69.7% vs 61.0%
  resolved within 3 samples, and materially fewer never-resolved cases (91
  vs 117 of 300). This is the expected, reassuring result.
- **On the single-step realized-entropy-reduction metric, ALL THREE
  EIG-informed policies (classical_eig, classical_plus_residual,
  learned_scout) show a NEGATIVE mean, while the two naive policies
  (random, fixed_order) show a small POSITIVE mean.** This is genuinely
  counter-intuitive and was investigated, not ignored: checked whether
  classical EIG's picks frequently land on a node the scenario ALREADY has
  a real (possibly cleaner) sensor reading for -- `generate_scout_label`'s
  candidates legitimately include already-instrumented nodes (`build_
  signature_artifact_for_network`'s own documented design: "every junction
  is both a possible source and a possible sample location"), and
  `reveal_sample_measurement` always draws a fresh, independently-noised
  value regardless of whether a clean prior reading already exists there --
  a plausible confound. Checked directly against 60 real stride-sampled
  scenarios: only 6/33 (18%) of classical EIG's actual picks land on an
  already-instrumented node -- present, but not the dominant driver of a
  -0.21 mean. **Root cause not fully resolved this pass** -- flagged
  explicitly as a real open question (candidate hypotheses: the large
  hypothesis space real bins produce, versus the reduced-bin space earlier
  smoke tests used; possible interaction between `noise_scale_mg_l` and a
  many-hypothesis posterior's sensitivity to any single noisy reading) --
  not silently smoothed over into a falsely tidy narrative.

**`learned_scout`'s real, clean, positive finding, independent of the
entropy-sign question above**: agreement with classical EIG jumped from
6.7% (untrained model, smoke-tested against a randomly-initialized
`sample_node_head`) to **56.7%** (the real trained checkpoint) -- direct,
strong evidence the trained head learned a genuinely useful approximation
of classical EIG's own targeting behavior, not merely a lower training
loss in isolation (`core-issues3.txt`'s own explicit "do not promote a
learned Scout merely because its supervised loss decreases" warning is
addressed by this operational, not-loss-based, comparison).

### Verdict

**Not yet promotable to lead/replace classical EIG in production.**
`learned_scout` shows a real, positive, operationally-grounded competence
signal (56.7% agreement with the established classical baseline, up from
6.7% untrained) -- promising, and worth continuing to develop. But it
cannot currently be evaluated on the multi-step operational metrics that
matter most for a real promotion decision (median samples to resolution,
resolved-within-k over a real sampling budget) due to the documented
architectural gap (no revealed-evidence conditioning in HydroCore's Scout
input). **Concrete, scoped next step, not attempted this pass**: add
`already_sampled`/revealed-evidence conditioning to HydroCore's Scout
input representation, enabling a genuine multi-step learned-Scout
trajectory comparable to `classical_eig`'s own 69.7%-resolved-within-3
result -- a real architecture change, not a training-configuration change,
so scoped as Phase 12 follow-up work rather than attempted under continued
time pressure in this same pass.

Reproduce:

```bash
export PYTHONPATH=src
# 1. Train Scout's heads (~10-15 min):
python scripts/train_scout_heads.py

# 2. Run the real 5-policy comparison (~3.5 min for 300 scenarios):
python scripts/run_stage_d_scout_policy_comparison.py \
  --limit 300 --maximum-samples 3 --mode both
# (--scout-checkpoint to override the auto-detected checkpoint from step 1;
# --mode step0|trajectory|both; smaller --limit for a fast smoke test)
```

### Full suite, Ruff, Pyright

`ruff check src/ scripts/ tests/` clean throughout. Full `pytest` suite
re-run after this pass: **698 passed, 0 failed** (575s) -- up by exactly 1
from the post-Phase-12-Stage-B baseline (697), matching the one new test
this pass added (`test_reindex_to_signature_grid_is_usable_directly_with_
localize_with_signatures`).

## Phase 12 Stage E (core-issues3.txt "PHASE 12", Stage E: Strategist) -- DONE, with a significant real defect surfaced

Continuation within the same overnight autonomous run, immediately after
the Phase 12 Stage D work above -- picked up per this session's own
instruction ("begin work on phase 12 section E unless something was
deferred"; nothing was deferred, and Stage D's own "Next steps" already
named Stage E as the natural next real training step). Two commits, both
pushed to `origin/agent/gcp-multitopology-v3`:

1. `73699f7` feat(training): train Strategist heads for real (Stage E prep)
2. `daa947e` feat(training): Stage E Strategist-policy comparison script and real run

No work on `main`. Every protected artifact untouched. Locked test not
opened; `final-selection.json` does not exist.

### Prerequisite: train the candidate-conditioned Strategist heads for the first time (real run)

Same situation Stage D found for Scout: `plan_value_head`/`plan_validity_head`
are unconditional base heads already present in the Stage-A Sentinel
checkpoint, but Stage-A trained with the default `strategist_mode=
"anonymous_queries"` and the stale `action_vocabulary_size=8` default (the
exact 8-vs-9 mismatch `checkpoint_identity.py`'s own Section D documents),
so those heads never saw a real candidate-plan representation --
`candidate_plan_encoder`/`consequence_proxy_heads` did not exist in that
checkpoint at all. New `scripts/train_strategist_heads.py` (frozen
backbone, matching `train_scout_heads.py`'s established pattern) trains
all 4 Strategist heads on Phase 10.3's real `cycle-b2-trajectories-v3/
strategist-tensors-normalized` dataset (9000 train / 1000 validation
candidates -- every real validation scenario carries all 9 canonical
candidates, all exactly WNTR-verified per Phase 3.1's own repair). The
teacher checkpoint's `action_head` weights are deliberately dropped before
loading (`load_state_dict(strict=False)` after popping those keys) --
vocab-size-incompatible (8 vs 9) AND excluded from the v4 output vocabulary
regardless (`checkpoint_identity.py` Section D item 6: deterministic
candidate plans own action-template/target identity; the learned model
only ranks/validates/prescreens).

**Real result** (727.2s, `experiments/runs/v4-strategist-heads/
20260807T173109Z-bf941ddc/checkpoints/checkpoint-0010`, 10 epochs):

| metric | value |
|---|---|
| `plan_validity` accuracy | **0.996** |
| `plan_validity` precision / recall / F1 | 0.994 / 1.000 / 0.997 |
| `plan_value` MSE | 3.4e-5 |
| `exposure_proxy` / `pressure_risk_proxy` / `service_loss_proxy` / `containment_time_proxy` / `plan_regret_proxy` MSE | all ~3e-6 to 3e-5 |

`plan_validity` is real signal (WNTR genuinely rejects ~26% of the 9000
training candidates on pressure/service grounds, and this head learns that
boundary well). The near-zero regression MSEs are **not** evidence of a
well-learned value function -- see the next section.

### A significant real defect this stage surfaced (not fixed in this pass)

While building the Stage E policy comparison, the near-perfect regression
MSEs above looked suspicious enough to check directly rather than accept
at face value (this project's own established discipline: "independently
verifying a failing assertion's root cause before either dismissing it as
a test bug or reporting it as a real regression" -- carried forward here
to a suspiciously GOOD number instead of a failing one). Checked directly
against the full validation split, not by inspection:

```
total_valid 6629
exposure_nonzero 0        (0.0% of valid candidates)
pressure_nonzero 0
service_nonzero 0
containment_lt1 0          (containment_time_proxy is 1.0 -- "never contained" -- always)
scenarios_with_cost_variation 0 / 1000
```

**Every one of the 6629 valid candidates across all 1000 validation
scenarios has an IDENTICAL cost, hence an IDENTICAL `plan_value` of exactly
1.0.** `plan_value_policy.evaluate_plan_value`'s formula
(`plan_value = 1/(1+regret)`, `regret = cost - min(pool costs)`) is correct
and well-tested (`tests/scientific/test_plan_value_policy.py`'s own
monotonicity tests pass) -- the defect is upstream, in what `cost` is
computed FROM.

**Root cause** (traced to source, not guessed): `HydraulicSimulator.
evaluate_plan()` (`src/hydroswarm/simulation/wrapper.py:792-865`) --
the function `PlanVerifier.verify()` calls, which `generate_strategist_
labels` (`src/hydroswarm/training/strategist_labels.py`) uses for
Strategist training-label generation, AND which the live
`POST /api/incidents/{id}/plans/{id}/verify` endpoint
(`src/hydroswarm/api/app.py:639`) uses for real-time plan verification --
only runs `self._run_hydraulics(model)`, a **pure hydraulic** simulation
(pressure, demand, flow). It builds its returned `ConsequenceMetrics` from
only `minimum_pressure_m`/`pressure_violation_minutes`/`unserved_demand_l`/
`service_availability`/`operation_count`. It never calls
`calculate_exposure_consequences()`/`HydraulicSimulator.
calculate_consequences()` (`src/hydroswarm/simulation/consequences.py`,
`wrapper.py:652-681`) -- the water-quality-aware function this SAME module
already implements, tests (`tests/scientific/test_consequences.py`), and
uses elsewhere. `ConsequenceMetrics`' `contaminant_mass_consumed_mg`
(default `0.0`) and `containment_time_minutes` (default `None`) are simply
never set by `evaluate_plan()`, so every plan -- regardless of what it
actually does -- silently gets the Pydantic field defaults.

**This is not a training-corpus-only gap.** `hydroswarm.evaluation.golden.
py` (the demo/golden-fixture builder, line ~244) already works around this
exact same incompleteness by computing `calculate_exposure_consequences`
SEPARATELY and merging it in by hand -- informal evidence someone already
recognized `PlanVerifier.verify()` alone is insufficient, but the fix was
only ever applied in one demo-fixture builder, never in
`HydraulicSimulator.evaluate_plan()`/`PlanVerifier` itself. Both
`strategist_labels.py` (training) and `api/app.py`'s `/verify` endpoint and
`evidence_bundle()` (`app.py:746`, surfaced directly to operators via
`EvidenceBundle.consequence`) trust `verification.consequences` as-is, with
no such enrichment step. **The live, deployed system's operator-facing
plan "consequence" therefore never reflects real contamination-exposure
reduction -- only hydraulic pressure/service impact** -- for a product
whose entire mission is contamination-incident response. This is
independent of, and larger in scope than, Stage E's own narrow task.

**Why this was not fixed in this same pass**: this is a large,
cross-cutting, safety-adjacent change (touches the live plan-verification
path, not just training-label generation), requires a real design decision
(how does injecting a contamination-transport simulation into plan
verification interact with the plan's own hydraulic actions -- run both a
hydraulic-only pass, per current behavior, AND a chemical-transport pass
per candidate? at what cost in simulator time, given `PlanVerifier.verify`
already runs once per bounded candidate?), and would require regenerating
the ENTIRE `cycle-b2-trajectories-v3` Strategist corpus (Phase 3.1's
labels) once fixed -- well beyond "Phase 12 Stage E" scope and in tension
with "prefer bounded, reviewable changes over broad rewrites." Recorded
here as a concrete, high-priority follow-up instead
(see "Next steps" below), not silently smoothed over or hidden behind the
good-looking MSE numbers.

### Stage E policy comparison, honestly interpreted given the defect above

`scripts/run_stage_e_strategist_comparison.py` compares the 4 required
policies using `cycle-b2-trajectories-v3/strategist-tensors-normalized`'s
already-exact WNTR-verified candidate targets directly -- no re-simulation
needed, since Phase 3.1 already verifies the FULL bounded candidate set
unconditionally for every training-label scenario (see the script's module
docstring, "What 'simulator calls' means here", for the full, deliberate
scoping rationale). WNTR remains authoritative in every policy: every
final selection is always the ground-truth `plan_validity`/`plan_value` of
whichever candidate(s) that policy chose to "check," never a raw predicted
score.

**Real result** (1000 validation scenarios, 4.3s):

| policy | mean simulator calls | selected-valid rate | found non-`NO_ACTION` plan rate | mean regret vs. oracle |
|---|---|---|---|---|
| `exact_all` (oracle) | 9.0 | 1.000 | 0.000 | 0.0 (reference) |
| `deterministic_heuristic` | 3.0 | 1.000 | 0.996 | 0.0 |
| `learned_prescreen` | 3.0 | 1.000 | 0.944 | 0.0 |
| `learned_ordering` | **1.0** | 1.000 | 0.960 | 0.0 |

**What is and is not a real finding here, given the plan_value defect
above**:

- **`mean_regret_vs_oracle == 0.0` for every policy is NOT a real finding.**
  With every valid candidate's `plan_value` tied at exactly 1.0, regret is
  mechanically 0 for any policy that selects any valid candidate -- this
  metric currently has zero discriminating power and must not be read as
  "every policy finds the truly best plan." It will only become meaningful
  once the root-cause defect above is fixed and `plan_value` genuinely
  varies by candidate.
- **`exact_all`'s `found_non_no_action_plan_rate == 0.000` is a tie-break
  artifact, not evidence NO_ACTION is usually best.** `policy_exact_all`
  ranks all-tied-at-1.0 candidates by a stable sort over `generate_response_
  plans`' own original candidate order (`NO_ACTION` always first), and
  `_select_from_shortlist`'s strict `>` comparison never displaces a
  first-seen tied value -- so the oracle, as currently defined, always
  "prefers" `NO_ACTION` among ties. `deterministic_heuristic`/
  `learned_prescreen`/`learned_ordering` check candidates in THEIR OWN
  ranked order instead (the heuristic's own tie-break explicitly sorts
  `NO_ACTION` LAST, matching `prescreen_top_plans`' real production
  behavior), so they almost always find a non-`NO_ACTION` valid candidate
  first and keep it (nothing later in a strict-`>` scan can displace an
  already-tied-for-best pick). This is a real property of each policy's
  own ranking order, but says nothing about which plan is actually BEST
  until `plan_value` carries real signal.
- **What IS a real, meaningful finding: `first_checked_was_valid_rate` ≈
  0.996 for every policy, and `selected_valid_rate == 1.000` for every
  policy.** WNTR's pressure/service-availability constraints (the part of
  plan verification that DOES vary meaningfully -- `plan_validity`'s own
  0.996 accuracy/0.997 F1 above) are rarely binding for this corpus's
  candidate plans -- almost any bounded candidate already satisfies them.
  Combined with the fallback-to-`NO_ACTION` structure every policy shares,
  every policy here achieves perfect operational safety (never proposes an
  invalid plan) by construction, not because of ranking quality.
- **What IS a real, promising efficiency signal: `learned_ordering`
  achieves the same 100% valid-selection outcome as every other policy
  while checking only 1 candidate (vs. 3 for both `deterministic_heuristic`
  and `learned_prescreen`, 9 for the oracle) -- a genuine 3x-9x reduction
  in "exact simulations checked" for the CURRENTLY-measurable dimension
  (validity), consistent with core-issues3.txt Phase 13's own "reduce the
  number of exact simulations needed" Strategist goal.** This result is
  real and worth carrying forward, but its practical value is currently
  bounded by the same defect: it demonstrates the learned ranker can find
  A hydraulically-valid plan efficiently, not yet that it can find the
  MOST EXPOSURE-REDUCING valid plan efficiently -- that claim requires the
  root-cause fix above before it can be tested at all.

Reproduce:

```bash
export PYTHONPATH=src
# 1. Train Strategist's heads (~12 min):
python scripts/train_strategist_heads.py

# 2. Run the real 4-policy comparison (~5s for the full 1000-scenario validation split):
python scripts/run_stage_e_strategist_comparison.py --limit 1000
# (--strategist-checkpoint to override the auto-detected checkpoint from step 1;
# --limit 0 for the entire split; smaller --limit for a fast smoke test)
```

### Full suite, Ruff, Pyright

`ruff check src scripts tests` and `pyright src scripts tests` both clean.
Full `pytest` suite re-run after this pass: **698 passed, 0 failed**
(583.9s) -- unchanged from the post-Phase-12-Stage-D baseline (this pass
added no new pytest test files; Stage D's own scripts had none either --
these Stage training/comparison entry-point scripts are validated by real
execution + a smoke run, matching that established convention, not by
dedicated unit tests of the underlying model/dataset code, which already
has its own coverage).

## Restrictions honored

No work on `main`. `data/learning-v2/cycle-b2`'s existing contents, all
promoted checkpoints, and every existing v3 result artifact are untouched.
`data/learning-v2/cycle-b2-control-v2`, `cycle-b2-trajectories-v3`, and
`cycle-b2-ood-extension` are all new, separately-versioned artifacts:
nothing was overwritten in place. Locked test not opened;
`final-selection.json` does not exist. No destructive git/filesystem
commands used (the one deletion performed -- Phase 10.4's discarded buggy
first run -- was this same session's own untracked, never-committed
scratch output). No sudo, no credential exposure. All commits pushed to
`origin/agent/gcp-multitopology-v3`.

## Next steps (current, as of the completed Phase 12 Stage E pass)

**Most important open item, found this pass (see "Phase 12 Stage E"
above for the full root-cause writeup): `HydraulicSimulator.evaluate_plan()`
(`src/hydroswarm/simulation/wrapper.py:792-865`) never computes
contamination-exposure consequences -- only hydraulic pressure/service
impact.** `contaminant_mass_consumed_mg`/`containment_time_minutes` are
Pydantic-default (`0.0`/`None`) for every plan, verified directly across
all 1000 validation scenarios / 6629 valid Strategist candidates (zero
cost variation, anywhere). This affects BOTH the Strategist training
corpus's `plan_value`/consequence-proxy labels (Phase 3.2/3.3's own
targets, believed correctly populated since their introduction, are
provably degenerate) AND the live `/api/incidents/{id}/plans/{id}/verify`
endpoint's operator-facing consequence (`EvidenceBundle.consequence`,
`app.py:746`) -- the deployed system currently cannot report or rank
plans by real contamination-exposure reduction, only by pressure/service
safety. Recommended fix direction: wire
`calculate_exposure_consequences()`/`HydraulicSimulator.
calculate_consequences()` (already implemented, tested, and used by
`evaluation/golden.py`'s own demo-fixture builder as a hand-merged
workaround) into `evaluate_plan()` itself, which requires deciding how a
per-plan chemical-transport simulation is incorporated into plan
verification (today `evaluate_plan` runs one pure-hydraulic pass per
candidate; a contamination-aware version needs the incident's source
profile threaded through `PlanVerifier`/`HydraulicSimulator` construction
and a second, more expensive water-quality pass), then regenerating the
entire `cycle-b2-trajectories-v3` Strategist corpus once fixed. Not
attempted this pass -- large, cross-cutting, safety-adjacent, and well
beyond "Phase 12 Stage E" scope; flagged here as the highest-priority
concrete follow-up rather than fixed hastily or left implicit in a good-
looking MSE number.

**Phase 8, Phase 9 (core-issues4.txt Sections A-I), Phase 10
(core-issues4.txt Section I / core-issues3.txt Phase 10, all 5 items),
Phase 11 (all 5 items), and Phase 12 Stages B/C/D/E are all fully DONE.**
See "core-issues4.txt continuation pass, part 2" above for the Section H
stop-gate checklist (all 16 items verified true), the "Phase 10" section
for its full item-by-item detail, "Phase 11" for 11.1-11.5, "Phase 12 Stage
B" for the real ablation results (`travel_time` is a clean win;
`sensor_reconstruction` needs a scale fix before it is a reliable
promotion candidate), "Phase 12 Stage D" for the real Scout-policy
comparison (`learned_scout` shows real, positive, operationally-grounded
competence -- 56.7% agreement with classical EIG, up from 6.7% untrained --
but is not yet promotable pending a real architecture change to support
multi-step evaluation; classical EIG clearly beats random/fixed-order on
the operational resolved-within-3-samples metric, 69.7% vs 61.0%), and
"Phase 12 Stage E" immediately above this one for the real Strategist-
policy comparison and the exposure-blind-consequence defect it surfaced
(`learned_ordering` finds a hydraulically-valid plan with 1 simulator call
vs. 3-9 for the other policies -- real and promising on the currently-
measurable validity dimension, but regret/value comparisons are not yet
meaningful pending the fix above). Phase 10 summary (unchanged from the
prior pass):

1. Regenerate the final non-provisional trajectory corpus -- **DONE**, all
   4 splits, 13150/13150 real scenarios processed (400 coastal-branch
   correctly skipped as unsupported topology), 0 errors throughout.
2. Build sharded Scout-state datasets and collators -- **DONE**, full
   `train`+`validation` scale, real end-to-end gradient proof. Found and
   fixed a real 3-instance-deep "hand-maintained list drifts from its
   source of truth" defect in `variable_collate.py`/`permutation.py`.
3. Build sharded Strategist-candidate datasets and collators -- **DONE**,
   full `train`+`validation` scale, real end-to-end gradient proof -- the
   candidate-conditioned architecture's first real training data in this
   project's history.
4. Generate the balanced supported-category OOD extension -- **DONE**,
   full scale (1600 real, individually-verified scenarios across the 4
   remaining reproducible categories). Found and fixed two real defects
   (a `CurriculumStage.CLEAN` config-override-zeroing gap, and a missing
   `ood_class_target` attachment) by actually running the generator, not
   by inspection.
5. Run real multi-topology gradient/training smoke tests for every
   retained v4 output -- **DONE**. Every retained v4 output (per
   `checkpoint_identity.py`'s Section D decisions) now has at least one
   real, committed, passing test proving a nonzero gradient from real
   multi-topology data.

Still open, correctly sequenced after Phase 10 (no promoted v4 checkpoint
exists yet to wire a runtime path to, and Phase 10's full-scale Scout/
Strategist datasets feed directly into any future architecture-selection
decision):

- **A full joint-multitask training run** (core-issues2.txt Phase 9's own
  staged-training plan) using the now-complete, full-scale Scout/
  Strategist/OOD-extension datasets -- explicitly NOT started this pass;
  this remains a real, substantial next step, not a formality. The
  datasets exist and are proven trainable; nothing has actually been
  trained on them at scale yet (only stride-sampled smoke batches).
- **Wire `output_governance`/checkpoint-identity into a live v4 runtime
  path**: `runtime/defaults.py`/`inference/pipeline.py` still only
  understand the v3 `trained_tasks` role-level gating.
- **Section F's optional low-LR joint fine-tune** for the control heads,
  only if a future evaluation pass materially improves
  `unsafe_non_abstention_count`/`INSPECT_FAULTY_SENSOR` recall enough to
  justify it -- not attempted this pass, see the honest-limitation note
  above.
- **`class_weights` (Phase 11.2) exist but are still not applied to any
  real training run** -- 11.2's own text ("use... when justified") gates
  that on a real evaluation showing it's warranted; Stage B's ablation was
  scoped to auxiliary objectives specifically, not class balance.
- **`sensor_reconstruction`'s target-scale fix** (log1p or similar,
  matching Phase 7.5's travel_time precedent) -- concrete, scoped,
  recommended by Stage B's real results but not attempted this pass.
- **HydroCore's Scout input needs `already_sampled`/revealed-evidence
  conditioning** before `learned_scout` can be evaluated (or trained) on
  genuine multi-step trajectories -- a real architecture change, not a
  training-configuration change; the concrete, scoped follow-up Stage D's
  own results point to.
- **The Stage D single-step realized-entropy-reduction sign anomaly**
  (classical_eig/classical_plus_residual/learned_scout all NEGATIVE mean,
  random/fixed_order both slightly positive) is a real, investigated-but-
  not-fully-explained open question -- checked and ruled out as the
  dominant cause: classical EIG re-sampling an already-instrumented node
  (only 18% of picks, in a real spot-check). Root cause still open; the
  multi-step operational metrics (which DO clearly favor classical EIG)
  are the more decisive signal in the meantime.
- **Phase 12 Stages F-G** (F: joint fine-tuning across roles; G:
  HydroCore-M, conditional on a measured capacity-limited case) -- Stage E
  (Strategist) is now DONE, see above. Stage F is the natural next step,
  but its value is currently bounded by the same plan-value defect: a
  joint fine-tune that includes the Strategist's plan_value/proxy losses
  would be tuning against a target with zero real variance until the
  root-cause fix lands.
- Phases 13-20 (required metrics/baselines, promotion gates, runtime
  integration, corpus gates, CI, artifact governance, architecture
  selection, locked-test boundary) not started.

A recurring pattern worth carrying forward, extended again this pass:
every phase touched so far has surfaced at least one real,
previously-unobserved defect only visible once exercised at real scale or
with a genuinely adversarial test case, not by inspection alone. This
pass's own instances: the raw-vs-normalized tensor-variant mismatch caught
by actually running `gate_normalization_ownership`; the
`CheckpointIdentity.fingerprint` bound-method-comparison bug in a new
test's own first draft, caught by an independent `dataclasses.asdict`
comparison before concluding it was a real defect rather than a test bug;
the third instance of the "hand-maintained list drifts from its source of
truth" defect class, in two DIFFERENT modules (`variable_collate.py` and
`permutation.py`), found only once a real per-node/per-plan target was
actually collated or permuted; the `CurriculumStage.CLEAN`
`missing_probability`-zeroing gap and the missing `ood_class_target`
attachment, both found only by actually running the OOD-extension
generator at real scale and writing a real test for its output, not by
reading the code. Continue prioritizing real-scale/real-data testing over
unit tests with synthetic arrays, independently verifying a failing
assertion's root cause before either dismissing it as a test bug or
reporting it as a real regression, and treating every "two things that
must agree" relationship in this codebase as a candidate for exactly this
drift class until it is derived from one shared source.

## important-issues.txt emergency fix pass — exposure-blind plan-verification defect

Stop-gate context: this pass was launched specifically to fix the defect
flagged as the highest-priority pre-freeze blocker at the end of the "Phase
12 Stage E" section above, per `important-issues.txt`'s explicit instruction
to complete it **before** resuming `core-issues3.txt` Stage F / joint
multitask training / architecture selection. Stage F has **not** been
started. Locked test not opened; `final-selection.json` does not exist.

### What was fixed (code — done, tested, committed, pushed)

Commit `96d945f` on `agent/gcp-multitopology-v3` (pushed to
`origin/agent/gcp-multitopology-v3`). Root cause, exactly as diagnosed at
the end of Stage E: `HydraulicSimulator.evaluate_plan()`
(`src/hydroswarm/simulation/wrapper.py`) ran a pure-hydraulic pass only —
`contaminant_mass_consumed_mg`/`containment_time_minutes` were Pydantic
defaults for every plan, so `plan_value` was mechanically tied at 1.0 for
every valid candidate across the entire Strategist corpus, and the live
`/verify` endpoint's `EvidenceBundle.consequence` never reflected real
contamination exposure.

Fix, matching `important-issues.txt` requirements 1-13:

- `HydraulicSimulator.simulate_incident_plan(plan, incident_profile, ...)`
  (requirement 2): copies the network, applies the plan's actions, injects
  the incident source into that SAME modified network, runs ONE EPANET
  chemical-transport pass. Shares `_inject_incident_sources`/
  `_build_incident_simulation` with `simulate_hypothesis` (requirement 2's
  "share the source-injection/EPANET implementation rather than
  duplicating it") and never runs an independent hydraulic-only pass
  followed by an unrelated chemical pass (requirement 8).
- `PlanEvaluationContext`/`WeightedSourceHypothesis` (requirement 5):
  explicit evaluation assumptions — a single ground-truth `source_profile`
  (training) or a bounded (`MAXIMUM_EVALUATION_HYPOTHESES = 3`, requirement
  7) `hypotheses` credible set (runtime), contamination threshold,
  population map, aggregation policy, consequence-policy version — carried
  alongside the plan rather than hidden inside `evaluate_plan()`'s own
  defaults.
- `evaluate_plan_consequences()`/`PlanExposureEvaluation` (requirements 3,
  4, 7): builds the complete `ConsequenceMetrics` (exposure AND
  pressure/service, from the SAME EPANET run) via the existing
  `calculate_consequences()`/`calculate_exposure_consequences()`.
  `PRESSURE_BELOW_MINIMUM`/`SERVICE_BELOW_MINIMUM` are derived per-
  hypothesis from exact simulator results and **unioned** across every
  evaluated hypothesis (requirement 4: a benign hypothesis can never
  average away real hydraulic safety authority). Reports both
  `posterior_weighted` and `worst_case` aggregates when hypotheses are used
  (requirement 7), never presenting either as ground truth.
- `ConsequenceMetrics.exposure_evaluated` (requirement 12): `False` on the
  preserved legacy hydraulic-only `evaluate_plan()` path so its Pydantic-
  default exposure fields (still used when no incident context exists at
  all) can never be read as measured; `True` whenever
  `calculate_exposure_consequences()` actually ran.
- **A real bug this pass's own first real run surfaced, not designed
  around in advance**: passing the full node set into
  `calculate_exposure_consequences` included reservoir/tank nodes, whose
  WNTR/EPANET "pressure" is their fixed head (~0 m by construction, not a
  real violation) — this silently rejected every plan, including
  `NO_ACTION`, the moment real execution was tried. Fixed once, centrally,
  in `HydraulicSimulator.calculate_consequences` (filter to
  `junction_name_list`, matching `evaluate_plan()`'s pre-existing pressure
  filter and `evaluation/golden.py`'s independently-discovered same fix,
  now unified into one place).
- Plan-evaluation cache key now includes profile/hypothesis-set identity,
  contamination threshold, population-map identity, and consequence-policy
  version (requirement 9) — was `profile=None` unconditionally.
- `evaluation_provenance["exact_simulation_count"]` records the real number
  of NEW EPANET runs one `verify()` call consumed (requirement 10) — up to
  4 for a 3-hypothesis runtime verification (1 cached baseline-demand run +
  up to 3 hypothesis runs), never silently 1. **Operational consequence
  worth flagging explicitly**: `ApiSettings.exact_plan_simulation_limit` is
  schema-capped at 3 (`le=3`) and is spent per-incident, not per-verify-call
  — a single 3-hypothesis runtime verification can now exhaust an
  incident's entire lifetime exact-simulation budget in one `/verify` call.
  This is the honest, correct behavior per requirement 10 (not a bug), but
  the operational budget policy (is 3 still the right per-incident limit
  now that one verification can cost up to 3-4 real simulations instead of
  1?) was not re-tuned this pass — flagged as a concrete follow-up, not
  silently absorbed.
- Training (`generate_strategist_labels`/`build_strategist_trajectory`,
  requirement 6): uses the scenario's exact ground-truth `IncidentTruth`
  converted to `IncidentSourceProfile` via the new
  `incident_truth_to_source_profile()` (factored out of, and now shared
  with, `simulate_all_node_truth`'s identical inline conversion — one
  source of truth, not two that could drift), using the exact reconstructed
  randomized scenario network. Never derives exposure labels from inferred
  source probabilities.
- `evaluation/golden.py`'s hand-merged `calculate_exposure_consequences`
  workaround (informal prior evidence, per Stage E's write-up, that someone
  already knew `PlanVerifier.verify()` alone was insufficient) is removed;
  the golden fixture now calls the same canonical evaluator as everything
  else (requirement 11).
- Live `/verify` (`api/app.py`, requirement 12): `_runtime_evaluation_context`
  builds a bounded hypothesis set from the incident's own **calibrated**
  `CandidateSet` (top 3 by probability; returns `None` — hydraulic-only
  fallback, `exposure_evaluated=False` — when candidates are absent or not
  yet calibrated, never a fabricated zero). Each hypothesis uses
  `IncidentSourceProfile`'s own governed default timing/strength
  (start=0min, duration=60min, strength=10 mg/min — the same baseline
  `evaluation/golden.py`'s fixture uses) at the candidate node: this
  codebase does not yet surface a validated per-node start-time/duration/
  relative-strength prediction to the API layer (core-issues4.txt's output-
  governance work has not promoted those Sentinel heads into
  `runtime_enabled_outputs`), so only node identity and calibrated
  probability vary between hypotheses — real, calibrated evidence, never
  presented as certain.
- `EvidenceBundle.exposure_reduction_mg` (requirement 13): computed for
  real from the exact verified plan vs. the exact `NO_ACTION` comparator
  (structural identity — a plan whose only action is `END_PLAN`, matching
  `planning/response.py`'s own template, not a name-string convention) —
  only when both sides have `exposure_evaluated=True`.

### Tests (requirement 18)

`tests/scientific/test_exposure_aware_plan_evaluation.py` (12 tests):
NO_ACTION canonical evaluator matches standalone `simulate_hypothesis` +
`calculate_consequences` (rel=1e-3: `simulate_incident_plan` runs on
`_prepared_network()`'s PDD demand mode, `simulate_hypothesis` on a plain
deepcopy — both deliver full demand on this fully-supplied fixture network,
so results agree to float32 solver precision, not bit-for-bit); a known
flush plan changes exposure relative to `NO_ACTION`; the same plan under
different source profiles has distinct exposure; profile-specific and
threshold-specific cache isolation; pressure/service rejection still
derives from the exact simulator; simulation failure abstains;
posterior-weighted vs. worst-case hypothesis aggregation; the legacy
hydraulic-only path marks `exposure_evaluated=False` rather than a measured
zero; `PlanEvaluationContext`'s own invariants (max 3 hypotheses, exactly
one of profile/hypotheses).

`tests/integration/test_live_exposure_verification.py` (2 tests): the
real (non-injected-verifier) `/api/incidents/{id}/plans/{id}/verify`
endpoint returns real, nonzero exposure with a real `evaluation_provenance`
when candidates are calibrated, and falls back to
`exposure_evaluated=False`/no fabricated worst-case/no provenance when they
are not.

`tests/e2e/test_golden_scenario.py`: strengthened to assert
`exposure_evaluated=True` on both the golden fixture's `no_response` and
`safe` verifications, proving the fixture uses the canonical evaluator, not
just that its old hand-computed numbers still happen to match.

Full suite **712 passed, 0 failed** (up from the pre-pass 698 baseline: +14
new tests, 0 regressions). `ruff check src scripts tests` and `pyright src
scripts tests` both clean.

### `scripts/run_strategist_corpus_gates.py` (requirement 15)

New structural non-degeneracy gates over a `cycle-b2-trajectories-vN`-style
corpus directory: `plan_value_variance`, `exposure_variance`,
`per_scenario_cost_variation`, `no_action_not_universally_identical`,
`some_valid_plan_improves_exposure` — plus plan_value/exposure distribution
reporting by action_template and by split (not pass/fail, required
reporting). **Verified against real data in both directions, not just by
inspection**: run against the existing (defective)
`data/learning-v2/cycle-b2-trajectories-v3/validation.jsonl`, it correctly
**fails 4/5 gates** and reproduces Stage E's own numbers exactly
(`plan_value_variance=2.79e-25`, `exposure_variance=0.0`, `6629` valid
candidates, `0/1000` scenarios with cost variation) — proof the gates
detect the real defect, not a synthetic one. Run against a freshly
regenerated 8-scenario smoke sample built through the corrected pipeline
(see below), it **passes 5/5** with real nonzero variance
(`plan_value_variance=8.3e-4`, `exposure_variance=3.69e15`,
`any_valid_plan_beats_no_action=True`).

```bash
export PYTHONPATH=src
python scripts/run_strategist_corpus_gates.py \
  --trajectory-dir data/learning-v2/cycle-b2-trajectories-v4 \
  --splits train validation calibration development_holdout \
  --report reports/results/v4/strategist-corpus-gates-v4.json
```

### Corpus regeneration — IN PROGRESS, resumable background jobs

`data/learning-v2/cycle-b2-trajectories-v3` is left completely untouched
(important-issues.txt restriction: "existing v3/v4 result artifacts
immutable"). The corrected corpus is being written to the new,
separately-versioned `data/learning-v2/cycle-b2-trajectories-v4/`
(requirement 14: "Create a new versioned Strategist artifact ... unless the
current artifact layout genuinely requires a full trajectory-version
bump" — no schema/field changes, only corrected VALUES, so the existing
`generate_trajectory_corpus.py`/`build_strategist_trajectory` pipeline
needed no changes beyond this pass's library fix already threading
`incident_truth` through).

**Real measured cost**: an 8-scenario smoke run (`--limit 8`) took ~2.25
s/scenario of real processing time (excluding one-time signature-artifact
fitting), noticeably slower than the ~0.5-1 s/scenario the Phase 0 audit
measured for the OLD (hydraulic-only) path — expected, since every
candidate plan now runs a full EPANET chemical-transport pass instead of a
quick WNTRSimulator hydraulics-only pass. At that rate, full regeneration
(13,150 scenarios across 4 splits) is projected at roughly **8 hours** of
real compute.

Launched as 4 parallel resumable background jobs (one per split, on this
16-vCPU host) rather than one 8-hour sequential job:

```bash
export PYTHONPATH=src
for split in validation calibration development_holdout train; do
  python scripts/generate_trajectory_corpus.py \
    --corpus-dir data/learning-v2/cycle-b2 \
    --output data/learning-v2/cycle-b2-trajectories-v4 \
    --split "$split" \
    > "experiments/jobs/cycle-b2-trajectories-v4/${split}.log" 2>&1 &
done
wait
```

Status/logs: `experiments/jobs/cycle-b2-trajectories-v4/{split}.log`.
**Resumable**: `generate_trajectory_corpus.py` skips any `scenario_id`
already present in `{split}.jsonl` on restart — if interrupted, re-running
the exact command above for the affected split(s) continues from where it
left off, no `--limit`/flags needed to resume. Being polled at the
requested 10-minute interval while other independent work continues.

### Not yet done (blocked on the regeneration above, or scoped after it)

Per important-issues.txt's own numbering:

- **14 (corpus regen)**: in progress, see above.
- **15 (structural gates)**: gate script done and validated in both
  directions (above); has not yet been run against the FULL regenerated
  v4 corpus (only the 8-scenario smoke sample) because the full
  regeneration is still running.
- **16 (retrain Strategist from clean Stage-A foundation)**: not started —
  correctly sequenced after 14/15 pass on the real full corpus, per
  important-issues.txt's own STOP GATE (requirement 20). `scripts/
  train_strategist_heads.py` exists from Stage E prep and is the right
  entry point; it will need pointing at
  `cycle-b2-trajectories-v4/strategist-tensors-normalized` once `scripts/
  build_strategist_candidate_dataset.py` has been re-run against the new
  corpus (that script itself needed no changes — it only reads
  `labels`/`targets` already produced upstream by the now-fixed
  `build_strategist_trajectory`).
- **17 (rerun Stage E)**: not started, same dependency.
- Requirement 20's STOP GATE is therefore **not yet satisfied** — Stage F
  / joint multitask training must not resume until 14-17 complete. This
  pass has not touched Stage F, architecture selection, the locked test, or
  `final-selection.json`.

### Exact continuation commands

If the background jobs above are no longer running (session restart, host
reboot, etc.), first check for partial progress and resume in place (no
data is lost or needs to be discarded — the script's own resume-by-
scenario-id logic handles this):

```bash
export PYTHONPATH=src
wc -l data/learning-v2/cycle-b2-trajectories-v4/*.jsonl  # progress so far
# Re-run the exact 4-job loop above; each job resumes its own split.
```

Once all 4 `{split}.jsonl` files reach their target counts (train=9000,
validation=1000, calibration=1000, development_holdout=2550 minus
unsupported-topology skips, matching v3's own real counts):

```bash
export PYTHONPATH=src
# 1. Structural non-degeneracy gates (requirement 15) -- must pass before training:
python scripts/run_strategist_corpus_gates.py \
  --trajectory-dir data/learning-v2/cycle-b2-trajectories-v4 \
  --splits train validation calibration development_holdout \
  --report reports/results/v4/strategist-corpus-gates-v4.json

# 2. Build the sharded candidate-conditioned tensors (needs the base
#    ShardedScenarioDataset tensors -- reuse cycle-b2-trajectories-v3's own
#    scout-tensors-normalized/strategist-tensors-normalized generation
#    command, pointed at the v4 trajectory JSONL and a NEW v4 tensor
#    output dir):
python scripts/build_strategist_candidate_dataset.py \
  --tensor-shard-dir data/learning-v2/cycle-b2/tensors/train \
  --trajectory-jsonl data/learning-v2/cycle-b2-trajectories-v4/train.jsonl \
  --output data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized/train \
  --split train
# (repeat --split validation/calibration/development_holdout against their
# own tensor-shard-dir/trajectory-jsonl pair)

# 3. Retrain Strategist heads from the clean Stage-A Sentinel/v4 foundation
#    (requirement 16 -- NOT the current degenerate-value Stage E checkpoint).
#    --teacher-checkpoint's own default already points at the Stage-A
#    Sentinel/v4 checkpoint (experiments/runs/v4-stage-a-sentinel/...), and
#    a fresh --run-root/--registry below means CandidatePlanEncoder and the
#    value/proxy heads are reinitialized, not resumed from Stage E's
#    degenerate-value checkpoint:
python scripts/train_strategist_heads.py \
  --corpus-root data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized \
  --run-root experiments/runs/v4-strategist-heads-v4corpus \
  --registry experiments/registry/v4-strategist-heads-v4corpus.jsonl \
  --output reports/results/v4/strategist-heads-training-v4corpus.json

# 4. Rerun the Stage E 4-policy comparison (requirement 17) against the
#    corrected checkpoint and corpus:
python scripts/run_stage_e_strategist_comparison.py \
  --strategist-checkpoint <path from step 3> --limit 0
```

This report will be updated again once the regeneration completes and
steps 1-4 above have actually been run (not merely planned).
