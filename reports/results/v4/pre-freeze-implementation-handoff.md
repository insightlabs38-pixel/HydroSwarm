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
| 7 | Auxiliary objectives / regression losses | **DONE** (7.1/7.2/7.3/7.4/7.5 done + tested; 7.6/7.7 scoped and deferred, see below) |
| 8 | Second-pass calibrated control targets | **step 6 PARTIAL** (steps 1-5, 7, 8, 9 done; step 6a -- per-scenario label persistence -- DONE and run against real data this pass; step 6b -- merge into a governed corpus + train control heads -- scoped as a documented, immediately-resumable follow-up, see "core-issues4.txt continuation pass" below) |
| 9 | Architecture v4 contract | **DONE for Sections A-E** (executable v4 checkpoint identity, granular output governance, head retain/demote decisions, candidate/vocabulary contract -- see "core-issues4.txt continuation pass" below; Section F step-6b/G training run/H full adversarial sweep remain, see Next steps) |
| 10-20 | Datasets/collation, loss config, staged training, metrics, promotion gates, runtime integration, corpus gates, CI, artifact governance, architecture selection, locked-test boundary | not started |

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

## Restrictions honored

No work on `main`. `data/learning-v2/cycle-b2`'s existing contents, all
promoted checkpoints, and every existing v3 result artifact are untouched.
Locked test not opened; `final-selection.json` does not exist. No destructive
git/filesystem commands used. No sudo, no credential exposure. All commits
pushed to `origin/agent/gcp-multitopology-v3`.

## Next steps (current, as of the core-issues4.txt continuation pass)

Phase 9 Sections A-E are DONE (see that section above). Phase 8 is DONE
through step 6a (per-scenario label persistence); step 6b (merge + train)
remains, exact resume commands in that section above. core-issues4.txt's
own stop-gate checklist (section H) status:

- [x] `ARCHITECTURE_VERSION_V4` defined for new v4 models (legacy
      `ARCHITECTURE_VERSION` untouched)
- [x] legacy v3 remains loadable through a separate explicit path
- [x] v4 identity reconstructs every behavior-critical field
- [x] v4 checkpoints persist and verify all schema/policy/artifact hashes
- [x] trained/validated/runtime output sets are implemented
      (`output_governance`) -- NOT yet wired into a live runtime path
      (no promoted v4 checkpoint exists to wire it to)
- [x] action vocabulary is canonical and nine-class where relevant
- [x] candidate-conditioned Strategist no longer depends on anonymous
      position (was already true from the prior pass; Section E hardened
      its input validation)
- [x] missing Scout heads exist and receive losses
- [x] orphaned outputs are removed or explicitly demoted (by vocabulary
      omission, documented in checkpoint_identity.py's Section D writeup)
- [x] second-pass control labels are persisted per scenario
- [ ] the corrected control corpus passes gates -- NOT DONE (merge script
      not yet written, Section F's remaining half)
- [ ] control-head training completes on real data -- NOT DONE (blocked
      on the merge above)
- [x] full tests, Ruff and Pyright pass (639 passed, both clean)
- [x] working tree is clean and commits are pushed
- [x] locked test remains unopened
- [x] `final-selection.json` does not exist

**Honest assessment**: Phase 9 is complete through Section E; the
continuation's own instructions frame Section F step 6b (train + persist
per-scenario, corpus merge, control-head training) and the full adversarial
test sweep in Section H as one continuous requirement before Phase 9 can be
called fully done. Step 6a is real, tested, and run against real data;
6b (the corpus merge + actual training run) is scoped and NOT attempted,
for the same reason this exact deferral happened once already in the prior
pass -- see Phase 8's section above for why, and for the exact commands to
resume it.

Remaining work, roughly in priority order:

1. **Phase 8 step 6b / Section F remainder** (deferred, not skipped, exact
   resume commands in that section above): write
   `scripts/merge_second_pass_control_labels.py`, run corpus gates on the
   result, train `event_control_heads=True` control heads (frozen backbone
   first), record real evidence-sufficiency/next-step/policy-agreement/
   unsafe-non-abstention metrics.
2. **core-issues4.txt Section I / core-issues3.txt Phase 10**: regenerate
   the final non-provisional trajectory corpus incorporating all Phase 1-9
   corrections; build sharded Scout-state and Strategist-candidate
   datasets/collators (the `scout_control_heads`/candidate-conditioned
   architecture landed this pass now has real heads to train, but still no
   dataset/collator wiring real data into them); generate the balanced
   supported-category OOD extension (Phase 6.3's still-open follow-up);
   run real multi-topology gradient/training smoke tests for every
   retained v4 output. Not started.
3. **Wire `output_governance`/checkpoint-identity into a live v4 runtime
   path**: `runtime/defaults.py`/`inference/pipeline.py` still only
   understand the v3 `trained_tasks` role-level gating. This has no
   promoted v4 checkpoint to point at yet (Phase 9's own training/
   selection work is not done), so building the v4 runtime path is
   naturally sequenced after step 1-2 above, not before.
4. Phases 11-20 (loss-system config, staged training/ablations, required
   metrics, promotion gates, CI, artifact governance, architecture
   selection, locked-test boundary) not started.
5. A recurring pattern worth carrying forward, extended this pass: every
   phase touched so far has surfaced at least one real,
   previously-unobserved defect only visible once exercised at real scale
   or with a genuinely adversarial test case, not by inspection alone --
   this pass's own new instance is Section E's padded-plan
   `CandidatePlanEncoder` crash (`IndexError: index out of range in
   self`), caught by a test that was expected to pass on first write and
   didn't. Continue prioritizing real-scale/real-data testing over unit
   tests with synthetic arrays, and re-checking a "no X exists" claim
   against the WHOLE codebase before committing it, not just the one
   module that prompted the question.
