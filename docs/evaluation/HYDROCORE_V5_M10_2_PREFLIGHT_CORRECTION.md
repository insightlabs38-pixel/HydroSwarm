# HydroCore-v5 Milestone 10.2 preflight / correction pass (before any M10.2 scientific evaluation)

Amends nothing in `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`, which remains frozen and unmodified. That
document's Section 2 already anticipated that Scout's schema-integration gap "is out of scope for M10.1;
M10.2 ... must first resolve this gap" and its Section 10 already anticipated that M10.2 "will likely need a
preflight-correction pass ... before a scientific comparison is even executable, not merely a data-generation
exercise" -- this document is exactly that pass, in the precedent set by
`docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md`. It does not reopen M9 or M10.0/M10.1 and does not
change any of their numeric results, checkpoints, or artifacts.

**This document runs the M10.2 PREFLIGHT/CORRECTION only. It does not execute the M10.2 scientific Scout
comparison, does not preregister M10.2 promotion thresholds, and does not proceed to M10.3/M10.4/M10.5.**

## Why M10.2 could not safely begin directly

Two independent conditions had to both hold for a scientific Scout comparison to be executable and valid:

1. A versioned, leakage-safe, well-defined evaluation-time state schema and output-decoding contract needed to
   exist, mapping a real sequential decision state onto `HydroCore.forward()`'s raw Scout heads and back onto a
   `HydroScout.deterministic_fallback`-comparable recommendation. Prior state: `checkpoint_identity.py`'s
   `SCOUT_STATE_SCHEMA_VERSION = "scout-state-v1-unbuilt"`, and no canonical masking/decoding helper existed
   anywhere in the repository -- every historical script that touched raw Scout heads
   (`scripts/run_stage_d_scout_policy_comparison.py`) invented its own ad hoc handling.
2. The specific, frozen, M10-selected checkpoint (M9.6 `ARM_B_M9_6` `FINAL_STEP_1350`, per the M10 protocol's
   own Section 1) needed to have actually trained the four raw Scout heads it structurally contains.

This preflight audited both. (1) was a real, closeable gap -- built below. (2) is **not** closeable within this
task's constraints (no retraining is permitted), and turned out to be false for the real, selected checkpoint.
That second finding is the dominant blocker and is reported distinctly, per this task's own instruction to do
so when a defect affecting experiment design is discovered.

## Finding A (closed by this pass): no evaluation-side Scout state/output contract existed

Confirmed by direct forward-pass audit (`hydroswarm.model.core.HydroCore.forward`, Scout-head section, lines
~1250-1279):

- `sample_node_logits`/`expected_information_gain` are already masked by `node_mask` (real vs. padding) inside
  `forward()` itself, using the exact same `torch.finfo(dtype).min` (logit) / `0.0` (regression) convention.
  `candidate_reduction_prediction` is masked the same way when `scout_control_heads=True`.
  `should_continue_sampling_logits` is a single incident-level scalar (from `incident_context`), not per-node.
- Node index `i` in every one of these tensors already maps deterministically to physical node `i` in
  `node_mask`/every other per-node `HydroBatch` channel -- no reordering happens anywhere between a batch's own
  node ordering and the Scout heads' output ordering (traced through `role_hidden["scout"] = adapters["scout"]
  (hidden)`, itself built directly from `hidden`, which is never permuted after `node_features` establishes the
  node axis).
- What `forward()` does **not** do: exclude already-sampled nodes from candidacy (it has no notion of
  "already sampled" -- only real-vs-padding), or define what a caller should do with the four raw outputs
  afterward. That is legitimately downstream of the model, and nothing in the repository defined it.

**Correction**: `hydroswarm.evaluation.scout_state` (new module), schema version `scout-eval-state-v1` --
deliberately a **separate, evaluation-adapter-only** version from `checkpoint_identity.py`'s
`SCOUT_STATE_SCHEMA_VERSION`, which remains `"scout-state-v1-unbuilt"` and is **not changed by this pass** (see
"What was intentionally NOT changed" below; that placeholder names a training-corpus dataset layout, a
different and still-genuinely-unbuilt thing). Provides:

- `ScoutEvaluationState`: a frozen, validated dataclass wrapping a per-decision-step `HydroBatch` plus
  `already_sampled_mask`/`accessible_mask`/`sampling_round`/`sample_budget_remaining`, with per-batch-item
  `node_ids` so multi-topology batches (different node counts/identities per item, padded to a shared `nodes`
  dimension the same way `hydroswarm.training.variable_collate.collate_variable_topology` already pads
  variable-topology training batches) keep correct node identity per item.
- `build_scout_evaluation_state(...)`: the canonical adapter, taking only plain node-id sequences (no
  `ScoutLabel`/`ScoutTrajectoryStep`/targets object can even be passed to it -- see the leakage audit below).
- `apply_scout_candidate_mask(...)`: **the one canonical masking helper**, applied after the forward pass, that
  additionally excludes already-sampled/inaccessible nodes (which `node_mask` alone never encodes) from the
  three per-node outputs, using the exact same fill-value convention `HydroCore.forward()` itself already uses.
- `select_candidate_node(...)`: fail-closed argmax -- a batch item with no eligible candidate returns `None`,
  matching `HydroScout.deterministic_fallback`'s own "no unsampled accessible nodes -> STOP" behavior, rather
  than silently picking an arbitrary node.
- `assert_finite_scout_outputs(...)`: explicit finiteness check over the raw (pre-mask) tensors.
- `LearnedScoutRecommendation` / `decode_learned_scout_recommendation(...)`: packages a decoded recommendation
  with `promotable: bool = False` always set (a frozen field nothing downstream can flip), and a docstring
  giving the exact M10.2 output semantics for all four heads (task requirement 7):

  | Output | M10.2 semantics |
  |---|---|
  | `sample_node_logits` | masked candidate-ranking signal only -- never authoritative |
  | `expected_information_gain` | diagnostic-only (untrained -- see Finding B) |
  | `candidate_reduction_prediction` | diagnostic-only (same) |
  | `should_continue_sampling_logits` | diagnostic-only, incident-level (same) |

No item is currently "directly usable as a candidate ranking signal" in the sense of reflecting a trained
policy, no item is "potentially promotable" today, and no item is "currently runtime-disabled" in the
governance-flag sense -- all four are simply never populated with real supervision (Finding B).

### Leakage isolation (task's required adversarial audit)

`build_scout_evaluation_state` has no parameter through which a training-target object could be passed, and
`assert_no_target_only_keys` additionally fails closed if any `targets_v2`-governed target name (ground-truth
labels: `sample_node`, `information_gain`, `candidate_reduction`, `should_continue_sampling`,
`source_node`, ...) appears inside the `HydroBatch` mapping itself -- with one deliberate, verified exception:
`travel_time` is simultaneously a real `HydroBatch` **input** feature (`HydraulicFeatureBuilder`-populated
known network travel times) and a governed **target** name (the auxiliary `travel_time_prediction` task); a
blanket name-based block would incorrectly reject every real feature batch. This is the only such collision in
the repository today (`tests/unit/test_scout_evaluation_state.py::
test_target_only_key_set_excludes_every_real_hydrobatch_field`).

Adversarial tests (`tests/unit/test_scout_evaluation_state.py`) prove, not merely assert:

- an enormous raw logit at an ineligible (already-sampled/inaccessible/padding) node can never win the masked
  argmax;
- already-sampled nodes are never selected, checked through a real forward pass;
- each of the four Scout target names, if smuggled into a batch dict, raises `ScoutStateLeakageError` before
  reaching the model;
- an unrelated, undeclared extra key appended to a batch (simulating a careless future/ground-truth field) has
  zero effect on model output or the decoded recommendation;
- `decode_learned_scout_recommendation(...).promotable` is always `False`;
- `hydroswarm.inference.authority.scout_certificate`'s function signature accepts only `analysis:
  IncidentAnalysisResult` -- structurally, nothing in this new module can be wired into it to bypass
  deterministic Scout authority.

## Finding B (NOT closed by this pass -- the dominant blocker): the selected checkpoint's raw Scout heads were never trained

`scout_control_heads=True` (part of `m9_1_common.SHARED_MODEL_CONFIG`, used by every M9.6 checkpoint)
constructs `sample_node_head`/`information_gain_head`/`candidate_reduction_head`/
`should_continue_sampling_head` as real parameters, and `configs/training-v5-causal.yaml` declares a nonzero
`task_weights` entry for all four (`sample_node=1.0`, `information_gain=0.5`, `candidate_reduction=0.5`,
`should_continue_sampling=0.5`). Neither fact means the heads were trained.

`hydroswarm.training.causal_prefix.scenario_to_prefix_example` is the **sole** source of the `targets` dict for
every M9.6 training example (`CausalPrefixDatasetView.__getitem__` calls it directly;
`scripts/hydrocore_v5/run_m9_6_train_arm_b.py` uses `CausalPrefixDatasetView` completely unmodified, imported
from `run_m9_0_arm_b`/`run_m9_0a_arm_b2`). Its `targets = {...}` dict literal produces exactly these 15 keys,
confirmed both by reading the source and by calling the real function against a real generated scenario
(`tests/scientific/test_m10_2_scout_preflight.py::test_m9_6_training_corpus_never_included_scout_targets`):

```
duration, duration_mask, event_cause, event_presence, evidence_sufficiency, relative_strength,
relative_strength_mask, sensor_fault, sensor_fault_mask, source_node, source_node_mask, source_region,
source_region_mask, start_time, start_time_mask
```

`sample_node`, `information_gain`, `candidate_reduction`, and `should_continue_sampling` are never among them.
`hydroswarm.training.losses.compute_multitask_loss` only computes -- and therefore only backpropagates through
-- a task when `task in targets and output_name in outputs` (both its classification and regression loops use
exactly this guard, proven directly at the loss-function level by
`tests/unit/test_scout_evaluation_state.py::test_compute_multitask_loss_skips_a_task_absent_from_targets`).

**Net effect: `sample_node_head`, `information_gain_head`, `candidate_reduction_head`, and
`should_continue_sampling_head` hold their random initialization in every canonical `FINAL_STEP_1350` M9.6
checkpoint.** This is a different, upstream, and materially stronger defect than
`SCOUT_STATE_SCHEMA_VERSION = "scout-state-v1-unbuilt"` names -- that placeholder is about a training-corpus
*input*-conditioning dataset layout (see `hydroswarm.training.scout_state_contract`, which is about whether
future *retrain* input conditioning is representable, not about whether *current* output supervision happened).
Even a perfectly-built evaluation-time schema (Finding A, above) cannot make an untrained head's output
scientifically meaningful: comparing it against `HydroScout.deterministic_fallback` would measure random
projection noise against a real heuristic, not "learned vs. deterministic Scout policy."

**Independent corroboration (context only -- M10.1 is not reopened and its results are not altered here):**
`ood_class`/`ood_category` is absent from this exact same `targets` dict for the identical structural reason,
and `reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json` already recorded
`"neural_ood_category_auroc": 0.5078...` -- indistinguishable from chance, consistent with (though this
document does not depend on) an untrained head.

**A precedented fix pattern exists in this repository and is explicitly NOT applied here**:
`scripts/train_scout_heads.py`'s own docstring describes fixing a structurally identical historical gap in the
legacy Stage-A pipeline via a narrow, frozen-backbone, Scout-heads-only fine-tune. Applying that same pattern to
the M9.6-selected checkpoint would still be a real retrain of checkpoint weights -- explicitly forbidden by this
task ("Do NOT: retrain HydroCore-S ... tune Scout heads ... run new hyperparameter searches") and by the M10
protocol's own Section 1 ("never retrained ... without an explicit refit amendment"). It is recorded here only
as the concrete shape a future, separately authorized amendment would most likely take.

## Exact corrections made

- New module `src/hydroswarm/evaluation/scout_state.py`: `ScoutEvaluationState`, `build_scout_evaluation_state`,
  `apply_scout_candidate_mask`, `select_candidate_node`, `assert_finite_scout_outputs`,
  `assert_no_target_only_keys`, `LearnedScoutRecommendation`, `decode_learned_scout_recommendation`,
  `SCOUT_EVAL_STATE_SCHEMA_VERSION = "scout-eval-state-v1"`.
- New module `src/hydroswarm/evaluation/scout_readiness.py`: `M9_6_SCOUT_HEAD_AUDIT` (the Finding B record, as
  data), `m10_2_readiness(...)`, `M10_2_READY_FOR_SCIENTIFIC_EVALUATION`/`M10_2_PREFLIGHT_BLOCKED` constants.
- `src/hydroswarm/evaluation/__init__.py`: exports both new modules' public names.
- New script `scripts/hydrocore_v5/run_m10_2_preflight.py`: exercises the schema/adapter against a real forward
  pass through the real, SHA-256-verified, unmodified frozen M9.6 checkpoints (all 3 seeds), writes this
  document's four machine-readable artifacts.
- Tests: `tests/unit/test_scout_evaluation_state.py` (34 focused tests -- schema, masking, leakage, governance,
  authority-boundary, loss-guard mechanism), `tests/scientific/test_m10_2_scout_preflight.py` (real-scenario,
  `real_simulation`-marked: empirical corpus-target-key re-verification, end-to-end schema/adapter exercise
  against a real `HydraulicFeatureBuilder` batch).

## What was intentionally NOT changed

- `hydroswarm.training.checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION` remains `"scout-state-v1-unbuilt"` --
  that placeholder names a training-corpus sharded-dataset layout that genuinely still does not exist; changing
  it would misrepresent that M9.6 training used a real `ScoutState` object (it did not -- it used ordinary
  single-snapshot `scenario_to_prefix_example` batches with no Scout targets at all).
- `hydroswarm.training.scout_state_contract` (the input-channel-conditioning contract for a hypothetical future
  retrain) is untouched -- its own scope (input representability) is orthogonal to this pass's schema (decoding
  today's already-existing outputs) and to Finding B (output supervision never happened).
- No M9.6 checkpoint weight, `configs/training-v5-causal.yaml` task weight, or
  `hydroswarm.training.causal_prefix.scenario_to_prefix_example` target dict was modified. All three seeds'
  canonical `FINAL_STEP_1350` checkpoint SHA-256 hashes are verified unchanged (see the closure artifact).
- `hydroswarm.agents.scout.HydroScout.deterministic_fallback` is unmodified and was found to have no correctness
  defect during this audit (see "Deterministic baseline audit" below) -- no change reported distinctly, since
  none was needed.
- `hydroswarm.inference.authority.scout_certificate` is unmodified; deterministic Scout authority
  (`source="CLASSICAL_EIG"`, `AuthorityLevel.DETERMINISTIC`) is unaffected and structurally cannot be bypassed
  by anything in the new module (see the leakage/authority tests above).
- M9/M10.0/M10.1 numeric results, closures, and artifacts are unmodified. `reports/evaluation/hydrocore-v5/m9-6/`
  and `reports/evaluation/hydrocore-v5/m10/m10-0/`, `m10-1/` are untouched by this pass.
- No M10.2 scientific promotion threshold is preregistered here -- none of Finding A or B constitutes a
  development result to tune against; this document only decides whether the interface is executable, not what
  a future comparison's outcome should be judged against.

## Deterministic baseline audit

`hydroswarm.agents.scout.HydroScout.deterministic_fallback` (`src/hydroswarm/agents/scout.py`): given
`sampling_history` (already-sampled node ids), `candidate_probabilities` (a posterior over remaining
candidates), and `candidate_region`/`node_ids`, it excludes already-sampled nodes, ranks the remainder by
posterior probability (ties broken by node id, deterministic), and returns the top candidate with
`expected_information_gain = min(1.0, 1.0 / max(1, len(region)))` and a `STOP` action when that value falls
below `0.01` or no unsampled accessible node remains. No defect found -- it reads only current, already-revealed
state (`sampling_history`, `candidate_probabilities`, `candidate_region`), never any future or ground-truth
field, and fails closed (`STOP`) rather than fabricating a recommendation when nothing eligible remains. A
future M10.2 scientific comparison must supply BOTH policies the identical incident, current evidence,
candidate mask, already-sampled set, sensor availability, sample budget, and measurement-noise realization --
`ScoutEvaluationState.candidate_mask()` and `HydroScout.deterministic_fallback`'s own already-sampled exclusion
are defined compatibly enough (both: real, non-padding, non-already-sampled, accessible) that the same
underlying incident state can drive both policies fairly; wiring that exact pairing is scientific-comparison
work for M10.2 itself, not this preflight.

## Checkpoint/output-governance summary

- All four raw Scout heads **are present** in every frozen M9.6 checkpoint (`scout_control_heads=True`).
- None of the four **were trained** according to the frozen training config's actual data path (Finding B).
- None are listed in any v4 `trained_outputs`/`validated_outputs`/`runtime_enabled_outputs` set today -- M9.6
  checkpoints do not use the v4 `checkpoint_identity.py` format at all (`scripts/hydrocore_v5/
  run_m9_6_train_arm_b.py` uses the legacy `hydroswarm.training.checkpoint.export_model`/`save_checkpoint`
  path, not `save_v4_checkpoint`); this is itself consistent with the underlying finding, not a separate gap
  this pass needs to close, since no v4 identity has ever claimed these outputs as trained.
- M10.2 needs no runtime-promotion change to evaluate the raw heads offline -- `decode_learned_scout_recommendation`
  reads `HydroCore.forward()`'s raw output dict directly and never touches `output_governance`'s
  runtime-enablement gate at all (that gate governs live production consumption, which this preflight does not
  perform or unlock).
- A schema-version bump for `checkpoint_identity.py`'s `SCOUT_STATE_SCHEMA_VERSION` is **not** warranted by this
  pass -- see "What was intentionally NOT changed."

## Remaining limitations

- Finding B is not resolvable without retraining, which is out of scope for this task and for M10.2 as
  currently authorized.
- `ScoutEvaluationState`'s `current_sampling_round`/`sample_budget_remaining`/accessibility-constraint items are
  represented at the evaluation-adapter level (this schema) but are **not** wired into `HydroCore`'s own
  `role_features`/`residual_features` input channels the way `hydroswarm.training.scout_state_contract` frozen
  the possibility of doing -- deliberately out of scope here, since even a fully-wired input would not change
  Finding B's conclusion (an untrained head produces uninformative output regardless of input content), and
  wiring it now would be speculative extra-scope work ahead of the retrain decision that would actually need it.
- `should_continue_sampling_logits` is incident-level (not per-node); this preflight's masking helper
  deliberately leaves it unmasked, per its own docstring -- any future multi-round trajectory evaluation must
  account for this when comparing against a per-step deterministic stop decision.

## Readiness decision

**`M10_2_PREFLIGHT_BLOCKED`.**

The minimal, precise, unresolved blocker: **the frozen, M10-selected M9.6 checkpoint's four raw Scout heads
never received a training gradient** (Finding B) -- not the schema gap (Finding A, closed by this pass). A
future M10.2 scientific evaluation cannot proceed against the current checkpoint without either (a) an
explicitly authorized, separately governed amendment that wires real Scout targets into the training corpus and
performs a narrow, frozen-backbone, head-only retrain (analogous to `scripts/train_scout_heads.py`'s
precedent), followed by a new, honestly re-versioned checkpoint identity, or (b) an explicit scientific decision
to characterize the current (untrained) heads' behavior as a deliberate negative/baseline result rather than a
"learned vs. deterministic" comparison -- a framing decision this preflight does not make, since it would be
setting M10.2's own scientific design after (not before) seeing this finding, which this task's instructions
correctly withhold from a preflight pass ("Do not preregister M10.2 result thresholds based on any M10.2
outcomes").

This document ends the M10.2 PREFLIGHT/CORRECTION pass. The M10.2 scientific Scout comparison is not executed by
this document. M10.3/M10.4/M10.5 are not addressed by this document.
