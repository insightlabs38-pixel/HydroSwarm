# HydroCore-v5 Milestone 10 downstream-supervision amendment (additive, frozen M10 protocol unmodified)

Amends nothing in `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`'s text, which remains historically frozen and
unmodified, and amends nothing in any closed M9/M10.0/M10.1 result artifact. This document records a factual
correction to that protocol's Section 1 statement that "Every head M10 characterizes was already jointly
trained as part of this same construction during M9.6 training" -- **that statement is incorrect** for every
head this document classifies as anything other than `TRAINED_WITH_REAL_TARGETS`, and is corrected here rather
than silently left standing.

**This document does not reopen or alter M9, M10.0, or M10.1's results.** M9's Sentinel/localization result
remains valid exactly as closed: the nine tasks classified `TRAINED_WITH_REAL_TARGETS` below, and the shared
hydraulic representation that produced them, genuinely were trained. M10.0's `SYSTEM_PREFLIGHT_PASS` and
M10.1's `LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED` remain the closed, correct results of those
milestones -- **only their interpretation changes**: M10.1's negative result must now be read as "a genuinely
unsupervised `ood_category` head was correctly not promoted," not as "a properly trained learned-OOD model
failed to beat the deterministic baseline." The deterministic/safe operational conclusion (learned OOD stays
suppressed, deterministic `OODDetector` stays authoritative) is unaffected either way.

## Method (mechanical, not inferential)

Every classification below is produced and cross-checked by
`scripts/hydrocore_v5/run_m10_2_supervision_audit.py`, writing
`reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-supervision-audit.json`. It:

1. loads a real, unmodified, SHA-256-verified canonical M9.6 checkpoint (seed `20260814`) with the exact
   selected construction (`m9_1_common.SHARED_MODEL_CONFIG`: `prior_mode="feature_only"`,
   `event_control_heads=True`, `scout_control_heads=True`, `strategist_mode="candidate_conditioned"`,
   `consequence_prescreening_heads=True`, `ood_category_head=True`);
2. runs one real forward pass on a real, `scenario_to_prefix_example`-built batch, in the EXACT shape (no
   `plan_*` tensors) M9.6 training ever actually supplied -- `hasattr`-checks which head modules exist, and
   inspects the real `outputs` dict returned;
3. reads the real `targets` dict `scenario_to_prefix_example` produces for that same scenario;
4. classifies every one of `hydroswarm.training.losses.ALL_TASK_NAMES`'s 27 governed task names (plus 5
   always-present legacy/ungoverned raw outputs and the `CandidatePlanEncoder` module) by the exact same
   `output_name in outputs` / `task in targets` conditions `compute_multitask_loss` itself uses;
5. cross-checks itself: calls the real `compute_multitask_loss(outputs, targets)` and asserts the set of tasks
   this audit calls `TRAINED_WITH_REAL_TARGETS` is EXACTLY the set that function actually produced a loss term
   for -- not merely consistent with it.

This is the same empirical method (call the real function, don't just read the source) the M10.2 preflight
correction (`HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`) already used for Scout; this document extends it to
every other gated output.

## Classification results

| Task / output | Classification | Why |
|---|---|---|
| `source_node` | `TRAINED_WITH_REAL_TARGETS` | target + output both present every batch |
| `source_region` | `TRAINED_WITH_REAL_TARGETS` | same |
| `start_time` | `TRAINED_WITH_REAL_TARGETS` | same |
| `duration` | `TRAINED_WITH_REAL_TARGETS` | same |
| `relative_strength` | `TRAINED_WITH_REAL_TARGETS` | same |
| `event_presence` | `TRAINED_WITH_REAL_TARGETS` | same |
| `event_cause` | `TRAINED_WITH_REAL_TARGETS` | same |
| `sensor_fault` | `TRAINED_WITH_REAL_TARGETS` | same |
| `evidence_sufficiency` | `TRAINED_WITH_REAL_TARGETS` | same (deterministic label, always valid) |
| `sample_node` | `PRESENT_BUT_UNSUPERVISED` | head present, output present, **no target ever supplied** |
| `information_gain` | `PRESENT_BUT_UNSUPERVISED` | same |
| `candidate_reduction` | `PRESENT_BUT_UNSUPERVISED` | same |
| `should_continue_sampling` | `PRESENT_BUT_UNSUPERVISED` | same |
| `ood_class` (`ood_category`) | `PRESENT_BUT_UNSUPERVISED` | same -- see M10.1 interpretation note above |
| `next_step` | `PRESENT_BUT_UNSUPERVISED` | same |
| `sensor_reconstruction` | `NOT_INSTANTIATED` | `auxiliary_heads=False` (default, never overridden by `SHARED_MODEL_CONFIG`) -- the head does not exist as a parameter at all |
| `future_concentration` | `NOT_INSTANTIATED` | same (also independently barred from `trained_outputs` by `checkpoint_identity.build_checkpoint_identity`'s own leakage guard, moot here since the head does not exist regardless) |
| `travel_time` | `NOT_INSTANTIATED` | same |
| `plan_validity` | `STRUCTURALLY_NOT_EXERCISED` | head instantiated (`strategist_mode="candidate_conditioned"`), but `forward()` only adds `plan_validity_logits` to `outputs` when `plan_hidden is not None`, which requires `plan_template_ids`/`plan_target_type`/`plan_mask`/`plan_features` -- a `grep` across `hydroswarm/training/causal_prefix.py` and `hydroswarm/preprocessing/` finds **zero** occurrences of any of those four field names; no code path ever supplies them |
| `plan_value` | `STRUCTURALLY_NOT_EXERCISED` | same gate |
| `exposure_proxy` | `STRUCTURALLY_NOT_EXERCISED` | same gate (`consequence_prescreening_heads=True` instantiates the module, but its own output line is additionally gated on `plan_hidden is not None`) |
| `pressure_risk_proxy` | `STRUCTURALLY_NOT_EXERCISED` | same |
| `service_loss_proxy` | `STRUCTURALLY_NOT_EXERCISED` | same |
| `containment_time_proxy` | `STRUCTURALLY_NOT_EXERCISED` | same |
| `plan_regret_proxy` | `STRUCTURALLY_NOT_EXERCISED` | same |
| `CandidatePlanEncoder` (module) | `STRUCTURALLY_NOT_EXERCISED` | instantiated, but its `forward()` is only invoked inside the same never-reached `plan_template_ids is not None` branch -- zero forward calls, zero gradient, for any M9.6 training batch |
| `action_template` (`action_logits`) | `LEGACY_UNGOVERNED` | excluded from `CANONICAL_OUTPUT_NAMES` by design (`checkpoint_identity.py` Section D item 6: deterministic candidate plans own action-template identity) -- ALSO structurally not exercised for M9.6 specifically, but the governing reason is the permanent vocabulary exclusion |
| `target_pointer` (`action_pointer_logits`) | `LEGACY_UNGOVERNED` | same |
| `uncertainty` | `LEGACY_UNGOVERNED` | unconditionally computed every forward pass, no governed target exists (`checkpoint_identity.py` Section D item 2) |
| `ood_logits` (old 3-logit `ood_head`) | `LEGACY_UNGOVERNED` | unconditionally computed, no governed target (Section D item 1) -- NOT the same head as `ood_category_logits` |
| `sentinel`/`scout`/`strategist` (anonymous role-hidden groups) | `LEGACY_UNGOVERNED` | unconditionally computed raw `RoleHead` outputs, no governed target (Section D item 9) |

**Totals**: 9 `TRAINED_WITH_REAL_TARGETS`, 6 `PRESENT_BUT_UNSUPERVISED`, 3 `NOT_INSTANTIATED`, 8
`STRUCTURALLY_NOT_EXERCISED` (7 governed targets + `CandidatePlanEncoder`), 7 `LEGACY_UNGOVERNED`.

Machine-readable: `reports/evaluation/hydrocore-v5/m10/m10-2-refit/m10-2-refit-supervision-audit.json`.

## Scope of this document

This document performs Part 1 of the M10.2 Scout supervision/representation refit amendment: it records the
correction and stops. It does not retrain anything. In particular:

- **OOD/`next_step`**: confirmed `PRESENT_BUT_UNSUPERVISED`. Not retrained by this task. Any future OOD refit
  is a separately authorized amendment, matching M10.1's own already-closed guidance that learned OOD stays
  suppressed regardless.
- **Strategist**: confirmed `STRUCTURALLY_NOT_EXERCISED` end to end (no candidate-plan tensors were ever
  supplied to any M9.6 training batch, so `CandidatePlanEncoder` and every consequence-proxy/plan-scoring head
  never even ran forward, let alone received a gradient). **Before any future M10.3 scientific evaluation of
  Strategist can be executed, M10.3 must undergo its own supervision/candidate-schema preflight-correction
  amendment**, analogous to this one and to `HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md` -- the gap there is
  structurally deeper than Scout's (Scout's heads at least ran forward on every batch; Strategist's
  candidate-conditioned path never ran forward at all during M9.6 training), and Strategist's `STRATEGIST_CANDIDATE_SCHEMA_VERSION = "strategist-candidate-v1-unbuilt"` placeholder remains accurate and unchanged
  by this document.
- **Scout** (`sample_node`/`information_gain`/`candidate_reduction`/`should_continue_sampling`) is the one
  family this document's parent task (the M10.2 Scout supervision/representation refit amendment) is
  authorized to act on -- see the refit protocol and Level-A/B execution documents this same task produces.

## Reusability

`scripts/hydrocore_v5/run_m10_2_supervision_audit.py`'s method (real construction + real forward pass + real
targets + cross-check against a real `compute_multitask_loss` call) is intentionally generic over
`hydroswarm.training.losses.ALL_TASK_NAMES` and is the correct tool to re-run, unmodified, against any future
training corpus change (Scout's refit corpus in this task, or a future OOD/Strategist corpus change) to prove
-- not assume -- that a task previously `PRESENT_BUT_UNSUPERVISED`/`STRUCTURALLY_NOT_EXERCISED` has actually
become `TRAINED_WITH_REAL_TARGETS`.
