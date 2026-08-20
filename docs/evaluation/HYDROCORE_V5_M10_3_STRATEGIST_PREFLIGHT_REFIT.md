# HydroCore-v5 Milestone 10.3A: Strategist candidate-schema + supervision + representation refit (frozen BEFORE any Level-A result is inspected)

Amends nothing in `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`,
`HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`, or any M10.2 document -- all remain frozen and
unmodified. This document is the Strategist analogue of the M10.2 Scout supervision/representation refit
amendment: it closes the readiness gap `HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md` already found
("Strategist's candidate-conditioned path never ran forward at all during M9.6 training... Before any future
M10.3 scientific evaluation of Strategist can be executed, M10.3 must undergo its own supervision/candidate-
schema amendment") and STOPS before the true M10.3 learned-vs-deterministic Strategist scientific comparison,
exactly like the M10.2 amendment stopped before true M10.2.

Protocol hash (frozen BEFORE Level-A training executed):
`f73accbf548e9b8987b8b1258efd7d3e61e052f802714ace3bb5f8b1b8d0f587`
(`scripts/hydrocore_v5/m10_3_refit_protocol.py::protocol_hash()`).

## Part 1: readiness audit (mechanically proved, `run_m10_3_readiness_audit.py`)

Machine-readable: `reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-readiness-audit.json`.

**Finding, CONFIRMED**: `strategist_mode="candidate_conditioned"` and `consequence_prescreening_heads=True`
(M9.6's own `SHARED_MODEL_CONFIG`) construct `CandidatePlanEncoder`/`plan_value_head`/`plan_validity_head`/
`consequence_proxy_heads` (5 proxy heads) as real parameters in every canonical M9.6 checkpoint. The real M9.6
training corpus path (`scenario_to_prefix_example`, the sole source of every M9.6 training batch) never
populates any of the four required candidate-plan input fields
(`plan_template_ids`/`plan_target_type`/`plan_target_node_index`/`plan_target_link_index`/`plan_features`) --
confirmed both by `grep`-level source inspection of `causal_prefix.py`/`corpus.py` (zero occurrences) and by
building a real example and inspecting its `.inputs`/`.targets` keys directly (absent from both). Therefore
`plan_hidden` was `None` for every M9.6 training batch (the candidate-conditioned forward branch requires all
four fields or none), so `action_logits`/`action_pointer_logits`/`plan_value`/`plan_validity_logits`/the five
consequence proxies were never present in any M9.6 training output, never received a gradient, and hold their
random initialization in every canonical checkpoint today.

The SAME frozen checkpoints, forwarded with real candidate-plan tensors (this amendment's own new corpus
builder), DO structurally execute the candidate-conditioned path and produce finite output for every one of
those nine keys -- the architecture is real and load-bearing, only unsupervised. `action_template`/
`target_pointer` are independently confirmed "v3-legacy head[s]... still trained by the unmodified default v3
model" by `configs/training-v5-causal.yaml`'s own comment -- repository evidence excluding them from this
amendment's canonical trainable scope (task weight exists in the frozen config, but is never applied here).
`HydroStrategist.deterministic_fallback` is confirmed structurally independent of the candidate-conditioned
path (no reference to `candidate_plan_encoder`/`generate_response_plans` anywhere in its source) -- unmodified
by this amendment, and unaffected by anything it does. `PlanVerifier.verify()` remains the sole source of
`plan_validity` (confirmed by direct inspection of `strategist_labels.generate_strategist_labels`, which never
assigns validity from anything else).

**Readiness decision**: `M10_3A_REFIT_READY`.

## A real, useful discovery this audit made: the missing glue already had two of its three pieces built

Unlike Scout (where M10.2 had to build training-state wiring, target generation, AND the eval-time adapter
essentially from scratch), Strategist's gap was narrower. Two of the three necessary pieces already existed,
real and governed, just never connected for training:

- `hydroswarm.planning.candidate_tensorizer.plan_proposals_to_candidate_tensors` -- the ONE canonical,
  already-live-production-used (`HybridInferencePipeline._score_candidate_plans`, PASS-2 runtime candidate
  scoring) converter from a real, bounded, deterministic `PlanProposal` set to `HydroBatch`'s
  candidate-conditioned INPUT fields. Its own module docstring already states "any future training-side
  consumer must import this rather than reimplementing the mapping" -- this amendment does exactly that,
  unmodified.
- `hydroswarm.training.strategist_trajectory.build_strategist_trajectory` -- the already-real,
  already-leakage-audited offline TARGET generator (exact WNTR verification via `PlanVerifier`, exact
  `plan_value`/regret/proxy computation via the governed `hydroswarm.planning.plan_value_policy.
  evaluate_plan_value`). Reused unmodified for the TARGET side.

The missing piece -- this amendment's own new module, `hydroswarm.training.strategist_candidate_corpus` -- is
the INPUT-tensor-construction glue connecting them for a real training run: independently reconstructing the
SAME `PlanGenerationContext`/`PlanProposal` list `generate_strategist_labels` builds internally (via the same
production static methods `strategist_trajectory.py` itself already reuses --
`HybridInferencePipeline._signature_observations`/`_credible_nodes`/`_planning_context`), so the INPUT tensors
and the offline TARGETS are provably aligned 1:1 by construction, mechanically re-verified by
`StrategistCandidateAlignmentError`'s fail-closed check every time (never merely assumed).

A legacy, now-superseded v4/cycle-b2 script (`scripts/build_strategist_candidate_dataset.py`) attempted a
weaker version of the INPUT half years earlier, operating on already-serialized trajectory JSONL with no live
generation-time context -- its own docstring explains it had to fall back to a purely structural
`plan_features` approximation because it could not reconstruct the real `PlanProposal` objects after the fact.
This amendment has live generation-time context (it builds the corpus directly), so it uses the real,
canonical, richer `PLAN_FEATURE_NAMES` the live PASS-2 runtime path itself uses -- one shared definition,
train and serve.

## Two documentation/comment drifts found (disclosed, not "fixed" -- out of this task's scope)

1. `hydroswarm.model.core`'s own `PLAN_FEATURE_DIM` comment and `hydroswarm.training.checkpoint_identity.
   PLAN_FEATURE_SCHEMA_VERSION`'s comment both describe `plan_features` as "predicted_value, predicted_validity,
   action count, 3 bounded action-parameter scalars" -- an EARLIER aspiration. The actual, live, governed,
   already-production-used implementation (`candidate_tensorizer.PLAN_FEATURE_NAMES`) is a different, simpler,
   purely-structural 6-dimensional vector (target-type one-hot x3, has-target, is-no-response-comparator,
   validity-target-present). Both are 6-dimensional (so no shape defect), but the stale comments' semantic
   description no longer matches the code that actually runs. This amendment trusts and reuses the LIVE code
   (per this task's own "trace actual code, not assume names/semantics" instruction), not the stale comment.
2. `targets_v2.py`'s own `exposure_proxy`/`pressure_risk_proxy`/`containment_time_proxy` definitions describe
   raw-unit proxies (mg, minutes, seconds) approximating `consequence_vector`'s own components. The actual,
   populated implementation (`plan_value_policy.evaluate_plan_value`) computes NORMALIZED, RATIO/SCALE-based
   proxies (see Part 4 below) -- a different, but equally real and consistently-used, definition. This
   amendment documents and reuses the ACTUAL formula, not the stale comment's stated units.

Neither drift is corrected by this amendment (unrelated-cleanup out of scope); both are disclosed here for
future readers, matching this task's own "report negative/surprising findings honestly" instruction.

## Part 2: candidate-plan training schema (frozen)

`hydroswarm.training.strategist_candidate_corpus.STRATEGIST_CANDIDATE_TRAINING_SCHEMA_VERSION =
"strategist-candidate-training-v1"`. Distinct from `checkpoint_identity.STRATEGIST_CANDIDATE_SCHEMA_VERSION`
(unchanged, still `"strategist-candidate-v1-unbuilt"` -- that placeholder names the ORIGINAL M9.6 checkpoint's
own [never-real] training-corpus claim, matching the exact convention `SCOUT_STATE_SCHEMA_VERSION` already
established in M10.2). Channel wiring:

| Field | Shape (padded) | Source |
|---|---|---|
| `plan_template_ids` | `[1, 9]` int64 | `candidate_tensorizer.plan_proposals_to_candidate_tensors`, unmodified |
| `plan_target_type` | `[1, 9]` int64 | same |
| `plan_target_node_index` | `[1, 9]` int64, `-1` sentinel | same |
| `plan_target_link_index` | `[1, 9]` int64, `-1` sentinel | same |
| `plan_features` | `[1, 9, 6]` float32 | same (`PLAN_FEATURE_NAMES`, live/production schema) |
| `plan_mask` | `[1, 9]` bool | same, `False` at padded positions |

`9` = `MAXIMUM_PLAN_COUNT` = `ACTION_TEMPLATE_COUNT` (the governed candidate-count upper bound
`generate_response_plans` itself never exceeds -- never invented). Real candidate count varies 3-9 per
incident (some templates conditionally skipped, e.g. `ISOLATE_SOURCE` requires `isolatable_links`); every
incident's tensors are padded to the fixed 9-candidate axis so batching is a plain `torch.cat`/`torch.stack`,
no variable-topology-style collation needed. Candidate identity/order is NEVER semantic (`candidate_
tensorizer.py`'s own docstring: "every tensor row is keyed back to its owning `PlanProposal` purely by
`action_template` name... reordering `proposals`... changes nothing about what a resulting learned score is
attributed to") -- verified, not merely asserted, by a real permutation test
(`tests/scientific/test_m10_3_strategist_refit_corpus.py::
test_candidate_order_does_not_change_per_candidate_model_output`): `CandidatePlanEncoder` has no positional
term, so its per-row output is provably a pure function of that row's own content.

## Part 3: candidate generation remains deterministic (unchanged, reused)

`hydroswarm.planning.response.generate_response_plans`, driven by `PlanGenerationContext`'s CURRENT-EVIDENCE-
derived fields only (`probable_source_nodes` from the classical localizer's current posterior;
`isolatable_links`/`critical_demand_nodes` from static network structure; `downstream_flush_nodes` from graph
successors of probable sources) -- reused unmodified, never touched by this amendment. Deterministic given a
fixed `context` (pure function); excludes nothing a priori (every template that structurally applies to the
network is proposed; `plan_validity` -- whether a proposal is actually feasible -- is the TARGET a real WNTR
verification determines, not a precondition for proposing it). Real, unverified prescreen heuristic
(`PlanProposal.predicted_value`/`predicted_validity`, hardcoded per-template constants) folds into
`plan_features` via the SAME canonical tensorizer production PASS-2 scoring uses. Exact WNTR outcomes are never
used to construct these INPUT tensors -- verified structurally (Part 5).

## Part 4: genuine Strategist targets (frozen, exact formulas -- from `hydroswarm.planning.plan_value_policy`,
`PLAN_VALUE_POLICY_VERSION = "plan-value-policy-v1"`, reused unmodified, never re-derived)

- **`plan_validity`** (binary, `1`=valid): `PlanVerifier.verify(...).decision == PlanDecision.VERIFIED`. WNTR/
  EPANET remains the sole authority; never assigned from a heuristic score.
- **`exposure_proxy`** (cost-like, LOWER better; unbounded above, typically near/below `1.0`):
  `contaminant_mass_consumed_mg / max(no_response.contaminant_mass_consumed_mg, 1e-9)` -- ratio to the
  no-action baseline's own exposure.
- **`pressure_risk_proxy`** (cost-like, LOWER better; unbounded above): `pressure_violation_minutes / 60.0`.
- **`service_loss_proxy`** (cost-like, LOWER better; bounded `[0, 1]` given `service_availability in [0,1]`):
  `1.0 - service_availability`.
- **`containment_time_proxy`** (cost-like, LOWER better; `1.0` at the 240-minute scale, unbounded above for
  longer containment): `(240.0 if containment_time_minutes is None else containment_time_minutes) / 240.0` --
  "never contained within the simulation window" maps to the worst-case `1.0`, not an unbounded/undefined
  value.
- **`plan_regret_proxy`** == **`regret`** (cost-like, LOWER better, `0.0` at the pool optimum):
  `max(0.0, cost - best_cost)`, where `cost = exposure_proxy + pressure_risk_proxy + service_loss_proxy +
  containment_time_proxy` (additive, nonnegative-weighted) and `best_cost = min(cost)` over EVERY exactly-
  verified valid candidate for this SAME incident state (including the no-response comparator itself).
- **`plan_value`** (HIGHER better, bounded `(0, 1]`, exactly `1.0` at the pool optimum):
  `1.0 / (1.0 + regret)`.

**Masking (frozen, per-candidate, never per-incident)**: `plan_value`/all five proxies are `None` (masked,
placeholder `0.0`) whenever the candidate is invalid, OR its consequences could not be computed, OR the
no-response comparator's own consequences are unavailable, OR the incident's valid-candidate pool is empty --
`targets_v2`'s own "undefined for invalid plans without a computed consequence vector" rule, applied
per-candidate (NOT "masked for incidents with zero candidates," which is `targets_v2.py`'s own stale
per-incident-level comment -- see the documentation-drift disclosure above). `plan_validity` itself is never
masked (every proposed candidate receives a real WNTR verification decision).

**No clipping** is applied anywhere in this formula -- `exposure_proxy`/`pressure_risk_proxy`/`plan_regret_
proxy` can exceed `1.0` for a genuinely bad candidate; this amendment does not add clipping (that would be
inventing a new definition, forbidden by this task's own "do not invent arbitrary objective weights/formula
changes" instruction).

`action_template`/`target_pointer` are NOT trained by this amendment (Part 1's own repository-evidence
finding).

## Part 5: leakage audit (structural, adversarially tested)

`hydroswarm.training.strategist_candidate_corpus.build_strategist_candidate_example` and its internal
`_reconstruct_context_and_proposals` helper have no parameter through which any WNTR-verification-derived
value (exact consequence vector, `plan_validity` decision, `plan_value`/regret/proxies, or the eventual
best-candidate identity) can reach the model INPUT tensors -- structurally, not merely by caller discipline.
`_reconstruct_context_and_proposals` never references `scenario.manifest.incident`/`incident_truth` at all
(verified by source inspection, `test_context_construction_never_reads_scenario_incident_ground_truth_for_
candidate_generation`) -- the exact ground-truth incident is used ONLY inside `generate_strategist_labels`
(the offline TARGET side), never for candidate generation. Adversarial tests (`tests/scientific/
test_m10_3_strategist_refit_corpus.py`) prove: INPUT tensors never contain any governed target key; the
alignment guard fails closed on a real forced mismatch; the builder function itself has no
label/target/outcome-shaped parameter at all.

## Part 6: gradient/supervision coverage (reused, extended)

Reuses `hydroswarm.training.gradient_coverage.compute_gradient_coverage`/`require_gradient_coverage`
UNMODIFIED (the exact mechanism M10.2's own amendment built, explicitly designed to be "reusable later for
... Strategist"). All seven canonical Strategist tasks' `TASK_OUTPUT_NAMES` entries already existed in
`hydroswarm.training.task_output_names` (confirmed by inspection -- no change needed). Certificates additionally
prove `CandidatePlanEncoder` itself receives real, nonzero, finite gradient and its parameters change after a
controlled optimizer step (Section 5's frozen allowlist includes every `candidate_plan_encoder.*` parameter in
every one of the seven tasks' `parameter_groups`, since it is the shared representation every head depends on).

## Part 7: frozen refit protocol (Section-numbered per `m10_3_refit_protocol.py`)

**Teacher checkpoints (Section 1)**: the three ORIGINAL canonical M9.6 checkpoints (seeds `20260814`/`31874`/
`20260815`) -- NEVER the M10.2 Scout-refit checkpoints (the authorizing task's own explicit instruction: Scout's
Level-A refit modified Scout-specific support pathways [`role_projection`/`residual_projection`] irrelevant,
and potentially confounding, to an independent Strategist characterization).

**Population (Section 2)**: family `golden-reference` only (matches the M10.2 Scout refit's own single-family
pilot-scope precedent). Seed namespace role `strategist_refit_m10_3`, base `1_300_000_000` (continues the
M10.1/`1_100_000_000`, M10.2/`1_200_000_000` per-milestone seed-block convention). Train: seed base
`1_300_000_000`, count `250`. Validation: seed base `1_300_100_000`, count `300` -- set directly to `300`
(never `100`) from the start, informed by the M10.2 Scout refit's own same-day support-driven amendment
history (a resourcing lesson applied BEFORE any Level-A result exists here, not a result-driven change within
this protocol's own execution). `source_round_robin=True`. Depth `25` (MATURE). Verified disjoint (by `grep`,
before this document was frozen) from every locked split and every other seed range in the repository,
including every M10.1/M10.2 range.

**Trainable target set (Section 4)**: the seven canonical outputs (Part 4), task weights copied verbatim from
`configs/training-v5-causal.yaml` (`plan_validity=1.0`, `plan_value=0.5`, all five proxies `=0.3`) -- never
re-derived or tuned.

**Level-A allowlist (Section 5, forward-graph-traced, mechanically verified by a real forward+backward pass
before this protocol was frozen)**: exactly 40 parameters -- every `candidate_plan_encoder.*` parameter (12),
`plan_value_head.*` (4), `plan_validity_head.*` (4), `consequence_proxy_heads.*` (20, 5 heads x 4). No shared
backbone component is needed at all: unlike Scout, no new backbone-injection layer exists on this path
(`plan_hidden` is built entirely from `CandidatePlanEncoder`, consuming the ALREADY-COMPUTED `pooled` incident
context) -- `adapters["strategist"]` is `nn.Identity()` under M9.6's own `use_adapters=False` construction
(zero parameters, nothing to freeze/unfreeze). `action_head`/`pointer_query` (backing the excluded
`action_template`/`target_pointer`) mechanically confirmed to receive exactly zero gradient from every one of
the seven trained tasks' losses -- correctly excluded, not merely by policy but by structural independence.

**Optimizer (Section 6)**: Adam, `lr=1e-3`, `weight_decay=0.0`, `batch_size=8`, `epochs=20`, FINAL_EPOCH
checkpoint selection -- copied verbatim from the M10.2 Scout refit's own frozen values, never re-tuned or
swept.

**Statistics (Section 7)**: 2,000-resample, 90% CI, bootstrap seed `20260819` (cross-milestone convention).
`GATE_MIN_SUPPORT = 20`.

**Level-A representation-sufficiency gate (frozen; evaluated on the VALIDATION population, final-epoch
checkpoint)**: `LEVEL_A_REPRESENTATION_SUFFICIENT` requires ALL of:

1. Gradient coverage passes for all seven tasks (Part 6).
2. Support: >= 20 valid (non-masked) validation examples for EACH of the seven targets, and >= 20 real
   within-incident ranking pairs; no NaN/Inf in any prediction.
3. `plan_validity`: AUROC's 90% bootstrap CI LOWER bound exceeds `0.5` (better than chance, CI-supported --
   never merely a point estimate).
4. `plan_value` and each of the five proxies: validation MSE lower than a constant-train-mean-prediction
   baseline's MSE, AND Spearman rank correlation's 90% bootstrap CI excludes zero on the positive side.
5. **Ranking** (the most important representation-level criterion, per this task's own instruction): within-
   incident pairwise ranking accuracy (does the predicted `plan_value` order agree with the true `plan_value`
   order for every non-tied, both-valid candidate pair within the same incident) has a 90% bootstrap CI
   (resampled over INCIDENTS, not raw pairs, to respect within-incident correlation) whose LOWER bound exceeds
   `0.5` (better than random pairwise ordering).
6. No NaN/Inf anywhere.

If ALL six hold for a seed: that seed's Level-A representation-sufficiency gate passes. Per the frozen M10.2
precedent, Level B is NOT run merely because it is authorized once Level A passes for the required seeds --
Level B fires only if Level A is mechanically valid but fails criteria 3-5 for a genuine frozen-representation-
capacity reason (never a bug/leakage/imbalance/masking/schema/optimizer defect, which blocks with a distinct,
honestly-labeled state instead).

**Level B (Section 8, frozen scope, defined here BEFORE Level-A results)**: Level-A's 40 parameters PLUS every
`backbone[3]` (the LAST of the 4 `LatentHydraulicBlock` modules in the `small` variant, `len(model.backbone) ==
4`, confirmed against a real model instance) parameter (24 named parameters) PLUS `final_norm.weight` -- 25
extra parameters, 65 total. Matches the M10.2 Scout refit's own Level-B scope definition exactly (same block
index, same one-bounded-tail rationale, never progressively widened). Warm-starts from the SAME canonical M9.6
teacher checkpoint used for Level A (never from Level A's own outcome-tuned weights). Level-B promotion
additionally requires (both): (A) Level B's own ranking/plan-value/proxy competence materially improves over
Level A's own validation numbers under the same paired-bootstrap procedure (CI lower bound exceeds Level A's
own point estimate for at least the criteria Level A failed); (B) M9 preservation -- development-only
re-evaluation of all nine `TRAINED_WITH_REAL_TARGETS` Sentinel tasks plus source-posterior/calibration-coverage
behavior, comparing Level B's checkpoint against the unmodified M9.6 teacher, under the existing M9-frozen
acceptance bounds (coverage floor `0.85`, alpha `0.1`, unchanged). No calibration refit is performed under any
circumstance.

## Part 8: checkpoint identity / provenance (frozen fields, never omitted)

Every refit checkpoint records: parent M9.6 teacher SHA-256; refit level (`"A"`/`"B"`); exact trainable
parameter allowlist; frozen/trainable parameter counts; candidate-training-schema version; target keys;
train/validation manifest hashes (scenario-id sets); seed; optimizer-config hash; checkpoint-selection policy;
gradient-coverage-certificate hash; git commit; the refit model's own SHA-256; `never_call_this_m9_6: true`.

## Part 9: output governance (unaffected regardless of outcome)

Learned Strategist remains runtime-disabled and non-authoritative in every case. Deterministic Strategist
(`HydroStrategist.deterministic_fallback`) remains unmodified and retained as runtime authority. WNTR/EPANET
verification remains final physical authority, unconditionally. Human approval remains mandatory; no
autonomous actuation occurs anywhere in this amendment's execution. No `runtime_enabled_outputs` promotion
occurs under this protocol regardless of Level-A/B outcome -- this amendment establishes scientific eligibility
only; the true M10.3 learned-vs-deterministic comparison is a separately authorized later task, which this
document does not execute.

## Part 10: locked-test policy (restated)

`locked_final_test`/`locked_topology_test` are never accessed by this protocol's population, training, gate, or
checkpoint-identity logic. `locked_test_opened` is asserted `False` before and after every phase in every
artifact this protocol's execution produces.
