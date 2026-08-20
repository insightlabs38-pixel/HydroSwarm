# HydroCore-v5 TRUE Milestone 10.2: learned-vs-deterministic Scout scientific comparison (frozen BEFORE any decision-utility result is inspected)

Amends nothing in `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`, `HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`,
`HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`, or `HYDROCORE_V5_M10_2_SCOUT_REFIT_PROTOCOL.md` -- all
four remain frozen and unmodified. This document is the separately-authorized TRUE M10.2 comparison the refit
protocol's own closure explicitly deferred (`M10_2_SCOUT_REFIT_A_ACCEPTED`'s own "next_recommended").

**Frozen and hashed BEFORE any decision-utility metric is computed.** `scripts/hydrocore_v5/m10_2_true_protocol.py`
is the machine-readable source of truth; this document is its prose companion. Protocol hash
`5dba094406aa6df7b85f50a6becdd4a092c171a7a766f75877584e7885548ad8`.

## Part 1: audit findings (before any science is run)

### Finding 1 -- "HydroScout" names two different deterministic policies in this codebase

The task's ARM D is `hydroswarm.agents.scout.HydroScout.deterministic_fallback` -- a simple, real,
already-tested, non-learned heuristic: exclude already-sampled nodes, rank the remainder by posterior
probability (ties by node id), recommend the top candidate, STOP when `1/len(candidate_region) < 0.01` or no
unsampled accessible node remains.

This is **not** the same function that backs `hydroswarm.inference.authority.scout_certificate`'s live
`source="CLASSICAL_EIG"` recommendation in the actual production pipeline
(`HybridInferencePipeline.analyze()` builds `sample_result` via `hydroswarm.sampling.active.
rank_sample_locations` -- a richer EIG-variance-based ranker with delay/cost/redundancy/separation terms,
also used internally by `hydroswarm.training.scout_labels.generate_scout_label`, which is what produced
Level A's own training TARGETS). Both are legitimate, real, deterministic (non-learned) policies; they are
simply two different pieces of code. This document uses `HydroScout.deterministic_fallback` for ARM D because
the authorizing task names it explicitly and specifically ("Inspect: ... deterministic `HydroScout`
implementation"). This is disclosed, not hidden, because it means ARM D is a CRUDER deterministic baseline
than the classical-EIG teacher that produced ARM L's own training labels -- worth remembering when interpreting
results (a result where ARM L beats ARM D is not automatically "learned beats classical-EIG"; it may partly
reflect ARM L having distilled a richer classical teacher than ARM D itself embodies). No code change follows
from this finding; it is reported for interpretive honesty only, per the task's own audit-first instruction.

### Finding 2 -- calibration is reused via the SAME frozen-support-refit pattern M10.1 already established

`hydroswarm.calibration.conformal.SplitConformalCalibrator.fit(...)` is deterministic given a fixed
`examples`/`alpha`/`minimum_group_size` -- calling `.fit()` again on the IDENTICAL frozen support set
reproduces the identical calibrator, not a new fit against new data. `scripts/hydrocore_v5/
run_m10_1_decide.py::_fit_frozen_calibrator` already established this exact pattern for M10.1
("reuses the frozen M9.6 ... calibration examples AS-IS (no refit)"). This document reuses the identical
pattern: refit from `reports/evaluation/hydrocore-v5/m9-6/m9-6-canonical-calibration.jsonl`'s `ARM_B_M9_6`
records, alpha=0.1, `network_id=f"{FAMILY}:{depth_bucket_of(DEPTH)}"` = `"golden-reference:MATURE"`, confirmed
present with 480 support examples (>= `minimum_group_size=10`). No calibration threshold, alpha, or grouping is
changed, tuned, or newly fit against this milestone's own data at any point.

### Finding 3 -- predictor consistency requires populating `role_features`/`residual_features` identically for BOTH arms, at every round, including round 0

`HydroCore.forward()` (`src/hydroswarm/model/core.py` lines ~983-997) skips `residual_projection`/
`role_projection` ENTIRELY (contributes exact zero, not even a bias term) whenever the batch does not include
the `residual_features`/`role_features` keys at all. Level A trained these two (previously-random-init, never
gradient-touched during M9.6) layers using nonzero round/budget/accessibility signal
(`hydroswarm.training.scout_training_state.build_scout_training_state_batch`, schema
`"scout-training-state-v1"`). For ARM L's Scout heads to make a genuinely round-aware decision, every round's
model input must populate these two channels. Because they feed into the hidden state BEFORE the (frozen)
backbone, populating them ALSO changes `source_node_logits`' computation relative to a batch that omits them --
this is an expected, disclosed consequence, not a defect: this document populates `role_features`/
`residual_features` identically for BOTH arms (D and L) at every round (round 0 included), so the SAME Level-A
refit checkpoint computes localization outputs the SAME way regardless of which arm's turn it is -- only the
NODE CHOSEN each round differs by arm, never how a given state is scored. **Consequence, stated explicitly**:
this evaluation's own round-0 source-localization numbers are NOT bit-identical to M9's own historical, frozen
Sentinel characterization (which never populated these two channels). This is a new, self-consistent,
WITHIN-this-evaluation comparison between two Scout policies sharing one predictor and one input convention --
never a restatement, replacement, or reopening of M9's own frozen Sentinel result, which remains untouched
(`reports/evaluation/hydrocore-v5/m9-6/`, `m9-final/` are not read or modified by this milestone's execution
beyond the read-only calibration-support file above).

### Finding 4 -- family/budget scope is inherited from the ACCEPTED Level-A checkpoint, not widened

Level A trained on `family="golden-reference"` only (a "deliberately bounded, single-family pilot scope," its
own protocol Section 4). Evaluating the accepted checkpoint against an UNSEEN family here would confound "does
learned Scout generalize past its trained scope" with this milestone's actual question ("does the ACCEPTED
Level-A checkpoint's learned Scout beat deterministic Scout"). This document evaluates `golden-reference` only,
matching Level A's own accepted scope exactly (`m10_2_refit_protocol.FAMILY`, `DEPTH=25`,
`MAXIMUM_SAMPLES=3`, `NOISE_SCALE_MG_L=0.5` -- all reused unmodified, never re-derived or widened). No
cross-topology-generalization claim is made or implied by this milestone's result.

### Finding 5 -- the historical Stage-D limitation this evaluation resolves

`scripts/run_stage_d_scout_policy_comparison.py`'s own module docstring documents a real, prior architectural
limitation: the pre-refit HydroCore Scout input had "no already_sampled/revealed-evidence conditioning," so a
trained model could only make ONE well-supported decision (its recommendation from the scenario's ORIGINAL
evidence) -- re-running it at step 2+ deterministically repeats the same recommendation, "not a real 'keep
sampling' policy." That script therefore explicitly excluded `learned_scout` from its own multi-step
`trajectory` mode. Level A's refit specifically closed this gap (training `role_projection`/
`residual_projection` against genuine round/budget/accessibility signal, passing its own frozen
representation-sufficiency gate). This document is therefore able to run a genuine MULTI-ROUND trajectory
comparison for BOTH arms -- something Stage D's own precedent could not do for a learned Scout policy.

### Validity conclusion

A clean paired comparison is implementable within the required constraints: no architecture change, no new
`HydroCore` parameter shape, no retraining/tuning, no calibration refit, no locked-split access, no change to
deterministic Scout authority, no runtime promotion. No scientific-validity blocker was found. Proceeding to
Sections 2-9 below (frozen BEFORE any result is inspected).

## 2. Checkpoints (frozen; Section A of `m10_2_true_protocol.py`)

All three Level-A refit checkpoints (seeds `20260814`/`31874`/`20260815`), SHA-256-verified against the
authorizing task's approved hashes before this protocol was frozen (see closure artifact for the verification
record). Parent M9.6 teacher hashes recorded for provenance; the ORIGINAL M9.6 checkpoints are never loaded as
"the learned policy" anywhere in this milestone's execution -- their raw Scout heads remain untrained
(`HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md` Finding B), so using them for ARM L would silently fall back to
random-initialization noise, which this protocol's execution mechanically asserts against (Section 9 below).

## 3. Population (frozen; Section B)

Family `golden-reference` only (Finding 4). Seed base `1_200_200_000`, count `100` (task's own "100 preferred"
bar, met exactly), `source_round_robin=True` (matches Level A's own TRAIN/VALIDATION convention -- guarantees
every junction is represented as a generator-assigned source at least once even at this population size, a
non-arbitrary, precedented choice, not invented for this milestone). Depth `25` (MATURE bucket, Level A's own
fixed depth -- Scout-round evidence layers on top via the same synthetic-sensor mechanism, never swept). Sample
budget (`MAXIMUM_SAMPLES`) `3` for both arms, matching Level A's own trained pilot bound. Noise scale `0.5
mg/L`, matching Level A's own frozen value. Seed range `1_200_200_000`-`1_200_209_900` verified disjoint (by
`grep`, before this document was frozen) from every other seed base in the repository, including Level A's own
TRAIN (`1_200_000_000`, count 250) and VALIDATION (`1_200_100_000`, count 300 post-amendment) ranges and from
every locked split. `locked_final_test`/`locked_topology_test` are never accessed by this population or its
generation.

## 4. Pairing (frozen; Section D, restated from Finding 3)

For each of the 100 incidents, BOTH arms run against the SAME `GeneratedScenario` (same source, same network
reconstruction, same original sensor evidence, same causal-prefix depth), the SAME Level-A refit checkpoint (no
arm ever loads a different checkpoint), the SAME calibration artifact, the SAME sample budget (3), and the
SAME deterministic per-(scenario, step, node) measurement-noise generator
(`hydroswarm.training.scout_trajectory.reveal_sample_measurement` -- identical realized value whenever both
arms happen to sample the same node at the same step index; a different, still-deterministic value when they
sample different nodes, which is the correct, expected behavior, not a pairing violation). Only the Scout
sampling/stopping POLICY differs between arms:

- **ARM D**: `HydroScout.deterministic_fallback`, given `candidate_probabilities` from
  `hydroswarm.classical.signatures.localize_with_signatures`'s per-node marginal `source_probabilities`
  (computed from CURRENTLY REVEALED evidence only -- the same real, leakage-audited Bayesian machinery
  `generate_scout_label` itself uses internally), `candidate_region`/`node_ids` = all junction ids (matching
  `generate_scout_label`'s own `total_candidate_count` denominator and the accessibility convention below),
  `sampling_history` = the arm's own already-sampled set.
- **ARM L**: the Level-A refit checkpoint's raw Scout heads, decoded via the already-frozen, already-tested
  M10.2-preflight evaluation adapter (`hydroswarm.evaluation.scout_state.build_scout_evaluation_state` /
  `decode_learned_scout_recommendation`, schema `scout-eval-state-v1`, unchanged) -- masked candidate ranking
  (`sample_node_logits`), fail-closed selection (`None` when no eligible candidate remains), stop decision from
  `sigmoid(should_continue_sampling_logits) >= 0.5` (matches the Level-A gate script's own decoding
  convention, `run_m10_2_level_a_gate.py`). `accessible` = junction ids only (reservoir/tank nodes are never
  eligible for either arm, matching `residual_features`'s own accessibility encoding and `HydroScout`'s own
  implicit junction-only candidate pool).

Neither arm's input at decision time may include: the true source node, any future observation, any node
either arm samples in a STRICTLY LATER step, the realized information gain/candidate reduction of any pending
choice, the eventual stop outcome, or any Strategist/Planner output. `generate_scout_label`'s own already-
leakage-audited module docstring ("uses only that scenario's own observations plus a signature artifact built
from exact network simulation") and `build_scout_evaluation_state`'s own `assert_no_target_only_keys` fail-
closed guard are both reused unmodified and re-exercised by this milestone's own adversarial tests (Section
10).

## 5. "Actionable / source-resolved" definition (frozen; Section C)

`candidate_gate_pass` (never called bare "actionable" -- M5's own precedent naming discipline, avoiding
overclaiming full product actionability): the frozen M9.6 B_DEPTH_AWARE `SplitConformalCalibrator` (alpha=0.1,
refit from frozen support, Finding 2)'s `candidate_set()` over THIS evaluation's own round-state
`source_node_logits` softmax has size in `[1, K=3]` (`K` = `hydroswarm.simulation.wrapper.
MAXIMUM_EVALUATION_HYPOTHESES`, the same governed production constant M5 itself reused for the identical
purpose). Computed identically for both arms at every round from the SAME checkpoint (Finding 3) -- only the
revealed-evidence TRAJECTORY leading to a given round's state differs by arm.

## 6. Primary scientific question and metrics (frozen; Section 4 of the task's own instruction)

**Decision utility** (primary): `actionable_within_budget` = fraction of incidents with `candidate_gate_pass`
achieved at or before round 3; also reported at rounds 1/2/3 separately; `samples_to_actionability` (rounds
elapsed, censored at budget exhaustion); `never_actionable_fraction`.

**Localization progression**: source top-1/top-3 accuracy per round (raw softmax argmax/top-3, uncalibrated);
true-source rank per round; `candidate_gate_pass` set size per round (contraction).

**Information quality**: per-round posterior entropy (from the calibrated-adjacent softmax distribution) and
its round-over-round reduction; for ARM L only, Spearman correlation between the trained
`expected_information_gain`/`candidate_reduction_prediction` head values (at the node actually chosen) and the
REALIZED entropy drop / calibrated-candidate-set-size drop after revealing that node (an offline, post-decision
scoring, never fed back as a model input -- Section 8).

**Stopping quality** (ARM L only, both frozen BEFORE results): `false_stop_rate` = fraction of ARM-L
trajectories that stopped at round `k < 3` without `candidate_gate_pass`, where ARM D's OWN full-budget
trajectory on the SAME incident later achieves `candidate_gate_pass` by round 3 (real, paired,
already-collected evidence that continued/different sampling would have helped -- not a fabricated
counterfactual run). `unnecessary_sampling_rate` = fraction of ARM-L trajectories that took an additional
sample AFTER `candidate_gate_pass` was already achieved at an earlier round. `budget_exhaustion_rate` (both
arms) = fraction of trajectories that used the full 3-sample budget without a voluntary stop.

## 7. Safety / governance hard gates (frozen; mechanically asserted during execution, never merely reported)

Recorded as pass/fail counts, asserted (not just observed) during the run: invalid/inaccessible node selected
(0 required); already-sampled node reselected (0 required); sampling budget exceeded (0 required); non-finite
Scout output (0 required, via `assert_finite_scout_outputs`, reused unmodified); fail-closed candidate masking
preserved (`select_candidate_node` returns `None` rather than an arbitrary node whenever no eligible candidate
remains -- reused unmodified); deterministic OOD/authority untouched (`hydroswarm.inference.authority.
scout_certificate`/`ood_certificate` are not called, modified, or bypassed anywhere in this milestone's
execution -- ARM L's recommendation is decoded via `decode_learned_scout_recommendation`, whose `promotable`
field is hardcoded `False` with no setter); WNTR/EPANET authority unchanged (all "truth" in this evaluation
comes from `hydroswarm.training.scenario_reconstruction.simulate_all_node_truth`, itself WNTR/EPANET-derived,
read-only); human approval requirement unaffected (this milestone performs no actuation of any kind); learned
Scout remains runtime-disabled (Section 9). Any single hard-gate violation blocks promotion regardless of any
decision-utility result (Section 8).

## 8. Statistics (frozen; Section E)

Paired bootstrap, 2,000 resamples, 90% CI, seed `20260819` (reused unmodified from the M9/M10/M10.1
cross-milestone convention) over the 100 matched incidents, computed independently per seed (never pooled
across seeds for the promotion decision itself -- pooling would hide single-seed non-consistency, which the
promotion rule explicitly checks for). Reported for: `actionable_within_budget`, `samples_to_actionability`,
`never_actionable_fraction`, source top-1/top-3 per round (paired differences, ARM L minus ARM D).

## 9. Promotion rule (frozen; Section F, BEFORE any result is inspected)

`M10_2_LEARNED_SCOUT_PROMOTION_SUPPORTED` requires ALL of:

1. Every Section 7 hard gate passes for all three seeds.
2. Primary metric (`actionable_within_budget`, ARM L minus ARM D) has a POSITIVE point estimate in all three
   seeds, AND a 90% paired-bootstrap CI lower bound exceeding zero in at least two of the three seeds (a
   majority-seed, CI-supported consistency bar -- not unanimous-CI, since one underpowered seed's CI touching
   zero while its point estimate still improves is not evidence against a real effect, but a unanimous-point-
   estimate requirement guards against being driven by a single seed).
3. Neither `NO_REGRESSION_METRICS` metric (`never_actionable_fraction`, `source_top1_final_round`) shows a
   CI-confident regression (90% paired-bootstrap CI lower bound of the regressing-direction difference does
   not exceed zero) in any seed.

Any single failed criterion => `M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED` -- an acceptable,
reportable, non-failure outcome for this milestone, exactly matching M10.1's own "negative result is a valid
result" precedent. If execution itself cannot be cleanly completed (a genuine implementation/data defect
discovered during execution, not a tuning opportunity) =>
`M10_2_SCIENTIFIC_EVALUATION_BLOCKED`, reported honestly rather than forced. No threshold above may be
changed, no population may be regenerated, and no metric may be substituted after any decision-utility number
is computed.

## 10. Output governance (unaffected regardless of outcome)

Learned Scout remains runtime-disabled and non-authoritative in every case. `scout_certificate` continues to
hardcode `source="CLASSICAL_EIG"`/`AuthorityLevel.DETERMINISTIC` unconditionally and is not called by this
milestone's execution at all. No `runtime_enabled_outputs` promotion occurs under this protocol, even if
`M10_2_LEARNED_SCOUT_PROMOTION_SUPPORTED` is the result -- that would be a SEPARATE, later, explicitly
authorized promotion milestone. This milestone only determines scientific eligibility.

## 11. Locked-test policy (restated)

`locked_final_test`/`locked_topology_test` are never accessed by this protocol's population, evaluation, or
decision logic. `locked_test_opened` is asserted `False` before and after every phase in every artifact this
protocol's execution produces.
