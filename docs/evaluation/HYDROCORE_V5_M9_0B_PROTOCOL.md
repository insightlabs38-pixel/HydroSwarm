# HydroCore-v5 Milestone 9.0b protocol (frozen before any calibration scheme is evaluated)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` (see its own new
Section 8.8). References but does not rewrite
`docs/evaluation/HYDROCORE_V5_M9_0A_PROTOCOL.md`. This document freezes the
M9.0b sub-protocol BEFORE any calibration scheme is fit or evaluated. It is
not altered after seeing results.

## 0. Why this milestone exists

M9.0a (`reports/evaluation/hydrocore-v5/m9-0a-summary.md`) validated that
`STEP_MATCHED_INTERLEAVED_MULTI_FAMILY` gives a real, optimizer-step-parity-
robust unseen-topology gain (pooled MATURE neural top1 +6.60pp, hybrid
+6.94pp, paired 90% bootstrap CI entirely > 0, all 3 unseen families
improved, all 3 seeds positive) but found its `B_DEPTH_AWARE`
(`network_id = f"{family}:{depth_bucket}"`) known-family calibration fails
the frozen 0.85 marginal-coverage floor on ALL THREE predictor seeds
(0.848 / 0.845 / 0.821, mean 0.838) --
`CALIBRATION_SYSTEMATICALLY_INCOMPATIBLE`. Decision:
`TOPOLOGY_GAIN_VALIDATED_CALIBRATION_BLOCKER_REMAINS`,
`M9_CAPACITY_STUDY_UNBLOCKED = NO`.

M9.0b tests, at fixed `alpha=0.1` and the FROZEN M9.0a `ARM_B2` predictor
checkpoints (zero retraining), whether a different Mondrian GROUPING/
FALLBACK construction over the SAME conformal machinery
(`hydroswarm.calibration.conformal.SplitConformalCalibrator`, unmodified)
can restore safety-valid coverage. It does not reopen the predictor,
topology-training, or representation questions.

## 1. Frozen predictor

The three M9.0a `ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY` checkpoints,
seeds 20260814/31874/20260815, loaded from
`reports/evaluation/hydrocore-v5/m9-0a-runs/ARM_B2_STEP_MATCHED_INTERLEAVED_
MULTI_FAMILY-seed{seed}.json`'s own `training_summary.export_path`. Before
any inference, each checkpoint's on-disk SHA-256 is recomputed and compared
against BOTH that file's own `training_summary.export_sha256` AND
`reports/evaluation/hydrocore-v5/m9-0a-results.json`'s
`arms.ARM_B2.per_seed.{seed}.checkpoint_sha256` (cross-provenance check --
both must match, or the run aborts before fitting anything). No gradient
computation occurs anywhere in M9.0b (`torch.no_grad()` throughout,
identical to `_infer`/`_calibration_examples`'s own existing convention);
`model.eval()` only.

## 2. Row generation -- reusing M9.0a's own methodology unmodified

M9.0a's calibration artifacts (`m9-0a-calibration.json`) store only
aggregated `CalibrationReport` statistics, not raw per-row probability
vectors used for quantile fitting -- those must be regenerated. M9.0a's
`m9-0a-topology-generalization.json` DOES store complete per-row data
(`neural_probs`, `condition`, `depth_bucket`, `truth_index`, `family`) for
the three UNSEEN families; those rows are REUSED VERBATIM, never
re-inferred (Section 12).

**Calibration-split rows** (golden-reference, branched-loop, loop-grid;
`CALIBRATION_PER_FAMILY=50`/family, 150 total; full `CAUSAL_PREFIX_DEPTHS`
grid): regenerated via the SAME code path `run_m9_0a_evaluate._calibration_
examples` used to fit M9.0a's own `ARM_B2` calibrators --
`scenario_to_prefix_example(..., **FEATURE_KWARGS)` (AGE_FIX_ONLY:
`unobserved_age_sentinel="fixed"`, `include_relative_gap_feature=False`),
`_family_scenario_pool("calibration", ...)` for every one of the three
families (golden-reference included -- `ARM_B2` never used
`build_scenario_pool`'s Arm-A-only special case), same `SEED_BASES`.
Produces `(probabilities, true_index, condition, family, depth_bucket)`
tuples -- the family/depth_bucket kept SEPARATE (not pre-baked into one
`network_id` string) so every scheme's grouping can be built from the SAME
underlying rows without re-running inference per scheme.

**Development-holdout rows** (known-family coverage evaluation, Section 9):
regenerated via `run_m7_topology._generate_eval_scenarios`/`_infer`, THE
SAME code path `run_m9_0a_evaluate._evaluate_on_family` used to produce
M9.0a's own known-network/trained-family evaluation rows -- same
`SEED_BASES[(family, "eval")]` (identical incidents to M9.0a), same
`EVAL_MAX_SOURCES=4`/`EVAL_SEED_REPEATS=4` (16 incidents/family), full
depth grid. `_infer` is reused completely unmodified, INCLUDING its
existing (M9.0/M9.0a-inherited) `HydraulicFeatureBuilder().build(...)` call
with no explicit `unobserved_age_sentinel`/`include_relative_gap_feature`
override -- this is a deliberate methodological-consistency choice, not an
oversight: M9.0b's Scheme A (`CURRENT_FAMILY_DEPTH`) must reproduce M9.0a's
OWN measured numbers exactly to serve as a valid control (Section 5), and
changing eval-time feature construction would silently alter the very
baseline being recalibrated against, which is out of this milestone's scope
(Section 23 forbids changing predictor/representation behavior). Only
neural probabilities are captured (no classical/hybrid fusion -- irrelevant
to calibration grouping).

**Unseen-topology rows** (Section 12, diagnostic only): read directly from
`m9-0a-topology-generalization.json`'s `arms.ARM_B2.UNSEEN_TOPOLOGY.
{family}.per_incident_rows.{seed}`, verbatim, no re-inference.

## 3. Split hygiene

Calibration schemes are FIT only on the golden-reference/branched-loop/
loop-grid CALIBRATION-split rows (Section 2). Evaluated on DEVELOPMENT-
holdout rows from the same three families (known-family coverage, the
promotion-relevant metric) and, diagnostically only, on the three unseen
families' already-computed development-holdout rows (never used to fit any
scheme; `coastal-branch`/`tree-branch`/`dense-loop` never contribute a
single calibration score). `locked_final_test`/`locked_topology_test`
untouched throughout (checked via `locked_test_opened` before and after the
run).

## 4. Alpha fixed

`alpha = 0.1` for every scheme, every seed, unconditionally. Not tuned, not
inflated, not chosen after seeing coverage.

## 5. Four schemes (frozen definitions, before evaluation)

All four are built from the SAME unmodified
`hydroswarm.calibration.conformal.SplitConformalCalibrator`/
`CalibrationExample` API -- only the STRING passed as `network_id` (the
Mondrian group key) differs per scheme, plus one small experimental wrapper
for Scheme D. `minimum_group_size=10` (the `SplitConformalCalibrator.fit`
default) is used unchanged for every scheme -- not tuned.

**A -- CURRENT_FAMILY_DEPTH** (control). `network_id =
f"{family}:{depth_bucket}"`, unchanged `SplitConformalCalibrator.selection`
fallback (`NETWORK_SPECIFIC -> CONDITION_SPECIFIC -> GLOBAL`). Reproduces
M9.0a's own fitted `ARM_B2` calibrator exactly (same rows, same construction
-- Section 22 test 1 verifies this directly against M9.0a's own recorded
per-seed coverage numbers). Expected: known-family marginal coverage
~0.82-0.85, reproducing the known failure.

**B -- POOLED_DEPTH_AWARE**. `network_id = depth_bucket` only (EARLY/MID/
MATURE, no family component) -- the union of all three trained families'
calibration scores is pooled within each depth bucket before fitting one
quantile per bucket. At evaluation every known-family row of a given depth
bucket receives that bucket's single pooled quantile, regardless of family.
Same `selection` fallback machinery, unmodified.

**C -- BROAD_FALLBACK_CONTROL**. `network_id = None` for every row, at both
fit and evaluation time -- no topology-family OR depth-bucket Mondrian
grouping at all. Falls through `SplitConformalCalibrator.selection`'s own
existing second tier, `CONDITION_SPECIFIC` (the real, already-governed
CLEAN/OPERATIONAL/DEGRADED evidence-quality condition every M1-M9.0a script
already computes via `classify_runtime_condition`), then `GLOBAL` if even
that group is underpowered. Predeclared choice, decided BEFORE any scheme is
evaluated: `CONDITION_SPECIFIC -> GLOBAL` was chosen over a hand-rolled
"GLOBAL only" construction because it requires ZERO special-casing of the
existing `SplitConformalCalibrator` API -- simply omitting `network_id`
already IS exactly this fallback chain, `SplitConformalCalibrator`'s own
built-in behavior, unmodified, whereas forcing GLOBAL-only would require
additionally suppressing the condition dimension too, a larger and less
minimal deviation from the existing, reused machinery.

**D -- HIERARCHICAL_CONSERVATIVE**. Fits BOTH Scheme A's calibrator
(family:depth quantiles) and Scheme B's calibrator (pooled-depth quantiles)
from the SAME calibration-split rows. For a row with family/depth_bucket:
compute `q_pooled` from Scheme B's calibrator via its own `selection`
(`CONDITION_SPECIFIC -> GLOBAL` fallback if even the pooled-depth group is
underpowered -- so `q_pooled` always resolves to SOME governed quantile).
If the row's `family:depth_bucket` group is present in Scheme A's
calibrator (i.e. had >= `minimum_group_size` calibration examples),
`q_used = max(q_family_depth, q_pooled)`; otherwise `q_used = q_pooled`
directly. Candidate set = `{i : 1 - p_i <= q_used}`. By construction
`q_used >= q_pooled` in EVERY case (Section 22 test 3) -- Scheme D can only
be as-or-more conservative than Scheme B, never less, while still letting a
genuinely wider family-specific distribution add extra conservatism where
the data supports it. No true label is used to choose between the family
and pooled branches -- the choice depends ONLY on calibration-split group
SIZE (a fit-time property), never on development/evaluation outcomes.

## 6. No adaptive/post-sample calibration

M9.0b does not reopen M7B. No adaptive-evidence recalibration, online/
rolling recalibration, label-dependent runtime adaptation, or learned
calibration network is tested. APS/RAPS is not implemented in M9.0b under
any circumstance (Section 24's gate only ever RECOMMENDS a follow-up, never
triggers implementation here).

## 7. Group-support audit

Reported BEFORE any held-out coverage number, for every scheme: `n`
calibration examples per group, score-distribution summary (min/median/max),
the fitted quantile, its rank (`ceil((n+1)(1-alpha))` per
`SplitConformalCalibrator`'s own `_quantile` formula, unmodified), and
whether the group met `minimum_group_size=10`. Not used to alter
`minimum_group_size` after the fact.

## 8. Three-seed evaluation

Every scheme is fit independently per predictor seed from that seed's own
calibration-split rows (Section 2) and evaluated on that seed's own
development-holdout rows. No quantile, group, or decision is shared across
predictor seeds.

## 9. Primary known-family coverage bar

`MIN_ACCEPTABLE_COVERAGE = 0.85` (frozen, matching M8.7/M9.0/M9.0a's own
bar). A scheme is `safety_valid` only if, for ALL THREE seeds: overall
known-family marginal coverage >= 0.85 AND each of EARLY/MID/MATURE pooled
coverage >= 0.85 AND each trained family's (golden-reference/branched-loop/
loop-grid) own marginal coverage >= 0.85 wherever that family has >= 10
held-out development rows in the relevant bucket (else reported, not used to
block or pass a decision by itself). No bucket or family is averaged away
behind the pooled marginal number.

## 10. Candidate-set / actionability guardrail

Reuses M9.0a's own frozen bar, expressed in the scale-invariant normalized
form Section 11 of the milestone instructions requests:
`candidate_set_size / eligible_source_node_count <= 0.5` per row (equivalent
to M9.0a's `mean_size <= 0.5 x mean_known_node_count` bar, but computed per
row against THAT row's own family's actual node count rather than an
unweighted cross-family mean -- a more precise application of the identical
threshold, not a different bar). Reported per family and pooled
(mean/singleton rate). Not loosened.

## 11. Wilson confidence intervals

Every primary coverage estimate is reported with `n` and a 95% Wilson score
interval (diagnostic context; the >= 0.85 empirical gate itself is never
redefined by the interval).

## 12. Unseen-topology calibration transfer

Diagnostic only (Section 12 of the milestone instructions). Never used to
fit or select a scheme, and only usable as an explicitly predeclared
tie-breaker if two schemes are otherwise tied on every known-family
criterion -- this protocol predeclares that NO such tie-breaker will be
needed/used (ties are resolved by Section 16's simplicity-preference rule
instead, Outcome D), so unseen-transfer numbers are reported for
completeness only.

## 13. Diagnosis of the M9.0a anomaly

Section 13 of the milestone instructions asks whether family x depth
fragmentation (vs. pooling) explains M9.0a's own unseen-transfer-coverage
exceeding known-family coverage. Diagnosed directly from Scheme A's vs.
Scheme B's own group quantiles/sizes (Section 7's audit) -- not inferred
from final coverage numbers alone.

## 14. Selection / promotion logic

Exactly Outcomes A-E as specified in the milestone instructions Section 16,
reproduced verbatim in `m9-0b-summary.md`'s own Decision section, not
restated here to avoid two documents drifting apart.

## 15. Scope discipline

No predictor retraining, no representation change, no topology-training
change, no alpha change, no locked-data access, no APS/RAPS implementation,
no M7B reopening, no production runtime wiring (experiment-scoped code
only), no M9.1/M9 capacity work begun in this milestone.
