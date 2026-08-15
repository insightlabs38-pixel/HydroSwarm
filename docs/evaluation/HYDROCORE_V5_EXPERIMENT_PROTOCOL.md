# HydroCore-v5 Experiment Protocol

Status: PRE-REGISTERED before any Milestone-1+ v5 training or evaluation
result was generated. Branch: `fix/v5-experiment-foundation`, based on
`main` @ `b7699387a1a825a9b69034ded74321bd1049fc19` (merge of PR #13,
`fix/capability-remediation`).

This protocol governs every experiment in `experiments.txt` Milestones 1–11.
It exists so that no later milestone can retroactively redefine a split,
metric, or promotion gate to make a specific candidate look better. Where
this document and a later milestone's own write-up disagree, this document
is authoritative unless it is explicitly amended here (not silently
overridden in a report).

HydroCore-v4 remains shipped and frozen throughout the v5 program. No v5
experiment may alter `models/hydrocore-v4-release/` or any production
runtime default. See [MODEL_CARD.md](../MODEL_CARD.md) and
[FINAL_SYSTEM.md](../FINAL_SYSTEM.md) for current shipped identity.

## 1. Split policy

v5 reuses the governed split-role machinery already enforced by
`hydroswarm.training.split_policy` (`SplitPolicyViolation`,
`authorize_locked_final_test`) and documented in
[EVALUATION_V3_POLICY.md](../EVALUATION_V3_POLICY.md); it does not invent a
parallel split system.

### 1.1 Development splits

`train`, `validation`, and `development_holdout` are used freely for
architecture, prefix-distribution, and objective-design iteration
throughout Milestones 1–10. `ood_development` is used only to develop
OOD threshold/behavior, never as a final OOD test.

### 1.2 Calibration split

`calibration` is used only for conformal/threshold fitting (Milestone 3),
after predictor selection is final for that fit (Milestone 3.1's "freeze
predictor first"). Never used for gradient updates or checkpoint selection.
Milestone 3's evidence-depth-aware grouping (`EARLY`/`MID`/`MATURE`) still
draws exclusively from this split; groups with insufficient support fall
back hierarchically rather than borrowing from another split.

### 1.3 Locked split (no-lock rule)

`locked_final_test` and `locked_topology_test` remain unopened for the
entire duration of Milestones 0–10. `locked_test_opened` is checked false
before and after every experiment script this protocol governs, exactly as
`hydroswarm.evaluation.live_robustness.locked_test_opened` already does for
capability-remediation scripts.

The locked final test may be opened only once, only in Milestone 11.6,
and only after all of the following, mirroring
`authorize_locked_final_test`'s Stage 6 preconditions:

1. A finalist candidate is selected and frozen (Milestone 11.1–11.2).
2. `reports/results/v3/final-selection.json`-equivalent record exists for
   the v5 finalist.
3. The full validation matrix in Milestone 11.5 is green.
4. No further architecture, hyperparameter, or calibration tuning is
   planned.

Until then, Claude must stop at the exact text specified in Milestone
11.6 and await explicit authorization. No number from the locked final
test or locked topology test may appear in any Milestone 0–10 report,
promotion decision, or model-selection rationale.

### 1.4 Topology-family split policy

Per `experiments.txt`'s scientific rules: split physical scenarios BEFORE
generating causal prefixes or augmentations; derived variants of one
physical scenario stay in the same split; topology-generalization
experiments (Milestone 7) split at the topology-**family** level, not the
individual-network level -- every network belonging to one structural
family (e.g. all `branched-loop` perturbations) is entirely inside one
split. Development unseen families for Milestone 7 must be genuinely
structurally distinct (different branching/looping topology, size, or
degree distribution), not roughness/demand perturbations of a known family.
The locked topology set is never used as a development "unseen" family.

## 2. Seed policy

Two seeds for rapid screening of any arm/ablation. A minimum of three seeds
before any finalist or promotion-relevant conclusion is drawn from an arm.
Where compute allows, three seeds are used for all three Milestone-1
primary arms (A/B/C) from the start rather than screening first. Per-seed
results are always preserved individually in the milestone's machine-
readable artifact, never collapsed to a mean before being recorded. No seed
is dropped, substituted, or re-rolled because its result was unfavorable;
an unstable arm (e.g. one seed diverging or producing a non-finite loss) is
reported as an unstable arm, not silently excluded.

## 3. Prefix-generation policy

Governs Milestone 1.1. For each retained physical training scenario:

1. The train/validation/calibration/development_holdout/topology-family
   split assignment happens first, on the full-history scenario.
2. Causal-prefix depths (approximately 1, 2, 3, 4, 6, 12, 25 available
   reports) are generated only after that split assignment, as views over
   the already-split scenario.
3. Prefix generation never moves a sample between splits: every prefix of
   scenario S inherits S's split membership exactly.
4. A prefix contains only reports whose timestamps are at or before its
   depth's decision time -- no future observation ever leaks into a prefix,
   matching the production causal-window contract in
   [PRODUCT_CAPABILITY_CONTRACT.md](../PRODUCT_CAPABILITY_CONTRACT.md).
5. Prefixes preserve the scenario's actual missing/degraded sensor states,
   real timestamps, and real topology; none of these are synthesized or
   normalized away by the prefix view.
6. Implementation prefers a lazy prefix view/wrapper over the underlying
   full-history tensors rather than physically duplicating them per depth,
   to keep the corpus artifact size and generation time bounded.

## 4. Metrics

### 4.1 Primary metrics (per causal depth, unless noted)

- top-1, top-3, MRR, NLL, Brier, posterior entropy, true-source rank
  (Milestone 1.5).
- Evidence sufficiency; candidate-region size (using development
  calibration only, never locked/final calibration, where a group has
  applicable support) (Milestone 1.5).
- Event presence / cause (Milestone 1.5).
- 3-step top-1, 6-step top-1, mature-history retention, evidence-
  sufficiency quality (Milestone 2.4).
- Actionable-within-1/2/3-samples, median samples to actionability,
  never-actionable fraction (Milestone 5.5).
- Empirical conformal coverage (+ CI), mean/median/p90 candidate-set size,
  singleton rate, source inclusion, planning-gate pass rate (Milestone 3.5).

### 4.2 Secondary metrics

- Model latency, memory (Milestone 1.5).
- Event metrics, OOD classification, sampling-head quality, training
  stability (Milestone 2.4).
- Top-1/top-3, candidate contraction, source-rank improvement, entropy
  reduction, expected-vs-realized IG correlation (Milestone 5.5).
- NLL, Brier, ECE, reliability curve, selective accuracy/risk-coverage
  curve (Milestone 3.4).

## 5. Promotion gates and failure criteria

Each milestone's own promotion rule in `experiments.txt` governs that
milestone's arms; this section restates the cross-cutting invariants every
gate shares:

- A gate is evaluated only on development-split (or calibration-split, for
  calibration work) evidence -- never on the locked split.
- "No material regression" bounds are predeclared per milestone (e.g.
  Milestone 1's "no >2-3pp mature-history regression"), not chosen after
  seeing results.
- Statistical comparisons that claim one policy beats another (Milestone 5)
  use paired bootstrap confidence intervals; a point-estimate difference
  alone never establishes superiority.
- A milestone that finds no benefit reports that as a negative result
  (e.g. Milestone 5's `ACTIVE SAMPLING REMAINS ADVISORY` outcome) rather
  than omitting the arm or softening the promotion bar to pass it anyway.
- Authority/safety thresholds (planning suppression, calibration
  invalidity, OOD gating, WNTR verification) are never weakened to improve
  a capability metric, and neural outputs never bypass deterministic/WNTR
  authority, in any arm.
- Conformal alpha (0.1) is preserved across all Milestone 1-10 experiments
  unless a separate, predeclared alpha study is explicitly approved --
  never adjusted merely to shrink candidate sets.

## 6. Model-size experiments

Governed by Milestone 9, run only after Milestones 1-8 establish the best
training distribution/objective at the current (~4M) size. Arms are
v5-S (~4M), v5-M (~12-16M), v5-L (~24-32M) on the same best causal recipe.
Promotion requires a *meaningful* capability gain from added capacity (see
Milestone 9.2's worked examples), not merely a non-negative one. Milestone
9.3's bounded Optuna search runs only if manual experiments show real
hyperparameter sensitivity, uses pruning, and stays within a small governed
search space -- it is not a substitute for this predeclared protocol.

## 7. Calibration experiments

Governed by Milestone 3. Exactly one frozen predictor checkpoint is
selected before calibration experiments begin (3.1); no further predictor
tuning happens while calibration schemes are compared. Alpha stays 0.1
(Section 5 above). Scheme B's evidence-depth buckets (EARLY 1-3, MID 4-6,
MATURE 7+) fall back hierarchically when a group's calibration-split
support is insufficient. An optional APS/RAPS-style construction (3.6) is
only attempted if depth-aware grouping still yields excessively broad sets,
and only while holding target coverage constant. Promotion requires similar
coverage AND smaller sets AND higher actionability together -- never smaller
sets alone.

## 8. Sampling experiments

Governed by Milestone 5, using the frozen best predictor + calibration
from Milestones 1-3 (no simultaneous predictor change during sampler
comparison, 5.1). Matched-incident design holds source, network, initial
evidence, sensor coverage, sample budget, measurement-noise seed, and
collection delay fixed across compared policies (5.4). At least 50 paired
incidents, 100 preferred. The decision-gain scoring function (5.3) is
predeclared using development data only and never includes the true source
at decision time. Promotion requires beating random on decision utility
(actionable-within-3-samples or samples-to-resolution distribution, not
entropy reduction alone); if no learned policy beats random, the milestone
exit is `ACTIVE SAMPLING REMAINS ADVISORY`, matching Milestone 0's baseline
finding in `reports/evaluation/hydrocore-v5/m0-baseline.json`
(`current_sampling_baseline`) that EIG did not beat random valid-unsampled
selection pre-v5.

## 8.5. Amendment: Milestone 8.7 (temporal representation correction)

Milestone 8.6 (`reports/evaluation/hydrocore-v5/m8-6-summary.md`) audited
the frozen Milestone-1 predictor's representation correctness and found:

- `ABSOLUTE_TIME_ORIGIN_LEAKAGE`: `HydraulicFeatureBuilder.build`'s
  never-observed-node `measurement_age` fallback depended on the incident's
  own timestamp origin rather than representing an origin-independent
  quantity.
- `TEMPORAL_FEATURE_USAGE_WEAK_OR_PARTIAL`: the explicit timestamp/
  positional-encoding pathway was measurably inert while the derived
  age-feature pathway was measurably used.

Milestone 8.7 (`docs/evaluation/HYDROCORE_V5_M8_7_PROTOCOL.md`, its own
frozen sub-protocol) trains and evaluates three matched-size representation
arms (CURRENT_CONTROL, AGE_FIX_ONLY, AGE_FIX_PLUS_RELATIVE_TIME) to
determine which representation, if any, corrects these findings without
materially regressing accuracy or calibration, and should become the
candidate small-size recipe for Milestone 9. See
`reports/evaluation/hydrocore-v5/m8-7-summary.md` for the outcome. This
amendment does not alter any prior milestone's historical results.

## 9. No-lock rule (restated)

Never use the locked final evaluation for development, tuning, model
selection, calibration fitting, architecture selection, or debugging, at
any point before Milestone 11.6's explicit authorization gate. Never use
`development_holdout` labels to train. Never use calibration examples to
optimize model weights. These three rules apply identically to every
milestone in this protocol's scope, with no per-milestone exception.
