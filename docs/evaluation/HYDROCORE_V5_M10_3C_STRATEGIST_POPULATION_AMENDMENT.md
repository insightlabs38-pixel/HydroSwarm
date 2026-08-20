# HydroCore-v5 Milestone 10.3C: Strategist expanded-population identifiability/oracle-gate amendment

Additive to `docs/evaluation/HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md` (closure
`M10_3B_POPULATION_AMENDMENT_REQUIRED`), which remains frozen and unmodified. Does not reopen, reverse,
retrain, or reinterpret-to-pass any M10.3A or M10.3B result. Diagnostic/population-governance only: trains
nothing, touches no checkpoint, opens no locked data, does not run true M10.3, does not run M10.4.

Protocol: `scripts/hydrocore_v5/m10_3c_population_protocol.py`. Diagnostic script:
`scripts/hydrocore_v5/run_m10_3c_population.py`. Frozen protocol hash (frozen BEFORE any candidate-
verification/diversity/identifiability/oracle result was inspected):
`56f1f62b2974699e4d7ee1acac02531f038208247a3728de0e6924c074f7cfb3`.

## Central question

Does expanding from the narrow golden-reference/depth-25 pilot M10.3A/M10.3B used to the already-governed
TRAINED_FAMILIES topology families (branched-loop, loop-grid) and earlier/mid causal-prefix depth regimes
(1, 2, 3, 4, 6, in addition to 25) reveal enough REAL, WNTR-verified within-incident candidate tradeoff and
oracle headroom to scientifically justify another learned Strategist training attempt (M10.3D)?

**Answer: No.** The expanded population shows real, measurable, statistically broader improvement over the
M10.3B golden-reference/depth-25 pilot baseline -- notably `ALTERNATE_VALVE_CUT` becomes 100% WNTR-VERIFIED
(180/180) on both looped families (`branched-loop`, `loop-grid`), versus 0/180 on `golden-reference`, and pooled
within-incident `plan_value` diversity roughly doubles (23.0% vs M10.3B's 17.5% for >=2 distinguishable
candidates; 5.6% vs 0.0% for >=3) -- but the improvement does not clear the preregistered M10.3C gate at either
the pooled or the per-family level. Closure: **`M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED`**.

## 1. Audit trail (Section 7 of the task spec)

Traced before any population was generated: `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`,
`HYDROCORE_V5_M10_DOWNSTREAM_SUPERVISION_AMENDMENT.md`, `HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`,
`HYDROCORE_V5_M10_3_STRATEGIST_REFIT_RESULTS.md`, `HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md`,
`scripts/hydrocore_v5/m10_3_refit_protocol.py` (M10.3A's frozen population/schema/target definitions),
`scripts/hydrocore_v5/run_m10_3_level_a_train.py` (`_build_corpus`), `scripts/hydrocore_v5/run_m10_3b_diagnosis.py`
(every M10.3B diagnostic function -- reused here unmodified, not reimplemented), and the actual candidate-
generation/target machinery: `hydroswarm.training.strategist_candidate_corpus.build_strategist_candidate_example`,
`hydroswarm.training.strategist_trajectory.build_strategist_trajectory`, `hydroswarm.planning.candidate_tensorizer`,
`hydroswarm.planning.plan_value_policy`, `hydroswarm.training.causal_prefix` (depth machinery),
`scripts/hydrocore_v5/run_m7_topology.py` (`TRAINED_FAMILIES`/`_family_scenario_pool`, the topology-family
generation machinery M10.3A/B already reused).

### 1.1 Locked-test terminology confirmed

No file or directory anywhere in this repository matches a `*locked*` name pattern. The repository's actual
terminology is `locked_final_test`/`locked_topology_test` (`docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`
Section 3/12; `hydroswarm.evaluation.live_robustness.locked_test_opened`). Neither has ever been materialized as
a data fixture in this repository (`run_m7_topology.py`'s own module docstring: "this repo currently has no
locked-topology fixture materialized at all"). `locked_test_opened` reads a static boolean flag from
`reports/results/v4/architecture-freeze.json["locked_test_opened"]` -- confirmed `False` both before and after
this task (Section 9 below).

### 1.2 Topology/depth generation machinery confirmed

`scripts/hydrocore_v5/run_m7_topology.py::TRAINED_FAMILIES = (("golden-reference", build_wntr_network),
("branched-loop", ...branched-loop.inp), ("loop-grid", ...loop-grid.inp))` -- the SAME three families M10.3B's
closure named, confirmed by direct read, not assumed. `_family_scenario_pool` (same file) is the SAME
deterministic, family-scoped scenario generator M10.3A/M10.3B already used for `golden-reference`; this task
calls it, unmodified, for the other two families too. `hydroswarm.training.causal_prefix.CAUSAL_PREFIX_DEPTHS =
(1, 2, 3, 4, 6, 12, 25)` is the SAME governed depth grid `scripts/hydrocore_v5/run_m3_calibration.py`'s
`DEPTH_BUCKET_OF` and every M9/M10 depth-stratified script already uses (`EARLY={1,2,3}`, `MID={4,6}`,
`MATURE={12,25}`). This task's population uses the exact task-specified subset `{1, 2, 3, 4, 6, 25}` -- no
invented depth value.

### 1.3 Load-bearing audit finding: depth is causally independent of Strategist candidate/target generation

Traced directly from source, not assumed: `build_strategist_trajectory`
(`src/hydroswarm/training/strategist_trajectory.py`) and `_reconstruct_context_and_proposals`
(`src/hydroswarm/training/strategist_candidate_corpus.py`) both build their classical-localization/plan-context
inputs from `build_sensor_series(scenario, feature_context)` -- the scenario's FULL, untruncated sensor
evidence -- and neither function accepts or reads a `depth` argument anywhere in the call chain
(`generate_response_plans`, `generate_strategist_labels`, exact WNTR verification via `PlanVerifier`, and
`plan_value_policy.evaluate_plan_value` likewise never receive one). `depth`
(`hydroswarm.training.causal_prefix.CAUSAL_PREFIX_DEPTHS`/`truncate_causal_prefix`) is consumed EXCLUSIVELY by
`scenario_to_prefix_example`, the step that builds a HydroCore causal-prefix MODEL INPUT for a training forward
pass -- a step this diagnostic never calls (no training, no model forward pass occurs anywhere in M10.3C).

**Consequence**: depth has zero causal effect on Strategist candidate proposals, WNTR verification outcomes, or
any of the 7 governed Strategist targets in this repository's current implementation. It affects only what
evidence-truncated input a future LEARNED Strategist would be trained/conditioned on. This does not weaken the
task's required family x depth reporting grid (every one of 18 cells is still generated from its own disjoint
seed block and reported individually below) -- it means depth functions as a disjoint-seed bookkeeping
partition (useful for a future M10.3D causal-prefix input construction) rather than a source of physical
candidate diversity in this diagnostic. The real diversity axis this population tests is topology family alone.
This is verified empirically, not merely asserted, in `m10-3c-invariance-audit.json`'s
`depth_invariance_empirical_check` (cross-depth mean `plan_value` within one family) and
`candidate_generation_determinism_probe`, and regression-guarded by
`tests/unit/test_m10_3c_population.py::test_build_strategist_candidate_example_signature_has_no_depth_argument`
and its two sibling tests.

## 2. Frozen population (Section 8/9/10 of the task spec, `m10_3c_population_protocol.py`)

- **Topology families**: `golden-reference`, `branched-loop`, `loop-grid` -- exactly `TRAINED_FAMILIES`, no new
  or unseen/locked family.
- **Depth buckets**: `1, 2, 3, 4, 6, 25` -- an already-governed subset of `CAUSAL_PREFIX_DEPTHS`, used here as a
  disjoint-seed bookkeeping label (Section 1.3 above), not a scenario-generation parameter.
- **Population size**: `PER_FAMILY_COUNT=180` scenarios/family, round-robin depth-labeled into exactly
  `PER_CELL_COUNT=30` scenarios/cell, `3 families x 6 depths = 18 cells`, `TOTAL_SCENARIO_COUNT=540` scenarios.
  Every cell receives the identical count -- golden-reference/depth-25 is not over-weighted merely because it
  has prior M10.3A/B data. 540 is the same order of magnitude as M10.3A/B's own 550-scenario single-family
  population (a genuine 3x-family / 6x-depth-label breadth expansion at comparable total generation cost, not
  an unrelated scale-up).
- **Candidate generator / target formulas**: SAME governed deterministic generator
  (`build_strategist_candidate_example`) and SAME 7 target formulas M10.3A/M10.3B used, completely unmodified --
  reused by direct import in `run_m10_3c_population.py`, not reimplemented.
- **Near-tie tolerances**: reused verbatim (the literal same Python object,
  `run_m10_3c_population.NEAR_TIE_TOLERANCE is run_m10_3b_diagnosis.NEAR_TIE_TOLERANCE`, regression-tested) --
  no protocol-level mathematical reason required a different tolerance (the target formulas/scales are
  byte-identical regardless of topology family).

### 2.1 Seed namespace / disjointness (Section 9)

`SEED_NAMESPACE_BASE = 1_400_000_000` -- continues the per-milestone seed-block convention (M10.1
`1_100_000_000`, M10.2 `1_200_000_000`, M10.3A/B `1_300_000_000`). Family offsets `golden-reference:+0`,
`branched-loop:+10_000_000`, `loop-grid:+20_000_000`. Verified disjoint from every historical seed range by
static grep (zero prior hits across `1_400_000_000..1_499_999_999` before this protocol was frozen, re-verified
programmatically in `m10-3c-seed-disjointness.json`) and from every other family's own range. A reserved,
NOT-materialized future block (`RESERVED_FUTURE_M10_3D_SEED_BASE = 1_450_000_000`) is set aside, disjoint from
every diagnostic range used here, so a future M10.3D refit (if authorized) can draw a fresh, independent
train/validation split without colliding with this diagnostic population -- consistent with this task's
instruction not to consume the future true-M10.3 comparison population and to preserve headroom for M10.3D.

## 3. Mechanical invariants (Section 13) -- regression, not new science

`m10-3c-invariance-audit.json` re-runs M10.3B's own `_ranking_alignment_audit` (unmodified, all 6 mechanical
checks) and `_leakage_audit` (unmodified) against this population's pooled records, plus two new checks specific
to this amendment: (1) the depth-causal-independence finding (Section 1.3), confirmed both by source inspection
and by an empirical cross-depth mean-`plan_value` comparison within each family; (2) a candidate-generation
determinism probe (rebuilding the SAME scenario twice through `build_strategist_candidate_example` and
confirming byte-identical INPUT/TARGET tensors).

**Result**: All checks passed. `ranking_alignment_audit.all_checks_passed=true` (6/6). `leakage_audit`:
candidate order is the fixed canonical template order in every incident (never truth-derived), input
construction never reads incident ground truth -- both `passed=true`. Candidate-generation determinism probe:
rebuilding the same scenario twice through `build_strategist_candidate_example` produced byte-identical
INPUT/TARGET tensors (`rebuild_byte_identical=true`). Depth-invariance empirical check: within each family, the
cross-depth standard deviation of mean `plan_value` is small and shows no monotonic trend
(`golden-reference` std=0.0070, range=0.022 across depths {1,2,3,4,6,25}; `branched-loop` std=0.0142,
range=0.039; `loop-grid` std=0.0112, range=0.036) -- consistent with Section 1.3's finding that depth carries no
causal signal for this population; the observed small spread is attributable to ordinary sampling noise at
n=30/cell, not a depth effect.

## 4. Candidate verification (Section 14)

`m10-3c-candidate-verification.json` reports, per cell/family/global: proposal count, verification rate,
rejection-code frequency per template, with special attention to `ISOLATE_SOURCE`/`ISOLATE_AND_FLUSH`/
`ALTERNATE_VALVE_CUT`/`FLUSH_DOWNSTREAM`/`NO_ACTION`.

**Result**: Every incident proposes all 9 templates (180/180 per family; zero skipped-no-candidates incidents in
any of the 540 generated scenarios). Isolation-template behavior differs sharply and meaningfully by family:

| Template | golden-reference | branched-loop | loop-grid |
|---|---|---|---|
| `ISOLATE_SOURCE` | 0/180 verified (`PRESSURE_BELOW_MINIMUM` x180) | 0/180 verified (`PRESSURE_BELOW_MINIMUM+SERVICE_BELOW_MINIMUM` x180) | 0/180 verified (same compound code x180) |
| `ISOLATE_AND_FLUSH` | 0/180 verified (`PRESSURE_BELOW_MINIMUM` x180) | 0/180 verified (compound code x180) | 0/180 verified (compound code x180) |
| `ALTERNATE_VALVE_CUT` | 0/180 verified (`PRESSURE_BELOW_MINIMUM` x180) | **180/180 verified (100%)** | **180/180 verified (100%)** |

`ALTERNATE_VALVE_CUT` flips from always-rejected (golden-reference, matching M10.3B exactly) to
always-VERIFIED on both looped topology families -- direct, exact-WNTR-verified confirmation of M10.3B's own
hypothesis that greater network redundancy can make an isolation-style action safely feasible.
`ISOLATE_SOURCE`/`ISOLATE_AND_FLUSH` remain always rejected in every one of the 18 cells; on the looped families
the rejection additionally trips `SERVICE_BELOW_MINIMUM` alongside `PRESSURE_BELOW_MINIMUM` (isolating the
source link on a more interconnected topology cuts off delivery to more of the network at once). This is a
genuine, exact-WNTR-verified safety-gate outcome and is not treated as a model failure or as evidence to
weaken the pressure/service thresholds.

Critically, `ALTERNATE_VALVE_CUT` becoming verified does not automatically mean it differs meaningfully from
`NO_ACTION`: on `branched-loop`, mean `exposure_proxy` is 0.767 (`NO_ACTION`) vs 0.801 (`ALTERNATE_VALVE_CUT`);
on `loop-grid`, 0.744 vs 0.753 -- real, WNTR-verified, non-machine-precision-identical differences (unlike
M10.3B's finding that 5 of 6 non-isolating templates were numerically identical to `NO_ACTION`), but small in
magnitude. This is the direct physical source of Section 5's improved-but-still-modest diversity numbers.

## 5. Candidate diversity / target identifiability / within-incident variance (Section 15/16)

`m10-3c-candidate-diversity.json`, `m10-3c-target-identifiability.json`, `m10-3c-within-incident-variance.json`,
`m10-3c-family-depth-summary.json` re-run M10.3B's own methodology (same functions, same tolerances) per cell,
per family (pooled over depth), and globally (pooled over everything).

**Result** (`plan_value`, pooled across all 18 cells, n=534 incidents with >=2 valid candidates out of 540
generated): 23.0% of incidents have >=2 meaningfully distinguishable candidates (vs M10.3B's 17.5% baseline),
5.6% have >=3 distinguishable clusters (vs M10.3B's 0.0%). Per-family pooled (over all 6 depth cells):

| Family | frac >=2 distinguishable | frac >=3 distinguishable | Support (n) |
|---|---|---|---|
| `golden-reference` | 14.9% | 0.0% | 174 |
| `branched-loop` | **30.6%** | **10.0%** | 180 |
| `loop-grid` | 23.3% | 6.7% | 180 |

A clean, monotonic family ordering emerges: `golden-reference` < `loop-grid` < `branched-loop` on both
diversity metrics, holding consistently across nearly every one of that family's 6 depth cells (per-cell
`plan_value` fraction->=2-distinguishable range: golden-reference 6.7%-32.1%, branched-loop 23.3%-40.0%,
loop-grid 13.3%-33.3% -- overlapping but family-ordered). Only 2 of 18 cells (`branched-loop:depth2`,
`branched-loop:depth4`, both at 40.0%) individually clear the frozen 35% per-cell diversity bar, so the
`DIVERSITY_MIN_CONTRIBUTING_CELLS=3` requirement is not met even though the pooled/family-level improvement is
real and broadly distributed within `branched-loop` specifically.

Other targets (Section 19 -- not every target is required to be non-degenerate): `exposure_proxy` shows the
highest diversity of any target (25.7% >=2-distinguishable, 14.4% >=3-distinguishable, pooled). `pressure_risk_
proxy`/`service_loss_proxy` remain exactly 0%/0% everywhere (expected -- structurally/near-structurally forced
by the same safety gate M10.3B identified; never weakened here). `containment_time_proxy` remains mostly
within-incident-tied (5.1%), consistent with M10.3B's between-incident-severity finding. `plan_regret_proxy`
tracks `plan_value` almost exactly (23.0%/6.6%), consistent with their known exact bijective redundancy.

## 6. Oracle decision-utility analysis (Section 17) -- NON-PROMOTABLE / DIAGNOSTIC ONLY

`m10-3c-oracle-utility.json`: exact best candidate vs `NO_ACTION`, vs random valid candidate, vs first proposed
candidate, per cell/family/global, using the SAME oracle methodology M10.3B established. Oracle truth never
enters any inference-time input; used here only for offline decision-utility measurement.

**Result** (best exact-WNTR-verified candidate vs `NO_ACTION`, `plan_value` scale, pooled n=534): mean gain
0.0269 (vs M10.3B's 0.022), median gain 0.0 (most incidents still have zero or negligible gain), fraction
meaningfully positive 13.7% (vs M10.3B's 9.6%), `NO_ACTION` already near-optimal in 86.3% of incidents (vs
M10.3B's 90.4%). Best-vs-random-valid-candidate and best-vs-first-proposed-candidate gains are nearly identical
to best-vs-`NO_ACTION` (median 0.0 in all three), since `NO_ACTION` is always the first proposed candidate and
usually already the pool optimum or near it.

| Family | mean gain | frac meaningfully positive | frac NO_ACTION near-optimal |
|---|---|---|---|
| `golden-reference` | 0.0138 | 6.3% | 93.7% |
| `branched-loop` | **0.0392** | **18.9%** | 81.1% |
| `loop-grid` | 0.0272 | 15.6% | 84.4% |

Every family shows a real, measurable improvement over M10.3B's single-cell baseline, `branched-loop` most of
all -- but no family reaches the frozen 20%-meaningfully-positive / 0.05-mean-gain / <=75%-NO_ACTION-near-optimal
bar. 13 of 18 cells individually clear the modest oracle per-cell contributing-cell bar (>=10% meaningfully
positive, n>=20), so oracle headroom -- while small in magnitude -- is fairly BROADLY distributed across the
population, unlike diversity's concentration in 2 cells.

## 7. M10.3C population-sufficiency gate (Section 18, frozen in `m10_3c_population_protocol.py` BEFORE any
result was inspected)

Requires BOTH population-diversity AND oracle-utility criteria, each evaluated pooled across all 18 cells, at
thresholds set roughly double (or more) the M10.3B negative-baseline numbers, plus a "not driven by one tiny
cell" contributing-cells requirement. Full frozen decision tree: `m10_3c_population_protocol.GATE_DECISION_TREE`.

**Result**: `global_pass=false`. Pooled diversity: 23.0% >=2-distinguishable (threshold 35%, FAIL), 5.6%
>=3-distinguishable (threshold 10%, FAIL), 2 contributing cells (threshold 3, FAIL) -> `diversity_pass=false`.
Pooled oracle: 13.7% meaningfully positive (threshold 20%, FAIL), mean gain 0.0269 (threshold 0.05, FAIL), 86.3%
NO_ACTION-near-optimal (threshold <=75%, FAIL), 13 contributing cells (threshold 3, PASS) ->
`oracle_pass=false`. Per-family: no family reaches `family_pass=true` (closest: `branched-loop` at
30.6%/10.0%/18.9%/0.039/81.1%, all just short of their respective thresholds of 35%/10%/20%/0.05/<=75% --
`fraction_3plus` alone clears its bar). `golden-reference` is `family_clear_fail=true` (14.9% < the 20%
clear-fail floor). Per the frozen decision tree: no family passes -> `M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED`
(Section 20-B), not CONDITIONAL (Section 20-C requires >=1 family to actually clear the FULL pass bar, which
none does here, even though a real family-ordering trend exists).

## 8. Decision

**`M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED`.**

Expanding to the already-governed `branched-loop`/`loop-grid` topology families and the earlier/mid causal-
prefix depth labels produced a real, physically grounded, exact-WNTR-verified improvement over the M10.3B
golden-reference/depth-25 pilot -- most concretely, `ALTERNATE_VALVE_CUT` becomes safely feasible (100%
verified) on both looped families, confirming M10.3B's own hypothesis that topology redundancy can restore
isolation-style action feasibility. Diversity and oracle-utility numbers improve roughly 1.3x-2x over the
M10.3B baseline in the best family (`branched-loop`), and a clean, consistent `golden-reference` <
`loop-grid` < `branched-loop` ordering emerges across nearly every depth cell within each family.

**But the improvement is not large enough.** Neither the pooled population nor any individual family clears the
preregistered M10.3C gate (Section 7 above) -- `branched-loop`, the best-performing family, falls short on
every one of its five criteria simultaneously (30.6% vs 35% threshold diversity; 18.9% vs 20% threshold /
0.039 vs 0.05 threshold oracle gain; 81.1% vs <=75% threshold NO_ACTION-near-optimal), not merely on one
marginal criterion. Per the frozen decision tree (Section 18/20 of the task spec, `GATE_DECISION_TREE`), because
no family reaches `family_pass=true`, Section 20-C (`M10_3C_POPULATION_IDENTIFIABILITY_CONDITIONAL`) does not
apply even though a real, physically meaningful family-ordering trend exists -- CONDITIONAL requires a family to
actually clear the full bar, not merely rank better than the others, precisely to prevent declaring victory on
a trend that is real but insufficiently strong. Section 20-A
(`M10_3C_POPULATION_IDENTIFIABILITY_PASS`) is directly contradicted by the pooled numbers. This is therefore a
genuine, well-evidenced instance of Section 20-B: **the expanded population remains substantially closer to
NO_ACTION-dominated near-ties than to a population with genuine, learnable, multi-candidate decision value**,
even though it is measurably less degenerate than the original pilot.

Per M10.3B's own closure guidance (Section 10 of `HYDROCORE_V5_M10_3B_STRATEGIST_DIAGNOSIS.md`): "If the
expanded population shows the same degeneracy, `M10_3B_LEARNED_STRATEGIST_NOT_JUSTIFIED` becomes the
well-evidenced conclusion, and the system should retain the deterministic candidate generator + deterministic
Strategist + exact WNTR verification permanently for this decision, proceeding to M10.4 on that basis." The
population is not literally "the same degeneracy" (it is measurably better), but it does not clear the bar this
task froze in advance specifically to distinguish a meaningful improvement from a marginal one -- so that same
guidance applies. **Recommendation: retain the deterministic candidate generator, deterministic Strategist, and
exact WNTR verification permanently for this decision (no further Strategist-population amendment is expected
to change this conclusion without either a materially larger population, a materially more redundant/diverse
topology regime than what is already governed, or a change to the underlying candidate-template/action
vocabulary -- none of which this task is authorized to pursue). Proceed to M10.4 on this basis.**

## 9. Locked-test status

`locked_test_opened_before=false`, `locked_test_opened_after=false` throughout every phase of this task
(protocol freeze, seed-disjointness verification, population generation, all diagnostic artifacts, gate,
closure). Never inspected. `locked_final_test`/`locked_topology_test` were not used to create candidate
diversity anywhere in this population -- only the 3 already-governed TRAINED_FAMILIES were used.

## 10. Output governance (unaffected)

No checkpoint was created, loaded, or modified. Learned Strategist remains runtime-disabled and
non-authoritative. Deterministic Strategist and WNTR/EPANET verification remain unmodified and retained as
runtime authority. No `runtime_enabled_outputs` promotion occurs. M9/M10/M10.3A/M10.3B historical artifacts and
checkpoints are unchanged (verified by `git status`/`git diff` on those paths at closure time). This task does
not authorize M10.3D, true M10.3, a full/shared HydroCore retrain, or M10.4 -- it only determines whether
M10.3D is scientifically justified.
