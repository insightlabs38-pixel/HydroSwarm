# HydroCore-v5 Milestone 10.4 results: governed full-trajectory end-to-end validation

Amends nothing in `HYDROCORE_V5_M10_4_FULL_TRAJECTORY_PROTOCOL.md`, which remains frozen exactly as
written before execution (protocol hash `cd0ac1f2d5a12a771cc441b4ea19bf0d76c672809b35d3d178f8893b768a177c`,
unchanged, reconfirmed identical in every artifact this run produced). No population, threshold, or
metric was changed after any result was inspected -- the one post-execution addition
(`m10-4-gate.json`'s `gate_e_evaluated`/`gate_e_detail` fields) is a transparency annotation explaining
that Criterion E was vacuously satisfied, not a threshold or population change (see "Honest limitations"
below).

## Result: `M10_4_FULL_TRAJECTORY_PASS`

All 7 frozen utility-gate criteria (A-G) pass, all 13 hard safety counters are exactly zero, and all 5
fail-closed cases resolve to bounded, deterministic outcomes. `reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json`
is the authoritative record.

The central M10.4 question -- does the retained production system (canonical HydroCore-S, frozen
uncertainty governance, deterministic active sampling, deterministic planning, exact WNTR verification)
operate as a causally valid, safe, useful end-to-end incident-response trajectory -- is answered **yes**,
with the specific, disclosed limitations below.

## Population executed

360 physical incidents (24 population cells x 5 incidents x 3 canonical model seeds), each realized as a
byte-identical-initial-evidence PAIR of production-API incidents (ARM_FULL / ARM_NO_EXTRA_SAMPLING) = 720
total API incidents, all through the real `hydroswarm.api.create_app` FastAPI application. `paired_initial_state_equal`
held for **100%** (360/360) of pairs. Execution took 342.6s wall-clock (3 model seeds x 120 incident-pairs
each). `m10-4-population-manifest.json` records every physical seed; `m10-4-seed-disjointness.json`
mechanically confirms the `[1_500_000_000, 1_599_999_999]` M10.4 range does not overlap any of the 9 prior
milestone seed ranges it was checked against.

## Preflight

`m10-4-preflight.json`: **`M10_4_PREFLIGHT_PASS`**, 22/22 checks, 2 disclosed non-blocking findings
(module-level `hydroswarm.api.app` still serves the older pre-M9 v4 checkpoint -- M10.4 uses its own
pipeline factory instead, see the protocol document; and `HybridInferencePipeline.analyze()`'s hard-coded
feature-builder kwargs, an unremarked characteristic already present at M10.0's own closed preflight).

## Source-inference metrics (`m10-4-source-trajectory.json`)

| slice | n | final top-1 | final top-3 | final MRR | calibrated | actionable |
|---|---|---|---|---|---|---|
| overall | 360 | 0.819 [0.784, 0.850] | 0.933 | 0.877 | 0.875 | 0.619 |
| golden-reference | 105 | 0.857 | -- | -- | 1.000 | -- |
| branched-loop | 105 | 0.771 | -- | -- | 1.000 | -- |
| loop-grid | 105 | 0.810 | -- | -- | 1.000 | -- |
| coastal-branch (dev topology-shift) | 15 | 0.800 | -- | -- | **0.000** | -- |
| tree-branch (dev topology-shift) | 15 | 1.000 | -- | -- | **0.000** | -- |
| dense-loop (dev topology-shift) | 15 | 0.800 | -- | -- | **0.000** | -- |

The three development-only topology-shift families show **0.000** calibrated rate across all 45 incidents
-- the deterministic topology-novelty OOD/calibration gate (`OODDetector.topology_novelty`,
`validated_topology_hashes` restricted to the 3 TRAINED_FAMILIES) correctly and consistently refuses to
treat an unseen topology as calibrated, exactly as intended. Source top-1 on unseen topologies is not
degenerate (0.80-1.00) -- the classical/neural belief itself still localizes reasonably -- but the system
correctly withholds the calibrated/actionable state that would be required to plan on it, matching the
non-negotiable safety requirement that an unfamiliar topology never be silently treated as trustworthy.

By condition (pooled across families/seeds; `LOW_COVERAGE_ACTIVE_SAMPLING` and `AMBIGUITY_DISAGREEMENT`
are the hardest conditions in this population):

| condition | n | final top-1 | actionable |
|---|---|---|---|
| NOMINAL | 90 | 0.933 | 0.400 |
| LOW_COVERAGE_ACTIVE_SAMPLING | 45 | 0.467 | 0.622 |
| SENSOR_DROPOUT | 45 | 1.000 | 0.711 |
| SENSOR_HEALTH_DEGRADED | 45 | 0.956 | 0.756 |
| MEASUREMENT_NOISE | 45 | 0.800 | 0.667 |
| SEVERITY_SHIFT | 45 | 0.867 | 0.933 |
| AMBIGUITY_DISAGREEMENT | 45 | 0.600 | 0.467 |

`NOMINAL`'s low actionable rate (0.400) is pulled down by the 45 unseen-topology incidents it contains
(which never calibrate, by design); its top-1 (0.933) is the best of any condition, consistent with it
being the easiest localization condition. `LOW_COVERAGE_ACTIVE_SAMPLING` (25% initial sensor coverage,
deliberately designed to exercise the Scout loop) has the lowest final top-1 (0.467) -- this is the
hardest condition in the population BEFORE any sampling and is discussed below.

By model seed: 20260814 0.817, 31874 0.817, 20260815 0.825 -- consistent across all three canonical
checkpoints, no seed-specific pathology.

## Scout / evidence-acquisition (`m10-4-scout-trajectory.json`)

18.1% of incidents (65/360) requested >=1 sample; mean 0.406 samples/incident overall (up to 3). 146 total
sample rounds. Stop-reason distribution: `PLANNING_ALLOWED` (219 -- already actionable, no sampling
needed), `"current analysis does not request sampling"` (94 -- calibration/OOD suppressed sampling,
correctly not attempted), `BUDGET_EXHAUSTED` (30), `marginal_value_below_threshold` (17 -- the
deterministic stop policy correctly recognized diminishing returns). Per sample round: mean true-source-rank
change +0.075 (positive = improves), 43.2% of individual sample rounds improved the true-source rank,
mean entropy reduction 0.271 bits. **Zero** safety-counter violations across all 146 rounds (no
reselection, no inaccessible selection, no budget overrun).

## Strategist / plan (`m10-4-strategist-trajectory.json`)

Mean 0.656 candidates generated/incident (budget-respecting `count=2`, verification stopped at the first
VERIFIED result to stay within the real, unmodified `exact_plan_simulation_limit=3` per-incident WNTR
budget -- see the protocol/comparator design). Mean 0.583 candidates WNTR-verified, 0 rejected on average
(most incidents that generate a candidate at all generate one that verifies). No-safe-plan rate: 3.6%
[2.3%, 5.6%]. `NO_ACTION` never appeared among the top-2 generated candidates in this population
(`no_action_available_rate = 0.0` -- see "Honest limitations"). Human-approved rate: 58.3% [54.0%, 62.5%]
of all 360 incidents ended with a human-approved, WNTR-verified plan.

## Comparator: ARM_FULL vs ARM_NO_EXTRA_SAMPLING (`m10-4-comparator.json`)

| | ARM_FULL | ARM_NO_EXTRA_SAMPLING |
|---|---|---|
| final top-1 | 0.819 [0.784, 0.850] | 0.792 [0.754, 0.825] |

Active Scout sampling is associated with a modest (+2.8pp, CIs overlapping) improvement in final top-1
accuracy, non-inferior by a wide margin against the frozen 5-point non-inferiority gate (Criterion B). Among
the 65 pairs where sampling actually occurred: 20 incidents had their top-1 correctness IMPROVED by
sampling, 10 had it WORSENED, 35 unchanged -- net positive (Criterion C), and the 90% margin (>= -7
incidents) is comfortably cleared. Sampling never changed which action template was ultimately approved in
this population: among the 210 paired incidents where BOTH arms reached a human-approved plan, the
approved action was **identical** in all 210; the remaining 150 pairs had NEITHER arm reach an approved
plan. In other words, in this population, active sampling helps localization accuracy without ever
flipping the final operational decision -- a genuine, non-cherry-picked finding, not a metric artifact
(confirmed by direct inspection of the raw per-pair `action_types`, not just the pooled
`fraction_final_decision_changed_by_sampling = 0.0` figure, which is consistent with it).

## Physical outcomes vs NO_ACTION (`m10-4-physical-outcomes.json`)

210/360 incidents produced a human-approved, WNTR-verified plan with real exact-simulation consequence
metrics (exposure, pressure, service availability, containment time -- confirmed populated and finite for
every verified plan sampled during development, e.g. `population_impacted`, `minimum_pressure_m`,
`service_availability`). **0 incidents had a NO_ACTION candidate available for paired comparison** --
Strategist's deterministic ranking never placed `NO_ACTION` in the top-2 generated candidates across this
entire population, so the selected-plan-vs-NO_ACTION delta table is entirely `null`. See "Honest
limitations."

## Fail-closed cases (`m10-4-fail-closed.json`)

All 5 cases (`MODEL_UNAVAILABLE`, `CALIBRATION_UNAVAILABLE`, `SENSOR_STATE_INSUFFICIENT`,
`SAMPLING_BUDGET_PREEXHAUSTED`, `NO_ACCESSIBLE_UNSAMPLED_CANDIDATE`) resolved boundedly and
deterministically: `MODEL_UNAVAILABLE` fell back to the classical-only posterior (`neural_belief=None`)
and correctly `ABSTAIN`ed rather than plan on an unavailable model; `CALIBRATION_UNAVAILABLE` correctly
reported `calibrated=False`/`ABSTAIN`; `SENSOR_STATE_INSUFFICIENT` (zero observations) was rejected at
incident creation (422); `SAMPLING_BUDGET_PREEXHAUSTED` (`maximum_samples=0`) rejected the incident at
creation (governed input validation, 422); `NO_ACCESSIBLE_UNSAMPLED_CANDIDATE` (100% initial sensor
coverage) correctly 409'd the sample-recommendation endpoint once no unsampled candidate remained.

## Hard safety counters (`m10-4-safety-counters.json`)

All 13 counters are **0**: `inaccessible_sample_selected`, `already_sampled_reselected`,
`sampling_budget_exceeded`, `unverified_plan_surfaced_as_actionable`, `wntr_rejected_plan_surfaced_as_safe`,
`human_approval_bypassed`, `autonomous_actuation_detected`, `learned_ood_overrode_deterministic`,
`learned_scout_selected_sample`, `learned_strategist_selected_plan`, `nonfinite_value_reached_decision`,
`locked_test_opened`, `invariant_failures`. `locked_test_opened` is `false` both before and after.

## Honest limitations (disclosed, non-blocking)

1. **Gate E (physical outcome vs NO_ACTION) is vacuous, not a real characterization.** Plan generation was
   deliberately bounded to `count=2` candidates, verified in ranked order and stopped at the first VERIFIED
   result, to respect the REAL, unmodified per-incident `exact_plan_simulation_limit=3` WNTR budget (see
   protocol Section 4) -- exactly matching `hydroswarm.evaluation.live_robustness`'s own established
   "measurement" lifecycle convention (`count=2`), not an invented smaller budget. Under this population's
   incidents, `NO_ACTION` never ranked in the top 2 candidates, so no selected-vs-NO_ACTION physical delta
   was ever observed. `m10-4-gate.json`'s `gate_e_evaluated=false` and `gate_e_detail` record this
   explicitly; Criterion E is not claimed as a positive demonstration of non-harm, only as "nothing to
   contradict it." A future milestone wanting a real NO_ACTION-paired physical-outcome characterization
   would need a dedicated, separately-frozen protocol amendment (e.g. explicitly requesting the NO_ACTION
   template as one of the generated candidates) -- not something this run's frozen design supports, and not
   something changed here after seeing this result.
2. **`fraction_final_decision_changed_by_sampling = 0.0`** is a real, confirmed finding (verified against
   raw per-pair data, not merely the pooled statistic), but the definition treats "neither arm approved a
   plan" as "unchanged" -- 150 of the 360 pairs fall in that bucket. Restricting to the 210 pairs where BOTH
   arms reached an approved plan, the approved action was still identical in all 210 -- the 0.0 finding is
   robust to this alternative framing, not an artifact of it.
3. **Development-only unseen-topology families (n=15 each) are small** relative to the trained families
   (n=105 each) -- adequate to demonstrate the deterministic topology-novelty gate fires consistently
   (0/45 calibrated), not intended as a precision estimate of unseen-topology localization accuracy.

## Utility gate (`m10-4-gate.json`)

| criterion | result |
|---|---|
| A: all hard safety gates pass | PASS |
| B: no material top-1 regression vs no-extra-sampling (5pp margin) | PASS (+2.8pp, non-inferior) |
| C: active sampling non-harmful where exercised | PASS (improved=20, worsened=10, of 65) |
| D: every approved plan WNTR-VERIFIED | PASS |
| E: no material exposure/service/pressure harm vs NO_ACTION | PASS (vacuous -- see limitation 1) |
| F: all outputs finite | PASS |
| G: fail-closed behavior valid | PASS |

## Closure

**`M10_4_FULL_TRAJECTORY_PASS`.** Integration is valid, every hard safety/authority gate passes, and the
retained system meets the frozen trajectory utility/quality gate. This does **not** authorize M10.5
(serving-path freeze, runtime promotion, or opening the locked test) -- `m10-4-closure.json`'s
`m10_5_authorized` is explicitly `false`. M10.5 requires separate, explicit authorization.
