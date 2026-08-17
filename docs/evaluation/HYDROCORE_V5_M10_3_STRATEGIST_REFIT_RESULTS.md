# HydroCore-v5 Milestone 10.3A Strategist refit results (executed under the frozen protocol)

Amends nothing in `HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`, which remains frozen exactly as written
before Level A executed (`protocol_hash` `f73accbf548e9b8987b8b1258efd7d3e61e052f802714ace3bb5f8b1b8d0f587`,
unchanged throughout Level A and Level B execution). This document records the result.

## Result: `M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED`

Level A is mechanically valid (all 21 gradient-coverage certificates -- 7 tasks x 3 seeds -- pass; adequate
support; no NaN/Inf) but fails the frozen representation-sufficiency gate for all three seeds. The Level-B
escalation trigger legitimately fired (Section 9's own rule: mechanically valid + genuine, non-degenerate
competence failure). Level B was executed under the already-frozen Section-8 scope (Level-A's 40 parameters +
`backbone[3]`'s 24 parameters + `final_norm.weight`, 65 total) and ALSO fails: it does not materially improve
Strategist competence over Level A, AND it damages M9 Sentinel/calibration preservation for all three seeds.
Per the frozen protocol, Level B is rejected on the preservation failure alone, regardless of its own
competence result. Level A's own checkpoint (never Level B's) is retained as the task's best available,
**not-promoted** artifact.

## Level-A competence: two objectives learn strongly, the rest do not

| Metric | Seed 20260814 | Seed 31874 | Seed 20260815 | Gate |
|---|---|---|---|---|
| `plan_validity` AUROC | 0.976 | 0.974 | 0.976 | **PASS** (>>0.5, CI-supported) |
| `containment_time_proxy` Spearman | 0.793 | 0.772 | 0.772 | **PASS** (CI excludes 0) |
| `plan_value` Spearman | -0.007 | 0.069 | -0.018 | FAIL |
| `exposure_proxy` Spearman | 0.042 | -0.057 | 0.043 | FAIL |
| `plan_regret_proxy` Spearman | -0.030 | -0.039 | -0.137 | FAIL |
| `pressure_risk_proxy` | n/a (degenerate) | n/a (degenerate) | n/a (degenerate) | EXCLUDED |
| `service_loss_proxy` | n/a (degenerate) | n/a (degenerate) | n/a (degenerate) | EXCLUDED |
| within-incident pairwise ranking | 0.404 | 0.486 | 0.202 | FAIL |

n=2700 candidates (`plan_validity`), n=1734 valid candidates (proxies/`plan_value`), n=835 real non-tied pairs
(ranking), from `reports/evaluation/hydrocore-v5/m10/m10-3-refit/m10-3-refit-level-a-gate.json`.

**`pressure_risk_proxy`/`service_loss_proxy` are mechanically excluded from the Level-B-eligibility
determination**, per the authorizing task's own explicit instruction ("Do NOT escalate to B because of ...
severe target imbalance invalidating metrics"): both targets have (numerically) zero variance in this
validation population -- `baseline_mse` is `0.0` / `3.6e-23` and Spearman is undefined (`NaN`) for every seed.
The `golden-reference` network's WNTR-verified valid candidates essentially never produce a measurable pressure
violation or service-availability loss in this population; "beat the constant-zero baseline" is not a
meaningful competence test here. This is reported as a population/metric-design limitation, not evidence
either for or against representation sufficiency.

The remaining failures (`plan_value`, `exposure_proxy`, `plan_regret_proxy`, pairwise ranking) are genuine,
non-degenerate (real variance, computable Spearman) competence gaps -- these are what legitimately fired the
Level-B trigger. Notably, `exposure_proxy`'s own training-population mean is `0.9997` (nearly every valid
candidate's exposure outcome is almost identical to doing nothing) and `plan_value`'s is `0.984` -- a real, if
not literally zero, low-variance/near-ceiling population, disclosed here as context for interpreting the
near-zero correlations, not as an excuse to reclassify them as degenerate.

## Level B: executed per the frozen trigger, does not resolve the gap, and damages M9 preservation

Level-B competence numbers are essentially unchanged from Level A's own pattern -- `plan_validity`
(AUROC 0.973-0.976) and `containment_time_proxy` (Spearman 0.758-0.814) still pass strongly; `plan_value`
(Spearman -0.190 to 0.031), `exposure_proxy` (-0.049 to 0.020), and pairwise ranking (0.187-0.485, still at or
below chance) still fail. Unfreezing one more backbone block did not give the model access to information it
was missing for these specific objectives -- consistent with (though not conclusive proof of) the low-target-
variance population hypothesis above, rather than a pure representation-capacity limitation.

**M9 preservation failed for all three seeds** (`reports/evaluation/hydrocore-v5/m10/m10-3-refit/
m10-3-refit-preservation.json`), independently disqualifying Level B regardless of its competence result:

| Seed | Teacher calibration coverage | Level-B calibration coverage | Floor | CI-confident Sentinel-task regressions |
|---|---|---|---|---|
| 20260814 | 0.860 | 0.823 | 0.85 | `source_region`, `start_time`, `event_presence`, `event_cause` |
| 31874 | 0.847 | 0.777 | 0.85 | `source_region`, `start_time`, `event_presence`, `event_cause`, `sensor_fault` |
| 20260815 | 0.803 | 0.667 | 0.85 | `start_time`, `duration`, `relative_strength` |

Calibration coverage drops below the frozen 0.85 acceptance floor in every seed, and multiple (varying by seed)
Sentinel tasks show a CI-confident paired-bootstrap regression against the unmodified M9.6 teacher on the same
development population. This is exactly the failure mode the M9-preservation gate exists to catch -- unfreezing
even one additional backbone block measurably perturbs the already-validated Sentinel/calibration behavior.
Per the frozen protocol, no calibration refit is performed to rescue this; Level B is rejected outright.

## Why this is a valid, complete M10.3A outcome, not an incomplete run

Both disqualifying findings (Level B fails to improve competence, and separately damages M9 preservation) are
independently sufficient to block promotion; together they leave no ambiguity. Per the frozen protocol's own
explicit language: "If Level B fails Scout[sic -- Strategist] competence, or regresses M9 preservation, or
would require recalibration/broader unfreezing to pass: `M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED`.
Level A's own (non-B) checkpoint is retained as the task's best available artifact in that case, clearly
labeled as not promoted." This document does exactly that. No full/joint retrain was performed (out of this
task's authorization) or attempted.

## Checkpoint / provenance

Three Level-A refit checkpoints and three Level-B refit checkpoints exist
(`reports/evaluation/hydrocore-v5/m10/m10-3-refit/checkpoints/level-{a,b}-seed{20260814,31874,20260815}/`),
each with a `checkpoint_identity.json` recording parent M9.6 teacher SHA-256, exact trainable-parameter
allowlist (40 for Level A, 65 for Level B), candidate-training-schema version, train/validation manifest
hashes, seed, optimizer-config hash, `"FINAL_EPOCH"` checkpoint-selection policy, gradient-coverage-certificate
hash, git commit, and the refit model's own SHA-256. `never_call_this_m9_6=true` on every one. All three
original M9.6 teacher checkpoint SHA-256 hashes verified unchanged before and after both Level A and Level B
executed. Level-A and Level-B checkpoints for the same seed are confirmed byte-distinct.

## Output governance (unaffected)

Learned Strategist remains runtime-disabled and non-authoritative. `HydroStrategist.deterministic_fallback`
was not modified and was independently confirmed structurally unaffected by (has no reference to)
`candidate_plan_encoder`/`generate_response_plans`. WNTR/EPANET verification remains final authority
throughout -- every `plan_validity` target used in this refit came only from `PlanVerifier.verify()`'s own
decision. No `runtime_enabled_outputs` promotion occurred; this result (not promoted) means no runtime-
promotion question even arises this cycle.

## Readiness

`M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED`. The true M10.3 learned-vs-deterministic Strategist
scientific comparison is **not** scientifically ready and is **not** authorized to proceed from this task's
output -- no refit checkpoint produced here is promoted. A future, separately authorized amendment would need
to determine whether broader shared-representation retraining is scientifically warranted for `plan_value`/
`exposure_proxy`/ranking specifically, or whether this population's disclosed low target variance for those
objectives makes them inherently hard to learn regardless of capacity -- this task does not resolve that
question and does not attempt the retrain. M10.4/M10.5 are not addressed by this task.
