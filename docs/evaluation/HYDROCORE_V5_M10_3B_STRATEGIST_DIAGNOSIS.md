# HydroCore-v5 Milestone 10.3B: Strategist target-identifiability + failure-diagnosis amendment

Additive to `HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`/
`HYDROCORE_V5_M10_3_STRATEGIST_REFIT_RESULTS.md`, which remain frozen and unmodified. Does not reopen,
reverse, retrain, or reinterpret-to-pass any M10.3A result. `M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED`
remains M10.3A's closure exactly as written. This document diagnoses **why** Level A/B failed, using only
already-authorized development data, with no training and no checkpoint access.

Diagnostic script: `scripts/hydrocore_v5/run_m10_3b_diagnosis.py` (+ a small supplementary rejection-code
probe folded into `m10-3b-candidate-diversity.json`, and `scripts/hydrocore_v5/write_m10_3b_root_cause_and_
closure.py`). Reuses the SAME frozen M10.3A population unmodified (`m10_3_refit_protocol.py`: golden-reference
family, `TRAIN_SEED_BASE=1_300_000_000`/`TRAIN_COUNT=250`, `VALIDATION_SEED_BASE=1_300_100_000`/
`VALIDATION_COUNT=300`) via the SAME `_build_corpus` M10.3A's own gate script already reuses.

## Central question, answered

**Did M10.3A fail because HydroCore needs broader retraining, or because the current Strategist
objective/population does not contain enough correct, legally observable, within-incident decision signal
to justify such retraining?**

**The latter.** Every failed Level-A criterion traces to `CANDIDATE_POPULATION_DEGENERACY` (with
`pressure_risk_proxy` additionally `TARGET_DEGENERACY` by mathematical construction), not to
`FROZEN_REPRESENTATION_INSUFFICIENT`. No mechanical target/ranking/sign/alignment/leakage defect was found
anywhere. `M10_3B_DECISION`: **`M10_3B_POPULATION_AMENDMENT_REQUIRED`** -- see Section 8 below.

## 1. Target-formula audit (`m10-3b-target-formula-audit.json`)

Every one of the 7 governed Strategist targets was traced from `strategist_labels.py` through
`plan_value_policy.py` and cross-checked against a controlled synthetic candidate pool (A clearly best, B
intermediate, C clearly worst, verified end to end through the real `evaluate_plan_value()`): plan_value
monotonicity `A > B > C` and regret monotonicity `A < B < C` both hold exactly. Every formula matches its
frozen M10.3A Part-4 definition verbatim -- no drift, no implementation defect.

**New finding not previously documented**: `pressure_risk_proxy` is not merely empirically near-zero in this
population -- it is **mathematically guaranteed exactly zero for every `plan_validity=True` candidate**, by
construction of the safety gate. `PlanVerifier` rejects (`PRESSURE_BELOW_MINIMUM`) any plan whose simulated
minimum pressure falls below `minimum_pressure_m` (10.0m default), and `pressure_violation_minutes` is
computed against that SAME threshold (`src/hydroswarm/simulation/wrapper.py:1341/1349` for the hydraulic-only
path, `:1477` for the exposure-aware path strategist-label generation actually uses). `pressure_violation_
minutes > 0` at any timestep implies the simulation-wide minimum pressure also falls below threshold, which
implies rejection. Confirmed both by source-code trace and empirically (`std=0.0`, `n_unique=1` across all
1,734 valid validation candidates) and by a direct unit test
(`tests/unit/test_m10_3b_diagnosis.py::test_pressure_risk_proxy_cost_component_present_when_nonzero`, which
proves the formula itself correctly incorporates a nonzero value when one reaches it -- the zero is a
population/gate property, never a formula bug). **This must never be "fixed" by weakening the pressure
safety threshold** (frozen governance rule).

`plan_value`/`plan_regret_proxy` are also confirmed **exact, deterministic, bijective (monotone-decreasing)
transforms of one another** (`plan_value = 1/(1+regret)`, `plan_regret_proxy = regret`, same `regret` local
variable, `plan_value_policy.py:127/136/145`) -- empirically verified (Spearman `-0.9999991`, max
reconstruction error `3.2e-8`) and by direct unit test. They are never independently informative.

## 2. Ranking/sign/alignment audit (`m10-3b-ranking-alignment-audit.json`)

Six independent mechanical checks, all passed: perfect-agreement-scores-100%, inverted-prediction-scores-0%,
permutation invariance of the pairwise-ranking metric itself, padded-slot exclusion from the pairwise count,
confirmation the gate ranks only `plan_value` (never accidentally mixed with the lower-is-better
`plan_regret_proxy`), and confirmation `candidate_tensorizer.py` keys every INPUT row to its owning
`PlanProposal` in a single content-preserving pass with no reordering step. **No mechanical sign, ordering,
masking, or indexing defect exists anywhere in the ranking pipeline.** The near-chance/sub-chance pairwise-
ranking accuracy M10.3A observed is not explained by any such defect (Section 5 below explains what it IS
explained by). Ten new unit tests added: `tests/unit/test_m10_3b_diagnosis.py`.

## 3. Within-incident target identifiability (`m10-3b-target-identifiability.json`,
`m10-3b-within-incident-variance.json`)

Preregistered near-tie tolerances (defined here, before any per-incident result was inspected; see the
artifact's own `tolerance_definitions`): `service_loss_proxy` reuses the repository's own governed
`SERVICE_AVAILABILITY_SENSITIVITY_EPSILON=0.02`; `exposure_proxy`/`plan_value`/`plan_regret_proxy` use a
1%-of-baseline / propagated-additive-cost argument; `pressure_risk_proxy`/`containment_time_proxy` use a
"1 simulator-reported minute is the smallest physically distinguishable difference" argument on each
proxy's own train-owned scale.

| Target | Validation: incidents w/ 2+ meaningfully distinguishable candidates | incidents w/ 3+ distinguishable clusters | incidents all-tied |
|---|---|---|---|
| `plan_value` | 17.5% | 0.0% | 82.5% |
| `exposure_proxy` | 23.0% | 0.0% | 77.0% |
| `pressure_risk_proxy` | 0.0% | 0.0% | 100.0% |
| `service_loss_proxy` | 0.0% | 0.0% | 100.0% |
| `containment_time_proxy` | 5.2% | 0.0% | 94.8% |
| `plan_regret_proxy` | 17.5% | 0.0% | 82.5% |

**Zero incidents, for any target, have 3+ meaningfully distinguishable candidate values.** Global (pooled)
statistics look far less degenerate than this (e.g. `containment_time_proxy` has 14 distinct global values
and a real IQR) -- the degeneracy is specifically a WITHIN-INCIDENT phenomenon, invisible to any pooled
metric.

## 4. Root physical mechanism: candidate-population degeneracy (`m10-3b-candidate-diversity.json`)

Every incident proposes all 9 templates (frequency 300/300 each on the 300-scenario validation split; 9
real candidates/incident, never fewer). But:

- `ISOLATE_SOURCE`, `ISOLATE_AND_FLUSH`, `ALTERNATE_VALVE_CUT` (the 3 link-target templates capable of a
  materially different physical outcome) are **`VERIFIED` in 0/300 incidents** -- always rejected.
- A targeted supplementary probe (20 golden-reference scenarios, `StrategistLabel.rejection_codes` inspected
  directly) confirms **60/60 rejections are exactly `('PRESSURE_BELOW_MINIMUM',)`** -- never
  `UNKNOWN_TARGET`/`INOPERABLE_TARGET` (which would indicate a prescreen/topology-availability defect) and
  never `SERVICE_BELOW_MINIMUM`. This is a genuine, exact-WNTR-verified safety-gate rejection: closing the
  isolating link(s) these templates require drives some node's pressure below `minimum_pressure_m` on this
  network's own topology, every time, at this severity regime.
- Of the 6 templates that DO pass verification, 5 (`NO_ACTION`/`PROTECT_CRITICAL`/`INCREASE_MONITORING`/
  `REQUEST_SAMPLE`/`WAIT_OBSERVE`) never modify the network at all and are **numerically identical** to
  `NO_ACTION`'s own consequences to machine precision (`exposure_proxy_mean=1.0`,
  `containment_time_proxy_mean=1.9441580756013745` for all five). Only `FLUSH_DOWNSTREAM` differs, and only
  marginally (`exposure_proxy_mean=0.994`).

**This is the direct physical cause of Section 3's near-total within-incident degeneracy**: in the vast
majority of incidents, every valid candidate is either exactly `NO_ACTION`-equivalent or a marginal variant
of it -- there is almost nothing left to rank. This is the authority/safety gate working exactly as governed
and must not be weakened; it is a property of the deterministic candidate generator's interaction with THIS
network topology and severity regime, not of the model.

## 5. Feature identifiability and oracle utility (`m10-3b-feature-identifiability.json`,
`m10-3b-oracle-utility.json`) -- NON-PROMOTABLE / DIAGNOSTIC ONLY

Legal, pre-verification template identity explains only **0.11%** of `plan_value`'s total variance -- not
because the signal is hidden from the model, but because there is almost no outcome variance for any signal
to explain (Section 4).

Oracle utility (perfect knowledge of exact WNTR-verified labels, never a deployable policy): **90.4%** of
incidents already have `NO_ACTION` within the near-tie tolerance of the pool-optimal candidate. Even a
perfect oracle beats trivial `NO_ACTION` by a meaningful margin in only **9.6%** of incidents, with a small
mean gain (`0.022` plan_value units) even then. **This population offers very little decision utility for
any Strategist -- learned or deterministic -- to capture**, independent of representation capacity.

## 6. `containment_time_proxy`: a gate-interpretation finding, not a genuine within-incident pass

M10.3A's Level-A gate criterion 4 computes MSE/Spearman **pooled across all validation candidates**
(`run_m10_3_level_a_gate.py`'s own `_spearman_ci`), never per-incident -- unlike the dedicated within-incident
pairwise-ranking criterion, which exists only for `plan_value`. `containment_time_proxy` has real
BETWEEN-incident variance (14 distinct global values, IQR 1.75) plausibly driven by incident-level
severity/topology, but only 5.2% of incidents show meaningful WITHIN-incident spread. Its strong pooled-
Spearman "pass" in M10.3A most likely reflects the model learning to predict per-incident severity (a real,
legitimate signal, obtainable even from `NO_ACTION`-only candidates), not genuine within-incident candidate
discrimination -- the same population degeneracy as everything else, simply not caught by a pooled metric the
way it was caught by `plan_value`'s dedicated within-incident criterion. Classified
`METRIC_OR_GATE_MISINTERPRETATION` + `TARGET_DEGENERACY`, not `FROZEN_REPRESENTATION_INSUFFICIENT`.

## 7. Calibration-preservation protocol-interpretation audit (`m10-3b-calibration-preservation-audit.json`)

PROTOCOL-INTERPRETATION AUDIT ONLY -- does not reopen or reverse M10.3A's Level-B rejection. The M9-wide
`OPERATIONAL_COVERAGE_FLOOR=0.85` (`m9_4_common.py`, governing the original multi-family/full-depth-grid M9
operational-evaluation population) was reused verbatim as an absolute floor for M10.3A's much smaller,
single-family/single-depth development population. The **unmodified M9.6 teacher itself** already scores
below 0.85 on 2/3 seeds on this specific population (seed 31874: 0.8467, seed 20260815: 0.8033) --
**before Level B touches anything**. This shows the floor, while valid for the population it was governed
for, is not automatically a well-calibrated absolute cutoff for every smaller development population a later
milestone evaluates on.

**This does not change the M10.3A Level-B rejection**: Level B is independently, sufficiently disqualified by
CI-confident paired-bootstrap regressions against its own unmodified parent teacher on the SAME population
(`source_region`/`start_time`/`event_presence`/`event_cause`, varying by seed) -- a relative criterion
unaffected by where the absolute floor sits. **Recommendation for future full/shared-refit preservation
gates**: require BOTH (A) no CI-confident paired regression against the checkpoint's own frozen parent
teacher on the same population (self-relative, population-invariant), AND (B) calibration validity under the
calibration regime's own governed population/support set, not an absolute floor number transplanted onto a
smaller population it was never calibrated against.

## 8. Leakage audit (`m10-3b-leakage-audit.json`)

Re-confirmed, extended: candidate order is the fixed canonical `ACTION_TEMPLATES` literal order in every one
of 300 validation incidents (`NO_ACTION` always first, zero violations) -- driven only by
`PlanGenerationContext`'s current-evidence-derived fields, never exact WNTR outcome or future truth.
`_reconstruct_context_and_proposals`'s own source contains no reference to `manifest.incident`/
`incident_truth`. No leakage found, consistent with M10.3A's own Part 5.

## 9. Root-cause classification (`m10-3b-root-cause.json`)

| Criterion | Classification |
|---|---|
| `plan_value` | `CANDIDATE_POPULATION_DEGENERACY` |
| `exposure_proxy` | `CANDIDATE_POPULATION_DEGENERACY` |
| `pressure_risk_proxy` | `TARGET_DEGENERACY` (mathematically forced) + `CANDIDATE_POPULATION_DEGENERACY` |
| `service_loss_proxy` | `CANDIDATE_POPULATION_DEGENERACY` |
| `plan_regret_proxy` | `CANDIDATE_POPULATION_DEGENERACY` (exact redundant transform of `plan_value`) |
| pairwise ranking | `CANDIDATE_POPULATION_DEGENERACY` |
| `containment_time_proxy` | `METRIC_OR_GATE_MISINTERPRETATION` + `TARGET_DEGENERACY` |

**`FROZEN_REPRESENTATION_INSUFFICIENT` and `MECHANICAL_TARGET_OR_RANKING_DEFECT` are not invoked for any
criterion** -- explicitly ruled out per-criterion with evidence (see the artifact), not merely because Level A
failed.

## 10. Decision (`m10-3b-closure.json`)

**`M10_3B_POPULATION_AMENDMENT_REQUIRED`.**

`M10_3B_LEARNED_STRATEGIST_NOT_JUSTIFIED` (Section 20-D) does not yet apply: the degeneracy traces to a
specific, narrow, already-disclosed pilot-scope choice in M10.3A's own frozen protocol (family=
`golden-reference` ONLY, depth=25 `MATURE` ONLY -- mirroring M10.2's precedent, never an exhaustive
population), not to a fundamental property of every realistic candidate population the system could
legitimately generate. Two already-governed, already-trained-on families (`branched-loop`, `loop-grid`) and
five already-governed depth buckets (1,2,3,4,6) exist and were never evaluated for Strategist candidate
diversity. `M10_3B_BROADER_REFIT_SCIENTIFICALLY_JUSTIFIED` (Section 20-C) is directly contradicted: root-cause
classification found no support for `FROZEN_REPRESENTATION_INSUFFICIENT` anywhere.
`M10_3B_CORRECTION_REQUIRED` (Section 20-A) is directly contradicted: zero mechanical defects found across 6
independent checks plus real-code unit tests.

**Recommended (not executed here) M10.3C population-amendment scope**: expand the frozen population to
include the other already-trained families and lower-severity depth buckets, using only already-existing
system semantics (no new candidate templates, no weakened safety thresholds). Freeze that population's own
protocol BEFORE inspecting any result, and re-run THIS SAME within-incident-identifiability/oracle-utility
diagnostic on it before any training, to confirm the expansion actually restores meaningful signal (never
assume it will). **If the expanded population shows the same degeneracy**, `M10_3B_LEARNED_STRATEGIST_NOT_
JUSTIFIED` becomes the well-evidenced conclusion, and the system should retain the deterministic candidate
generator + deterministic Strategist + exact WNTR verification permanently for this decision, proceeding to
M10.4 on that basis.

Does **not** authorize: true M10.3, Strategist retraining of any kind, another Level A or Level B, a
full/shared HydroCore retrain, opening locked final/topology tests, altering closed M9/M10 results, or the
M10.3C population amendment itself (a separately authorized future task).

## 11. Locked-test status

`locked_test_opened_before=false`, `locked_test_opened_after=false` throughout every phase of this task
(protocol, formula audit, ranking audit, calibration audit, corpus build, all diagnostic artifacts, closure).
Never inspected.

## 12. Output governance (unaffected)

No checkpoint was created, loaded for inference beyond what M10.3A already produced as read-only reference,
or modified. Learned Strategist remains runtime-disabled and non-authoritative. Deterministic Strategist and
WNTR/EPANET verification remain unmodified and retained as runtime authority. No `runtime_enabled_outputs`
promotion occurs. M9/M10 historical artifacts and checkpoints (M9.6 canonical, M10.1, M10.2, M10.3A) are
unchanged.
