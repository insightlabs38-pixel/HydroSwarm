# Source-identifiability analysis: final report

Branch: `exp/source-identifiability-analysis`. Status: **experimental,
post-hackathon, analysis-only**. Protocol:
`docs/evaluation/SOURCE_IDENTIFIABILITY_ANALYSIS_PROTOCOL.md`. This
analysis trains nothing, touches no checkpoint, and never opens, alters,
or re-executes the M11.6 locked evaluation -- every number attributed to
"HydroCore-v5" below comes from read-only replay against the already-frozen
scenario specs in `data/locked/m11-6/**`, joined to the already-frozen
outcomes in `reports/evaluation/hydrocore-v5/m11/m11-6-final/
m11-6-raw-incidents.jsonl`. All oracle/template numbers are
**NON-PROMOTABLE / DIAGNOSTIC ONLY**, matching the repo's own convention
(`reports/evaluation/hydrocore-v5/m10/m10-3b-diagnosis/
m10-3b-oracle-utility.json`) -- never a production safety claim, never a
substitute for the governed OOD/calibration/human-approval gates, which
remain untouched.

## 0. Question and short answer

**Are HydroCore-v5's hard source-localization failures information-limited
(the sensors genuinely can't tell candidates apart) or
representation-limited (the model fails to use evidence that is
physically present)?**

**On this evidence: predominantly REPRESENTATION_LIMITED, with a real but
secondary information-limited/stress-collapse component.** A non-learned
physics oracle, given the exact same real (including stressed/noisy)
sensor observations HydroCore-v5 saw, succeeds at Top-1 in **96.4%** of the
cases where HydroCore-v5's own recorded Top-1 prediction was wrong (54 of
56 confirmatory failures). Only 2 of 125 confirmatory incidents are cases
where even the oracle fails -- true information-limited cases are rare in
this locked evaluation. See Section 10 for the full decision and Section 11
for ranked next directions.

## 1. Implementation summary

- **Confirmatory tier** (n=125, authoritative): every incident in
  `data/locked/m11-6/{locked_final_test,locked_topology_test}/
  scenarios.jsonl` is replayed exactly (`scripts/hydrocore_v5/
  source_identifiability/common.py::reconstruct_incident`, verified via
  recomputed `network_sha256` against the recorded value for all 7 network
  identities used) to obtain the same randomized hydraulic state, sensor
  set, and (for the true source) the same bit-for-bit degraded observation
  the original M11.6 run used. For every OTHER junction in that network, a
  counterfactual `simulate_incident` call under identical conditions
  builds a full candidate signature library
  (`scripts/hydrocore_v5/source_identifiability/library.py`).
- **Exploratory tier** (self-generated, no HydroCore-v5 prediction
  attached, seeds verified disjoint from every locked seed): a larger
  corpus on the same 7 networks for statistical power on the
  clean-vs-stress and known-vs-unseen identifiability questions alone
  (`run_build_exploratory.py`).
- **Pairwise distances / identifiability metrics**: `signatures.py`
  (pure, unit-testable). **Structural features**: `centrality.py` (new --
  no betweenness/closeness/degree centrality existed anywhere in the repo
  before this branch; confirmed by full-repo survey). **Oracle**: reuses
  the repo's own existing classical Bayesian signature-residual localizer
  (`hydroswarm.classical.prior.bayesian_source_posterior`) rather than a
  new ranking rule. **Bootstrap CIs** for paired comparisons reuse
  `scripts/hydrocore_v5/m9_1_common.py::paired_bootstrap` verbatim; a new
  `stats_utils.unpaired_bootstrap_diff` (same 2,000-resample/seed
  20260815/90%-interval convention) is used for between-group comparisons,
  since `paired_bootstrap` assumes equal-length paired sequences that
  don't exist for a centrality-group or topology-group comparison.
- Total EPANET/WNTR simulator cost: confirmatory tier ~85s (under 1,000
  calls); exploratory tier ~330s (525 incidents, up to 12 candidates each).
  All read-only; nothing in `data/locked/**` or `models/**` was written.

## 2. Source-signature definition

Three signature views are computed per incident's candidate pool, all from
`log1p(concentration_mg_l)` at the incident's own sensor nodes and sample
times (`signatures.py::build_signature_set`):

1. **RAW** -- keeps injected strength as part of the signature.
2. **NORMALIZED** -- RAW divided by its own L2 norm; marginalizes strength,
   isolating temporal/spatial shape.
3. **ARRIVAL-ORDER** -- per-sensor first-crossing time of a fixed
   1e-4 mg/L threshold; scale-invariant timing fingerprint.

Every candidate in one incident's pool is simulated under *bit-for-bit*
identical randomized hydraulics, timing, strength, demand, and sensor set
(Section 6 of the protocol) -- nuisance variables are held fixed by
construction within each incident, not marginalized after the fact.

**A genuine, unplanned finding**: RAW-signature **cosine** distance
saturates at 1.0 (orthogonal) for nearly every candidate pair in the
smaller networks. This is not a bug -- with only 25 sample times and short
injection pulses, different candidates' few nonzero timesteps frequently
don't overlap at all, so raw-magnitude cosine similarity is measuring
"did these two pulses land in the same time bins" rather than anything
about source identity. This is exactly why NORMALIZED+correlation, not
raw cosine, is used as the primary identifiability metric throughout, and
is reported here as a concrete, reproducible illustration of why "evaluate
multiple defensible signature definitions" (per the protocol) matters in
practice, not just in principle.

## 3. Pairwise source-distinguishability

Per incident, per signature definition, per metric (normalized
RMSE/cosine/correlation on RAW+NORMALIZED, L1 on ARRIVAL-ORDER): nearest/
second-nearest competitor distance, mean/median separation, an
`identifiability_score` (nearest-competitor distance / that incident's own
mean pairwise distance -- comparable across networks of different scale),
and two ambiguity-count definitions (a physical sensor-noise-floor
threshold and a data-driven within-incident percentile). Full per-incident
table: `reports/evaluation/hydrocore-v5/source-identifiability/
confirmatory/confirmatory-identifiability.jsonl`.

**17.6% of confirmatory incidents (22/125)** have `identifiability_score
== 0` under NORMALIZED+correlation -- every candidate produces a
literally identical (usually all-below-detection) shape signature within
the observation window. This is the single clearest piece of genuine
information-limitation in the dataset, and it is concentrated in networks
with more junctions than sensors (branched-loop, loop-grid,
locked-topology-*) and observation windows too short for the injected
pulse to reach differentiate-able sensor combinations.

## 4. Oracle/template localization results

Non-learned nearest-signature ranking (`oracle.py`, reusing
`bayesian_source_posterior`), evaluated on the SAME real (bit-for-bit
reproduced) observation HydroCore-v5 was scored against:

| tercile (by identifiability_score) | n | mean score | oracle Top-1 | HydroCore-v5 Top-1 | HydroCore-v5 MRR |
|---|---|---|---|---|---|
| T1 (least identifiable, shape-only) | 42 | 0.006 | **1.000** | 0.214 | 0.433 |
| T2 | 42 | 0.572 | 0.952 | 0.810 | 0.866 |
| T3 (most identifiable, shape-only) | 41 | 1.021 | 0.878 | 0.634 | 0.747 |

The oracle is at or near ceiling in every tercile, **including the
tercile where shape-only identifiability is essentially zero** -- because
strength/timing/magnitude cues (deliberately excluded from the
NORMALIZED+correlation score to isolate shape) still separate candidates
via RMSE residual almost everywhere. This is itself an important, honest
methodological finding: "shape-ambiguous" does not mean "physically
indistinguishable" once magnitude and arrival timing are allowed back in
-- see Section 8 for why this matters to the final decision.

Clean-vs-stress progression (`comparison_8`, confirmatory tier,
`locked_final_test` only, n=15 per condition): no condition's
identifiability-score shift from NOMINAL has a bootstrap CI excluding
zero at this sample size (all 90% CIs cross zero). The exploratory tier
(Section 8b) has far more power for this question.

## 5. HydroCore-v5 vs. oracle, paired (same 125 incidents)

| metric | oracle − HydroCore-v5 | 90% CI |
|---|---|---|
| Top-1 | +0.392 | [+0.304, +0.472] |
| Top-3 | +0.248 | [+0.184, +0.312] |
| MRR / reciprocal rank | +0.289 | [+0.228, +0.346] |

All three CIs are entirely positive. Of the 56 confirmatory incidents
where HydroCore-v5's own recorded Top-1 was wrong:

- **54/56 (96.4%)**: the oracle, using the identical real observation, is
  correct.
- **2/56 (3.6%)**: both the oracle and HydroCore-v5 fail (the only
  incidents this analysis calls genuinely information-limited by the
  strongest available test).
- (For reference: HydroCore-v5 succeeds while the oracle fails in 5/125
  incidents overall -- HydroCore-v5 is not strictly dominated by the
  oracle, consistent with it having access to a full learned
  representation the oracle deliberately does not use.)

## 6. Identifiability vs. centrality/observability

- Correlation between true-source betweenness centrality and
  identifiability_score across the 125 confirmatory incidents: **-0.016**
  (essentially zero) -- centrality and physical shape-identifiability are
  largely independent axes in this data, consistent with the motivating
  prior finding that "observability and centrality carry largely
  independent signal."
- **Unconditioned**, high-betweenness sources have higher HydroCore-v5
  Top-1 than low-betweenness sources: +0.147 [90% CI +0.001, +0.292],
  entirely positive.
- **Conditioned on identifiability tercile**, the centrality effect's CI
  no longer excludes zero in any of the three strata (T1: -0.144 [-0.346,
  +0.058]; T2: +0.114 [-0.171, +0.429]; T3: +0.051 [-0.208, +0.290]) -- on
  this evidence, the raw centrality penalty is **not robustly independent
  of identifiability**: once identifiability is held roughly fixed, the
  centrality association with HydroCore-v5 failure weakens to
  statistical noise at this sample size. This should be read as
  "attenuated, not eliminated" given how wide these per-stratum CIs are
  (n as low as 7 per group in T2) -- a larger, purpose-built follow-up
  would be needed to say more.
- **Source-to-sensor distance** (direct hops to the nearest sensor) shows
  the same pattern more cleanly: unconditioned, directly-instrumented
  sources (0 hops) beat indirectly-inferred sources by +0.176 [+0.033,
  +0.317] (entirely positive); conditioned on identifiability tercile, all
  three strata's CIs cross zero. Correlation between sensor-distance and
  identifiability_score is -0.40 (farther-from-a-sensor sources are
  meaningfully less identifiable, the most intuitive relationship found in
  this analysis).

## 7. Failure taxonomy (56 confirmatory Top-1 failures)

| category | n | % of failures |
|---|---|---|
| A -- information-limited (oracle also fails) | 2 | 3.6% |
| B -- representation-limited (oracle succeeds, margin ≥ noise floor) | 20 | 35.7% |
| C -- stress-induced collapse (subset of A, non-NOMINAL + good clean separability) | 0 | 0.0% |
| D -- ranking-limited (true source in Top-3, not Top-1) | 25 | 44.6% |
| E -- OOD/governance-limited (system did not fully commit: `SUPPRESSED`/`ABSTAINED`, or uncalibrated) | 55 | 98.2% |
| F -- inconclusive (oracle succeeds, but margin < noise floor) | 34 | 60.7% |

Categories overlap by design. Two readings matter most:

- **Confident vs. borderline representation-limited**: of the 54 failures
  where the oracle succeeds, only 20 (37%) do so with a residual margin
  clearly outside the sensor noise floor (category B); the other 34 (63%,
  category F) succeed by a margin close enough to the noise floor that the
  evidence, while present, is not overwhelmingly decisive. The headline
  "representation-limited" conclusion should be read with this
  qualification: strong evidence existed in over a third of failures,
  present-but-marginal evidence in most of the rest.
- **Governance almost always intervenes first**: 55/56 failures (98.2%)
  coincide with HydroCore-v5's own governance NOT fully committing
  (`SUPPRESSED` or `ABSTAINED`, rather than `VERIFIED`). Exactly **one**
  confirmatory incident (`seed=3531334002386096233`, loop-grid,
  `SENSOR_DROPOUT`) has HydroCore-v5 confidently `VERIFIED` a wrong Top-1
  -- and it also has `conformal_truth_coverage: false` (a real conformal
  miscoverage event) and near-zero physical identifiability
  (`identifiability_score ≈ 8.7e-7`), while the oracle still recovers the
  true source (margin ≈ 0 -- a razor-thin oracle win, not a confident one).
  This single incident is the closest thing in the confirmatory set to a
  "silently, confidently wrong" case, and it is exactly the kind of case
  the existing conformal/governance layer is designed to catch -- it is
  flagged here as one concrete example worth follow-up scrutiny, not as
  evidence of a systemic governance defect (n=1).
- **Known vs. unseen**: unseen-topology failures skew more clearly
  representation-limited (67% vs. 30% for known-topology failures) with
  zero information-limited unseen failures observed (n=9, small).

## 8. Known vs. unseen topology

### 8a. Confirmatory tier (n=105 known / 20 unseen, real HydroCore-v5 outcomes)

| | known | unseen | diff (known − unseen) | 90% CI |
|---|---|---|---|---|
| HydroCore-v5 Top-1 | 0.552 | 0.550 | +0.002 | [-0.198, +0.210] |
| HydroCore-v5 MRR | -- | -- | see `required-comparisons.json` | -- |
| identifiability_score | 0.512 | 0.615 | -0.102 | [-0.250, +0.054] |
| oracle Top-1 | -- | -- | see `required-comparisons.json` | -- |

On this confirmatory evidence alone, there is **no statistically robust
known-vs-unseen HydroCore-v5 Top-1 gap** (CI includes zero, and n_unseen
=20 is small), and if anything, unseen-topology incidents in this specific
locked set have slightly (not significantly) **higher** physical
identifiability than known-topology incidents -- the opposite direction
from "unseen networks are intrinsically harder." This is a genuinely
different (smaller, differently-conditioned) sample from whatever prior
condition-matched comparison motivated this analysis, and should not be
read as contradicting it -- only as this analysis's own, independently
computed answer on this specific frozen evidence.

### 8b. Exploratory tier (n=225 known / 300 unseen, self-generated, condition-stratified, much higher power)

`reports/evaluation/hydrocore-v5/source-identifiability/joined/
topology-split-decision.json`. Across CLEAN, MEASUREMENT_NOISE, and
SENSOR_DROPOUT, identifiability known-minus-unseen is small and its 90% CI
crosses zero in every single condition:

| condition | known mean | unseen mean | diff | 90% CI |
|---|---|---|---|---|
| CLEAN | 0.549 | 0.584 | -0.035 | [-0.141, +0.069] |
| MEASUREMENT_NOISE | 0.526 | 0.569 | -0.043 | [-0.153, +0.064] |
| SENSOR_DROPOUT | 0.577 | 0.514 | +0.063 | [-0.051, +0.175] |

Oracle Top-1 is **exactly 1.0 in both known and unseen networks** at this
sample size (n=225/300). This is the highest-power evidence in this
analysis, and it is unambiguous: **on these 7 networks, at these incident
parameters, unseen topology is not intrinsically less physically
identifiable than known topology.**

### Decision

Per the protocol's A/B/C framing (worse identifiability vs. poorer
representation vs. both): **B — comparable identifiability, not worse.**
The confirmatory tier could not robustly detect a HydroCore-v5 known/unseen
gap at all (n_unseen=20); the exploratory tier, with far more power,
finds no known/unseen identifiability gap either. If a HydroCore-v5
known/unseen performance gap exists (as referenced in this analysis's
motivating context, from a different, condition-matched comparison this
analysis did not itself reproduce), the evidence here rules out
"physically less identifiable" as the explanation and is consistent
instead with a representation-transfer gap — matching Section 10's overall
conclusion.

## 9. Exploratory clean-vs-stress (n=175/condition, higher power)

`reports/evaluation/hydrocore-v5/source-identifiability/exploratory/
exploratory-stress-comparison.json`. Pooled across all 7 networks:

| condition | identifiability diff vs. CLEAN | 90% CI | oracle Top-1 |
|---|---|---|---|
| MEASUREMENT_NOISE (`sensor_noise_std` 0.01→0.05) | -0.018 | [-0.092, +0.056] | 1.000 |
| SENSOR_DROPOUT (30% missingness) | -0.028 | [-0.105, +0.050] | 1.000 |

Neither stress condition produces a CI-robust identifiability shift, and
the oracle's Top-1 rate is **exactly 1.0 under every condition tested,
including both stress conditions**. This is a genuine, higher-power null
result, not a contradiction of Section 4/8a's harder-to-detect signal at
n=15 -- it should be read narrowly: **at the specific stress magnitudes
tested here (0.05 sensor-noise std, 30% missingness) and the long
observation windows this corpus otherwise shares with the confirmatory
tier, physical separability and oracle recoverability survive the applied
stress almost entirely intact.** It does NOT establish that stress can
never collapse identifiability -- only that these two specific,
moderate-intensity stress mechanisms did not, on these networks. The four
M11.6 conditions this analysis could not reproduce exactly
(SENSOR_DROPOUT/LOW_COVERAGE_ACTIVE_SAMPLING/SENSOR_HEALTH_DEGRADED/
AMBIGUITY_DISAGREEMENT's real mechanism, Section 6 of the protocol) may be
more severe than this approximation and are not covered by this null
result.

## 10. Decision: dominant bottleneck

**REPRESENTATION_LIMITED, with a real minority INFORMATION_LIMITED /
stress-collapse component.**

Supporting evidence:
- 96.4% of HydroCore-v5's confirmatory Top-1 failures are solved by a
  non-learned physics oracle using the identical real observation.
- The oracle is at or near ceiling in every identifiability tercile,
  including the least shape-identifiable one -- physical information
  (magnitude + timing) is present almost everywhere in this locked
  evaluation.
- Centrality's association with failure weakens once identifiability is
  conditioned on, suggesting some but not all of the previously observed
  centrality penalty is mediated by identifiability rather than being an
  independent representation gap -- but the CIs here are wide.
- Against this: 17.6% of incidents have zero shape-based separation
  between candidates, 2/125 are failures where even the oracle can't
  recover the source, and the taxonomy's "F" (borderline oracle success)
  bucket is the single largest category (60.7% of failures) -- a
  meaningful fraction of the evidence is present but not decisive, not
  overwhelming.

## 11. Ranked next research directions

1. **Candidate-conditioned graph-native architecture is justified by this
   evidence** -- the dominant finding (oracle recovers 96% of failures)
   directly supports investing in a representation that can exploit
   evidence that is physically present but currently missed, rather than
   assuming the sensor network itself needs to change first.
2. **A parallel, smaller investment in evidence acquisition remains
   worthwhile, not a full pivot**: the 22 incidents (17.6%) with zero
   shape-based separation, and the 42% of hard incidents where the
   deterministic single-extra-sensor analysis (Section 12) resolves the
   ambiguity, both indicate a real, bounded information-limited
   population that a graph-native model cannot fix by architecture alone.
3. **Investigate the exact single VERIFIED-but-wrong incident** (Section
   7) and the conformal-coverage relationship to identifiability more
   broadly -- an n=1 finding, but the kind of case worth a dedicated,
   larger-sample follow-up before ruling out a governance-tuning
   opportunity.
4. **Do not pursue DTW/Wasserstein signature-comparison metrics or a
   sensor-placement redesign as the PRIMARY next step** -- this analysis's
   own comparison of RAW/NORMALIZED/ARRIVAL-ORDER x
   RMSE/cosine/correlation found no case where a more exotic metric was
   needed to reach a clear answer, and the identifiability-vs-outcome
   relationship, while real, explains only part of HydroCore-v5's error.

## 12. Bounded counterfactual: one additional sensor

For the bottom tercile by identifiability_score (n=41; golden-reference
incidents excluded automatically since every junction there is already a
sensor), adding the single best additional junction sensor (computed
deterministically from already-simulated full-node traces -- zero
additional EPANET calls):

- **24.4% (10/41)** cross back over the "as identifiable as a typical
  incident" threshold (`identifiability_score` > 1.0) with one added
  sensor.
- **0/41** oracle Top-1 flips from wrong to right -- because, consistent
  with Section 4/10, the oracle was already correct in 100% of this
  tercile using magnitude/timing cues alone; the extra sensor strengthens
  an otherwise fragile *shape*-only margin, it does not fix an oracle
  failure that didn't exist in this population.
- Most frequently the single best node to add is `J1`-equivalent
  (network-specific; see `counterfactual-sensor-summary.json`'s full
  ranked list) -- i.e. a small number of specific junctions repeatedly
  carry the most additional shape-discriminating value across many
  incidents on the same network, a concrete, actionable starting point for
  a future sensor-placement study rather than a claim that placement is
  the dominant lever.

## 13. Explicit answers

- **Are low-centrality sources physically less identifiable from current
  sensors?** Only weakly, if at all, in this data (correlation ≈ -0.016
  with betweenness). Source-to-sensor *distance* is a cleaner physical
  driver of identifiability (correlation -0.40) than centrality per se.
- **Does centrality remain associated with HydroCore-v5 failure after
  controlling for identifiability?** The unconditioned association is
  real (CI entirely positive); conditioned on identifiability tercile, the
  CIs no longer exclude zero in any stratum -- attenuated on this
  evidence, not cleanly independent, but not conclusively eliminated
  either given small per-stratum n.
- **How much of HydroCore-v5's error is information-limited vs.
  representation-limited?** By the strongest test (oracle also fails):
  3.6% information-limited. By the taxonomy's broader signal (oracle
  succeeds only marginally): up to 60.7% of failures have *available but
  not decisive* evidence. The clear majority is not "no information
  available," but the fraction with strongly decisive information (35.7%,
  category B) is a minority of failures, not all of them.
- **Does the oracle materially outperform HydroCore-v5 on unseen
  topology?** Yes in aggregate direction (unseen failures skew more
  representation-limited, 67% vs. 30% known), but n=9 unseen failures is
  too small to claim this robustly on its own.
- **Are unseen networks intrinsically harder, or is HydroCore-v5 failing
  to transfer despite adequate signal?** On the confirmatory tier: no
  robust Top-1 gap detected at all, and identifiability is not lower for
  unseen networks (if anything, slightly higher, not significantly). See
  Section 8b for the higher-power exploratory read.
- **How much does stress destroy otherwise-useful source separability?**
  Confirmatory tier (n=15/condition): no condition shows a CI-robust
  identifiability shift from NOMINAL. See Section 9 for the
  exploratory tier's much higher-power answer (n=175/condition): also no
  CI-robust shift, at the specific stress magnitudes tested.
- **Could one additional strategically chosen observation resolve a
  meaningful portion of ambiguous cases?** Yes for about a quarter of the
  hardest (shape-ambiguity) cases, but it does not flip any oracle
  failures in this population because there were none to flip in that
  subset -- see Section 12.
- **Is a full graph-native/GNN localization architecture now justified?**
  Yes, primarily -- see Section 10/11.
- **Or should the next research effort focus on sampling/sensor
  placement?** As a secondary, bounded investment (Section 11.2), not the
  primary one.

## Reproducibility

All scripts, configs, and derived tables are committed under
`scripts/hydrocore_v5/source_identifiability/` and
`reports/evaluation/hydrocore-v5/source-identifiability/`. Seeds:
confirmatory tier reuses the exact recorded M11.6 seeds (read-only replay,
never regenerated); exploratory tier seeds are drawn from
`np.random.default_rng(20260826)` and asserted disjoint from every locked
seed at build time (`run_build_exploratory.py`'s own assertions). Bootstrap
convention: 2,000 resamples, seed 20260815, 90% percentile interval,
matching `m9_1_common.py` throughout.

Commands, in order:
```
python scripts/hydrocore_v5/source_identifiability/run_build_confirmatory.py
python scripts/hydrocore_v5/source_identifiability/run_counterfactual_sensor.py
python scripts/hydrocore_v5/source_identifiability/run_join_and_analyze.py
python scripts/hydrocore_v5/source_identifiability/run_taxonomy.py
python scripts/hydrocore_v5/source_identifiability/run_build_exploratory.py --per-network 25
python scripts/hydrocore_v5/source_identifiability/run_topology_split_analysis.py
python scripts/hydrocore_v5/source_identifiability/run_exploratory_stress_analysis.py
```
