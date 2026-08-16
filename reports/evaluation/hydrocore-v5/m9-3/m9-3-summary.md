# Milestone 9.3 summary: interleaved-predictor calibration root-cause study

DIAGNOSTIC / ANALYSIS-ONLY. No predictor was trained, tuned, fine-tuned, or promoted. No checkpoint was modified (SHA256-reverified identical before and after). `alpha` never changed from 0.1. The `>=0.85` operational coverage floor was never weakened. `locked_final_test`/`locked_topology_test` remained unopened throughout (`locked_test_opened_before=false`, `locked_test_opened_after=false`).

`m9_0a_protocol_frozen_commit`: `d8439830f70e9922f4d7d7e94e2378d94a232efe`
`m9_0b_protocol_frozen_commit`: `3b353167d598d76efc6fbde303387388fbc3ccbf`
`M9.3 start_commit`: `ef3383a939e01b820b66d6be25d08f829ade572d` (M9.2 closure)

## Data provenance

All 6 predictor checkpoints (ARM_A/CURRENT and ARM_B2/INTERLEAVED x 3 seeds) were SHA256-verified against their recorded M8.7/M9.0a provenance, cross-checked against `m9-0a-results.json`, and re-verified unchanged after diagnostics completed. Per-example calibration and development rows were reconstructed via deterministic, read-only inference against these exact frozen checkpoints, reusing the unmodified M9.0a/M9.0b evaluation machinery (`run_m9_0a_evaluate._evaluate_on_family`, `run_m9_0b_evaluate`'s row builders, `m9_0b_calibration_schemes`) -- the only addition was retaining incident identity, which those functions' own return types otherwise drop. Unseen-topology rows were read directly from already-persisted M9.0a artifacts, not re-inferred.

**Section-5 reproduction gate**: M9.0a known-network Top-1 by maturity (both arms, all 3 seeds) and the M9.0b `CURRENT_FAMILY_DEPTH` scheme's known-family marginal coverage were recomputed from the canonical table and matched the original artifacts to float tolerance 1e-6. Status: **EXACT_OR_WITHIN_DECLARED_FLOAT_TOLERANCE**. Gate passed; interpretation below proceeded.

**Implementation audit** (Section 21): independent read of the conformal code path (nonconformity score, finite-sample quantile formula, fallback hierarchy, group-key construction, `minimum_group_size` behavior, split-seed disjointness) found no defect. This is corroborated, not merely asserted, by the exact reproduction gate above.

Canonical table: 9,660 rows (2 arms x 3 seeds x known+unseen families x 7 depths), `reports/evaluation/hydrocore-v5/m9-3/m9-3-canonical-calibration-diagnostics.jsonl`.

## Family-pooled known-family coverage (the headline number)

CURRENT_FAMILY_DEPTH scheme, ARM_B2 (INTERLEAVED), n=112/seed/family (all 7 depths pooled):

| family | n junctions | seed 20260814 | seed 31874 | seed 20260815 | Wilson 90% CI (worst seed) |
|---|---|---|---|---|---|
| golden-reference | 4 | 0.9464 | 0.9464 | 0.9554 | [0.900, 0.978] |
| branched-loop | 7 | 0.8750 | 0.8571 | 0.7946 | [0.725, 0.850] |
| loop-grid | 8 | 0.7232 | 0.7321 | 0.7143 | [0.640, 0.795] |

**FACT / ROBUST**: golden-reference is CI-entirely-above the 0.85 floor on all 3 seeds. loop-grid is CI-entirely-below the floor on all 3 seeds -- no overlap. branched-loop is the mixed/borderline case (2 seeds close to/above, 1 seed clearly below).

## Root cause: a dataset/eval-generator representativeness defect, not a conformal-code or predictor defect

**FACT**: `run_m7_topology._generate_eval_scenarios` restricts development-holdout scenarios to `EVAL_MAX_SOURCES=4` source junctions, selected as `junctions[:4]` after an alphabetical sort. For golden-reference (exactly 4 junctions) this exercises the full source-node space. For branched-loop (7 junctions) and loop-grid (8 junctions), it does not: development-holdout for loop-grid exercises only `{J1, J2, J3, J4}` (25% each; J5-J8 never appear), and branched-loop only `{JA, JB, JC, JD}` of its 7 junctions. The calibration pool (`_family_scenario_pool`) draws `source_node` from the scenario generator's own broader, less-truncated process across the full junction set.

**ROBUST**: within loop-grid, per-source-node coverage is wildly uneven and identical across all 3 seeds (development scenarios are a shared, non-resampled pool by M9.0a's own design): J1 = 0.214 (6/28) on every seed, J2 = 0.82-0.86, J3 = 0.857, J4 = 0.964-1.0. One node -- J1 -- single-handedly drags the family average down; the case-study export shows J1 consistently mispredicted as J7 or J8 (rank 8 of 8, ~6% true-source probability) at every prefix depth 1 through 25 for the same incident, on every seed. branched-loop shows the identical pattern in miniature (JC at 0.679 vs JA/JB/JD at 0.89-1.0).

**ROBUST**: the counterfactual decomposition (Section 18) shows every alternative M9.0b grouping scheme -- CURRENT_FAMILY_DEPTH (actual), a family-only pooled-depth variant, POOLED_DEPTH_AWARE, and BROAD_FALLBACK_CONTROL -- yields the *same* loop-grid MATURE coverage (0.75) on every seed. Only the development-oracle quantile (diagnostic only, `DEVELOPMENT_ORACLE_NOT_VALID_FOR_DEPLOYMENT`, never a candidate calibration output) reaches near-nominal coverage, requiring a robust +0.040 to +0.057 quantile shift over the actual frozen quantile on all 3 seeds. **This proves the M9.0b failure is not a grouping-choice problem** -- regrouping calibration-split scores differently cannot fix a development population that is itself non-representative of the family's full source-node space.

## What was ruled out

- **H1 (insufficient calibration support)**: WEAKLY_SUPPORTED. Calibration n is 100-150/bucket (not tiny); the loop-grid MATURE quantile is stable under incident-level bootstrap resampling (relative span 0.04-0.11); the counterfactual decomposition shows more/differently-pooled support does not change the outcome.
- **H3 (interleaved-predictor overconfidence)**: NOT_SUPPORTED on golden-reference, the only family where ARM_A and ARM_B2 can be directly compared on identical incidents. At depth 1, ARM_B2 is *less* confident than ARM_A both when correct and incorrect; at depth 6, ARM_B2's wrong predictions have higher entropy and lower max-probability than ARM_A's wrong predictions. MATURE depths have zero incorrect examples for either arm on golden-reference, so no failure-mode comparison exists there.
- **H8 (nonconformity score fundamentally incompatible with this predictor)**: NOT_SUPPORTED. The identical score, quantile formula, and alpha achieve robustly-passing coverage (0.90-0.98) for golden-reference under the same predictor -- the score works fine where the development population is representative.
- **H5 (depth-conditional failure)**: SUPPORTED but family-conditional, not universal. golden-reference's only weak depth is MID (depth 6, 0.812 on all seeds; MATURE is a perfect 1.0). branched-loop is weakest at EARLY depths. loop-grid is uniformly weak at every depth alike -- H5 does not explain the dominant contributor (loop-grid).

## Cross-seed consistency

Every finding cited above as ROBUST is identical in direction and comparable in magnitude on all 3 seeds (20260814, 31874, 20260815). The J1-within-loop-grid finding is bit-for-bit identical across seeds because development scenarios are a shared pool by M9.0a's own protocol design, not independently resampled per seed -- flagged as a limitation, not treated as 3 independent confirmations.

## H1-H8 evidence ratings

| Hypothesis | Rating |
|---|---|
| H1 Insufficient calibration support | WEAKLY_SUPPORTED |
| H2 Calibration/development shift | SUPPORTED |
| H3 Interleaved overconfidence | NOT_SUPPORTED |
| H4 Family score heterogeneity | STRONGLY_SUPPORTED |
| H5 Depth-conditional failure | SUPPORTED (family-conditional) |
| H6 Specific-family failure | STRONGLY_SUPPORTED |
| H7 Implementation/data defect | SUPPORTED (dataset/eval-generator, not conformal-code) |
| H8 Score incompatibility | NOT_SUPPORTED |

Full evidence-for/against/limitations for each hypothesis: `m9-3-root-cause.json`.

## M9_3_RECOMMENDATION: D (DATASET_EXCHANGEABILITY_OR_GENERATOR_FIX_REQUIRED)

**Reason**: the dominant, robust driver of the known-family calibration failure is a concrete, deterministic dataset/eval-generator representativeness defect (`EVAL_MAX_SOURCES=4` truncating development-holdout source-node coverage for any trained family with more than 4 junctions), not a conformal-code defect, a support/sample-size problem, interleaved-predictor overconfidence, or a fundamentally incompatible nonconformity score.

**Strongest evidence**: golden-reference (4 junctions, full source-node coverage in development) passes robustly on all 3 seeds; loop-grid (8 junctions, only half tested in development) fails robustly on all 3 seeds; the specific untested-vs-tested-node pattern predicts exactly which family passes and which fails. Every alternative calibration grouping scheme gives the same failing coverage, ruling out a grouping-choice fix.

**Evidence against**: the J1-specific difficulty within loop-grid is not fully explained by truncation alone -- J1 may be intrinsically harder to discriminate for genuine hydraulic/structural reasons, coexisting with the dataset defect.

**Alternative hypothesis**: H5 (depth-conditional failure) is real but secondary and does not explain loop-grid, the dominant contributor.

**Robust across all 3 seeds**: yes, for every finding underpinning this recommendation.

**Next experiment, if any**: none authorized by this closure. A future, separately-scoped milestone could examine whether `run_m7_topology._generate_eval_scenarios` can be made to draw development-holdout source nodes from the same (or a comparably representative) population the calibration pool already uses, for families with more than `EVAL_MAX_SOURCES` junctions -- and only THEN re-assess whether M9.0b's calibration schemes (or a new one) achieve the `>=0.85` floor against a representative development population. This milestone does not implement, authorize, or select that fix.

## Governance

No training, tuning, fine-tuning, checkpoint modification, or alpha change occurred. `locked_final_test`/`locked_topology_test` never opened. M9.0a/M9.0b outcomes and raw artifacts are unmodified (verified unchanged by SHA256 in `m9-3-manifest.json` and by test). No field-performance claim is made. No promotion decision is made. No calibration scheme was selected or implemented as a fix.

## Artifacts

- `m9-3-manifest.json` -- provenance, checkpoint SHA256s, reproduction gate, topology metadata
- `m9-3-canonical-calibration-diagnostics.jsonl` -- 9,660-row canonical per-example table
- `m9-3-reproduction.json` -- Section 5 gate detail
- `m9-3-support-analysis.json` -- Section 7 (calibration group support, finite-sample resolution, fallback frequency)
- `m9-3-coverage-uncertainty.json` -- Section 8 (Wilson 90% CIs per family/depth/seed)
- `m9-3-score-shift.json` -- Section 9 (calibration vs development nonconformity distributions, KS/Wasserstein)
- `m9-3-quantile-stability.json` -- Section 10 (incident-level bootstrap quantile stability)
- `m9-3-support-learning-curves.json` -- Section 11 (nested calibration-support learning curves)
- `m9-3-family-heterogeneity.json` -- Section 12 (cross-family score-distance matrices)
- `m9-3-depth-analysis.json` -- Section 13 (exact-depth calibration behavior)
- `m9-3-confidence-analysis.json` -- Section 14 (CURRENT vs INTERLEAVED confidence/overconfidence)
- `m9-3-source-conditional.json` -- Section 15 (per-source-node coverage)
- `m9-3-miscoverage-cases.json` -- Sections 16-17 (deterministic case studies + miscoverage severity)
- `m9-3-counterfactual-diagnostics.json` -- Sections 18-19 (read-only quantile decomposition + sample-size estimation)
- `m9-3-exchangeability-audit.json` -- Section 20 (calibration vs development covariate audit)
- `m9-3-implementation-audit.json` -- Section 21 (conformal code-path audit)
- `m9-3-root-cause.json` -- Section 23 (H1-H8 evidence ratings)
- `m9-3-closure.json` -- this milestone's governance/recommendation record
- `figures/` -- supplementary plots (coverage by family/depth, quantile vs support, score CDFs, family heatmap)
