# Milestone 9.2 summary: post-M9.1 diagnostic study

DIAGNOSTIC / ANALYSIS-ONLY. No model was trained, tuned, fine-tuned, or promoted. No checkpoint or calibration was modified. `locked_final_test`/`locked_topology_test` remained unopened throughout (`locked_test_opened_before=false`, `locked_test_opened_after=false`).

`m9_1_protocol_frozen_at_commit`: `0f05be1d47258a8c3d19e3a0d0e1122e3e560069`
`m9_1_closure_commit` / `M9.2 start_commit`: `3b6ee2a3529faffdcc2288cda294a4ef4d6f0765`
M9.1 final decision preserved: **CURRENT_HYDROCORE_RETAINED** (zero arms reached PROMOTION_CONFIRMED)
Screening seeds used (only): `20260814`, `31874`. CURRENT's third seed (`20260815`) is documented in the source artifacts but excluded from every cross-arm paired comparison in this milestone (no matching novel-arm seed exists).

## Data provenance

All per-incident/per-depth rows (probabilities, truth index, per-row metrics, runtime condition, solver-health flags) were **already persisted** by M9.1 in `m9-1-results.json` and reused as-is -- no development_holdout forward pass was re-run. Two quantities were reconstructed via deterministic, read-only inference against the exact frozen M9.1 checkpoints (SHA256-verified byte-identical to the recorded `checkpoint_sha256` for all 8 arm x seed combinations before use):

1. Per-development-row conformal candidate sets (M9.1 only persisted aggregate coverage/set-size). Reconstructed by re-running the `calibration`-split forward pass per (arm, seed) -- GRAPH_SDE using its exact frozen MC=4 Brownian-seed schedule -- refitting `B_DEPTH_AWARE`, and applying it to the already-persisted dev-row probabilities. **Verified exactly equivalent** (marginal + by-maturity coverage/set-size) to `m9-1-calibration.json` for all 8 combinations before any row was trusted (`m9-2-manifest.json["inference_reconstruction_equivalence"]`).
2. Topology metadata: the corpus uses one fixed `golden-reference` network for every scenario in every split, built once from the network object (not scenario-dependent).

**Section-4 reproduction gate**: early/mid/mature Top-1, MRR, and the guardrail regression-pp formulas (via the unmodified `run_m9_1_decide.py._step1_guardrails`) were recomputed from the canonical table and matched `m9-1-results.json`/`m9-1-guardrails.json` to float tolerance 1e-6. Status: **REPRODUCED_EXACTLY**. Gate passed; interpretation below proceeded.

Canonical table: 6,720 rows (4 arms x 2 seeds x 120 development_holdout incidents x 7 depths), `reports/evaluation/hydrocore-v5/m9-2/m9-2-canonical-diagnostics.jsonl`.

## Answers to the four scientific questions

### Q1 -- Where do CURRENT and the continuous-time arms begin to diverge?

Two-seed mean Top-1 by depth:

| depth | CURRENT | GRAPH_ODE | GRAPH_CDE | GRAPH_SDE |
|---|---|---|---|---|
| 1 | 0.271 | 0.267 | 0.267 | 0.267 |
| 2 | 0.496 | 0.492 | 0.446 | 0.492 |
| 3 | 0.592 | 0.596 | 0.575 | 0.596 |
| 4 | 0.813 | 0.850 | 0.796 | 0.846 |
| 6 | 0.979 | 0.946 | 0.946 | 0.942 |
| 12 | 0.983 | 0.938 | 0.958 | 0.946 |
| 25 | 0.996 | 0.967 | 0.954 | 0.950 |

**FACT / ROBUST**: EARLY depths (1-3) are near-parity, sign-MIXED across seeds -- no robust divergence. Depth 4 is arm-specific and ROBUST: GRAPH_ODE and GRAPH_SDE actually **improve** over CURRENT (+2.5 to +5.0pp on both seeds independently); GRAPH_CDE mildly regresses (-0.8 to -2.5pp). From depth 6 onward, all three arms show a **ROBUST** negative delta (~-2 to -5pp) reproduced independently on both seeds, continuing through depths 12 and 25. The gap does not grow sharply with more evidence -- it plateaus. Mechanistically, CURRENT keeps consolidating accuracy as evidence accrues (0.979 -> 0.996 from depth 6 to 25) while all three CT arms roughly plateau or mildly oscillate over the same range. Full per-depth/per-seed numbers, deltas, and bootstrap CIs: `m9-2-depth-metrics.json`.

### Q2 -- Complementary mistakes, or mostly losing evidence CURRENT already solves?

**ROBUST, and the central finding of this milestone**: at MATURE depths (12, 25), novel-only wins (paired cell C: CURRENT wrong, novel correct) are **0 in 11 of the 12** arm x seed x depth combinations; the single exception (GRAPH_ODE, seed 20260814, depth 25, C=1) is one incident, not reproduced at the paired seed. CURRENT-only wins (cell B) are consistently 3-6 per combination. Net paired advantage (C - B) is negative in all 12 combinations examined, range -1 to -6. One incident, `902004500`, is lost by **all three** continuous-time arms while CURRENT solves it, reproduced at both seeds and both MATURE depths -- a shared structural blind spot, not seed noise. Zero incidents exist anywhere at a MATURE depth where all three CT arms win and CURRENT loses. Full 2x2 tables, incident-ID lists, and cross-arm overlap: `m9-2-disagreements.json`.

### Q3 -- What predicts CURRENT failure or continuous-time regression?

- **Rank movement (ROBUST)**: at MATURE depths, median rank_delta = 0 for every arm/seed; mean is small (+0.02 to +0.07 -- typically nudging the true source from rank 1 to rank 2, not further). Large regressions (rank worsens by >=3) occur in only 1 of 24 arm x depth x seed combinations and 0 large improvements anywhere. The MATURE regression is a narrow, near-miss error mode, not a catastrophic mislocalization. `m9-2-rank-analysis.json`.
- **Topology (EXPLORATORY, network-size-limited)**: when wrong, every arm lands within 2 hops of the true source 100% of the time; within-1-hop accuracy when wrong is 50-83% with no consistent CURRENT-vs-CT ordering. The golden-reference network has only 4 candidate junctions and a max junction-junction distance of 2, so "100% within 2 hops" is largely a mechanical property of network size, not strong confirmation of a hydraulically-informed near-miss structure. `m9-2-topology-analysis.json`.
- **Missingness / irregularity (ROBUST NULL RESULT)**: no arm shows a robust cross-seed advantage under high missingness or gap irregularity. `fraction_missing` is heavily right-skewed in development_holdout (predeclared quartile bins: 750/79/9/2 rows per seed) -- the two highest-missingness bins are too small (9, 2 rows) to support any claim; the large deltas there (+0.11, +0.5) are single-/few-incident artifacts. `gap_coefficient_of_variation` and `mean_gap_seconds` are degenerate (most rows have zero valid-reading gap variance). Stratifying instead by `n_valid_observations` (well-populated: 257/175/199/209 rows) shows **no monotonic CT advantage under sparse evidence** -- deltas in the lowest-evidence quartile are near zero and sign-mixed, while the highest-evidence quartile (which overlaps MATURE depths) is consistently negative for all three arms on both seeds. More evidence does not help the CT arms relative to CURRENT; it is where they fall behind. `m9-2-missingness-analysis.json`.
- **Calibration (ROBUST, all-depths-pooled)**: in CURRENT-correct/novel-wrong paired cases, the novel arm's own true-source probability is low (0.10-0.19) while CURRENT's is high (0.77-0.85) in the *same* incidents -- CT arms are usually confidently wrong when they lose, not narrowly edged out, even though rank movement shows the true source is typically still ranked #2. Spearman rank-probability correlation between CURRENT and each CT arm is consistently strong (0.77-0.82). `m9-2-calibration-diagnostics.json`.

### Q4 -- Does the evidence justify one specific HydroCore-S follow-up, or should architecture optimization stop?

The evidence does not support **H3** (complementary continuous-time signal): novel-only MATURE wins are essentially zero and not cross-seed-reproduced; the missingness/irregularity inductive-bias story that would motivate H3 (or a bounded CT-residual pathway) shows a robust *null* result. **H2** (spatial/hydraulic discrimination bottleneck) receives only weak, network-size-confounded support and is not established strongly enough by this milestone to itself justify a new experiment. **H1** (prefix-objective misalignment) is not directly tested by M9.2 -- this milestone diagnosed continuous-time *architecture substitution*, not CURRENT's own training-objective design -- and remains an open, untested hypothesis, not refuted.

## Cross-seed consistency

Per-depth Top-1-delta classifications (`m9-2-depth-metrics.json["cross_seed_consistency_top1_delta"]`): EARLY depths are MIXED for all three arms (near-zero, no consistent sign); depths 4, 6, 12, 25 are ROBUST for all three arms (GRAPH_CDE is additionally ROBUST at depths 2-3). The Q2 complementarity finding (near-zero novel-only wins at MATURE) is ROBUST on both seeds for all three arms. The Q3 missingness null result is ROBUST (same qualitative absence of benefit on both seeds). The Q3 topology near-miss finding is EXPLORATORY, limited by network size, not a promotion-relevant ROBUST finding.

## Hypothesis mapping

| Hypothesis | Evidence for | Evidence against | Verdict |
|---|---|---|---|
| H1 (prefix-objective misalignment) | Not tested by this milestone (would require probing CURRENT's own training supervision, out of scope for an architecture-substitution diagnostic) | -- | UNTESTED |
| H2 (spatial/hydraulic discrimination bottleneck) | Rank regressions are narrow (rank 1->2); errors stay graph-close when wrong | Network has only 4 candidate junctions / max distance 2 -- the near-miss signal is largely mechanical, not a strong confirmed structural pattern | WEAK, NOT ACTIONABLE ON THIS EVIDENCE |
| H3 (complementary continuous-time signal) | One small, arm-specific depth-4 improvement (GRAPH_ODE, GRAPH_SDE) | Novel-only MATURE wins ~0 and not cross-seed-reproduced; no missingness/irregularity advantage; the one MATURE-depth CT-only win found is not reproduced at the paired seed | NOT SUPPORTED |
| H4 (no actionable model-side headroom, this axis) | Central Q2 finding (robust, both seeds); Q3 missingness null result (robust, both seeds); MATURE regression plateaus rather than growing | Rank movement is narrow, suggesting *some* structure remains, just not one this milestone's evidence can turn into a specific next experiment | SUPPORTED for the continuous-time-substitution axis specifically |

## M9_2_RECOMMENDATION: D (STOP_ARCHITECTURE_OPTIMIZATION)

**Scope**: applies specifically to the continuous-time temporal-dynamics substitution axis (GRAPH_ODE/GRAPH_CDE/GRAPH_SDE replacing CURRENT's `temporal_dynamics` module), as explored across M9.0-M9.1. Makes no claim about, and does not evaluate, CURRENT's own training-objective design (H1) or a topology/hydraulic-aware discrimination objective (H2) -- both remain open, untested hypotheses for a possible separate future milestone, not authorized by this closure.

**Strongest supporting evidence**: the Section 6 disagreement analysis -- near-zero, non-cross-seed-reproduced novel-only MATURE wins against consistent CURRENT-only wins, with one incident (902004500) lost by all three CT arms at every seed x MATURE-depth combination examined while CURRENT solves it every time.

**Evidence against**: the regression itself is narrow in rank terms (median rank_delta = 0) and stays graph-close, consistent with a fixable discrimination gap rather than a fundamentally unsuitable architecture family; GRAPH_ODE/GRAPH_SDE show a small robust improvement at depth 4 that this recommendation does not further investigate.

**Alternative hypothesis**: H2 remains plausible but is neither confirmable nor refutable with this milestone's evidence, because the golden-reference network's small size gives the topology-distance diagnostic very little discriminating power.

**Robust across both seeds**: yes, for every finding cited above as ROBUST.

**Next experiment, if any**: none authorized by this closure. If architecture-axis work resumes in the future, the most scientifically motivated next step (not authorized here) would be re-running this same M9.2 diagnostic protocol on a larger, higher-junction-count topology family, where the topology-distance and missingness-quartile stratifications would carry real statistical power -- not training a new model variant.

## Governance

No training, tuning, fine-tuning, checkpoint modification, or calibration refit-for-promotion occurred. `locked_final_test`/`locked_topology_test` never opened. M9.1 outcomes and raw artifacts are unmodified (SHA256-reverified in `m9-2-manifest.json`). No field-performance claim is made. No promotion decision is made. No M9 S/M/L capacity scaling is authorized by this document.

## Artifacts

- `m9-2-manifest.json` -- provenance, reproduction gate, checkpoint SHA256s, reconstruction-equivalence proof
- `m9-2-canonical-diagnostics.jsonl` -- 6,720-row canonical paired diagnostic table
- `m9-2-depth-metrics.json` -- Section 5
- `m9-2-disagreements.json` -- Section 6
- `m9-2-rank-analysis.json` -- Section 7
- `m9-2-topology-analysis.json` -- Section 8
- `m9-2-missingness-analysis.json` -- Sections 9-10
- `m9-2-calibration-diagnostics.json` -- Section 11
- `m9-2-case-studies.json` -- Section 12
- `m9-2-closure.json` -- this milestone's governance/recommendation record
- `figures/` -- supplementary plots (Top-1/MRR vs depth, paired wins by depth, rank-delta histogram)
