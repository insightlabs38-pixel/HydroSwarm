# Milestone 9 final closure: model-size / temporal-architecture / training-recipe search

**M9_STATUS = CLOSED.**

Milestone 9 governed HydroCore-v5's temporal-architecture search, S-scale training-recipe search, and model-size (capacity scaling) search. All three are now closed with a single selected predictor.

## Selected predictor

- **Size**: HydroCore-**S**
- **Parameters**: 4,182,612
- **Variant**: `small`
- **Recipe**: `CLASSICAL_HYDROCORE_S` + `AGE_FIX_ONLY` + `EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING` + `B_DEPTH_AWARE_CALIBRATION` + `ALPHA_0_1` + `SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_20_PER_SOURCE`
- **Checkpoint policy**: `FINAL_STEP_1350` (never best-validation), canonical exports at `reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed{20260814,31874,20260815}.json`

## HydroCore-M / HydroCore-L

- **HYDROCORE_M_STATUS = NOT_PROMOTED.** 13,919,572 parameters (3.328x S), ~1.590x median inference latency, no statistically or practically meaningful unseen-topology capacity gain over S (M9.8: delta -0.00343, 90% CI [-0.0128, +0.0063] vs. +0.02 threshold -- a clean fail, not borderline/inconclusive).
- **HYDROCORE_L_AUTHORIZED = FALSE**, unaffected by the M capacity result per the frozen policy.
- **CAPACITY_SCALING_RESULT = NO_MEANINGFUL_CAPACITY_GAIN.**

## Evidence chain

1. `AGE_FIX_ONLY` selected (M8.7) -- corrected absolute-time-origin leakage and weak/partial temporal-feature usage without regressing accuracy or calibration.
2. Classical (discrete-time) temporal architecture retained (M9.1) -- CURRENT outperformed GRAPH_ODE/GRAPH_CDE/GRAPH_SDE continuous-time arms on the frozen primary metric; temporal-architecture search closed.
3. Multi-topology interleaved training validated (M9.0/M9.0a/M9.0b) -- interleaved ARM_B training across golden-reference/branched-loop/loop-grid confirmed over single-family ARM_A, with optimizer-step-matched comparability and a calibration-grouping study resolving the multi-topology support question.
4. Full-source evaluation defect corrected and calibration support issue resolved (M9.4/M9.5/M9.5R) -- independently reconfirmed under a disjoint population.
5. Exact compute parity confirmed (M9.6, `M9_6_DECISION=A`) -- HydroCore-S frozen at exactly 1,350 optimizer steps under the interleaved recipe; S-scale training-recipe search closed.
6. HydroCore-M capacity experiment preregistered (M9.7), checkpoint-policy inconsistency corrected before any M performance was viewed (M9.7A).
7. 3.33x larger M produced no meaningful capacity gain (M9.8, `M9_8_DECISION=B`) -- a clean, non-cherry-picked negative result, reported honestly per the standing "report negative results" rule; model-size search closed.
8. S retained as the selected predictor.

## Metadata corrections applied before closure (non-scientific, disclosed)

1. **M9.8 closure `start_commit` conflation** -- `run_m9_8_decide.py`'s decide-stage `m8.current_commit()` call had written the same HEAD value into `start_commit`, `execution_manifest_commit`, and `execution_commit` alike. Corrected to disambiguate the pre-experiment freeze commit (M9.7A HEAD) from the training-execution commit and the true final artifact-commit SHA. No metric, decision, guardrail, or checkpoint identity was altered. (`reports/evaluation/hydrocore-v5/m9-8/m9-8-closure.json`, commit `8284e48`)
2. **M9.1 provenance tripwire refresh** -- M9.7's audited, additive-only `core.py` change (registering `MODEL_VARIANTS["small_v5_capacity_m"]`) correctly tripped M9.1's frozen-path guard. Mechanically audited as additive-only and semantically irrelevant to M9.1's frozen paths; `CODE_UNDER_TEST_COMMIT_FLOOR` re-superseded to the audited commit via the guard's own existing mechanism, with regression tests re-proving the audit on every run. No M9.1 scientific rerun; no M9.1 conclusion changed; the guard remains fully intact for future changes. (`reports/evaluation/hydrocore-v5/m9-1-provenance-refresh/`, commit `0e27e61`)

## Verification at closure

Full repository suite: **1539 passed, 1 pre-existing skip, 0 failed**. `locked_final_test` / `locked_topology_test`: unopened throughout M9 and both metadata corrections.

## Historical milestones preserved unmodified

This closure does not rewrite or delete any individual historical M9 milestone outcome. M9.0-M9.8 (and their preflight/correction documents) remain the authoritative historical record; this document only aggregates their already-frozen conclusions into one final decision.
