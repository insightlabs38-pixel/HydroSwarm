# Experiment plan (living document)

Tracks the plan's execution order, dependency gates, and experiment matrix (E0-E10).
Updated as bundles complete; see `experiment-registry.json` for the machine-readable
per-run provenance once Task 0.2 lands.

## Execution order (per overnight-plan.txt and operator instructions)

1. Phase 0 — baseline (this report) — **complete**
2. Bundle A (0.2-0.8) — experiment registry, job runner, lazy/sharded data, label audit,
   normalization, balanced sampler, split policy — **complete**
3. Bundle B (1.1-1.5) — variable-topology metadata/signatures/collation/hydraulic
   context/permutation tests — **complete**. Bundle B (2.1-2.6) — targets_v2 contract +
   Sentinel/Scout/Strategist/OOD/trajectory labels — **not started, next up**
4. Bundle C (3.1-3.8) — live/demo UI separation — **not started**
5. Bundle D (4.0-4.6) — configurable HydroCore architecture — **not started**
6. Cycle A corpus + E0/E3/E4/E9 smoke jobs — **not started**
7. Cycle B corpus + governed S screening/finalist training/calibration/eval — **not started**
8. HydroMono/no-adapter/baseline controls — **not started**
9. Updated HydroCore-M (winning S architecture only) — **not started**
10. Calibration, OOD, Scout, Strategist, full-trajectory evaluation — **not started**
11. Product/UI improvements (Phase 8) — **not started**

## Experiment matrix (from plan section "Experiment matrix")

| ID | Description | prior_mode | incident_pooling | message_direction | auxiliary_heads |
|---|---|---|---|---|---|
| E0 | Current architecture baseline | feature_and_logit | mean | forward_only | off |
| E1 | Prior as feature only | feature_only | mean | forward_only | off |
| E2 | Prior as logit correction only | logit_only | mean | forward_only | off |
| E3 | Source-conditioned pooling | feature_and_logit | source_conditioned | forward_only | off |
| E4 | Dual message channels | feature_and_logit | mean | dual_gated | off |
| E5 | Source-conditioned + dual channels | feature_and_logit | source_conditioned | dual_gated | off |
| E6 | Transport auxiliary objectives | feature_and_logit | source_conditioned | dual_gated | reconstruction,future_concentration,travel_time |
| E7 | Source-only diagnostic | (same as winning candidate, other task weights zeroed) | — | — | — |
| E8 | Full targets_v2 multitask | (same as winning candidate, all targets active) | — | — | — |
| E9 | Prior/fusion diagnosis | none / feature_only / logit_only / feature_and_logit / external-fusion-only | mean | forward_only | off |
| E10 | Curriculum/sampler diagnosis | (fixed architecture) uniform / topology-balanced / curriculum-stage / hard-negative-oversampled sampling | — | — | — |

Overnight scheduling guidance (from plan): Cycle A → E0/E3/E4/E9 smoke → Cycle B (parallel with
review) → E0 baseline + top 2 updated S variants → source-only vs multitask diagnosis → second
seed for leading variant → one updated M candidate → calibration + dev/OOD/full-trajectory eval →
locked test only when `final-selection.json` conditions are satisfied.

## Notes

- No experiment has been launched yet. This document will gain a "Results" section per
  experiment ID as Stage 1-3 training begins in Bundle F, cross-referenced by run ID once
  the registry (Task 0.2) exists.
- Scope reality check: the full plan (multi-topology dataset generation up to 40k examples,
  full architecture screening, HydroCore-M retraining, locked test) represents multiple
  days of real CPU-bound compute even on 16 vCPUs. This run will proceed through the bundles
  in dependency order, using resumable background jobs, and will honestly report how far it
  got rather than fabricate completion of stages that did not run.
