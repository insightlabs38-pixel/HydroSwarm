# HydroCore-v5 Milestone 10.4: Governed Full-Trajectory End-to-End Validation Protocol

Frozen BEFORE any trajectory result is inspected, per `run_m10_4_preflight.py` (`M10_4_PREFLIGHT_PASS`)
and `scripts/hydrocore_v5/m10_4_protocol.py`. Protocol hash: **`cd0ac1f2d5a12a771cc441b4ea19bf0d76c672809b35d3d178f8893b768a177c`**
(`scripts/hydrocore_v5/m10_4_protocol.protocol_hash()`, machine-recomputable from the frozen constants
in that module; also recorded in `reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-protocol.json`).

Amends `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`. Does not reopen M9's model-size/architecture
search, nor M10.1/M10.2/M10.3A/M10.3B/M10.3C's closed learned-specialist promotion decisions.

## 0. Closed upstream state (taken as-is)

M9 CLOSED (selected predictor: HydroCore-S, canonical M9.6 `FINAL_STEP_1350` checkpoints).
M10.0 `SYSTEM_PREFLIGHT_PASS`. M10.1 `M10_1_LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED`.
M10.2 `M10_2_LEARNED_SCOUT_NOT_PROMOTED_DETERMINISTIC_RETAINED`. M10.3A `M10_3_STRATEGIST_REFIT_BLOCKED_FULL_RETRAIN_REQUIRED`.
M10.3B `M10_3B_POPULATION_AMENDMENT_REQUIRED`. M10.3C `M10_3C_LEARNED_STRATEGIST_NOT_JUSTIFIED`. The
Strategist decision is scientifically closed: M10.4 does not execute M10.3D, does not retrain any
learned specialist, and does not run a learned-vs-deterministic comparison for Scout, Strategist, or OOD.

## 1. System under test

Canonical M9.6 HydroCore-S predictor, frozen M9 `B_DEPTH_AWARE`/alpha=0.1 calibration, deterministic
OOD, deterministic fusion, the REAL production deterministic Scout path, the REAL production
deterministic Strategist/candidate-generation/PlanVerifier path, driven end-to-end through the real
FastAPI production application (`hydroswarm.api.create_app`), exactly as
`hydroswarm.evaluation.live_robustness` already does for a different (older, v4) checkpoint. See
`scripts/hydrocore_v5/m10_4_protocol.SYSTEM_UNDER_TEST` for the complete frozen identity record and
`scripts/hydrocore_v5/m10_4_common.py`'s module docstring for the full rationale of why a dedicated
`M10_4_PipelineFactory` is used instead of the module-level default `hydroswarm.api.app.app` (which
still serves the older, pre-M9 `hydrocore-v4` architecture-freeze checkpoint -- disclosed, non-blocking
finding `M10-4-DISCLOSED-02` in `m10-4-preflight.json`).

**Canonical checkpoint SHA-256** (unchanged from M9.6/M10.0-M10.3):

| seed | SHA-256 |
|---|---|
| 20260814 | `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5` |
| 31874 | `527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332` |
| 20260815 | `b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c` |

**Frozen calibration**: same examples, same alpha=0.1, same `B_DEPTH_AWARE` grouping, same
`minimum_group_size=10` as `run_m10_1_decide.py`/`run_m10_2_true_evaluation.py`'s already-established
"frozen calibration reuse" convention. M10.4 is the first M10 milestone to drive the real
`HybridInferencePipeline.analyze()` production code path, which enforces
`CalibrationArtifact.validate_runtime(model_hash=..., feature_schema_hash=..., fusion_config_hash=...)`
as a hard identity gate; M10.4 stamps the SAME frozen fit with the REAL runtime identity values
(the loaded checkpoint's actual SHA-256, `DEFAULT_FEATURE_SCHEMA.fingerprint`,
`DYNAMIC_TRUST_FUSION_CONFIG`) instead of the symbolic placeholder strings M10.1/M10.2's own scripts
use for their own (non-`HybridInferencePipeline`) evaluation harnesses. This is a required correctness
fix to the identity metadata, not a refit -- no example, alpha, or grouping changes.
`validated_topology_hashes` is additionally populated with the 3 TRAINED_FAMILIES network hashes so the
deterministic topology-novelty OOD check can distinguish a trained topology from a development-only
topology-shift family (this check is a no-op when this set is empty, which is how M10.1/M10.2 left it,
since they never exercised OODDetector.topology_novelty through the real pipeline either).

**Governance constant**: `trained_tasks = frozenset({"sentinel"})` -- identical to
`hydroswarm.runtime.v4_defaults.V4_TRAINED_TASKS`, independent of which checkpoint is loaded. This
structurally keeps every learned Scout/Strategist/OOD head non-authoritative
(`hydroswarm/inference/pipeline.py`: `"scout"`/`"strategist"`/`"ood" not in self.trained_tasks` gates).

## 2. Deterministic Scout / Strategist authority (traced, not assumed)

- **Scout**: `POST /api/incidents/{id}/samples/recommend` reads `analysis.sample_result`, which
  `HybridInferencePipeline.analyze()` computes via `self.sampling_ranker(...)` =
  `hydroswarm.sampling.rank_sample_locations` (expected-information-gain ranking over the classical
  signature posterior, with accessibility/already-sampled/budget constraints) -- **not**
  `HydroScout.deterministic_fallback`, which True M10.2 already disclosed is a narrower comparator, not
  the production path.
- **Strategist**: `POST /api/incidents/{id}/plans/generate` returns `analysis.plan_proposals`, computed
  by `self.planner(...)` = `hydroswarm.planning.generate_response_plans` (deterministic candidate
  generation). `POST /api/incidents/{id}/plans/{id}/verify` invokes the real, unmodified WNTR/EPANET
  `PlanVerifier`. M10.0's own closed preflight already mechanically confirmed the learned Strategist's
  NAMED candidate-conditioned proxy heads (`plan_value`, `plan_validity`, etc.) do not even exist in the
  M9.6 checkpoint's forward output (`strategist_named_proxy_heads_present=False`) -- learned Strategist
  scoring is doubly non-authoritative (schema-unbuilt AND `trained_tasks`-gated).

## 3. Population (development-only, fresh disjoint seed namespace)

Seed base `1_500_000_000`, range `[1_500_000_000, 1_599_999_999]`. Verified disjoint (mechanical range
overlap check, `m10_4_common.verify_seed_disjointness`) from: M9.4-and-later floor (`990_000_000+`),
M10.1 (`1_100_000_000+`), M10.2 refit train/validation (`1_200_000_000`/`1_200_100_000`), true M10.2 eval
(`1_200_200_000+`), M10.3 refit train/validation (`1_300_000_000`/`1_300_100_000`), M10.3C population
(`1_400_000_000+`), and the reserved-but-never-executed M10.3D block (`1_450_000_000+`, M10.3 is closed).
`locked_final_test`/`locked_topology_test` are never opened (`hydroswarm.evaluation.live_robustness.
locked_test_opened`, a static flag in `reports/results/v4/architecture-freeze.json`, unchanged).

**Topology families**: TRAINED_FAMILIES (`golden-reference`, `branched-loop`, `loop-grid`) get the full
7-condition matrix; UNSEEN development-only topology-shift families (`coastal-branch`, `tree-branch`,
`dense-loop`) get the `NOMINAL` condition only (the topology shift itself is the perturbation). All 6
loaders are the SAME already-governed `m10_common.ALL_FAMILY_LOADERS` used throughout M9/M10 -- no new
topology generator.

**Condition matrix** (reuses `hydroswarm.evaluation.live_robustness.Condition`/`_scenario_config`/
`_payloads` verbatim -- no new perturbation framework):

| kind | perturbation |
|---|---|
| `NOMINAL` | clean, full sensor coverage |
| `LOW_COVERAGE_ACTIVE_SAMPLING` | 25% initial sensor coverage (exercises the Scout evidence-acquisition loop) |
| `SENSOR_DROPOUT` | 30% forced missingness |
| `SENSOR_HEALTH_DEGRADED` | 50% of sensors frozen |
| `MEASUREMENT_NOISE` | moderate (std=0.05) measurement noise |
| `SEVERITY_SHIFT` | elevated source-strength bin (hydraulic mismatch) |
| `AMBIGUITY_DISAGREEMENT` | contradictory/disagreeing sensor readings |

`INCIDENTS_PER_CELL = 5`, `MAXIMUM_SAMPLES = 3` (existing production default, unchanged). 24 population
cells x 5 incidents x 3 canonical model seeds = **360 physical incidents**, each run as a byte-identical
initial-evidence PAIR (ARM_FULL / ARM_NO_EXTRA_SAMPLING) = 720 production-API incidents total.

## 4. Comparator

**ARM_FULL**: retained end-to-end system, production deterministic Scout sampling engaged.
**ARM_NO_EXTRA_SAMPLING**: identical checkpoint/calibration/OOD/fusion/candidate-generation/Strategist/
WNTR verification, identical initial evidence -- no active Scout sample request is ever issued; plans
are generated directly from the initial analysis. Each physical incident is realized as two separate
production-API incidents built from byte-identical initial observations; `paired_initial_state_equal` is
asserted for every pair. This isolates the value/cost of the sequential evidence-acquisition loop without
reopening any closed learned-vs-deterministic promotion question (Scout/Strategist/OOD are identical,
deterministic, and unmodified in both arms).

## 5. Fail-closed / failure-injection cases

Small, targeted, deterministic (not part of the statistical population): `MODEL_UNAVAILABLE`,
`CALIBRATION_UNAVAILABLE`, `SENSOR_STATE_INSUFFICIENT`, `SAMPLING_BUDGET_PREEXHAUSTED`,
`NO_ACCESSIBLE_UNSAMPLED_CANDIDATE`. Seed base `1_590_000_000` (within the M10.4 range, offset from the
main population).

## 6. Metrics

Source-inference (top-1/top-3/MRR/true-source-rank/entropy/candidate-set-size/calibrated state/
actionability, by seed/family/condition), Scout/evidence-acquisition (samples requested, rounds, stop
reasons, entropy/rank change per sample, safety counters), Strategist/plan (candidates generated,
WNTR-verified, rejection reasons, selected-plan consequences vs NO_ACTION), end-to-end decision utility
(localization correctness, actionable-calibrated-state rate, sampling's effect on the final decision,
exposure/service/pressure vs NO_ACTION). Exact WNTR truth is used OFFLINE only, to score outcomes after
selection -- never as a runtime feature.

## 7. Hard safety gates (non-negotiable)

`m10-4-safety-counters.json` must show every counter equal to zero: inaccessible sample selected,
already-sampled reselected, sampling budget exceeded, unverified plan surfaced as actionable,
WNTR-rejected plan surfaced as safe, human approval bypassed, autonomous actuation, learned
OOD/Scout/Strategist authority, nonfinite value reaching a decision, locked test opened.

## 8. Utility / quality gate

Frozen in `scripts/hydrocore_v5/m10_4_protocol.UTILITY_GATE` (A-G) BEFORE any trajectory result is
inspected: (A) all hard safety gates pass; (B) no material top-1 regression vs the no-extra-sampling
reference (5-point absolute non-inferiority margin); (C) active sampling is non-harmful in aggregate
where it changes evidence; (D) every approved plan is WNTR-VERIFIED; (E) no material systematic exposure/
service/pressure harm vs NO_ACTION; (F) every output finite; (G) every fail-closed case resolves boundedly
and deterministically. No threshold in this gate may be adjusted after trajectory results are computed.

## 9. Closure vocabulary

`M10_4_FULL_TRAJECTORY_PASS`, `M10_4_FULL_TRAJECTORY_UTILITY_NOT_ESTABLISHED`, or
`M10_4_FULL_TRAJECTORY_BLOCKED` -- exactly one, per `run_m10_4_closure.py`.

## 10. M10.5 is out of scope

Even a PASS here does not authorize serving-path freeze, runtime promotion, or opening the locked test.
M10.5 requires separate authorization.
