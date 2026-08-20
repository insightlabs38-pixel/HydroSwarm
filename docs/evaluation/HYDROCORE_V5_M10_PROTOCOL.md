# HydroCore-v5 Milestone 10 protocol: system-level downstream validation (frozen before any M10 development result is inspected)

Amends `docs/evaluation/HYDROCORE_V5_EXPERIMENT_PROTOCOL.md`, which governs Milestones 1-11 and is not otherwise modified by this document. M9 is formally closed (`reports/evaluation/hydrocore-v5/m9-final/`) before this document is frozen: HydroCore-S (4,182,612 parameters, `CLASSICAL_HYDROCORE_S` + `AGE_FIX_ONLY` + `EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING` + `B_DEPTH_AWARE_CALIBRATION` + `ALPHA_0_1` + `SOURCE_REPRESENTATIVE_CALIBRATION_SUPPORT_20_PER_SOURCE`) is the selected predictor; HydroCore-M is not promoted; HydroCore-L remains unauthorized. This document does not reopen model-size, architecture, or temporal-representation search.

## 0. Numbering-conflict disclosure

A local, uncommitted planning file (`/workspace/experiments.txt`, outside this git repository, no commit history at all) uses "Milestone 10" for an unrelated automatic-commissioning feasibility experiment and "Milestone 11" for finalist selection/production promotion, ordered directly after a brief capacity-scaling "Milestone 9". The actually-committed, git-tracked master protocol (`HYDROCORE_V5_EXPERIMENT_PROTOCOL.md`) references "Milestone 11.6" by that number for the locked-final-test authorization gate, but defines no Milestone 10 content anywhere in the repository, and M9 itself was executed as a much deeper multi-part program (M9.0-M9.8) than that file's brief sketch anticipated -- the file is a rough early plan later superseded, milestone by milestone, through this repository's own governed amendment process, not a binding numbering contract.

Per this task's explicit governing instruction (received directly, not via that file): system-level downstream validation of OOD/fusion, Scout, Strategist, full-trajectory, and serving-path freeze is assigned the number **Milestone 10** with the `M10.0`-`M10.5` sub-structure below. This is a genuinely new milestone, not previously defined in any committed repository document. The uncommitted file's own "Milestone 10" (commissioning feasibility) and "Milestone 11" (finalist selection) concepts are real and not discarded, but their eventual milestone numbers are unresolved by this document and must be decided explicitly before either is frozen -- most likely renumbered to follow this M10 (e.g. as M12/M13), since `HYDROCORE_V5_EXPERIMENT_PROTOCOL.md`'s own Section 9 "No-lock rule" already anchors "Milestone 11.6" specifically to the locked-final-test gate, which is inconsistent with re-using "11" for commissioning. This document does not resolve that renumbering; it only avoids compounding the conflict by not silently claiming "11" for anything here.

## 1. Selected predictor identity (frozen, taken as-is, never retrained in M10 without an explicit refit amendment)

- **Variant**: `small` (`HydroCore.from_variant("small", ...)`), 4,182,612 trainable parameters.
- **Recipe**: `CLASSICAL_HYDROCORE_S` (AGE_FIX_ONLY representation, M8.7) + `EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING` (M9.0/M9.0a/M9.6).
- **Canonical checkpoints**: `reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed{20260814,31874,20260815}.json` `canonical_export_path`/`canonical_export_sha256` (M9.7A-authoritative `FINAL_STEP_1350` policy -- never best-validation). These are the SAME checkpoints M9.8 reused (`REUSED_M9_6_CHECKPOINT` provenance) and are the ONLY checkpoints M10 may use as "the selected predictor."
- **Calibration**: `B_DEPTH_AWARE`, alpha=0.1, source-representative support (20/source). M10 reuses M9's frozen calibration fit; M10 does not refit calibration except as explicitly triggered by Section 8 below.
- **Model construction kwargs**: `m9_1_common.SHARED_MODEL_CONFIG` (`prior_mode="feature_only"`, `event_control_heads=True`, `scout_control_heads=True`, `strategist_mode="candidate_conditioned"`, `consequence_prescreening_heads=True`, `ood_category_head=True`) + `CURRENT_MODEL_KWARGS` (M8.7 `AGE_FIX_ONLY`). Every head M10 characterizes was already jointly trained as part of this same construction during M9.6 training -- M10 introduces no new training objective and reuses these weights as-is.

## 2. Downstream component candidates and deterministic comparators (frozen, existing modes only -- no new architecture)

M10 characterizes value-add over deterministic/classical alternatives for the following already-implemented subsystems. No subsystem is redesigned; only existing, already-committed code paths are compared.

| Subsystem | Deterministic/classical comparator (already live) | Learned/neural comparator (already implemented, current promotion state) |
|---|---|---|
| OOD / confidence | `hydroswarm.inference.ood.OODDetector` (multi-signal: latent distance, energy, network novelty, demand shift, sensor consistency -- entirely non-learned) | `HydroCore.ood_category_head` (11-class, jointly trained with every M9.6 `small` checkpoint via `ood_category_head=True`) -- output name `ood_category`, governed by `hydroswarm.training.output_governance.OOD_CONTROL_OUTPUTS`; per `pipeline.py`'s own comment, "advisory only -- never read by OODDetector/the deterministic controller... every real checkpoint identity built so far excludes `ood_category`" from `runtime_enabled_outputs`, i.e. **currently NOT promoted, resolves to `None` in production today** |
| Fusion / disagreement | `hydroswarm.inference.fusion.fuse_source_probabilities` (`DYNAMIC_TRUST_FUSION_CONFIG = "fuse_source_probabilities-v1"`, dynamic-trust weighting over `TrustFeatures` including the classical `ood_score`) -- already the live production fusion policy | Same function; no separate "learned fusion" implementation exists beyond feeding the classical `ood_score` in -- M10.1 characterizes whether substituting/augmenting `TrustFeatures.ood_score` with the neural `ood_category` signal would help, without promoting it |
| Scout | `hydroswarm.agents.scout.HydroScout.deterministic_fallback` (nearest-unsampled-candidate heuristic) | `scout_control_heads=True` raw model outputs (`sample_node`/`information_gain`/`candidate_reduction`/`should_continue_sampling`, `SCOUT_OUTPUTS`) -- schema integration marked `SCOUT_STATE_SCHEMA_VERSION = "scout-state-v1-unbuilt"` in `checkpoint_identity.py`: **not yet built end-to-end**. Out of scope for M10.1; M10.2 (not run by this task) must first resolve this gap. |
| Strategist | `hydroswarm.agents.strategist.HydroStrategist.deterministic_fallback` | `strategist_mode="candidate_conditioned"` raw model outputs (`plan_validity`/`plan_value`/`exposure_proxy`/`pressure_risk_proxy`/`service_loss_proxy`/`containment_time_proxy`/`plan_regret_proxy`, `STRATEGIST_OUTPUTS`) -- schema integration marked `STRATEGIST_CANDIDATE_SCHEMA_VERSION = "strategist-candidate-v1-unbuilt"`: **not yet built end-to-end**. Out of scope for M10.1; M10.3 (not run by this task) must first resolve this gap. |
| Physical authority | WNTR/EPANET exact verification (`hydroswarm.simulation`) | none -- WNTR/EPANET remains final authority in every mode, unconditionally (Section 6) |

M10.1 (this task's scope) therefore characterizes exactly the OOD/fusion row; the Scout/Strategist rows are recorded here for roadmap completeness but explicitly deferred, consistent with Section 12 of this document ("staged gates").

## 3. Evaluation populations (development-only, disjoint from every historical range and from locked splits)

M10.1 generates a fresh, disjoint development-only OOD population at generation time -- it does not reuse M9.8's or any prior milestone's development/calibration incidents (avoids re-using rows the frozen calibration/S recipe has already been indirectly exposed to via development-representativeness audits).

- **Seed namespace**: role `ood_development_m10_1`, base `1_100_000_000`, `source_stride = 10_000`, families = all six (`golden-reference`, `branched-loop`, `loop-grid`, `coastal-branch`, `tree-branch`, `dense-loop`). Verified disjoint from every existing seed base in the repository (`grep` over `1_100_000_000..1_199_999_999`: zero hits before this document).
- **Split discipline**: physical scenarios are split before any causal-prefix/augmentation generation (Section 1 of the master protocol, unchanged); derived variants of one physical scenario remain in the same split; topology-family-level split preserved.
- `locked_final_test` / `locked_topology_test`: **not used**. Where an OOD condition would otherwise overlap a locked population (e.g. "severe topology shift"), a separate development-only governed perturbation is substituted instead (Section 5).
- Calibration reused as-is from M9 (Section 1) -- M10.1 does not refit calibration unless Section 8's invalid-calibration condition is itself the object of study, in which case only the fail-closed BEHAVIOR is measured, never a new fit.

## 4. Seeds

Three frozen seeds, reused from every M9 milestone for cross-milestone comparability: `20260814`, `31874`, `20260815`. No cherry-picking: all three reported per-seed in every M10.1 artifact (Section 10 below), matching M9.8's own per-seed reporting discipline.

## 5. OOD / shift conditions (development-only, frozen before execution)

| Condition | Construction |
|---|---|
| Topology shift | unseen development families (`coastal-branch`, `tree-branch`, `dense-loop`) vs. trained families (`golden-reference`, `branched-loop`, `loop-grid`) -- same unseen/trained split M9.8 already used |
| Sensor dropout / degraded availability | `missing_probability` elevated per `causal_prefix._degradation_probabilities`'s existing `DEGRADED`/`SHIFT` curriculum stages (already-implemented scenario curriculum, not a new mechanism) |
| Missingness | `sensor_mask`/`quality_mask` fraction swept via the same existing curriculum stages |
| Noise | existing sensor-noise scenario parameters at `SHIFT`/`ADVERSARIAL` curriculum stages |
| Cadence/timing perturbation | reuse M6's existing cadence/detection-delay scenario construction (`reports/evaluation/hydrocore-v5/m6-cadence.json`, `m6-detection-delay.json` show this mechanism already exists and was characterized pre-v5-causal) |
| Severity shift | `relative_strength`/`duration_minutes` tail of the existing `IncidentSourceProfile` distribution, development-only |
| Source-location difficulty | candidate-set size / calibration-coverage-miss incidents, already surfaced by the existing conformal calibration machinery |
| Model/classical disagreement | incidents where neural top-1 and classical (WNTR-informed) top-1 disagree, using the existing `fuse_source_probabilities` disagreement signal |
| Invalid/unavailable calibration | synthetic fail-closed condition: calibration artifact intentionally marked stale/absent for a subset of development incidents, to characterize fail-closed behavior only (never used to refit) |

No condition uses `locked_topology_test`. Any condition that would require it is replaced by the development-only equivalent above.

## 6. Metrics (frozen before execution; all reported per family / seed / causal-depth bucket / OOD condition-severity, per Section 10 of this task's own instructions)

**Predictive**: Top-1, Top-3, MRR, NLL, Brier.
**Uncertainty**: entropy, calibration coverage, candidate-set size, abstention rate, false-confidence rate, true-source rank.
**OOD**: AUROC/AUPRC (where labels make this meaningful), TPR/FPR at the preregistered threshold(s) below, in-distribution false-positive rate, OOD miss rate, severe-OOD miss rate (by severity group).
**System behavior**: deterministic-fallback activation rate, learned-vs-classical disagreement rate, invalid-calibration fail-closed behavior, fraction of incidents where learned fusion changes final ranking and whether that change helps or hurts (against ground truth).

## 7. M10.1 promotion rule (frozen BEFORE evaluation, per the standing "keep learned components only if they beat or complement simpler deterministic baselines" design principle)

`LEARNED_OOD_PROMOTED` requires ALL of:

- no regression in calibrated in-distribution behavior beyond the existing M9-frozen allowable bounds (coverage floor 0.85, alpha 0.1, unchanged);
- measurable OOD detection or decision-quality improvement over the deterministic `OODDetector` baseline (nonzero, CI-supported, using the same 2,000-resample / 90% paired-bootstrap procedure M9 used, bootstrap seed `20260819` reused for consistency);
- no unsafe-confidence increase (false-confidence rate does not increase);
- no increase in invalid-calibration acceptance (fail-closed behavior preserved or improved, never weakened);
- deterministic fail-safe remains available and is exercised in the invalid-calibration condition;
- all outputs finite (no NaN/Inf under any condition);
- no authority-boundary regression (WNTR/EPANET remains final authority in every arm; no arm allows a neural output to bypass it).

Any single failed criterion => `LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED` (an acceptable, reportable outcome, not a failure of this milestone). If the evaluation itself cannot be cleanly executed (e.g. an unresolved dependency on Scout/Strategist schema, or a governance blocker), the outcome is `OOD_VALIDATION_BLOCKED`, reported honestly rather than forced. Historical numeric OOD thresholds are reused verbatim where they exist (M9's coverage floor/alpha); no threshold is invented or loosened after seeing results.

## 8. Calibration fit/refit policy

M10.1 does not refit calibration. The frozen M9 `B_DEPTH_AWARE`/alpha=0.1/20-per-source calibration is used as-is for every arm compared. If a learned OOD/fusion configuration were ever promoted in a future milestone, any resulting calibration refit would be a SEPARATE, explicitly frozen, dated amendment -- never folded silently into a promotion decision.

## 9. Authority boundaries and fallback semantics (frozen, unchanged from existing production design; M10 only characterizes, never relaxes)

- WNTR/EPANET remains final physical authority in every M10 arm, unconditionally.
- Human approval remains mandatory; no learned component may autonomously actuate in any M10 arm.
- `HydroScout`/`HydroStrategist` remain `DeterministicAgent` subclasses with a `deterministic_fallback` that does not depend on any learned head; M10.0 verifies this fallback path is reachable and produces finite, valid output independent of model availability.
- The neural `ood_category` head, even where evaluated in M10.1's comparator C, is NEVER wired to gate/suppress the deterministic `OODDetector`/planning-suppression path during this milestone -- it is scored offline, out-of-band from any live decision.
- Audit/provenance events (`locked_test_opened_before/after`, checkpoint SHA-256, seed, commit) are recorded in every M10 artifact, matching the M9 convention.

## 10. Order of execution (staged, gated -- Section 12 restated)

1. `M10.0` -- non-tuning system/predictor preflight (interfaces, checkpoint/config/calibration identity, authority-boundary audit). No metric computed.
2. `M10.1` -- OOD/fusion validation (this task's scope). Ends in exactly one of `LEARNED_OOD_PROMOTED` / `LEARNED_OOD_NOT_PROMOTED_DETERMINISTIC_RETAINED` / `OOD_VALIDATION_BLOCKED`.
3. STOP. `M10.2` (Scout), `M10.3` (Strategist), `M10.4` (full-trajectory), `M10.5` (serving-path freeze) are NOT executed by this task and require their own explicit authorization, each after interpreting the prior gate's result -- per Section 2's finding that Scout/Strategist schema integration is itself `*-v1-unbuilt`, `M10.2`/`M10.3` will likely need a preflight-correction pass (mirroring `HYDROCORE_V5_M9_1_PREFLIGHT_CORRECTION.md`'s own precedent) before a scientific comparison is even executable, not merely a data-generation exercise.

## 11. M11 entry criteria (restated, not altered)

Per `HYDROCORE_V5_EXPERIMENT_PROTOCOL.md` Section 9 / `experiments.txt`'s original 11.6 language (informational, non-binding numbering as noted in Section 0 above): the locked final evaluation is never opened automatically. M10 completion (all of M10.0-M10.5) is a precondition for M11 finalist-selection entry, not a sufficient one -- M11 additionally requires every development gate green and an explicit human authorization before the one-time locked evaluation. This document does not authorize M11 and does not open any locked split.

## 12. Locked-test policy (restated)

`locked_final_test` and `locked_topology_test` remain unopened throughout M10.0 and M10.1. Asserted and recorded before and after each phase in every M10 artifact.
