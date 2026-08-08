# Phase 14 — Promotion Gates

core-issues3.txt "PHASE 14 — PROMOTION GATES". Evaluated per-`HydroOutput`
key against the phase's 10 general gates plus the Scout/Strategist/
OOD-control-specific requirements, using Phase 13's measurements
(`phase13-metrics-and-baselines.md`) and the prior-pass Stage B/D/E/F
results. Role-level gating is too coarse per Phase 9.2 (a checkpoint may
have a validated validity head but an unvalidated pointer head) — this
report is deliberately output-level, matching the granularity Phase 9.2's
`trained_outputs`/`validated_outputs`/`runtime_enabled_outputs` metadata
scheme is meant to express (the metadata plumbing itself is Phase 9/15
work; this report is the evaluation basis for it).

The 10 general gates (numbered as in core-issues3.txt):
1. governed reproducible labels — 2. masks/valid counts correct —
3. nonzero gradient in real multi-topology batches — 4. beats/complements
deterministic baseline — 5. calibration valid where dependent — 6. no
safety regression — 7. ≥2 finalist seeds agree — 8. checkpoint records
trained+validated — 9. runtime integration has explicit fallback —
10. clean-clone replay reproduces the evaluation.

## Summary table

| output | gates 1–2 (labels/masks) | gate 3 (gradient) | gate 4 (beats baseline) | gate 7 (≥2 seeds) | verdict |
|---|---|---|---|---|---|
| `source_node_logits` | PASS | PASS | PASS (top1 0.72 vs. classical-only baseline already in production fusion) | PASS (0.7247/0.7149 Stage-A; 0.7205/0.7331 Stage-F, all close) | **already runtime-enabled (v3 path)**; re-verify under v4 metadata in Phase 15 |
| `source_region_logits` | PASS | PASS | not separately measured (piggybacks on source_node) | — | trained, not independently evaluated — **do not runtime-enable until measured on its own** |
| `event_presence_logits` | PASS (never masked) | PASS | PASS, but modest margin (F1 0.895 vs. a constant-positive baseline's own F1≈0.83) | PASS (0.895/0.897) | **methodologically promotable as advisory; blocked only on Phase 15 wiring — no runtime consumer exists yet** |
| `event_cause_logits` | PASS, with caveat (~5% HYDRAULIC_MISMATCH label noise in `cycle-b2`) | PASS | PASS on 3 supported classes (macro F1 0.698); **AMBIGUOUS/HYDRAULIC_MISMATCH have zero real examples** | not cross-checked per-class across seeds | **promotable for CONTAMINATION/SENSOR_FAULT/NORMAL only — the other 2 classes must be masked/suppressed at runtime, never surfaced as live predictions (Phase 6.5/9.3)** |
| `start_time_logits`/`duration_logits`/`relative_strength_logits` | PASS | PASS | PASS for start_time/relative_strength (accuracy 0.65–0.75 vs. ~25–33% chance); duration weak (0.50 vs. ~33% chance — real but modest) | PASS (close across seeds) | **promotable as diagnostic/advisory profile info; flag `duration` as lower-confidence** |
| `sensor_fault_logits` | mask PASS (scoped correctly); label population **degenerate** | PASS | **INDETERMINATE** — evaluated population has zero true negatives (Phase 13 finding #1) | can't assess meaningfully | **DO NOT PROMOTE — re-evaluate on a balanced population first** |
| `evidence_sufficiency` | PASS | PASS | PASS (F1 0.946, ECE 0.0085) | not cross-checked across seeds this pass | **promotable, ADVISORY ONLY per spec — deterministic controller stays authoritative regardless of quality** |
| `next_step_logits` | PASS, `ABSTAIN` has **zero support** in this evaluation batch | PASS | PARTIAL (macro F1 0.658; INSPECT_FAULTY_SENSOR recall 0.137 — weak) | not cross-checked | **promotable ADVISORY ONLY; suppress/flag low-confidence for INSPECT_FAULTY_SENSOR and ABSTAIN specifically** |
| `sample_node_logits`, `expected_information_gain`, `candidate_reduction_prediction` (Scout) | PASS | PASS | **FAIL** — `learned_scout`'s realized entropy reduction is *negative* (−0.219 bits) and *worse* than `random` (+0.007) or `fixed_order` (+0.015); only 56.7% agreement with classical EIG | n/a | **DO NOT PROMOTE — clear, direct fail of Scout's own promotion requirement ("no invalid... improved operational sample efficiency"); classical EIG remains the runtime policy** |
| `should_continue_sampling_logits` | PASS | PASS | not independently measured (bundled into Scout's negative result above) | — | **DO NOT PROMOTE alongside the rest of Scout** |
| `action_logits` / `action_pointer_logits` (legacy anonymous Strategist heads) | N/A — no governed target maps to these under `strategist_mode=candidate_conditioned` | untrained under the adopted architecture path | N/A | N/A | **orphaned under the adopted candidate-conditioned path (Phase 3.5/9.4) — recommend explicit removal from the final architecture, not carried forward as "trained"** |
| `plan_value`, `plan_validity_logits` (candidate-conditioned Strategist) | PASS (Phase 3.1 full-candidate-set verification) | PASS | **PASS, strong margin** — F1 0.997, NDCG@3 0.993 (`learned_prescreen`) vs. oracle 1.0, 67% fewer simulator calls than `exact_all` (3.0 vs 9.0) at `selected_valid_rate=1.0` | **FAIL — only 1 seed trained** (`v4-strategist-heads-v4corpus-corrected`, single run) | **conditionally ready — train and check a 2nd seed before promotion; WNTR remains authoritative regardless (structural safety net already in place)** |
| `exposure_proxy`/`pressure_risk_proxy`/`service_loss_proxy`/`containment_time_proxy`/`plan_regret_proxy` | PASS | PASS | PASS as ranking aids (physical-unit MAE now measured, Phase 13) — these are explicitly non-authoritative proxies by design | same 1-seed gap as above | **promotable as non-authoritative ranking aids only (never a substitute for exact WNTR consequences), same 2nd-seed blocker** |
| `ood_category_logits` | labels PASS (4 real-labeled categories) | **FAIL — zero real training gradient this run** (Phase 13 finding #2) | near-chance (macro F1 0.095) | n/a | **DO NOT PROMOTE — textbook case of the gate this phase exists to catch** |
| `ood_logits` (legacy 3-class severity) | deterministic severity is authoritative per Phase 6.1; this head is optional/non-authoritative by design | not evaluated this pass (superseded by `ood_category` work) | — | — | **not runtime-enabled by design regardless of quality — matches current (v3) runtime behavior already** |
| `uncertainty` | **no governed target/loss exists** (Phase 9.3's own audited concern) | untrained | N/A | N/A | **DO NOT PROMOTE — remove or define a governed target first** |
| `sensor_reconstruction_prediction`, `future_concentration_prediction`, `travel_time_prediction` (auxiliary) | PASS, masks per Phase 7.3–7.5 | PASS | Stage B ablation: all 3 retained, zero measured degradation to primary tasks | — | **training-only by design (spec: "must not be operator-authoritative") — correctly never intended for `runtime_enabled_outputs`** |

## Role-specific promotion requirements (explicit re-check)

**Scout** — requires "no invalid candidate selections; improved operational
sample efficiency or justified residual reranking; no budget-policy
violations." The middle condition **fails directly and measurably**
(negative realized entropy reduction, worse than doing nothing clever).
The other two cannot even be assessed given the architectural gap
documented in Phase 13 (`already_sampled` missing from `HydroBatch`).
**Verdict: FAIL. Scout stays fully non-authoritative; classical EIG /
fixed-order sampling remains the deployed policy.**

**Strategist** — requires "no increase in unsafe accepted plans; no
reduction in safe valid-plan discovery relative to the configured gate;
fewer exact simulator calls or lower regret; exact WNTR still verifies
every action considered operational." First three: **met** (0 unsafe
selections across 1000 scenarios in Stage E, `selected_valid_rate=1.0`
matches `exact_all`, 67% fewer calls). Fourth: **structurally guaranteed**
regardless of promotion decision — WNTR verification is not something the
learned ranker can bypass in the current runtime design. **Verdict:
substantively ready, formally blocked only on the general gate-7 2-seed
requirement.**

**OOD/control promotion** — requires "bounded false-normal rate; zero
unsafe planning outside validated range in the governed evaluation;
deterministic authority preserved." Deterministic authority: **preserved**
— `deterministic_plan_suppression_correctness_rate=1.0` (Phase 13),
independent of the learned head. False-normal rate/unsafe-planning:
**cannot be meaningfully certified** while `ood_category` sits at
near-chance (a near-chance classifier's low false-normal number, Phase
13's measured 4.6%, is not evidence of safety — it reflects rarely
predicting any single class, not correctly detecting normal operation).
**Verdict: FAIL for the learned category head; the deterministic OOD/
calibration/severity machinery, which is what actually gates planning
today, is unaffected and remains authoritative.**

## Net `runtime_enabled_outputs` recommendation (pre-Phase-15)

**Promote (advisory/non-authoritative only, deterministic gates unchanged):**
`event_presence_logits`, `event_cause_logits` (3 supported classes only),
`start_time_logits`/`relative_strength_logits` (duration flagged low-confidence),
`evidence_sufficiency`, `next_step_logits` (INSPECT_FAULTY_SENSOR/ABSTAIN
flagged low-confidence), `plan_value`/`plan_validity_logits`/consequence
proxies (pending 2nd Strategist seed).

**Do not promote:** `sensor_fault_logits` (degenerate eval population),
Scout's entire head group (`sample_node_logits`, `expected_information_gain`,
`candidate_reduction_prediction`, `should_continue_sampling_logits`),
`ood_category_logits`, `ood_logits`, `uncertainty`, `action_logits`/
`action_pointer_logits`.

**Always training-only by design:** `sensor_reconstruction_prediction`,
`future_concentration_prediction`, `travel_time_prediction`.

**No promotion in this report is final** — gates 8–10 (checkpoint
output-level metadata, explicit runtime fallback wiring, clean-clone
replay) are Phase 15/17 deliverables and have not been executed yet. This
report is the evaluation basis those phases consume, not a substitute for
them.
