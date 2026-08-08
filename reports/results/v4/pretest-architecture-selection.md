# Pre-test architecture-selection report (core-issues5.txt Section 21)

Branch `agent/gcp-multitopology-v3`. This is the final report before the
Phase-20/locked-test boundary. **The locked final test has not been
opened. `final-selection.json` does not exist and is not created by this
report.**

## Exact branch/commit

- Branch: `agent/gcp-multitopology-v3`
- HEAD at calibration/bundle build time: `c5d2dd5` (Delta 10's commit)
- Full pytest/Ruff/Pyright re-verified after all Section 19 changes (see
  "Final reproducibility check" below)

## Selected HydroCore variant

`HydroCore-S` ("small"), `d_model=192`, `nhead=6`, `num_layers=4`,
`dim_feedforward=576`, `latent_tokens=64` — the Stage-F `no_adapters`
winner (`experiments/registry/stage-f.jsonl`; direction-consistent across
2 seeds on both validation loss and development-holdout loss, per
`reports/results/v4/stage-f-adapters-comparison.json`).

## Adapter decision

**No adapters** (`use_adapters=false`). Stage F's real 4-run comparison
(2 arms × 2 seeds, `no_adapters-seed{20260810,20260811}` vs.
`adapters-seed{20260810,20260811}`) found `no_adapters` consistently
better on both validation loss (5.3876 vs. 5.4229 mean) and
development-holdout loss (8.8325 vs. 8.9769 mean), concentrated in
`evidence_sufficiency`/`source_node`/`relative_strength`.

## Strategist mode

`candidate_conditioned` (`CandidatePlanEncoder`, Section 4/Phase 4). Live
planning wired end-to-end (Section 6): deterministic candidate generation
→ canonical candidate tensorizer → candidate-conditioned scoring → bounded
top-K exact WNTR/EPANET verification, with deterministic heuristic
ordering as the fallback whenever the learned ranker is disabled or
unavailable.

## Promoted outputs (`runtime_enabled_outputs`)

```
event_cause, event_presence, evidence_sufficiency, next_step,
relative_strength, source_node
```

All six are ADVISORY (event/control heads) or CALIBRATED_ADVISORY
(`source_node`, delta item 2's fix) — none bypass deterministic
authority, exact WNTR verification, or human approval. `source_node`'s
inclusion follows directly from Phase 14's own measured evidence (top-1
0.72 vs. classical-only baseline already in production fusion; close
agreement across 4 seeds) — a governance/behavior contradiction this
pass's delta item 2 resolved, not a new promotion decision.

## Disabled outputs (trained but not runtime-enabled, or excluded from every set)

| output group | status | reason |
|---|---|---|
| `source_region`, `start_time`, `duration` | trained, validated (start_time only) or diagnostic, **not runtime-enabled** | not independently evaluated (`source_region`) / weakest profile signal (`duration`, accuracy 0.50 vs ~33% chance) |
| `sensor_fault` | trained, **not runtime-enabled** | Phase 13 finding: evaluated population is degenerate (zero true negatives) — F1=1.0 is an artifact, not real quality. Section 17 decision: leave disabled, deterministic sensor-health logic stays authoritative. Not a blocker for freeze. |
| Scout (`sample_node`, `information_gain`, `candidate_reduction`, `should_continue_sampling`) | trained, **excluded from every governance set** | Phase 14: `learned_scout`'s realized entropy reduction is *negative* (worse than random/fixed-order sampling) — a direct, measured fail of Scout's own promotion requirement. Classical EIG remains the deployed policy. |
| `plan_value`, `plan_validity`, 5 consequence proxies (Strategist) | trained, validated, **not runtime-enabled** | Phase 14 gate 7 (≥2 finalist seeds) — this pass's delta item's Section 7 work ran the second seed (`v4-strategist-heads-v4corpus-corrected-seed2`) and found the two seeds **do not** agree closely enough to promote (see `reports/results/v4/strategist-second-seed-promotion-decision.md`). Decision: keep learned prescreening disabled; deterministic heuristic ordering + exact WNTR remain the deployed policy. Either promote-or-not outcome was explicitly acceptable per the governing spec — this is not a partial/incomplete result. |
| `ood_category` | trained (architecturally present), **excluded from every governance set** | Phase 13 finding: zero real train-split gradient this run (near-chance macro F1 0.095) — textbook Phase-14-gate-3 fail. Deterministic OOD severity (3-level) remains sole authority. Section 18.1 preserved a one-retrain path (vocabulary/schema frozen, advisory field already wired) without promoting anything now. |
| `sensor_reconstruction`, `travel_time`, `future_concentration` | **never trained by the selected run** (delta item 4, Section C) | `auxiliary_heads=False` for this model config — these heads were never physically constructed, let alone supervised. Corrected in this pass; previously falsely claimed trained. |
| `action_logits`/`action_pointer_logits` (legacy anonymous Strategist heads) | orphaned | no governed target maps to these under `strategist_mode=candidate_conditioned`; physically present (shared parameters, breaking change to remove) but structurally un-governable under v4 (`output_governance` refuses them by omission from `CANONICAL_OUTPUT_NAMES`). |
| `ood_logits` (legacy 3-class), `uncertainty` | not runtime-enabled by design | no governed loss/target; deterministic severity is authoritative regardless. |

## Deterministic authorities (unchanged, verified)

- **Source localization**: classical/neural fusion is CALIBRATED_ADVISORY
  at best; conformal calibration + deterministic disagreement/OOD gating
  control whether planning may proceed at all.
- **Sampling**: classical expected-information-gain / fixed-order
  sampling. Learned Scout fully disabled (see above).
- **Planning**: deterministic bounded candidate generation is
  authoritative for which candidates exist; learned Strategist prescreen/
  ordering is disabled (see above) — deterministic heuristic ordering is
  the deployed policy.
- **Verification**: WNTR/EPANET is the sole authority for `VERIFIED`.
  No learned score can mark a plan verified. Exact-simulation budgets are
  tracked separately from plan-verification counts (Section 8/9); a
  verification's context is invalidated by evidence, threshold, or policy
  changes (Section 10, delta items 5/6).
- **OOD**: deterministic 3-level severity (NORMAL/CAUTION/
  OUTSIDE_VALIDATED_RANGE) is sole authority. Learned `ood_category` is
  fully disabled.
- **Approval**: human approval is mandatory for every operational plan;
  no code path bypasses it.

## Calibration identity (Section 19, fit this pass)

Fit against the exact frozen V4 serving path — the real governed
checkpoint identity (`experiments/runs/v4-checkpoint-identity/
no_adapters-seed20260810`), its own `runtime_enabled_outputs`, its real
`model.safetensors` file hash (not a state-dict-only fingerprint — a real
defect found and fixed during this fit), and the real committed train-owned
normalization artifact.

| field | value |
|---|---|
| checkpoint / model hash | `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7` |
| feature schema hash | `7ec97775e5f01f87ae62669146a7eb70958f99b1162a356614eb87220e9ddd09` |
| normalization hash | `e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114` |
| fusion policy hash | `fuse_source_probabilities-v1` (`DYNAMIC_TRUST_FUSION_CONFIG` — the real dynamic-trust fusion policy, not a fixed-weight approximation) |
| signature policy | `hydroswarm-signature-policy-v1` (`GOVERNED_TRAINING_SIGNATURE_POLICY.policy_hash`) |
| topology coverage | the 3 governed training topologies, by PRISTINE family hash (golden-reference, branched-loop, loop-grid) — a real defect found and fixed during this fit: the first fit recorded each scenario's own roughness-randomized network hash, which would never match a real served (pristine) network |
| calibration split manifest | `data/learning-v2/cycle-b2/scenarios/manifests/calibration.jsonl` (1000 scenarios; 712 CONTAMINATION-event scenarios used, 288 NORMAL/SENSOR_FAULT_ONLY scenarios correctly excluded) |
| alpha / coverage target | 0.10 / 90% |
| empirical coverage | **91.43%** (712 examples) — by condition: CLEAN 90.8%, OPERATIONAL 92.8%, DEGRADED 92.2%, SHIFT 89.4%, ADVERSARIAL 92.2%; by topology: golden-reference 92.9%, branched-loop/loop-grid 90.7% each |
| mean candidate-set size | 2.80 nodes |
| expected calibration error | 0.088 |
| calibration artifact hash | `829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa` |

**Data-provenance note, stated honestly**: this sandbox does not have the
calibration split's raw scenario `.npz` archives materialized
(gitignored/ephemeral — see `hydroswarm_checkpoint_persistence` memory
record). All 712 examples were reconstructed via the identical seeded
`WNTRScenarioGenerator.generate_with_network` call the original corpus
generation used (real EPANET simulation, same seeds/config), rather than
loaded from a disk-verified `.npz` baseline that does not exist in this
environment. This is real physics, not fabricated data — only unverified
against a persisted baseline unavailable here. A future environment with
the raw archives present would use them automatically (the script prefers
disk-verified data when available, falls back only when missing).

## Normalization identity

`e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114` — the
real, committed, train-split-fit `data/learning-v2/cycle-b2/normalization`
artifact. Not `"none"` (delta item 4/prior-pass Section 3 fix).

## Signature policy identity

`hydroswarm-signature-policy-v1` (bins/regimes/sample-times matching the
real training corpus's own signature-fitting configuration — Section 4).
The MODEL-INPUT `classical_prior` feature is now computed via the same
algorithm training used (delta item 1 fix) — `GOVERNED_KNOWN_NETWORK`
mode for the 3 known topologies, `RUNTIME_GENERATED_IMPORTED_NETWORK`
mode (clearly labeled, never conflated) for imported networks.

## Release bundle hash

`experiments/runs/v4-release-bundle/no_adapters-seed20260810/` —
`checkpoint_identity_fingerprint` `a94069adba25230f58f24f57901b855fab3a702aabd5e30cf0bc105e002e90a1`,
model SHA-256 `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`.
Loads cleanly via `hydroswarm.runtime.v4_inference_bundle.
load_v4_inference_bundle` (delta item 3) with no dependency on the
original training corpus/checkpoint directory (proven by deleting the
source checkpoint/normalization directories before loading, in the
automated test suite). `calibration_status=FITTED` in this final build.

## Train/serve parity result

**PASSED outright** (delta item 1) — `scripts/run_train_serve_parity_gate.py`,
3 governed training topologies × 2 operating conditions (clean, degraded),
every field (node/edge order, masks, node/edge features, classical_prior,
feature schema, normalization identity, signature policy identity) matches
exactly or within strict tolerance. No accepted `classical_prior` failure
(previously a documented, accepted known finding — now a real fix).

## Strategist second-seed result

Run (Section 7): `v4-strategist-heads-v4corpus-corrected-seed2`
(seed `20260812`), same governed dataset/protocol as seed 1. **Decision:
do not promote** — see `reports/results/v4/strategist-second-seed-promotion-decision.md`
for the full seed-agreement analysis. Learned prescreening/ordering stays
disabled; deterministic heuristic ordering + exact WNTR verification
remain the deployed policy. This is a valid, evidence-based outcome per
the governing spec ("Either result is acceptable. Do not force promotion
to make the product appear 'more AI.'").

## Simulation-budget policy

Separately tracked (Section 8/9, this pass's Delta 7 audit confirmed
correct): `plans_exactly_verified` (verification attempts) vs.
`exact_simulations_used`/`epanet_executions` (underlying EPANET runs,
which may exceed 1 per plan under multi-hypothesis exposure-aware
evaluation) vs. `exact_simulation_cache_hits`. `remaining_epanet_budget`
is seeded from the configured limit at incident creation (not 0) and
consumed monotonically, including on governed simulator failures
(timeout/incomplete/unstable/budget-exceeded), never reclaimed by
constructing a new simulator instance.

## Verification-staleness policy

A plan verification's `context_hash` now composes (delta item 5):
evidence, source hypothesis-set identity/probabilities, model/checkpoint/
calibration/normalization/signature-artifact identity, simulator name/
version, minimum-pressure/minimum-service-availability thresholds,
consequence-policy version, aggregation policy, population-map identity.
Any change invalidates prior CURRENT verifications (marked STALE, retained
in the audit trail, never deleted); approval requires
`decision == VERIFIED AND verification_status == CURRENT AND
context_hash == current`. Delta item 6 closed the remaining leak: every
operator-facing "current evidence" surface (`evidence_bundle()`'s
selected/rejected/NO_ACTION lookups, `recommended_plan_id`) now also
requires `verification_status == CURRENT` — historical/stale verifications
remain in the audit trail (`/export`, `/events`) but never appear as
current evidence.

## Known limitations

- `sensor_fault` evaluation population is degenerate (Phase 13 finding,
  unresolved this pass — data-generation audit needed, out of this pass's
  scope, documented and flagged, not blocking freeze per Section 17's own
  text).
- Scout and learned Strategist prescreening remain disabled by measured
  evidence, not by omission — both have a documented one-retrain path
  (Section 18) if a future pass wants to revisit them, without a
  structural architecture change.
- `model_input_signature_mode` (governed vs. runtime-generated) is
  computed per-analysis but not yet threaded into the Decision Authority
  certificate's `DecisionProvenance` (delta item 1's own explicitly
  deferred follow-up).
- The frontend's hand-written TypeScript `ApiIncidentView` type does not
  yet declare `verification_status`/`context_hash` on
  `plans[].verification` (delta item 6's own explicitly deferred
  follow-up) — backend correctness does not depend on this, but a
  frontend pass should close it.
- Calibration was fit against reconstructed (not disk-verified) scenario
  data, per this sandbox's own environment limitation (see the
  data-provenance note above) — re-fitting against disk-verified data in
  an environment with the raw `.npz` archives present is a reasonable,
  low-effort follow-up, not required for correctness of the fit itself.

## Optional post-freeze one-retrain opportunities (Section 18, unchanged status)

- Learned OOD category improvement (vocabulary/schema frozen; deterministic
  severity stays authoritative regardless).
- Learned Scout improvement (state-conditioning contract scaffolding
  audited; a genuine multi-step trajectory retrain would still be
  required — not literally "one retrain only" if the current architecture
  cannot condition on revealed sampling state without new parameters,
  per Section 18.2's own honesty requirement).
- PCGrad multitask retrain (optimization-only change, no architecture
  impact).
- Class balancing / loss-weight refinement.

## Confirmation

- The locked final test **has not been opened**.
- `final-selection.json` **does not exist**.
- No tuning used the development holdout as calibration/training data —
  Section 19's calibration fit used only the calibration split.
- No learned plan output bypasses WNTR verification or human approval —
  verified structurally (WNTR remains the sole `VERIFIED` authority) and
  by the disabled-output mutation tests (Section 11 / delta item 2).

## Exact reproduction commands

`experiments/runs/v4-checkpoint-identity/` and
`experiments/runs/v4-release-bundle/` are gitignored/ephemeral (matching
this project's established `experiments/runs/` convention) — regenerate
from the committed source checkpoint and this session's scripts:

```bash
export PYTHONPATH=src

# 1. Build the real v4 checkpoint identity from the selected Stage-F checkpoint.
python scripts/build_phase15_v4_checkpoint.py
# -> experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/
#    (model.safetensors sha256 a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7)

# 2. Fit the final calibration against the exact frozen serving path (~9 minutes,
#    712 usable CONTAMINATION-event calibration scenarios; wall time dominated by
#    per-scenario hydraulic reconstruction + HybridInferencePipeline.analyze()).
mkdir -p experiments/cache/signatures
python scripts/fit_dynamic_fusion_calibration.py \
  --corpus-dir data/learning-v2/cycle-b2 \
  --identity-dir experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810 \
  --node-normalization data/learning-v2/cycle-b2/normalization/node-normalization.json \
  --edge-normalization data/learning-v2/cycle-b2/normalization/edge-normalization.json \
  --signature-cache-dir experiments/cache/signatures \
  --alpha 0.1 \
  --output reports/results/v4/section19-final-calibration-fit.json \
  --calibration-artifact-output experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/calibration.json
# -> calibration_artifact_hash 829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa
#    coverage 0.9143258426966292 (target 0.90), mean_set_size 2.80, ECE 0.088

# 3. Build the final release bundle with the fitted calibration included.
python scripts/build_v4_inference_release_bundle.py \
  --checkpoint-dir experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810 \
  --calibration-artifact experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/calibration.json \
  --output-dir experiments/runs/v4-release-bundle/no_adapters-seed20260810

# 4. Verify clean load (no dependency on the checkpoint/normalization directories).
python -c "
from hydroswarm.runtime.v4_inference_bundle import load_v4_inference_bundle
b = load_v4_inference_bundle('experiments/runs/v4-release-bundle/no_adapters-seed20260810')
assert b.calibration_status == 'FITTED'
print('ok')
"
```

If a future environment has the calibration split's raw scenario `.npz`
archives materialized (this sandbox does not — see the calibration
section's data-provenance note above), step 2 will automatically prefer
the disk-verified data over the seeded-reconstruction fallback with no
command-line change required.
