# Section 20 — Architecture-freeze acceptance gates

core-issues5.txt Section 20. Every gate below was checked for real against
HEAD (branch `agent/gcp-multitopology-v3`) after Section 19's final
calibration fit and release-bundle build — not asserted from memory.

**Do not open the locked test. Do not create `final-selection.json`. This
report stops at the authorized pre-test architecture-freeze boundary.**

## A. Runtime equivalence

| Gate | Status | Evidence |
|---|---|---|
| Candidate-conditioned model supports incident-only inference | ✅ | `tests/unit/test_incident_only_inference.py` (10 tests), core-issues5.txt Section 2, prior pass |
| Real V4 pipeline performs neural incident analysis without missing-plan-field failure | ✅ | Same test suite + `test_hybrid_pipeline.py`'s PASS-1/PASS-2 separation tests |
| Train/serve parity gate passes | ✅ | `scripts/run_train_serve_parity_gate.py` — 3 topologies × 2 conditions, all fields match; delta item 1 fix |
| Runtime normalization is the train-owned normalization | ✅ | `normalization_hash=e0808f21...` (real committed artifact, not `"none"`) in the release bundle; `load_v4_inference_bundle` validates it |
| Signature policy/artifact provenance is explicit and validated | ✅ | `signature-policy-manifest.json` in the release bundle; `resolve_model_input_signature_library`'s explicit `GOVERNED_KNOWN_NETWORK`/`RUNTIME_GENERATED_IMPORTED_NETWORK` modes (delta item 1) |

## B. Strategist

| Gate | Status | Evidence |
|---|---|---|
| Live planning uses real candidate-conditioned tensors | ✅ | Section 6, prior pass — `plan_proposals_to_candidate_tensors` wired into `HybridInferencePipeline._score_candidate_plans` |
| One canonical 9-template vocabulary | ✅ | `action_template_count=9` in checkpoint identity; `ACTION_TEMPLATE_COUNT` single source of truth |
| Learned Strategist only prescreens/orders candidates | ✅ | `plan_value`/`plan_validity` excluded from `runtime_enabled_outputs`; even if enabled, `_score_candidate_plans` only ever produces score deltas, never a `VERIFIED` decision |
| Exact WNTR/EPANET verification remains mandatory | ✅ | Structural — no code path sets `PlanDecision.VERIFIED` outside `PlanVerifier`/injected verifier |
| Second corrected Strategist seed evaluated | ✅ | `v4-strategist-heads-v4corpus-corrected-seed2`, seed `20260812` — Section 7 |
| Promotion decision is evidence-based and recorded | ✅ | `reports/results/v4/strategist-second-seed-promotion-decision.md` — decision: do not promote |

## C. Safety

| Gate | Status | Evidence |
|---|---|---|
| Exact-plan count and raw EPANET execution count are distinct | ✅ | `plans_exactly_verified` vs. `exact_simulations_used`/`epanet_executions`/`exact_simulation_cache_hits` — Section 8, `tests/integration/test_live_exposure_verification.py` |
| Simulator failures still consume/persist budget | ✅ | Section 9 — governed failure categories persist `exact_runs` before failing closed |
| Verification failure is auditable and fail closed | ✅ | `PLAN_VERIFICATION_FAILED` audit event type, distinct from `PLAN_REJECTED` |
| New evidence invalidates old verification | ✅ | Section 10 + delta item 5 (verification-context identity now includes simulator/threshold/policy identity, not only evidence) |
| Approval requires a current verification context | ✅ | `approve_plan` requires `decision == VERIFIED AND verification_status == CURRENT AND context_hash == current_hash` |
| Numeric threshold sensitivity can be represented | ✅ | Section 16 — numerical-sensitivity flag near verification thresholds |

## D. Governance

| Gate | Status | Evidence |
|---|---|---|
| `runtime_enabled_outputs` is operationally authoritative | ✅ | Section 11 + delta item 2 closed the one remaining gap (`source_node` was previously unconditional) |
| Disabled outputs provably cannot change downstream decisions | ✅ | `test_hybrid_pipeline_v4_gating.py`'s extreme-logit mutation tests, including this pass's new `source_node` cases |
| Learned OOD does not override deterministic OOD | ✅ | `ood_certificate` sources only `ood_level` (deterministic); `ood_category` excluded from every governance set regardless |
| Learned Scout remains disabled unless promoted | ✅ | Excluded from every governance set (Phase 14: negative realized entropy reduction) |
| Sensor-fault learned head remains disabled unless repaired/promoted | ✅ | Section 17 decision recorded; excluded from `runtime_enabled_outputs`/`validated_outputs` |
| Final V4 inference release bundle contains real hashes and truthful provenance | ✅ | Delta items 3/4 — real content hashes throughout (fusion policy, corpus manifest, trained-output claims all corrected); calibration now `FITTED` with real hashes (Section 19) |
| No placeholder "none" normalization identity for a normalized model | ✅ | `normalization_hash=e0808f21...` (real), verified by `load_v4_inference_bundle` |

## E. Reproducibility

| Gate | Status | Evidence |
|---|---|---|
| Full pytest suite passes | ✅ | **859 passed**, 0 failed (final run, HEAD after Section 19 changes) |
| Ruff passes | ✅ | `ruff check src tests scripts` — clean |
| Pyright passes | ✅ | 0 errors, 0 warnings |
| Corpus/trajectory gates pass | ✅ | `scripts/run_trajectory_corpus_gates.py` — all gates passed (`cycle_b2_original_nine` reports `passed_except_environment_limitation`, a known sandbox characteristic per Phase 17's own established handling, not a regression) |
| Train/serve parity gate passes | ✅ | Re-verified after Section 19 changes — still passes |
| Clean-clone inference bundle load succeeds | ✅ | `load_v4_inference_bundle` against the real Section 19 bundle; `tests/integration/test_v4_inference_bundle_loader.py` (6 tests) |
| Clean-clone non-locked self-test succeeds | ✅ | `test_v4_release_bundle.py::test_clean_runtime_loads_and_analyzes_from_the_bundle_alone` (deletes source checkpoint/normalization dirs, loads only from the bundle, analyzes a real incident); manually re-verified against the real Section 19 calibrated bundle (see `pretest-architecture-selection.md`'s "release bundle hash" section) |
| Secret/artifact governance remains clean | ✅ | `scripts/scan_secrets.py` — 0 findings across 1368 files considered |
| No locked-test data opened | ✅ | Confirmed — no code in this pass reads any `*/test/*` or locked-split path |
| `final-selection.json` absent | ✅ | Confirmed absent from the repository |

## F. Product-facing contracts

| Gate | Status | Evidence |
|---|---|---|
| Decision Authority / Applicability Certificate API exists | ✅ | Section 13 — `hydroswarm.inference.authority` |
| Verification context/current-vs-stale status exists | ✅ | Section 10 + delta items 5/6 — `verification_status`, `context_hash`, `_invalidate_stale_verifications` |
| Verified Pareto frontier API/domain object exists | ✅ | Section 14 + delta item 9 fix — `hydroswarm.planning.pareto`, now with `FrontierGroup` separation |
| Evidence Value / Stop Certificate exists | ✅ | Section 15 + delta item 8 fix — `hydroswarm.inference.evidence_certificate`, now with `candidate_region_calibrated` |
| Numeric-sensitivity status exists | ✅ | Section 16 |

## Overall result

**Every mandatory gate is green.** The architecture is ready to be
considered frozen per this specification's own "Final operating
principle" — a system that can honestly report calibration validity,
deterministic-vs-learned authority, verification currency, and threshold
sensitivity, rather than one that maximizes the count of enabled learned
components.

Per this pass's explicit instructions: **stop here.** Do not open the
locked test. Do not create `final-selection.json`. See
`reports/results/v4/pretest-architecture-selection.md` for the full
Section 21 report.
