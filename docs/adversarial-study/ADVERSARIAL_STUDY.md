# HydroSwarm adversarial validation study

## Frozen test plan

This plan was frozen before adversarial execution. `PASS` means the stated
fail-safe behavior was observed; it is not evidence of field safety.

| ID | Hypothesis | Adversarial input/action | Expected fail-safe behavior | Invariant |
|---|---|---|---|---|
| ADV-01 | Missing data cannot advance evidence | all observations missing/zero quality | analysis or controller stops/samples; no plan generation | insufficient evidence |
| ADV-02 | Malformed observations are rejected | missing fields, unknown node, backwards timestamps | 4xx/schema rejection; no persisted incident mutation | invalid observations |
| ADV-03 | Numeric poison cannot become evidence | NaN, infinity, negative and extreme fields | request rejected or planning suppressed | invalid observations |
| ADV-04 | Duplicate/stale telemetry cannot silently strengthen evidence | duplicate sensor/time and old received samples | no false planning eligibility; provenance changes visibly | evidence integrity |
| ADV-05 | Contradictory/bias/outliers are not trusted | mutually inconsistent, drift-flagged observations | suppression, sampling, or conservative result | evidence integrity |
| ADV-06 | High ML confidence cannot override gates | high neural score plus uncalibrated/high-disagreement fixture | planning remains suppressed | calibration/disagreement |
| ADV-07 | Conformal boundary/artifact failure is safe | threshold-edge posterior; missing/corrupt calibration | no planning under invalid calibration | calibration |
| ADV-08 | OOD topology/state cannot plan | unseen topology hash, demand/pipe/valve change | OOD caution/outside result suppresses planning | OOD |
| ADV-09 | Suppression cannot be bypassed | direct plan-generation and workflow API attempts | 409/no plan or no approvable plan | planning authority |
| ADV-10 | Model plans cannot bypass static constraints | malformed/unknown/inoperable/extreme actions | rejected before/at verifier | deterministic constraints |
| ADV-11 | Simulator failures fail closed | exception, timeout, incomplete, unstable, budget failure | ABSTAINED/REJECTED; never VERIFIED | verification |
| ADV-12 | Hydraulic/service violation cannot verify | low pressure/service exact evaluation fixture | REJECTED; never approvable | verification |
| ADV-13 | Verification binds a plan | mutate/reidentify plan after verification | approval rejected or separately re-verifies exact plan | provenance binding |
| ADV-14 | New evidence invalidates verification | add a valid sample after VERIFIED | old verification STALE and approval 409 | freshness |
| ADV-15 | Network/model provenance invalidates verification | mutate network/provenance after VERIFIED | old verification stale/unapprovable | freshness |
| ADV-16 | Direct/rejected approval is blocked | approve before verify, REJECTED, malformed request | 409/422, no approval receipt | approval gate |
| ADV-17 | Duplicate approval is idempotent/safe | replay same approval request concurrently and serially | at most one approval event/receipt | approval lifecycle |
| ADV-18 | Approval/evidence race is safe | synchronize approval with sample mutation | approval cannot close stale state | race safety |
| ADV-19 | Verification/approval race is safe | concurrent verify/approve | approval only sees a complete current verification | race safety |
| ADV-20 | Restart preserves safety state | restart at VERIFIED/STALE/APPROVED | stale stays stale; no resurrection | persistence |
| ADV-21 | Reference is checksummed replay | corrupt/replay reference artifact | checksum/chain rejection | reference integrity |
| ADV-22 | LIVE/REFERENCE/cached provenance cannot blur | call cached live inputs and reference endpoints | provenance/mode remains distinct | provenance |
| ADV-23 | API resource abuse is bounded | oversized/malformed requests and headers | 4xx/413 without state changes | API resilience |
| ADV-24 | State-machine direct transitions are rejected | invalid controller transitions/approve state | exception; no state advance | authority boundaries |
| ADV-25 | Fuzzed API observations retain safety | property-generated valid/invalid payloads | invalid never persists; valid stays typed | input integrity |
| ADV-26 | Replay is deterministic and checksummed | repeated reference/live replay reads | reference chain/checksum stable, LIVE labelled LIVE | provenance |

## Status

## Executive summary

This local-only study falsified several claimed authority and integrity
boundaries. It does **not** prove or disprove field safety. The most serious
finding is that a planning-suppressed `DEMO_FALLBACK` incident can reach a
`HUMAN_APPROVAL` workflow response containing a `VERIFIED` plan through
`POST /api/incidents/{id}/workflow`. Other reproduced failures let duplicate
approvals succeed, accept positive-infinite evidence, retain duplicate/stale
evidence, approve plan content changed under the same ID, and approve after a
network record replacement.

The baseline was clean before the study: 1,026 passed in 561.24 seconds,
87.56% coverage. The focused support suite passed 72 tests and the dedicated
adversarial harness passed 19 probes. “Harness passed” means that its probes
reproduced the documented behavior; it does not mean the system passed the
safety expectations.

## Environment and scope

- Commit under test: `fe8a09952735dfbf1d6f80da4904627c3cccc898` (`main` HEAD).
- Study branch: `review/adversial-validation`; no production source was
  changed and `main` was not modified.
- Runtime: Python 3.12.13, pytest 9.1.1, FastAPI TestClient, SQLite temporary
  databases, local WNTR/EPANET test paths only.
- No real infrastructure, external telemetry, or actuator was contacted.

## Threat model and invariants

The attacker is a malformed or adversarial local API client, stale/replayed
data source, corrupted artifact/cache input, or an actor able to trigger an
ordinary API route while an incident is in flight. Local persistence tampering
was also modeled where the invariant explicitly requires a verification to be
bound to exact plan/network/evidence content. The frozen matrix above contains
the exact pre-execution expected behavior for every case.

## Methodology and evidence

1. Created the study branch from `main` HEAD and inspected the verified
   controller, API, Pydantic contracts, persistence, inference/OOD gate,
   verifier, reference replay, live-input cache, and frontend paths.
2. Ran the unmodified baseline:
   `.venv/bin/python -m pytest --cov=hydroswarm --cov-branch` → **1026 passed**.
3. Added the frozen test matrix and local test harness only.
4. Ran the focused support suite and dedicated probes. Exact commands and each
   result are preserved in [results.json](results.json); reproducible probes
   are in [test_adversarial_validation.py](../../tests/adversarial/test_adversarial_validation.py).
5. No scientific thresholds, suppression rules, or production code were
   modified in response to a failure.

Post-study baseline rerun: `.venv/bin/python -m pytest --cov=hydroswarm
--cov-branch` → **1045 passed** in 624.98 seconds at 87.57% coverage. The
increase from 1,026 is exactly the 19 test-only adversarial probes; no
production behavior was modified.

## Results

| Result | Cases |
|---|---|
| FAIL | ADV-03, 04, 09, 13, 15, 17, 21, 22 |
| PASS | ADV-01, 02, 06, 07, 08, 10, 11, 12, 14, 16, 20, 23, 24, 26 |
| INCONCLUSIVE | ADV-05, 18, 19, 25 |

The machine-readable record gives actual behavior, exact test function, and
severity for every case. The four INCONCLUSIVE cases are deliberate scope
limits, not passes: full contradictory-sensor trajectories and deterministic
concurrency interleavings were not proven, and property fuzzing was not added.

## Findings

### HIGH — planning-suppression bypass (ADV-09)

`/analysis` accurately reports `planning_allowed: false` for `DEMO_FALLBACK`.
`/workflow` then constructs a Sentinel with `evidence_sufficient` forced true
when runtime mode is demo fallback, allowing the controller to return
`HUMAN_APPROVAL` and a `VERIFIED` plan. The ordinary plan-generation route
behaves differently. This is an authority-boundary violation even though the
separate API approval endpoint rejects that particular workflow verification
because it lacks an API context hash.

Reproduce: run `test_planning_suppression_is_bypassable_through_workflow_endpoint`.

### HIGH — non-finite evidence accepted (ADV-03)

The `NonNegative` concentration contract accepts `+Infinity`. Raw JSON
`Infinity` returns `201 Created` and persists/retrieves the incident. It is
not valid physical evidence and must fail before persistence.

Reproduce: run `test_positive_infinity_json_is_accepted_and_persisted`.

### HIGH — duplicate and stale evidence accepted (ADV-04)

The sample API accepts an identical sensor/timestamp/value and an observation
from 2000, then increments `sample_count` for both. There is no API
deduplication identity or maximum evidence age. This can influence posterior
weight and sample budget before later quality logic gets a chance to suppress.

Reproduce: run `test_duplicate_and_stale_samples_are_accepted_without_deduplication_or_age_gate`.

### HIGH — verification is not bound to plan content or current network (ADV-13, ADV-15)

Approval verifies a `plan_id` and context hash but no canonical plan digest.
Replacing actions under the same ID leaves approval possible. Separately,
replacing the network record under the same `network_id` through validation
does not change approval context and approval succeeds. The tested validation
route requires injected-verifier mode; nevertheless it demonstrates the
missing network-content binding, and direct local network-file/persistence
mutation has the same gap.

Reproduce: run `test_plan_content_mutation_with_same_id_remains_approvable`
and `test_network_replacement_does_not_invalidate_verified_plan`.

### HIGH — replayed approval accepted (ADV-17)

Two identical serial approval requests both return 200 and append approval
events after the incident has already reached `CLOSED`. No actuation occurred,
but approval/audit authority is replayable and receipt semantics are not
idempotent.

Reproduce: run `test_duplicate_approval_requests_are_not_idempotent`.

### MEDIUM — reference integrity is asserted but not enforced at serving (ADV-21)

The reference endpoint serves syntactically valid JSON without recomputing
`artifact_sha256` or validating its event chain. A forged artifact with an
intentionally wrong checksum returned 200. The committed artifact's build
tests are deterministic; the serving boundary is the missing check.

Reproduce: run `test_reference_endpoint_serves_checksum_mismatched_artifact`.

### MEDIUM — cached LIVE inputs lack explicit cache/computation provenance (ADV-22)

`/api/live-example-inputs` intentionally caches a deterministic WNTR-derived
payload but returns no machine-readable cache status, computed-at time, input
hash, or data-mode/provenance envelope. Its endpoint name and code comments
are not an API-level provenance contract.

Reproduce: run `test_live_example_cache_response_has_no_explicit_cache_or_provenance_label`.

## Expected fail-closed controls observed

Malformed unknown-node/negative requests were rejected; injected timeout,
incomplete, and unstable simulator outcomes became `ABSTAINED`; direct and
rejected-plan approval returned 409; new evidence staled verification across a
restart; OOD/calibration/disagreement gates passed their existing focused
tests; response constraints passed; direct invalid FSM transitions failed; and
body-size/invalid-length requests were rejected. These are focused software
observations only.

## Recommended remediation (not implemented)

1. Make `/workflow` require the already-computed `analysis.planning_allowed`
   unconditionally; remove the demo-fallback override and do not return a
   verified/approval-capable workflow state from suppressed analysis.
2. Enforce finite numeric values at every external numerical contract using
   explicit `math.isfinite` validation; reject before audit/persistence.
3. Define an evidence identity and freshness policy: idempotency key or
   sensor/node/observed-time digest, accepted age window, and an explicit
   override/audit mechanism for historical imports.
4. Stamp verification with canonical SHA-256 digests for the full plan and
   actual network-state bytes/hash; recompute and compare both at approval.
   Invalidate active verifications on any network replacement or model/
   calibration provenance change.
5. Make approval a one-way, transactional transition: require status
   `APPROVAL`, record an immutable approval idempotency key, and reject any
   second approval once `CLOSED`.
6. Verify reference artifact semantic checksum and ledger/event chain on every
   serve (or verify once at startup and fail readiness). Add a live input
   provenance envelope containing frozen input hashes, computed-at timestamp,
   cache status, and `data_mode: LIVE`.

## Proposed regression tests

- A non-finite numeric corpus (`NaN`, `±Infinity`, overflow, nested fields)
  must return 422 and leave all SQLite tables untouched.
- Duplicate/stale observations must be rejected or recorded as non-evidence;
  the posterior and sampling budget must not increase.
- `/workflow`, `/plans/generate`, queued analysis, and UI flow must all reject
  a planning-suppressed state equivalently.
- Altering plan JSON, network bytes/state hash, calibration/model hash, or
  evidence after verification must make approval return 409.
- Parallel approval/verify/sample tests with controllable barriers must prove
  exactly one approval event and no stale approval.
- Tampered reference artifacts and cache payload/provenance mismatches must
  fail closed at the endpoint.

## Limitations

This was not a penetration test of the host, EPANET numerical validation
campaign, live utility validation, or field-safety proof. The direct plan
mutation probe models persistence/internal corruption because no public
plan-update API exists. The network replacement route is directly exposed only
when an injected verifier is configured; its root cause (identity not bound in
approval) remains relevant to local artifact mutation. Race cases are
explicitly INCONCLUSIVE rather than assumed safe.

## Final verdict

Safety invariants were violated. A verification/authority bypass was found:
planning-suppressed analysis can reach a `HUMAN_APPROVAL` response through the
workflow API, though this study did not demonstrate its completion through the
separate approval endpoint. Fail-open behavior was found for `+Infinity`,
duplicate/stale evidence, verification-content/network binding, repeated
approval, and reference artifact serving. The highest-severity unresolved
issue is the planning-suppression bypass combined with verification state
exposure. Next action: block release promotion, review the authority state
machine, and implement the first four remediation items with deterministic
concurrency tests before any broader safety claim.
