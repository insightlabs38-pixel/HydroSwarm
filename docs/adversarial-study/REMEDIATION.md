# Adversarial remediation

This document records remediation work performed after the immutable
adversarial study. It does not alter the historical study or claim field,
utility, or external validation.

## Scope

The remediation preserves scientific model behavior, calibration/OOD
thresholds, WNTR/EPANET physics, and frozen reference measurements. Changes
are confined to API authority, persistence, evidence integrity, and
provenance boundaries.

## Remediated findings

| Finding | Root cause | Remediation | Invariant restored |
|---|---|---|---|
| ADV-09 planning bypass | `/workflow` promoted `DEMO_FALLBACK` suppression to sufficient evidence | Shared `_require_planning_authority` gate for generation, verification, and workflow; fallback plan construction removed | Suppressed evidence cannot reach plan/verification/approval |
| ADV-03 nonfinite evidence | Float fields accepted infinity; validation error rendering could fail | `allow_inf_nan=False` on domain/API models and sanitized 422 validation responses | Invalid numerical evidence never persists |
| ADV-04 duplicate/stale samples | No sample identity or incident-time guard | Sensor/node/observed-at identity; idempotent exact retransmission; conflict on changed duplicate; reject predating samples | Evidence cannot silently multiply or predate incident |
| ADV-13 plan mutation | Verification bound plan ID but not content | SHA-256 canonical plan binding, compared on recording and approval | Approval applies only to verified plan bytes |
| ADV-15 network mutation | Network IDs were replaceable and context used only ID | Immutable IDs, current record/INP-byte SHA-256 identity, verification/approval binding | Verification applies only to authoritative network content |
| ADV-17 replayed approval | Approval route did not require authority state; persistence replaced receipts | APPROVAL/pending guard, per-incident lock, `INSERT OR IGNORE` receipt persistence | APPROVAL → CLOSED is one-way and non-replayable |
| ADV-21 reference integrity | Endpoint parsed JSON without semantic validation | Verify semantic artifact hash, contiguous milestones/final hash, and companion manifest | Reference replay fails closed on corruption |
| ADV-22 LIVE provenance | Cached frozen inputs lacked response provenance | LIVE/frozen source/cache/timestamp/input/network/scenario hashes in response and frontend mapping | Cached inputs remain visibly distinct from newly computed incident results |

## Change and regression map

| Finding | Changed files | Regression test | Result |
|---|---|---|---|
| ADV-09 | `src/hydroswarm/api/app.py` | `test_planning_suppression_is_not_bypassable_through_workflow_endpoint` | PASS |
| ADV-03 | `src/hydroswarm/domain/schemas.py`, `src/hydroswarm/api/state.py`, `src/hydroswarm/api/app.py` | `test_all_externally_reachable_observation_numbers_reject_nonfinite` | PASS |
| ADV-04 | `src/hydroswarm/api/app.py` | `test_duplicate_and_stale_samples_are_non_evidence` | PASS |
| ADV-13 / ADV-15 | `src/hydroswarm/api/app.py`, `src/hydroswarm/domain/schemas.py`, `src/hydroswarm/storage/scenario_store.py` | `test_approval_requires_current_exact_plan_network_and_context_bindings` | PASS |
| ADV-17 | `src/hydroswarm/api/app.py`, `src/hydroswarm/api/state.py`, `src/hydroswarm/storage/scenario_store.py` | `test_approval_is_one_way_and_duplicate_receipts_are_not_persisted`; race regressions | PASS |
| ADV-21 | `src/hydroswarm/evaluation/reference_demo.py`, `src/hydroswarm/api/app.py` | `test_reference_endpoint_rejects_checksum_mismatched_artifact` | PASS |
| ADV-22 | `src/hydroswarm/evaluation/live_example.py`, `frontend/src/api/liveExampleFlow.ts` | `test_live_example_inputs_is_cached_across_requests_within_one_app_instance` | PASS |

## Executed regression evidence

- `tests/adversarial/test_adversarial_validation.py`: 24 passed, including
  property-based nonfinite inputs and deterministic barrier-driven races.
- Focused authority, scientific, reference, network, and exposure set: 68
  passed.
- Frontend: 162 passed; production TypeScript/Vite build passed.
- `hydroswarm.cli self-test --strict`: passed, including frozen bundle and
  WNTR smoke validation.
- Complete Python baseline: 1,050 passed.

## Validation limits

These are adversarial software regressions for a research prototype. Passing
them does not establish field safety, utility validation, chemical identity,
or external operational authorization.

The frozen reference artifact retains milestone ranges and the final ledger
hash, but not the full event-chain payload. This remediation therefore
validates the semantic artifact checksum, contiguous range coverage, final
event identity, and companion manifest; it cannot independently recompute
every historical event link from that artifact alone. Adding the complete
event chain to a future frozen artifact would permit that additional check.
