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

Execution results, evidence, findings, and remediation are recorded below after
the frozen matrix has been run.
