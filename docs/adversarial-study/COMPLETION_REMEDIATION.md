# Completion-study remediation: ADV-27 through ADV-29

This record remediates findings preserved in `COMPLETION.md` and
`completion-results.json`. Those historical records have not been altered.
This is local software validation of a research prototype; it does not claim
field safety, utility validation, or external validation.

## Scope and tested commit

- Remediation branch: `fix/adversarial-completion-findings`
- Base: `test/adversarial-completion` at `6937a00`
- Runtime: Python 3.12, FastAPI TestClient, SQLite temporary databases,
  Hypothesis, and local WNTR/EPANET only.
- Preserved without change: HydroCore weights and feature-schema hash,
  calibration/OOD/disagreement/conformal/fusion thresholds, WNTR/EPANET
  physics, frozen reference measurements, held-out evaluation results, and
  product claims.

## Remediated findings

| Finding | Severity | Root cause | Exact remediation | Invariant restored | Regression result |
|---|---|---|---|---|---|
| ADV-27 | HIGH | `POST /api/incidents` accepted its initial tuple directly; the canonical identity guard existed only in `POST /samples`. | Added `_evidence_identity` and `_validate_initial_evidence` in `api.app`. Creation rejects an identical identity with `422 DUPLICATE_INITIAL_EVIDENCE` and changed contents with `422 CONFLICTING_INITIAL_EVIDENCE`, before runtime, audit, or SQLite persistence. `/samples` now calls the same identity helper and retains idempotent exact retransmission / `409` changed-duplicate behavior. | Initial evidence cannot silently multiply evidence history, observation count, posterior influence, or planning authority. | PASS |
| ADV-28 | HIGH | `frozen_flag` was persisted in `SensorObservation` but discarded while constructing `SensorSeries`; frozen evidence therefore shared an analysis identity and healthy authority behavior. | Added provenance-only `SensorSeries.frozen`, included it in the canonical evidence payload, preserved `FROZEN` in the incident-view contract, and maps frozen API telemetry to the existing trained `sensor_health` channel at `min(quality, 0.25)`, consistent with the existing corpus convention. An explicit `ALL_SENSORS_FROZEN` deterministic suppression prevents planning when every latest reading is frozen. | Frozen evidence has distinct provenance, reduced deterministic trust, and cannot retain full planning authority. Evidence changes reanalyze and stale old verification. | PASS |
| ADV-29 | MEDIUM | The `RequestValidationError` handler removed raw `input` but returned Pydantic `ctx.error` objects (such as `ValueError`), which JSON encoding turned into 500. | Added an explicit allow-list sanitizer for error fields and recursively JSON-safe context. It omits raw input, exception objects, nonfinite values, and unsupported values; datetimes become ISO timestamps. | Every request-validation response is controlled JSON `422` before handler-side persistence/audit mutation, without exception reprs or local paths. | PASS |

## Changed files and adjacent coverage

- `src/hydroswarm/api/app.py`: validation sanitizer, canonical evidence
  identity, frozen-series propagation, effective frozen health, and `FROZEN`
  operator state.
- `src/hydroswarm/preprocessing/builder.py`: provenance-only
  `SensorSeries.frozen`; no added HydroCore input column.
- `src/hydroswarm/inference/pipeline.py`: frozen evidence hashing and
  `ALL_SENSORS_FROZEN` deterministic planning suppression.
- `src/hydroswarm/api/state.py`, `frontend/src/types.ts`, and
  `frontend/src/api/incident.ts`: explicit `FROZEN` sensor-health contract.
- `src/hydroswarm/agents/sentinel.py`: deterministic fallback does not count
  frozen evidence as usable.
- `tests/adversarial/test_adversarial_completion.py`: strict reproducers
  converted to required regressions plus adjacent tests.

Adjacent tests cover identical/conflicting initial duplicates by concentration,
pressure, quality, and flags; distinct sensor/node identities; atomic zero
runtime/incident/observation/audit state; idempotent and concurrent duplicate
samples; single/multiple/all frozen sensors; frozen with low-quality, drift,
and missing metadata; freeze/unfreeze reanalysis; stale verification; restart;
evidence hashes; cross-field and nested validation errors; numeric poison;
wrong primitive types; extra fields; and invalid approval literals.

## Extreme-finite-outlier follow-up

**OBSERVATION.** The investigated `1,000,000 mg/L` trajectory remains finite,
with `OOD=NORMAL` and planning allowed in the frozen deterministic fixture.
The codebase has no governed runtime sensor/concentration applicability range:
the only located `expected_range=(0, 100)` is a default for the separate
`SensorTelemetry` diagnostic utility, not an API contract, model metadata,
calibration applicability bound, or runtime OOD policy. No arbitrary maximum
was added and no ADV-30 finding is assigned.

## Second-pass adversarial result

The targeted second pass exercised duplicate initial variants, concurrent
duplicate samples, freeze/unfreeze after analysis and verification, multiple
frozen sensors, nested/cross-field validation errors, and restart behavior.
No new invariant violation was reproduced: **new findings: NONE**.

## Validation executed

- Completion adversarial tests: `41 passed` (including the 315 existing
  bounded Hypothesis examples).
- Existing adversarial study tests: `24 passed`.
- Focused API, persistence, incident-view, inference, preprocessing, sensor
  health, and control-label tests: `126 passed`.
- Complete Python suite: `1090 passed` in 10m42s.
- `hydroswarm self-test --strict`: passed.
- Frontend: `162 passed`; production `tsc -b && vite build`: passed.

The frontend test run emitted React `act(...)` warnings, but no test failure;
they are unrelated to this response-contract expansion.
