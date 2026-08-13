# Adversarial study completion: ADV-05 and ADV-25

## Objective and scope

This local-only completion study attempted to falsify the remaining
evidence-integrity and structured API-boundary claims. It adds test-only
fixtures and documentation; it does not change HydroCore, calibration/OOD
thresholds, fusion weights, WNTR/EPANET physics, reference measurements, or
product claims. Historical audit records remain unchanged.

- Branch: `test/adversarial-completion`
- Base remediation commit: `ca938a2`
- Completion test commit: `454610c`
- Runtime: Python 3.12, pytest/Hypothesis, FastAPI TestClient, SQLite temp
  databases, local WNTR/EPANET only.

## ADV-05 methodology

The deterministic trajectory matrix uses a real `HybridInferencePipeline`,
the real feature builder, classical localizer, calibration artifact path,
fusion, OOD detector, and local `HydraulicSimulator`. A deliberately simple,
fixed model merely makes neural outputs reproducible; it is not used to
replace the pipeline. The post-verification case uses the real API route,
real reanalysis, and an injected verifier solely to avoid an unrelated
expensive plan simulation.

| Scenario | Before → injected evidence | Actual result | Classification |
|---|---|---|---|
| Contradictory healthy pair | `J1=0.78` → add `J3=1.0` | disagreement `0.472→0.755`; `planning_allowed true→false`; control changes to `INSPECT_SENSORS`; OOD remains `NORMAL` | PASS |
| Extreme finite outlier | coherent `J1=0.78` → `J1=1,000,000` | finite normalized posterior, but top source becomes 1.0, OOD remains `NORMAL`, planning remains allowed | OBSERVATION |
| Repeated biased trajectory | one `J1=0.78` → repeated same sensor trajectory | posterior/authority did not grow in this fixture | OBSERVATION |
| Near tie | `J1=0.4`, `J2=0.65` | top beliefs `0.708/0.292`; conformal region has two nodes; planning is suppressed | PASS |
| Conflict after VERIFIED | planning-eligible API incident → verified plan → add `J3=1.0` | reanalysis suppresses planning; verification becomes `STALE`; approval returns 409 | PASS |
| Quality/drift/frozen | high-quality vs low-quality vs drift/frozen flags | quality `1.0→0.01` suppresses planning; drift does not deterministically suppress in this fixture; `frozen_flag` is lost before `SensorSeries` and leaves planning allowed | FAIL: ADV-28 |

All tested fused beliefs were finite, in `[0,1]`, summed to 1, and had
candidate sets drawn from their belief keys. The extreme-outlier result is
not called a threshold failure: this prototype does not declare a robust
outlier model, and its limitations already warn that frozen/drifting sensors
can yield misleading likelihoods. It is nevertheless recorded because the
existing gates did not suppress it.

## ADV-25 methodology

Structured Hypothesis strategies were bounded and shrinkable, not arbitrary
byte fuzzing. The final completed run executed **315 generated examples**:

- 180 domain observation combinations: finite/nonfinite values, concentration
  and quality boundaries, and missingness combinations.
- 50 malformed sample requests: unusual IDs/Unicode, wrong primitive types,
  nested unexpected keys, numeric poison, and timestamp shapes except the
  separately minimized known failure below.
- 45 incident-creation combinations: valid/missing network IDs, known/unknown
  nodes, sample-budget bounds, and duplicate initial observations.
- 40 short suppressed-state sequences of 1–8 API operations.

Additional deterministic properties cover plan serialization/content hashes,
reference checksum mutation, and LIVE cache provenance identity across
MISS/HIT responses.

Passing properties established that accepted domain observations are finite
and bounded; malformed sample/incident requests that returned 4xx made no
state mutation; suppressed sequences produced neither `CLOSED` incidents nor
approval receipts; plan serialization retained canonical content identity;
reference mutation failed checksum validation; and cache status changed
without changing frozen input/network/scenario identities.

## New minimized findings

### ADV-27 — HIGH: duplicate initial evidence is accepted

Invariant: an exact retransmission must not enter durable evidence twice.

Minimal reproducer: create an incident with two byte-identical observations
sharing `sensor_id`, `node_id`, and `observed_at`. Actual result: `201`, two
durable observations, and analysis reports `observation_count: 2`. Expected:
422/conflict or idempotent single evidence record before incident persistence.

Suspected cause: the identity guard is implemented only in `POST /samples`;
`POST /incidents` persists the initial observation tuple without applying the
same identity check. The strict expected-fail reproducer is
`test_adv27_duplicate_initial_observation_is_rejected_atomically`.

### ADV-28 — HIGH: frozen sensor metadata does not reach authority gating

Invariant: degraded frozen evidence must not retain the same operational
authority as healthy evidence.

Minimal reproducer: a real API/pipeline incident with one `J1=0.78`
observation is planning eligible; changing only `frozen_flag` to `true`
leaves it planning eligible. `SensorSeries` has no frozen field and the API
maps only `drift_flag` into its `drift` vector. Expected: frozen evidence is
represented in evidence/trust/OOD/authority logic or conservatively
suppressed. The strict expected-fail reproducer is
`test_adv28_frozen_sensor_cannot_retain_planning_authority`.

### ADV-29 — MEDIUM: timestamp validation returns 500

Invariant: malformed API input must produce a controlled 4xx response and no
mutation.

Minimal reproducer: submit a sample with `received_at` one second before
`observed_at`. Actual result: 500. The custom request-validation handler
removes `input` but leaves `ctx.error`, a `ValueError` that JSON serialization
cannot encode. Expected: 422 with no persisted mutation. The strict
expected-fail reproducer is
`test_adv29_received_before_observed_is_controlled_422_without_mutation`.

No production remediation was applied for any of these findings.

## Results and limitations

- **ADV-05: FAIL.** Contradiction, freshness, finite-state, low-quality, and
  near-tie gates behaved as required, but ADV-28 shows that a frozen sensor
  can retain planning authority.
- **ADV-25: FAIL.** Most bounded structured properties passed, but ADV-27
  and ADV-29 violate creation-time evidence integrity and controlled malformed
  request handling.

This does not test field sensor behavior, exhaustive adversarial numerical
conditions, internet exposure, or utility operation. Passing components do
not establish field safety, utility validation, or external validation.

## Executed validation

- Completion and existing adversarial tests: **36 passed, 3 expected
  failures**.
- Relevant scientific/API/state/persistence suite: **94 passed, 3 expected
  failures**.

The expected failures are deliberate, strict minimized reproductions of
ADV-27 through ADV-29; an unexpected pass would fail the suite and require
review.

