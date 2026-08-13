# HydroSwarm Robustness and Scale Characterization Protocol

## Objective

Characterize the frozen, shipping HydroCore-v4 system as evidence quality,
sensor availability, topology familiarity, operating conditions, and network
scale change. This is an evidence-generation study, not a selection,
training, calibration, or remediation exercise. Unfavourable outcomes are
preserved as results.

## Frozen system and split boundary

- Base/system commit: `e45f72cf730d3f12c13dbcb9403c64f185510173`.
- Model SHA-256: `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`.
- Calibration SHA-256: `829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa`.
- Feature-schema SHA-256: `7ec97775e5f01f87ae62669146a7eb70958f99b1162a356614eb87220e9ddd09`.
- Normalization SHA-256: `e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114`.
- Locked final evaluation: excluded. `locked_test_opened` must be `false`
  before and after the campaign. No `test`, `locked_final_test`, or
  `locked_topology_test` directory is accepted as an input.

## Populations and deterministic matrix

The runner reads the governed, checksummed `cycle-b2-joint-v4` validation and
development-holdout tensor populations only. It never writes a corpus, trains,
fits calibration, or changes a runtime artifact. Rows are deterministically
stratified with seed `20260813` and a fixed cap of 24 rows per population.

| Dimension | Predeclared conditions |
|---|---|
| Baseline / availability | `validation-baseline`; `ood-SEVERE_MISSINGNESS` (existing governed severe-missingness condition) |
| Sensor health | `ood-FROZEN_DRIFTING_SENSOR`; no new health semantics are introduced |
| Operating shift | `ood-EXTREME_DEMAND`, `ood-ROUGHNESS_MISMATCH`, `ood-TANK_STATE_SHIFT` |
| Topology familiarity | validation/in-distribution and `ood-UNSEEN_TOPOLOGY` |
| Evidence conflict | rows with JS disagreement above the existing runtime threshold, reported as a derived deterministic stratum |
| Scale | the valid topology populations present in the above manifests; inference is measured with the shipping weights after one warm-up, with five repetitions per topology/configuration |

Exact 10/25/50/75% missingness, isolated/multiple/majority sensor health
subsets, and additional synthetic network sizes are known exclusions from
this frozen campaign: the current committed development-holdout tensors do
not label those levels independently, and generating them would create a new
characterization corpus. They are not silently represented by a different
condition.

## Metrics and authority checks

Each row records the identity, scenario metadata, source top-1/top-3/MRR,
true-source probability, candidate-set proxy size/coverage, entropy, JS
disagreement, deterministic OOD applicability, calibration validity,
evidence sufficiency, planning permission, control action, suppression
reasons, sampling fields when available, and timing/RSS fields. Unavailable
measurements are JSON/CSV null/empty, never fabricated as zero.

The required safety invariant is checked for every row:

```
planning_allowed == false  =>  control_action != GENERATE_PLANS
```

Unsupported/OOD conditions are expected to be `CAUTION`/suppressed by the
governed deterministic behavior table. Localization accuracy is an
observation, not a safety failure. A violation receives a `ROB-XX` identifier
and is preserved without production-code modification.

## Commands

```
python scripts/run_robustness_scale_characterization.py --verify-only
python scripts/run_robustness_scale_characterization.py
python -m pytest tests/evaluation/test_robustness_scale_characterization.py -q
```

## Criteria

- **PASS:** the explicit authority invariant holds; locked-test guard holds.
- **FAIL:** either explicit invariant is violated.
- **OBSERVATION:** measured change in localization, uncertainty, sampling, or
  cost without an invariant violation.
- **INCONCLUSIVE:** a measurement is unavailable or a predeclared exclusion
  prevents answering that part of the question.

Results are written under `reports/evaluation/robustness-scale/` and are
reproduced from this protocol without modifying production behavior.
