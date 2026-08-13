# HydroSwarm Robustness and Scale Characterization

## Executive summary

This frozen, 168-run offline replay characterizes the current HydroCore-v4
release bundle against seven predeclared governed validation/development-
holdout populations. The locked final evaluation was not opened. The explicit
authority invariant held in every row: no row with `planning_allowed=false`
emitted `GENERATE_PLANS`.

The weakest measured localization regime was the existing unseen-topology
population (27.8% top-1, MRR 0.523). It was simultaneously fail-closed:
calibration was inapplicable and every row suppressed planning. This is an
observation about a small, synthetic, development-only population, not a
cross-topology capability claim.

## Frozen identities

| Field | Value |
|---|---|
| Tested system commit | `e45f72cf730d3f12c13dbcb9403c64f185510173` |
| Model SHA-256 | `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7` |
| Calibration SHA-256 | `829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa` |
| Feature schema SHA-256 | `7ec97775e5f01f87ae62669146a7eb70958f99b1162a356614eb87220e9ddd09` |
| Normalization SHA-256 | `e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114` |

Protocol: [ROBUSTNESS_SCALE_PROTOCOL.md](ROBUSTNESS_SCALE_PROTOCOL.md).
Machine-readable rows and aggregation: [results.json](../../reports/evaluation/robustness-scale/results.json),
[results.csv](../../reports/evaluation/robustness-scale/results.csv), and
[summary.json](../../reports/evaluation/robustness-scale/summary.json).

**Study status correction.** This is **Study 1: Offline frozen tensor
characterization**. It did not invoke the LIVE API, dynamic live fusion, live
OOD calculation, active sampling, plan generation, or WNTR verification. Its
corrected raw-artifact terminology is documented in
[CORRECTION.md](../../reports/evaluation/robustness-scale/CORRECTION.md).
The separate **Study 2: LIVE API end-to-end characterization** is documented
in [LIVE_ROBUSTNESS_EVALUATION.md](LIVE_ROBUSTNESS_EVALUATION.md); its results
must not be back-projected into this offline replay.

## Tested populations and perturbation matrix

Each condition contains a deterministic 24-row content-addressed sample from
the committed, checksum-verified `cycle-b2-joint-v4` corpus: nominal
validation plus existing severe-missingness, frozen/drifting-sensor, extreme-
demand, roughness-mismatch, tank-state-shift, and unseen-topology populations.
They are development-only fixtures; none is training, calibration, or locked
final-test data.

The committed populations do not independently label 10/25/50/75% missingness
or isolated/multiple/majority sensor degradation. Those cells are
**INCONCLUSIVE**, not inferred from the severe condition. No new favourable
synthetic distribution was generated to fill them.

## Localization results

| Condition | Top-1 | Top-3 | MRR | Mean candidate size | Mean entropy (bits) |
|---|---:|---:|---:|---:|---:|
| Nominal validation | 76.2% | 90.5% | 0.831 | 2.96 | 1.21 |
| Severe missingness | 71.4% | 85.7% | 0.810 | 0.00* | 1.70 |
| Frozen/drifting sensor | 66.7% | 72.2% | 0.747 | 0.00* | 1.47 |
| Extreme demand | 55.6% | 77.8% | 0.700 | 0.00* | 1.42 |
| Roughness mismatch | 66.7% | 77.8% | 0.761 | 0.00* | 1.36 |
| Tank-state shift | 78.9% | 100.0% | 0.895 | 0.00* | 1.15 |
| Unseen topology | 27.8% | 77.8% | 0.523 | 0.00* | 1.88 |

`*` A zero candidate set is the intentional representation of calibration
inapplicability for the non-nominal OOD conditions, not a zero-width
uncertainty claim. The offline posterior used for localization is a clearly
labeled tensor-replay proxy; it does not replace the live dynamic-trust
pipeline calculation.

## Calibration, OOD, and authority

**PASS — authority invariant.** All 168 rows satisfied
`planning_allowed=false => control_action != GENERATE_PLANS`.

**OBSERVATION — nominal caution.** Eighteen of 24 nominal rows were also
suppressed, primarily by the existing JS disagreement/candidate-breadth
guards. This is measured conservative behavior; it is not changed here.

**PASS — governed-policy replay.** In the offline governed-policy replay,
all 144 pre-labeled OOD rows were calibration-inapplicable and
planning-suppressed. This does not claim that the LIVE runtime independently
detected those OOD states; Study 2 evaluates that question.

## Sampling behavior

Rows record a request when the current deterministic uncertainty controller
selects it. Learned Scout values are unavailable because that head is not
runtime-promoted; `expected_information_gain` is therefore null rather than
invented. The runner does not perform live grab sampling against these stored
tensors, so sampling latency and realized information gain are **INCONCLUSIVE**.

## Performance and scalability

This is a CPU, offline tensor-replay characterization on Linux/aarch64,
Python 3.12.13, 16 logical CPUs, and 64,163 MB RAM. After one warm-up and five
repetitions per row, median model-inference latency across conditions ranged
from 10.7 to 16.0 ms. The point-in-time `process_rss_mb` recorded per row is
process-level rather than model-only or peak memory.

**INCONCLUSIVE — full workflow scale.** The stored fixtures do not preserve
live raw telemetry/hydraulic state needed to invoke the exact current WNTR
pipeline without generating a new corpus. Accordingly, analysis/sampling/plan/
verification fields that would require a live WNTR run are null and exact
simulator calls are zero. This report does not claim utility-scale deployment
or exact-verifier latency from tensor inference timing.

## Failure regimes and unexpected observations

- **OBSERVATION:** unseen topology was the poorest localization regime while
  remaining fully suppressed.
- **OBSERVATION:** severe missingness increased posterior entropy relative to
  nominal validation, while its predeclared OOD policy suppressed authority.
- **OBSERVATION:** tank-state shift had high localization in this limited
  sample but was still treated as out of validated range; no favourable
  accuracy result was allowed to restore authority.
- **FAIL:** none. No `ROB-XX` finding was created because no explicit
  invariant was violated.

## Known limitations

This uses only synthetic, governed, development-side data; it is not a field
performance study. It cannot establish exact 10/25/50/75% missingness trends,
sensor-count health strata, live WNTR workflow cost, or scalable utility
network behavior. Full detail, including unavailable fields represented as
null, is in the result artifacts.

## Locked-test status

**PASS:** `locked_test_opened` was `false` before and after the campaign. The
runner rejects any protocol that names `test`, `locked_final_test`, or
`locked_topology_test`.

## Reproduction

```bash
python scripts/run_robustness_scale_characterization.py --verify-only
python scripts/run_robustness_scale_characterization.py
python -m pytest tests/evaluation/test_robustness_scale_characterization.py -q
```
