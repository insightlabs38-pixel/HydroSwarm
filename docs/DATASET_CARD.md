# HydroCore-v4 Synthetic Dataset Card

## Scope and current corpus

The current shipped HydroCore-v4 bundle was trained from the governed
`cycle-b2-joint-v4` synthetic corpus, not primarily from the historical
learning-v1 corpus. It contains deterministic WNTR/EPANET-generated
contamination, normal, and sensor-fault scenarios represented as versioned
feature tensors (`hydroswarm-features-v2`) plus JSONL metadata and checksum
manifests. The corpus fingerprint is
`32b0528f569d27a0e51e6285d9da696c794c720d8ffc5c6d702042532d67f93c`.

| Population | Role | Committed examples |
|---|---|---:|
| `train` | optimization only | 9,000 |
| `validation` | selection/validation evidence | 1,000 |
| `calibration` | split-conformal fitting only | 1,000 |
| `development_holdout` | repeated development characterization, never optimization | 1,750 |
| `ood-*` development populations | governed OOD/authority experiments | 400 each (six populations) |
| locked final test | one-time final evaluation only | intentionally not enumerated or inspected here |

The split policy is [evaluation_policy_v3.json](../configs/evaluation_policy_v3.json).
The v4 locked final evaluation remains unopened; no result in this repository
uses it.

## Topologies and generation

Training/validation/calibration include governed synthetic network families
such as `golden-reference`, `branched-loop`, and `loop-grid`; topology hashes,
node IDs, edge IDs, hydraulic-state hashes, and signature-library identity are
recorded per example. The current calibration artifact declares exactly those
validated topology hashes. The existing unseen-topology population is a
development-only safety experiment, not a training topology or transfer claim.

Scenarios are generated under governed WNTR/EPANET hydraulics with source,
timing, duration, strength, demand regime, tank state, roughness, and network
state controls. Sensor observations can include nominal noise, missing values,
quantization, signed drift, frozen/carry-forward readings, communication
outage/delay, and unit mismatch. The generator records masks and provenance
instead of silently treating missing or frozen readings as healthy evidence.

## Split isolation and provenance

Seeds, seed families, scenario IDs, split labels, checksums, and network
identities are persisted. Split integrity prohibits duplicate scenario IDs and
seed-family overlap within the governed corpus. Training, validation,
calibration, development-holdout, and locked-final roles are intentionally
separate; development populations are not authorized for fitting calibration
or weights. New characterization fixtures must remain outside all of them.

All data in this card are synthetic. The project does not contain utility
telemetry, consumer records, or a public utility-network dataset. WNTR and
EPANET model assumptions, not field measurements, generate concentration and
hydraulic outcomes.

## Data quality, imbalance, and limitations

- Contamination/source labels and response targets are simulator-derived;
  they can encode modeling assumptions and historical label defects. The
  repository retains label-audit and leakage reports rather than hiding them.
- Event/cause and OOD categories are not uniformly represented. In particular,
  the v4 OOD classifier received no real training-split gradient and is not a
  runtime-promoted head.
- Sensor fault, missingness, and topology shifts are governed synthetic
  mechanisms, not a claim that their real-world prevalence or failure mode has
  been measured.
- The robustness-scale study reports a limited 168-row development replay;
  it does not establish utility-scale performance, field accuracy, or all
  missingness/sensor-coverage strata.

## Storage, licenses, and legacy data

Committed tables/manifests use JSON/JSONL, arrays use safetensors/NPZ, and
network definitions use EPANET `.inp` files. Code is Apache-2.0; consult each
third-party tool/dataset source before redistributing any external material.

`learning-v1` remains a clearly historical 1,320-incident corpus (800 train,
160 validation, 160 calibration, 200 test) retained for research provenance.
It is not the primary description of the shipped HydroCore-v4 system.
