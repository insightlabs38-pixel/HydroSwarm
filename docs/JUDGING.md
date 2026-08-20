# Judging evidence map

This page is a reviewer router, not a marketing scorecard. Every strong claim below points to a frozen artifact or current technical document, and every capability is paired with its authority boundary.

## Two-minute review

1. [Top-level README](../README.md) — what the system does and the final V5 result.
2. [Scientific evidence](SCIENTIFIC_EVIDENCE.md) — one-time locked matrix, including weak stress slices.
3. This page — where each judging claim is proven.

### The strongest concise story

HydroSwarm is unusual not because “AI predicts contamination,” but because the repository makes the **entire decision authority chain testable and auditable**: learned Sentinel evidence is advisory; calibration applicability is explicit; deterministic OOD/Scout/planning controls remain authoritative; WNTR/EPANET must verify a candidate; a human must separately approve; and the final held-out evaluation was opened exactly once with all 15 hard safety counters at zero.

## Evidence by criterion

| Criterion | Evidence | Important boundary |
|---|---|---|
| Technical depth | [Architecture](ARCHITECTURE.md), [Final system](FINAL_SYSTEM.md) | Optional model heads are not equated with trained/runtime authority |
| Scientific rigor | [Evaluation](EVALUATION.md), [Scientific evidence](SCIENTIFIC_EVIDENCE.md) | Locked test opened once; no rerun/post-lock tuning |
| Reproducibility | [Reproducibility](REPRODUCIBILITY.md) | Reproduce hashes/protocols without reopening M11.6 |
| Safety engineering | [Authority and safety](AUTHORITY_AND_SAFETY.md), [M11.6 safety counters](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json) | Zero measured counters are not a field-safety guarantee |
| Model governance | [Model card](MODEL_CARD.md), [M11.2 freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json) | Sentinel-only training; 5 learned runtime outputs |
| Data governance | [Dataset card](DATASET_CARD.md) | Synthetic data; locked seed/topology isolation documented |
| Product workflow | [Reference demo](REFERENCE_DEMO.md), [Operator guide](USER_GUIDE.md) | Reference replay demonstrates workflow, not V5 benchmark performance |
| Honest limitations | [Limitations](LIMITATIONS.md) | Stress degradation and calibration-inapplicable topology are explicit |
| Claim traceability | [Claims and evidence](CLAIMS_AND_EVIDENCE.md) | Claims are mapped to generated artifacts |

## Final V5 identity a judge can verify

| Item | Value |
|---|---|
| Model | HydroCore-v5 M10 frozen release |
| Parameters | 4,182,612 |
| Seed | 20260814 |
| Model SHA-256 | `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5` |
| Calibration SHA-256 | `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d` |
| Runtime learned outputs | 5 Sentinel outputs |
| Deterministic controls | OOD, Scout, planning |
| Physical verifier | WNTR/EPANET |
| Human approval | required |
| Autonomous actuation | none |

## Final locked evidence a judge can verify

- 105 locked-final incidents + 20 locked-topology incidents.
- 125/125 complete.
- exactly one authorized/opened evaluation.
- no locked rerun.
- no post-lock tuning.
- locked-final PASS.
- locked-topology PASS.
- all 15 hard safety counters zero.

Predictive headline, with denominators:

- nominal final (`n=15`): Top-1 73.3%, Top-3 86.7%, coverage 93.3%;
- all final stress conditions (`n=105`): Top-1 55.2%, Top-3 76.2%, coverage 88.6%;
- novel topology (`n=20`): Top-1 55%, Top-3 70%, but calibration/actionability/approval all 0%.

This is stronger evidence presentation than quoting the most favorable slice alone.

## Five-minute technical review

Read:

[TECHNICAL_BRIEF](TECHNICAL_BRIEF.md) → [FINAL_SYSTEM](FINAL_SYSTEM.md) → [ARCHITECTURE](ARCHITECTURE.md) → [AUTHORITY_AND_SAFETY](AUTHORITY_AND_SAFETY.md) → [SCIENTIFIC_EVIDENCE](SCIENTIFIC_EVIDENCE.md).

Questions that path answers:

- Which checkpoint actually serves?
- Which outputs were actually supervised?
- Which learned outputs actually run?
- Who/what has decision authority?
- What was selected before the lock?
- What did the final lock measure?
- What failed or degraded under stress?
- What does the evidence not establish?

## Demo path

For a UI/workflow demonstration, use the **REFERENCE INCIDENT** after launching the current source. It gives a deterministic walkthrough from initial uncertainty through sample collection, unsafe-plan rejection, exact verification, human approval, and replay.

Do not present the reference replay as final V5 performance evidence. The benchmark evidence is M11.6.

For current V5 container behavior, build the current checkout with `docker compose build && docker compose up`. The pinned release-compose image is an older V4 package.

## What makes the documentation itself reviewable

- one canonical final-system page;
- one detailed scientific evidence dossier rather than duplicated numbers everywhere;
- authority labels used consistently;
- exact hashes where identity matters;
- direct links to generated artifacts;
- current versus historical research clearly separated;
- negative/stress results surfaced alongside favorable ones;
- no claim that descriptive novel-topology metrics are calibrated.

## Claims judges should **not** make on the project's behalf

- “field validated”;
- “safe for utilities”;
- “AI autonomously chooses/executes the response”;
- “55% unseen-topology accuracy is calibrated”;
- “90% conformal coverage means 90% confidence on this incident”;
- “WNTR verification proves real-world safety.”

Those are outside the evidence.
