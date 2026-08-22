# Judging evidence map

This page is a reviewer router, not a marketing scorecard. Every strong claim below points to a frozen artifact or current technical document, and every capability is paired with its authority boundary.

## Five-minute review

1. [Executive summary](EXECUTIVE_SUMMARY.md) — the problem, the system, the final result, and the limitations, in one self-contained document.
2. [Final 3:23 demo video](https://vimeo.com/1220385465?share=copy&fl=sv&fe=ci#t=0) — end-to-end walkthrough of the workflow and authority boundary.
3. [Scientific evidence](SCIENTIFIC_EVIDENCE.md) — one-time locked matrix, including weak stress slices.
4. This page — where each judging claim is proven.

### The strongest concise story

HydroSwarm is unusual not because “AI predicts contamination,” but because the repository makes the **entire decision authority chain testable and auditable**: learned Sentinel evidence is advisory; calibration applicability is explicit; deterministic OOD/Scout/planning controls remain authoritative; WNTR/EPANET must verify a candidate; a human must separately approve; and the final held-out evaluation was opened exactly once with all 15 hard safety counters at zero.

## Evidence by rubric criterion

This maps to the six judging categories plus the bonus, each with its strongest current evidence and the claim boundary that keeps that evidence honest.

### 1. Real-World Problem & Impact

| Evidence | Claim boundary |
|---|---|
| [Problem and product boundary](PROBLEM.md) — research basis, target interval, measured-vs-plausible impact | EPA sources establish context for the problem space, not endorsement of HydroSwarm or utility validation |
| [Executive summary §1](EXECUTIVE_SUMMARY.md) — why sparse/delayed evidence and hydraulic consequence make this hard | Motivates the design; not a substitute for field evidence |
| [Measured vs. plausible impact](PROBLEM.md#measured-vs-plausible-impact) | Only the "measured" tier is backed by M11.6; everything else is explicitly labeled plausible or unvalidated |

### 2. Technical Execution — Software Development

| Evidence | Claim boundary |
|---|---|
| Published release [`v0.2.1`](https://github.com/insightlabs38-pixel/HydroSwarm/releases/tag/v0.2.1) via `docker-compose.release.yml` | A working demo path; not evidence of field performance |
| [Architecture](ARCHITECTURE.md), [Final system](FINAL_SYSTEM.md) | Optional model heads are not equated with trained/runtime authority |
| Exact WNTR/EPANET verification (all of the above + [Authority and safety](AUTHORITY_AND_SAFETY.md)) | "Verified" means configured simulator checks passed, not real-world safety |
| Tests/CI: `HydroSwarm CI`, `HydroSwarm Native Cross-Platform Verification`, `HydroSwarm Release` workflows; `hydroswarm self-test --strict` | Green CI is software correctness evidence, not scientific-accuracy evidence |
| Release engineering: versioned tags, multiarch published image, runtime-ZIP asset, [Reproducibility](REPRODUCIBILITY.md) | Reproducible artifacts/hashes, not a promise the locked test can be rerun |
| Docker (`linux/amd64`, `linux/arm64`) and native (Linux x86_64/ARM64, Windows x86_64, macOS Apple Silicon) support | Native macOS Intel/x86_64 is not supported (no upstream `torch>=2.5` wheel) |

### 3. Innovation & Originality

| Evidence | Claim boundary |
|---|---|
| “[What is actually novel?](../README.md#what-is-actually-novel)” in the README | Individually, simulation/localization/graph-learning/conformal prediction/active sampling are not claimed as novel |
| [References and prior art](REFERENCES.md) | Prior-art table makes narrow claims about cited systems, not that they lack every HydroSwarm feature |
| [Authority and safety](AUTHORITY_AND_SAFETY.md) — calibration applicability, deterministic OOD/sampling/planning, physical-verification separation | The novelty claim is about the governed integration chain, not any single component |

### 4. User Experience & Design

| Evidence | Claim boundary |
|---|---|
| [Reference Incident](REFERENCE_DEMO.md) — deterministic, checksummed, staged workflow walkthrough | Demonstrates workflow/UX, not V5 benchmark performance; post-completion Replay is intentionally unavailable (see [Reference demo](REFERENCE_DEMO.md#replay-availability)) |
| [Operator guide](USER_GUIDE.md) — Decision Inspector, Technical Dock, left-rail workflow | Describes the intended operator flow, not a usability study |
| Explicit text status labels (`PROPOSED`, `VERIFYING`, `REJECTED`, `VERIFIED`, `APPROVED`) rather than color-only state | Labels are visible in the documented API/UI vocabulary; this is not an accessibility audit |
| Playwright end-to-end screenshots/tests across the workflow | Screenshots predate the current documentation rebase and are UI/workflow evidence, not model-identity evidence |

### 5. Sustainability & Scalability

| Evidence | Claim boundary |
|---|---|
| [Sustainability and path to scale](PROBLEM.md#sustainability-and-path-to-scale) | Describes current design properties and an intended validation path, not a completed deployment |
| Local/offline runtime, no hosted-LLM dependency, Apache-2.0, versioned release artifacts | Real deployment still has unmeasured hardware, maintenance, calibration, integration, and staffing costs |
| Standard EPANET `.inp` interoperability (Import Your Own Network) | Interoperability is a practical scaling hook, not proof of cross-utility generalization |
| Published multiarch image + native install across the supported platform matrix | Supported platforms only; native macOS Intel/x86_64 is out of scope |

### 6. Presentation & Communication

| Evidence | Claim boundary |
|---|---|
| [Final demo video (3:23)](https://vimeo.com/1220385465?share=copy&fl=sv&fe=ci#t=0) | A recorded walkthrough, not a substitute for running the release yourself |
| [Executive summary](EXECUTIVE_SUMMARY.md) | Self-contained explanation; deeper evidence lives in the linked dossiers |
| [README judge quick path](../README.md#judge-quick-path) | A navigation aid, not a duplicate of the underlying documents |
| [Scientific evidence](SCIENTIFIC_EVIDENCE.md) router | One dossier rather than numbers scattered/duplicated across pages |

### Bonus: Exceptionality

| Evidence | Claim boundary |
|---|---|
| Locally trained/served learned model with no hosted inference dependency | Local execution, not a claim of superior accuracy |
| Exact physical (WNTR/EPANET) verification gating every actionable plan | Verification is conditional on the supplied network model, not a real-world safety certificate |
| One-time locked M11.6 evaluation, opened exactly once, no rerun, no post-lock tuning | Governance rigor about the test process, not a claim about generalization beyond the tested populations |
| Governed real LIVE sampling abstention (the `v0.2.1` fix) shown truthfully rather than forced to a fabricated result | One real, reproducible governed-stop example; not evidence that abstention is rare in general |
| 0 of 15 hard safety counters violated across 125 locked incidents | See [Safety claim rules](CLAIMS_AND_EVIDENCE.md#safety-claim-rules) for exactly what this does and does not support |
| Reproducible, promoted-after-test multiarch release (`v0.2.1`) | Reproducibility of the software artifact, not of the one-time locked scientific result |

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
| Published release | `v0.2.1` (`ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`) |

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

## Deep technical review

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

For a UI/workflow demonstration, use the **REFERENCE INCIDENT** after launching the current release. It gives a deterministic walkthrough from initial uncertainty through sample collection, unsafe-plan rejection, exact verification, and human approval. Post-completion Replay of the Reference Incident is intentionally unavailable, because the reference artifact does not contain individual event-ledger records to populate it — see [Reference demo](REFERENCE_DEMO.md#replay-availability).

Do not present the reference replay as final V5 performance evidence. The benchmark evidence is M11.6.

For V5 container behavior, either pull the published release (`docker compose -f docker-compose.release.yml up`, `ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`) or build the current checkout (`docker compose build && docker compose up`). Both resolve to the same frozen V5 identity.

## What makes the documentation itself reviewable

- one canonical final-system page;
- one detailed scientific evidence dossier rather than duplicated numbers everywhere;
- authority labels used consistently;
- exact hashes where identity matters;
- direct links to generated artifacts;
- current versus historical research clearly separated;
- negative/stress results surfaced alongside favorable ones;
- no claim that descriptive novel-topology metrics are calibrated;
- no claim that the Reference Incident produces a populated post-completion replay when the underlying artifact does not support one.

## Claims judges should **not** make on the project's behalf

- “field validated”;
- “safe for utilities”;
- “AI autonomously chooses/executes the response”;
- “55% unseen-topology accuracy is calibrated”;
- “90% conformal coverage means 90% confidence on this incident”;
- “WNTR verification proves real-world safety”;
- “EPA endorses or validated HydroSwarm”;
- “the Reference Incident replay shows a full audit ledger.”

Those are outside the evidence.
