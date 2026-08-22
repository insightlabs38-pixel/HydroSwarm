# Judging evidence map

This page is a reviewer router, not a marketing scorecard. Every strong claim below points to a frozen artifact or current technical document, and every capability is paired with its authority boundary.

## Five-minute review

1. **Watch the [final 3:23 demo video](https://vimeo.com/1220385465?share=copy&fl=sv&fe=ci#t=0).**
2. **Read [The strongest concise story](#the-strongest-concise-story) and [Final locked evidence a judge can verify](#final-locked-evidence-a-judge-can-verify) on this page.**
3. **Use the rubric tables below only for the category you want to verify.**

### The strongest concise story

**HydroSwarm is designed to produce a more defensible response decision, not simply a more confident prediction.**

HydroSwarm is unusual not because “AI predicts contamination,” but because the repository makes the **entire decision authority chain testable and auditable**: learned Sentinel evidence is advisory; calibration applicability is explicit; deterministic OOD/Scout/planning controls remain authoritative; WNTR/EPANET must verify a candidate; a human must separately approve; and the final held-out evaluation was opened exactly once with all 15 hard safety counters at zero.

## 10–15 minute review

1. Read the [Executive summary](EXECUTIVE_SUMMARY.md) for the problem, governed workflow, final result, and limitations.
2. Read [Scientific evidence](SCIENTIFIC_EVIDENCE.md) for the one-time locked matrix, stress slices, gates, provenance, and claim boundaries.
3. Use the rubric tables on this page to jump directly to the evidence for any category you want to inspect further.

## Evidence by rubric criterion

This maps the six submission-review categories plus the bonus to explicit subcriteria, without self-scoring. Each row points to the strongest current evidence and the claim boundary that keeps that evidence honest.

### 1. Real-World Problem & Impact

| Rubric subcriterion | Evidence | Claim boundary |
|---|---|---|
| **Problem significance** | [Problem and product boundary](PROBLEM.md) — research basis, target interval, and why sparse/delayed evidence plus hydraulic consequence make contamination response difficult | EPA sources establish context for the problem space, not endorsement of HydroSwarm or utility validation |
| **Evidence & understanding** | [Executive summary §1](EXECUTIVE_SUMMARY.md) + [Measured vs. plausible impact](PROBLEM.md#measured-vs-plausible-impact) — separates measured synthetic/runtime evidence from plausible field impact | The “measured” tier is backed by M11.6 and current runtime/release evidence; the LIVE governed abstention is a `v0.2.1` runtime observation, not an M11.6 result |
| **Solution impact** | Governed workflow combines localization, evidence collection, response comparison, physical verification, and explicit human approval rather than stopping at prediction | Demonstrates a decision-support design and synthetic measured behavior; real utility impact remains unvalidated |

### 2. Technical Execution — Software Development

| Rubric subcriterion | Evidence | Claim boundary |
|---|---|---|
| **Working functionality** | Published release [`v0.2.1`](https://github.com/insightlabs38-pixel/HydroSwarm/releases/tag/v0.2.1), `docker-compose.release.yml`, strict self-test, Reference Incident, Live Example, and exact WNTR/EPANET verification | A working/reproducible software path; not evidence of field performance |
| **Code quality & architecture** | [Architecture](ARCHITECTURE.md), [Final system](FINAL_SYSTEM.md), [Authority and safety](AUTHORITY_AND_SAFETY.md), typed runtime boundaries, deterministic authority separation, tests/CI | Optional model heads are not equated with trained/runtime authority; green CI is software-correctness evidence, not scientific-accuracy evidence |
| **Appropriate tools & version control** | GitHub history/tags, CI workflows, versioned releases, multiarch image, runtime-ZIP asset, [Reproducibility](REPRODUCIBILITY.md), Python/PyTorch/WNTR/FastAPI/React stack | Reproducible artifacts and hashes; not a promise that the one-time locked M11.6 evaluation can be rerun or that every platform is supported |

### 3. Innovation & Originality

| Rubric subcriterion | Evidence | Claim boundary |
|---|---|---|
| **Originality of the idea** | “[What is actually novel?](../README.md#what-is-actually-novel)” — governed integration of learned evidence, calibration applicability, deterministic controls, exact physical verification, and human approval | Hydraulic simulation, contamination localization, graph learning, conformal prediction, and active sampling are not individually claimed as novel |
| **Creativity in approach** | [Authority and safety](AUTHORITY_AND_SAFETY.md) — prediction and permission-to-act are deliberately separated across learned, deterministic, simulator, and human authorities | The novelty claim is about the integration and governance chain, not any single algorithm |
| **Potential to inspire** | [References and prior art](REFERENCES.md) + auditable claim/evidence structure show how uncertain ML can be embedded inside a fail-closed engineering workflow | This is a design contribution and research-software example, not evidence of broad adoption or field validation |

### 4. User Experience & Design

| Rubric subcriterion | Evidence | Claim boundary |
|---|---|---|
| **Ease of use** | [Operator guide](USER_GUIDE.md) — primary left-rail workflow, persistent Decision Inspector, Technical Dock, explicit states, and first-launch Reference/Live paths | Designed for the intended professional context; not a formal operator-usability study |
| **Aesthetic appeal** | Professional decision-support information hierarchy — primary workflow separated from Decision Inspector/Technical Dock; evidence, provenance, verification, and authority intentionally remain visible | Technical density is intentional engineering design; no claim of validated human-factors optimization |
| **Accessibility** | Explicit text status labels (`PROPOSED`, `VERIFYING`, `REJECTED`, `VERIFIED`, `APPROVED`), provenance banners, burned-caption demo, and documented workflow states rather than color-only meaning | These are concrete accessibility-oriented choices; this is not a formal accessibility audit |

### 5. Sustainability & Scalability

| Rubric subcriterion | Evidence | Claim boundary |
|---|---|---|
| **Long-term viability** | [Sustainability and path to scale](PROBLEM.md#sustainability-and-path-to-scale), local/offline runtime, Apache-2.0, versioned release artifacts | Describes current design properties and an intended validation path, not a completed deployment or maintenance program |
| **Environmental, social, or economic sustainability** | Offline/local execution avoids a hosted-LLM runtime dependency; explicit human authority and fail-closed behavior target responsible operational use | Real deployments still have unmeasured hardware, staffing, integration, maintenance, and energy costs |
| **Scalability** | Standard EPANET `.inp` interoperability, published multiarch image, native install matrix, reproducible V5 bundle | Interoperability and packaging are scaling hooks, not proof of cross-utility generalization; native macOS Intel/x86_64 is unsupported |

### 6. Presentation & Communication

| Review subcriterion | Evidence | Claim boundary |
|---|---|---|
| **Narrative clarity** | [Executive summary](EXECUTIVE_SUMMARY.md) + [README judge quick path](../README.md#judge-quick-path) — problem first, then governed workflow, final evidence, and limitations | Concise routing does not replace the underlying technical/scientific dossiers |
| **Demo quality** | [Final demo video (3:23)](https://vimeo.com/1220385465?share=copy&fl=sv&fe=ci#t=0) — burned captions, explicit distinction between Reference replay and Live `v0.2.1` computation, an actual governed abstention, then the locked scientific evidence | A recorded walkthrough and current runtime observation, not a substitute for running the release or benchmark evidence |
| **Documentation & evidence traceability** | [Scientific evidence](SCIENTIFIC_EVIDENCE.md), [Claims and evidence](CLAIMS_AND_EVIDENCE.md), [Final system](FINAL_SYSTEM.md) — one canonical detailed scientific dossier, with headline summaries elsewhere linking back to it | Repeated headline metrics are summaries; immutable/generated artifacts remain the authority when prose conflicts |

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

For a UI/workflow demonstration, use the **REFERENCE INCIDENT** after launching the current release. It gives a deterministic walkthrough from initial uncertainty through sample collection, modeled constraint-violating plan rejection, exact verification, and human approval. Its post-completion Replay limitation is documented once in [Reference demo](REFERENCE_DEMO.md#replay-availability).

Do not present the reference replay as final V5 performance evidence. The benchmark evidence is M11.6. For current runtime behavior, **Live Example** runs frozen `v0.2.1` on the bundled scenario inputs and may validly stop at a governed abstention.

For V5 container behavior, either pull the published release (`docker compose -f docker-compose.release.yml up`, `ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`) or build the current checkout (`docker compose build && docker compose up`). Both resolve to the same frozen V5 identity.

## What makes the documentation itself reviewable

- one canonical final-system page;
- one canonical detailed scientific dossier, with headline summaries elsewhere linking back to it;
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
- “WNTR verification proves real-world safety”;
- “EPA endorses or validated HydroSwarm.”

Those are outside the evidence.
