# HydroSwarm — Devpost Submission

> **Offline, physics-first decision support for drinking-water contamination incidents.**

This is the repository copy of HydroSwarm's Reverie Hacks 2026 Devpost submission. The Devpost page uses its dedicated project, video, repository, and Built With fields; those details are retained here so this document remains self-contained.

HydroSwarm helps an operator move from an uncertain water-quality alert to a defensible next decision: localize plausible sources, determine whether more evidence is worth collecting, compare possible responses through hydraulic simulation, and keep final authority with a human.

## Inspiration

A water-quality alert can reveal that something is wrong without revealing where it began.

In a drinking-water distribution network, evidence can be sparse and delayed. Several upstream locations may explain the same observations, while changing tank levels, pumps, valves, and demand alter flow paths over time. Collecting another field sample takes time and effort, but acting too quickly creates another risk: a response that appears useful may also reduce pressure or disrupt service elsewhere.

This problem maps directly to existing utility-response practice. The U.S. EPA's [Water Quality Surveillance and Response System](https://www.epa.gov/waterresilience/fact-sheet-about-water-quality-surveillance-and-response-system) treats monitoring, sampling and analysis, and contamination response as connected parts of incident response. EPA-regulated public water systems provide drinking water to roughly [90% of Americans](https://www.epa.gov/dwreginfo/information-about-public-water-systems).

That led to the central question behind HydroSwarm:

> **How can machine learning contribute useful evidence without allowing a prediction to silently become permission to act?**

Rather than making a neural model the decision-maker, HydroSwarm separates prediction from operational authority. Learned evidence can influence localization, but deterministic controls decide whether the workflow may continue, WNTR/EPANET checks response candidates against the modeled hydraulic system, and a human retains final approval authority.

HydroSwarm is a simulation-validated research prototype. It has not been field-validated and does not replace laboratory analysis, utility procedures, public-health judgment, or infrastructure operators.

## What it does

HydroSwarm is a local operator console built around a governed incident workflow:

> **Observe → Localize → Check uncertainty → Sample or abstain → Generate responses → Verify physics → Human decision**

An operator can import a standard EPANET `.inp` network, create an incident, and add timestamped observations. HydroSwarm reconciles those observations with the hydraulic network and produces a ranked source advisory using classical physics-derived evidence together with HydroCore-v5, a locally served learned model.

The learned prediction is deliberately not treated as a verdict. HydroSwarm exposes the candidate source region, calibration applicability, out-of-distribution state, sensor health, classical/neural disagreement, and evidence sufficiency together. When the frozen calibration artifact is applicable, conformal calibration provides population-level coverage semantics; when it is not applicable, HydroSwarm says so instead of presenting an unsupported confidence claim.

If another measurement is expected to be useful, a **deterministic sampling policy** can recommend where to collect it. If no remaining measurement has enough value, the correct output can instead be to stop.

Response planning follows the same principle. Candidate actions are generated deterministically and remain proposals until they pass WNTR/EPANET simulation. A candidate that violates configured modeled pressure or service constraints is rejected rather than surfaced as verified.

Even a verified candidate still cannot execute. HydroSwarm has no infrastructure-actuation connector. Approval is a separate human event.

### Two deliberately different demo paths

The **Reference Incident** is a deterministic, checksummed replay of a frozen WNTR-backed scenario. It demonstrates the product workflow consistently: initial uncertainty, evidence collection, posterior contraction, response generation, simulator rejection, a verified alternative, and the human-approval boundary.

The **Live Example** is different. It runs the actual frozen `v0.2.1` system on the same reference inputs instead of replaying a predetermined successful workflow.

In the final LIVE case, HydroSwarm finds that no remaining sample has enough marginal value while the evidence still does not justify planning.

**So it stops rather than manufacturing a result.**

That behavior captures a core design principle of the project: **abstention is a valid system output.**

## How we built it

The key architectural decision was to give different components different levels of authority.

| Stage | Final authority |
|---|---|
| Source and event evidence | HydroCore-v5 + classical evidence — **advisory** |
| Calibration / fusion | **Calibrated advisory** when applicable |
| OOD and evidence gating | **Deterministic** |
| Sampling | **Deterministic** |
| Response generation | **Deterministic** |
| Physical verification | **WNTR / EPANET** |
| Final approval | **Human** |

The resulting authority chain is:

**ADVISORY → CALIBRATED ADVISORY → DETERMINISTIC → SIMULATOR_VERIFIED → HUMAN_APPROVED**

### HydroCore-v5

HydroCore-v5 is a locally trained graph/time model with **4,182,612 parameters**.

The final release permits exactly five learned runtime outputs:

- source node
- event presence
- event cause
- evidence sufficiency
- relative strength

The architecture also contains experimental Scout, Strategist, OOD, and consequence-related heads. Those structures were not automatically treated as product capabilities. The final system only enables learned outputs supported by the training and governance evidence; operational OOD control, sampling, and planning remain deterministic.

### Application architecture

The scientific and service layer uses Python, PyTorch, WNTR/EPANET, NetworkX, NumPy, pandas, FastAPI, Pydantic, and SQLite. The operator console uses React, TypeScript, Vite, MapLibre GL JS, and Playwright-based end-to-end testing.

HydroSwarm runs locally without a hosted model API. The model, calibration artifacts, network model, incident records, and hydraulic simulation can all remain in the operator environment.

The project also includes hash-bound release artifacts, a strict application self-test, CI/static analysis, and a published multiarchitecture Docker release. Native verification covers Linux x86_64/ARM64, Windows x86_64, and Apple Silicon macOS.

The current published release is **v0.2.1**.

## Challenges we ran into

### Making topology diversity real, not cosmetic

Early training data varied hydraulic conditions more than actual graph structure. Once HydroSwarm began mixing genuinely different networks—with different node counts, tanks, branches, and loop structures—assumptions that had been invisible on a single topology immediately broke.

One data-audit path attempted to stack per-node targets across an entire split, which only works when every graph has the same number of nodes. The training loop also hardcoded a fixed-shape collator, so a batch containing different graph sizes could not train at all.

We replaced those assumptions with topology-aware statistics, explicit padding and masking, and variable-topology collation. That changed how we thought about the model itself: supporting “multiple networks” cannot mean replaying different hydraulic regimes on one graph. The data and training pipeline has to preserve real graph structure and identity.

### Proving that a neural head actually learned before giving it authority

One of the most important discoveries came from auditing the model after training.

Several Scout heads physically existed in the architecture and even had nonzero task weights configured, but the canonical training-target pipeline had never supplied their targets. They therefore received no training gradients. The candidate-conditioned Strategist path had a related issue: its modules existed, but the required candidate-plan inputs had never been populated during the canonical training run.

That forced us to separate **architecture presence, supervision, validation, runtime enablement, and operational authority** instead of treating them as the same thing.

We added explicit supervision and gradient-coverage checks, built real Scout and Strategist training-state contracts, and ran governed follow-up evaluations. Even after specialist components became technically trainable, they still had to satisfy predeclared promotion criteria.

When they did not, we kept deterministic sampling and planning.

### Turning a research checkout into a reproducible offline product

Code that works from a repository checkout is not automatically a reliable release.

At one point, the API and CLI self-test resolved the frozen model bundle using different assumptions. They agreed in an editable checkout launched from the repository root but could diverge after a normal packaged install. That created a dangerous failure mode: the application could fall back to a degraded path while a superficially plausible self-test still looked healthy.

We replaced the duplicate path logic with one shared resolver and added release gates that require the actual frozen model bundle to load inside the packaged environment.

Portability exposed another issue: WNTR ships no Linux ARM64 EPANET binary. Instead of dropping ARM64 support, the setup path now builds the official EPANET 2.2 engine for the host architecture, and CI verifies a real water-quality simulation rather than merely checking that the application starts.

## Accomplishments that we're proud of

HydroSwarm became much more than a model notebook.

We built a complete offline operator workflow in which learned prediction, deterministic decision logic, physical verification, and human authority are explicitly separated and inspectable.

The final one-time evaluation completed all **105 locked-final incidents** and all **20 separately locked novel-topology incidents**. The frozen evaluation was opened once, with no rerun or post-lock tuning.

On the nominal locked subset (`n=15`), HydroSwarm reached **73.3% Top-1** and **86.7% Top-3** source localization.

Across all 105 locked-final cases, including stress conditions, Top-1 was **55.2%**, Top-3 was **76.2%**, and applicable conformal coverage was **88.6%**.

On the 20 novel-topology incidents, localization retained **descriptive** signal at **55% Top-1 / 70% Top-3**, but calibration was inapplicable. HydroSwarm therefore produced **0% actionable plans and 0% human-approved plans** on that population.

All **15 frozen hard authority/safety counters recorded zero violations** across the complete locked evaluation.

Those results do not prove real-world utility safety or field accuracy. They show that predictive degradation can remain visible while the software continues to enforce the authority boundaries it was designed to enforce.

We also shipped HydroSwarm as a real `v0.2.1` release with tested multiarchitecture containers, native setup paths, reproducible artifacts, and strict self-tests rather than leaving it as an unpackaged development environment.

## What we learned

The biggest lesson from HydroSwarm is that **prediction quality and decision authority are different engineering problems**.

A model can carry useful information without being reliable enough to determine the next action.

A response can look plausible and still fail a modeled hydraulic constraint.

A simulator-verified response can still require human judgment.

And sometimes the correct output of the entire system is simply to stop.

We also learned that a larger or more complex learned system is not automatically a better product. A neural component should not gain authority merely because it can produce an output. In a safety-sensitive workflow, supervision, validation, applicability, physical feasibility, and decision authority need to remain distinguishable.

The most important question became less:

> *How can we make the model more confident?*

and more:

> **What evidence is strong enough to justify the next decision?**

## What's next

HydroSwarm is currently a **simulation-validated research prototype, not a field-validated utility system**.

The next meaningful step is not adding more AI features. It is validation.

A credible path forward is:

1. retrospective evaluation with utility partners on historical network and incident data;
2. network- and utility-specific calibration;
3. operator usability studies with hydraulic and water-quality professionals;
4. broader transport, sensor, and network stress characterization;
5. prospective shadow-mode testing alongside existing utility procedures;
6. only after those stages, consideration of operational integration.

The long-term goal is not to replace utility engineers or produce a more confident model.

**It is to make the evidence behind a consequential decision easier to inspect, challenge, verify, and defend.**

## Technical evidence

- **Published v0.2.1 release:** https://github.com/insightlabs38-pixel/HydroSwarm/releases/tag/v0.2.1
- **Scientific evidence:** https://github.com/insightlabs38-pixel/HydroSwarm/blob/main/docs/SCIENTIFIC_EVIDENCE.md
- **Judging evidence map:** https://github.com/insightlabs38-pixel/HydroSwarm/blob/main/docs/JUDGING.md

## Built with

Python · PyTorch · WNTR · EPANET · NetworkX · NumPy · pandas · FastAPI · Pydantic · SQLite · React · TypeScript · Vite · MapLibre GL JS · Playwright · Vitest · pytest · Ruff · Pyright · safetensors · Docker Buildx · GitHub Actions

## Submission and evidence links

- **Demo video:** https://vimeo.com/1220385465?share=copy&fl=sv&fe=ci#t=0
- **Repository:** https://github.com/insightlabs38-pixel/HydroSwarm
- **Published v0.2.1 release:** https://github.com/insightlabs38-pixel/HydroSwarm/releases/tag/v0.2.1
- **Executive summary:** https://github.com/insightlabs38-pixel/HydroSwarm/blob/main/docs/EXECUTIVE_SUMMARY.md
- **Scientific evidence:** https://github.com/insightlabs38-pixel/HydroSwarm/blob/main/docs/SCIENTIFIC_EVIDENCE.md
- **Judging evidence map:** https://github.com/insightlabs38-pixel/HydroSwarm/blob/main/docs/JUDGING.md
