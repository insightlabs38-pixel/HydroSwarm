# Problem and product boundary

When a water utility sees an abnormal quality observation, responders must reason from
sparse, delayed, and sometimes faulty evidence. Flow direction changes with tanks, pumps,
valves, and demand; a plausible upstream source at one time can become implausible later.
Meanwhile, an intervention that reduces contaminant exposure can create unacceptable
pressure or service consequences.

HydroSwarm is local decision support for this interval between detection and field action.
It estimates a source region, calibrated when the frozen calibration artifact is applicable,
and a deterministic sampling policy may recommend the next sample expected to reduce
uncertainty most or abstain when the evidence does not support one. It compares typed
response plans against both a no-response baseline and authoritative WNTR simulation. It
exposes evidence, disagreement, and failure states to an operator instead of issuing
autonomous commands.

The primary users are utility incident commanders, hydraulic engineers, water-quality
staff, and field-sampling coordinators. HydroSwarm does not identify chemistry, certify
potability, replace laboratory analysis, connect to SCADA, or control infrastructure.
Every operational action remains subject to utility procedures and human approval.

## Research basis

This section establishes the scale and shape of the problem using official sources. It does
not claim EPA endorsement, utility validation, or field-tested impact for HydroSwarm itself —
only that the problem HydroSwarm targets is real and already treated as connected by the
agency that regulates it.

- Public water systems serve most of the country: EPA states that "the public drinking water
  systems regulated by EPA and delegated states and tribes provide drinking water to 90
  percent of Americans," across more than 148,000 public water systems (EPA, [Information
  about Public Water Systems](https://www.epa.gov/dwreginfo/information-about-public-water-systems)).
  A contamination-response gap at this scale is not a niche concern.
- EPA's Water Quality Surveillance and Response System (SRS) framework explicitly treats
  monitoring, sampling and analysis, and contamination response as connected components of one
  system rather than isolated functions: online/physical monitoring and customer/public-health
  surveillance feed detection, sampling and analysis verify it, and a documented
  water-contamination-response ("consequence management") stage follows (EPA, [Fact Sheet about
  Water Quality Surveillance and Response
  System](https://www.epa.gov/waterresilience/fact-sheet-about-water-quality-surveillance-and-response-system)).
- HydroSwarm targets one narrow segment inside that connected chain: the interval **after** an
  abnormal quality observation has been flagged and **before** a utility commits to additional
  field sampling or a hydraulic response. It does not perform detection (online monitoring),
  laboratory confirmation, or consequence execution; it is decision support for the sampling and
  response-comparison step in between.

See [References](REFERENCES.md) for the full source list, including EPA's WNTR, TEVA-SPOT, and
CANARY tools and the prior-art table this project builds on.

## Measured vs. plausible impact

Claims about HydroSwarm's impact fall into three tiers. Only the first is backed by the current
locked evaluation; the rest are explicitly labeled as not yet measured.

**Measured in the current research prototype** (source: [Scientific
evidence](SCIENTIFIC_EVIDENCE.md), [Authority and safety](AUTHORITY_AND_SAFETY.md)):

- source-localization metrics (Top-1/Top-3/MRR) across nominal, stress, and novel-topology
  populations;
- conformal coverage where calibration is applicable;
- actionable/planning rates per population;
- exact hydraulic constraint rejection (modeled constraint-violating or infeasible plans are rejected before human review);
- 0 of 15 hard safety counters violated across the complete 125-incident M11.6 locked evaluation;
- novel-topology calibration inapplicable, 0% actionable — the correct fail-closed behavior under
  genuine topology shift, not a hidden failure;
- real governed LIVE sampling abstention: against this repository's own bundled Live Example
  scenario, the deterministic active-sampling policy correctly abstains on the first analysis, and
  the `v0.2.1` frontend now shows that governed stop truthfully instead of forcing a sample or plan
  through it.

**Plausible operational benefit — not yet field-measured**:

- prioritizing confirmatory sampling toward the location expected to reduce uncertainty most;
- rejecting hydraulically infeasible response options before they reach a human approver;
- preserving local infrastructure data (network models and incident data can stay on-premises);
- exposing uncertainty and abstention explicitly instead of forcing a confident-looking answer;
- providing an auditable evidence trail (what was known, recommended, verified, and approved, and
  when) for engineers and after-action review.

**Requires real-world validation — not claimed**:

- operator decision-quality improvement;
- time saved during an incident;
- confirmatory samples avoided;
- response-time improvement;
- exposure or service-impact reduction;
- utility deployment cost or return on investment.

## Sustainability and path to scale

### Long-term viability

- Local/offline runtime: no hosted LLM or model API dependency, and no recurring inference-API
  fee — HydroCore-v5 is a small, locally trained and locally served scientific model.
- Apache-2.0 license.
- Versioned, hash-identified model/release artifacts (see [Final system](FINAL_SYSTEM.md),
  [Reproducibility](REPRODUCIBILITY.md)).
- Both Docker (published multiarch image and from-source build) and native deployment across the
  supported platform matrix — see [Installation](INSTALLATION.md).
- Roughly 4 GiB RAM for the small/default local demonstration, currently.
- Real utility deployment still has hardware, maintenance, calibration, integration, validation,
  and staffing costs that this project has **not** measured.

### Responsible impact

- Sensitive infrastructure data (network models, incident records) can remain entirely local.
- No autonomous actuation connector exists anywhere in the system.
- Every response candidate requires a separate, explicit human-approval event.
- Fail-closed behavior: exact simulation is required before a plan is `VERIFIED`, and calibration
  or evidence inadequacy can suppress planning rather than silently proceeding.
- Limitations are stated explicitly rather than implied away — see [Limitations](LIMITATIONS.md).

### Path to scale

The intended progression, none of which has happened yet beyond the first step:

1. current synthetic research prototype (this repository);
2. retrospective utility-partner validation against real historical incidents;
3. network/utility-specific calibration;
4. operator usability testing;
5. prospective shadow-mode evaluation (running alongside, not instead of, existing procedures);
6. only then, consideration of operational integration.

A practical scaling hook that already exists: HydroSwarm imports standard EPANET `.inp` network
files (see "Import Your Own Network" in [Operator guide](USER_GUIDE.md)), so bringing a new
network in does not require a bespoke data pipeline — the same interoperability the broader
water-modeling ecosystem (WNTR, TEVA-SPOT, EPANET) already relies on. This is a practical
integration point, not a claim that HydroSwarm generalizes to every network topology or utility
context without further validation; the locked novel-topology result above is direct evidence of
where that generalization currently breaks down.
