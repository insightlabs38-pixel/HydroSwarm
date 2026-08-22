# HydroSwarm v0.2.1 — Final Frozen Demo Narration Script

**Final video runtime:** ~3:22.7
**Tone:** calm, technical, evidence-driven
**Music:** none

## 0:00–0:07 — Physical scale

A water-quality alert reveals a problem.

Across a municipal network, it rarely reveals where it began.

## 0:07–0:29 — Problem animation

Sparse sensors can leave several upstream sources plausible.

Another sample costs time, while a reasonable-looking response can reduce pressure or disrupt service elsewhere.

So the question is not only what a model predicts.

It is what evidence is strong enough to justify the next decision.

A proposed response must survive physics, then stop at a human boundary.

## 0:29–0:35 — HydroSwarm reveal

HydroSwarm is offline, physics-first decision support built around that distinction.

## 0:35–0:51 — Source localization

This checksummed Reference Incident exposes the complete workflow.

From the first observation, HydroCore-v5 contributes learned evidence to rank plausible sources, while uncertainty remains visible instead of collapsing into one answer.

## 0:51–1:05 — Sampling

When evidence is insufficient, HydroSwarm does not simply ask the model again.

A deterministic sampling policy identifies a useful next measurement, and the workflow pauses for that evidence.

Only then does the workflow advance.

## 1:05–1:15 — Posterior contraction

When that sample arrives, the posterior changes sharply.

The earlier prediction was evidence, not a permanent conclusion.

## 1:15–1:40 — Response verification

Next, response candidates are generated deterministically and checked by exact WNTR and EPANET simulation.

The first candidate is rejected because it violates the configured pressure constraint.

A second satisfies the modeled constraints and is marked verified.

**Proposal is not permission.**

Physics can veto a plausible-looking response.

## 1:40–1:57 — Human approval

And verified still does not mean executed.

HydroSwarm has no infrastructure-actuation authority.

The consequential step remains a separate human approval boundary, with verification context and provenance available for inspection.

## 1:57–2:05 — Provenance

The Technical Dock preserves the network identity and verification context behind that decision.

## 2:05–2:21 — Actual LIVE runtime

That was the reproducible Reference Incident.

This is the frozen v0.2.1 runtime on the same inputs.

Here, no useful sample remains while planning is unjustified.

**HydroSwarm stops.**

Rather than forcing a result.

## 2:21–2:35 — Technical authority

HydroCore-v5 is advisory.

Calibration and deterministic out-of-distribution checks bound its authority.

Sampling and planning stay deterministic.

WNTR and EPANET verify physics.

Final approval stays human.

## 2:35–3:03 — Validation

The evaluation was locked before opening.

Nominal Top-1 was 73.3 percent; Top-3, 86.7.

Across 105 locked-final cases, Top-1 was 55.2 percent, with 88.6 percent conformal coverage.

On novel topology, localization signal remains, but calibration is inapplicable and actionable rate is zero.

None of fifteen hard safety counters were violated.

No rerun.

No post-lock tuning.

## 3:03–3:11 — Feasibility

HydroSwarm runs locally and accepts standard EPANET network models, keeping infrastructure data in the operator environment.

## 3:11–3:22.7 — Closing

Every path ends in a governed decision.

This is simulation-validated, not field-validated.

The goal is not greater confidence.

**It is a more defensible decision.**
