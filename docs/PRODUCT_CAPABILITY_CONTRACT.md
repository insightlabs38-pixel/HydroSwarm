# HydroSwarm product capability contract

HydroSwarm is a local-first, physics-grounded decision-support platform for
drinking-water contamination incidents. It accepts EPANET-compatible networks,
establishes a network-specific hydraulic and physics context, combines
deterministic and learned inference to localize likely contamination sources
from temporally sparse but realistic evidence, requests informative additional
samples when evidence is insufficient, generates candidate interventions only
when authority conditions are satisfied, verifies those interventions through
exact WNTR/EPANET simulation, and leaves final approval to a human operator.

Its lifecycle is:

```text
NETWORK ONBOARDING / READINESS
        ↓
INCIDENT EVIDENCE
        ↓
LOCALIZATION + UNCERTAINTY
        ↓
ACTIVE SAMPLING IF NEEDED
        ↓
RESPONSE GENERATION
        ↓
EXACT PHYSICS VERIFICATION
        ↓
HUMAN APPROVAL
```

## Independent readiness concepts

Network compatibility answers whether HydroSwarm can parse the submitted
EPANET representation and run the required hydraulic/quality simulation. It
does not assert that a learned model or calibration supports the network.

Model/calibration applicability answers whether the canonical structural
identity and deterministic operating-condition classification are supported by
the fitted model-input, calibration, and OOD reference artifacts. An unknown
or unsupported network can be compatible while still being inapplicable for
calibrated inference.

Operational readiness answers whether currently available evidence and the
applicability/uncertainty result permit planning. It is a per-incident decision
and is never inferred merely from a compatible topology.

## Evidence contract

At analysis time HydroSwarm uses every realistically available *causal*
telemetry report in its bounded history. It never requires or consumes future
observations. Continuously monitored sensors contribute their reports through
the decision time; a grab sample contributes only after it has actually been
collected and reported. One report, multiple historical reports, and
incremental reports over time are all valid inputs. The runtime must not invent
history when only one report exists and must not collapse available history to
the latest report merely for convenience.

The canonical initial feature window has a maximum of 25 ordered report steps,
matching HydroCore-v4's trained representation. When fewer than 25 causal
reports exist, all available reports are used. When more exist, the latest 25
reports whose timestamps are at or before the analysis decision time are used.
This is a bounded causal window, not a future trajectory.

## Network identity contract

For governed networks, the canonical EPANET `.inp` representation is the
identity boundary. Corpus generation, signature fitting, calibration fitting,
tests, and serving must derive structural identity from that same canonical
representation. Structural identity intentionally excludes mutable demand,
tank, and simulator state. Those state values have separately tracked runtime
identity and condition semantics. An uploaded network cannot acquire governed
identity through a display name alone.
