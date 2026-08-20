# Operator guide

## 0. Start

On first launch choose **Run Reference Incident** for a deterministic workflow replay, or proceed to a LIVE incident with a local network. The reference replay demonstrates product state/authority; final V5 performance evidence is in [Scientific evidence](SCIENTIFIC_EVIDENCE.md).

## 1. Prepare a network

Import a local EPANET `.inp`. Review validation errors, units, disconnected elements, node/link counts, tanks, reservoirs, controls, and the content hash. A failed hydraulic run blocks analysis.

## 2. Open an incident

Create an incident against a validated network and add timestamped measurements with units/quality state. Preserve raw timestamps and units so preprocessing remains auditable.

## 3. Interpret localization

Read the posterior together with candidate region, calibration applicability, OOD state, classical/neural disagreement, sensor health, and evidence sufficiency. A broad set or abstention is a valid outcome.

HydroCore-v5 learned outputs are advisory. See [Final system output governance](FINAL_SYSTEM.md#what-is-and-is-not-runtime-enabled).

## 4. Collect evidence

When the deterministic Scout recommends a sample, review node, expected information value, collection-delay/access assumptions, alternatives, and remaining budget. Record the actual arriving sample; reanalysis creates a new evidence/posterior revision.

A learned Scout head does not select the authoritative sample in the final system.

## 5. Compare response plans

Compare generated candidates and modeled consequences. `VERIFIED` means a candidate completed WNTR/EPANET verification and passed the configured modeled constraints. It does not mean the plan is approved or proven safe in the real utility.

Learned plan/consequence heads are not operational authorities; authoritative candidate generation is deterministic and exact verification remains physical-simulator based.

## 6. Approve or reject

Only a human can record approval of an eligible verified plan. HydroSwarm does not actuate infrastructure. Rejections, stale verification, and simulator failures remain part of the evidence history.

## 7. Replay and export

Use the timeline/audit record to inspect evidence and decisions in order. Replay should preserve recorded state/hash provenance; it is not a chance to recompute a more favorable result.

## Status meanings

- `PROPOSED`: candidate generated; not verified.
- `VERIFYING`: exact simulation pending.
- `REJECTED`: verification/constraint/completeness failure.
- `VERIFIED`: modeled exact verification completed and configured constraints passed.
- `APPROVED`: separate human approval recorded; no automatic execution.
- abstention / planning suppression: evidence, calibration, OOD, or safety conditions do not justify continuing.

## Always keep the authority chain in view

**ADVISORY → CALIBRATED ADVISORY → DETERMINISTIC → SIMULATOR_VERIFIED → HUMAN_APPROVED**

See [Authority and safety](AUTHORITY_AND_SAFETY.md) and [Limitations](LIMITATIONS.md).
