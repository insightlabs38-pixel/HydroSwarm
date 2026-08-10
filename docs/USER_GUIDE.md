# Operator guide

## 0. Start

On first launch with no incident configured, choose **Run Reference Incident** to see the
full workflow below play out against a real, checksummed replay with no setup required, or
proceed to step 1 to work with a real network. See [Installation](INSTALLATION.md) for
setup and [Final system](FINAL_SYSTEM.md) for the experience-state labels (`LIVE`,
`REFERENCE INCIDENT`, `ILLUSTRATIVE DEMO`, `INCIDENT UNAVAILABLE`) you may see in the mode
banner.

## 1. Prepare a network

Import a local EPANET `.inp` file from the Network view. Review validation errors,
warnings, units, disconnected elements, node/link counts, tanks, reservoirs, controls,
and the content hash. The system never fetches a remote file. A failed hydraulic run
blocks incident analysis.

## 2. Open an incident

Create an incident against a validated network, enter the observation time, and add
measurements with sensor, analyte/channel, value, unit, timestamp, and optional quality
flags. Keep raw timestamps and units: normalization and fault assessment are auditable.

## 3. Interpret localization

The map distinguishes posterior probability from the calibrated candidate region. Read
the runtime mode, calibration status, OOD status, classical/neural disagreement, sensor
health, and abstention reason before interpreting a rank. A broad region is a valid
outcome—not an error to hide.

## 4. Collect the recommended sample

The Evidence Value / Stop Certificate reports the recommended node, expected information
gain, expected candidate reduction, remaining sample budget, and whether the recommended
node is currently accessible. (The deterministic ranking underneath also weighs
detectability, delay, cost, and redundancy -- but only the fields above are surfaced to the
operator per recommendation.) Record an arriving sample against its request. Reanalysis
must create a new posterior revision; the timeline and Evidence Changed panel show
contraction or expansion and why.

## 5. Compare response plans

Compare no response, Plan A, Plan B, and rejected candidates side by side. Check exposure
mass/volume/population proxies, minimum pressure, service/unserved demand, action count,
verification completeness, and Pareto-dominance status. `VERIFIED` means a completed WNTR
run passed every configured hard constraint. It does not mean a plan is approved or safe in
the real utility. (A trained plan-regret head exists but is not runtime-enabled -- see
[Final system](FINAL_SYSTEM.md#what-is-and-is-not-runtime-enabled) -- so no numeric regret
score is shown; pump energy is computed by the simulator internally but is not currently
surfaced to the operator either.)

## 6. Approve or reject

Only a human operator can approve a verified plan. Record the reason, identity label, and
time. HydroSwarm does not execute the action. Rejections and simulator failures remain in
the immutable audit history.

## 7. Replay and export

Use the timeline to revisit each evidence and decision revision. Export the incident
review report and audit data for engineering review. A deterministic replay should
reproduce state transitions and hashes with the same software, data, and configuration.

## Status meanings

- `PROPOSED`: generated but not simulated.
- `VERIFYING`: exact simulation is pending.
- `REJECTED`: a hard constraint, timeout, instability, or completeness check failed.
- `VERIFIED`: simulation completed and all configured constraints passed.
- `APPROVED`: a human recorded approval; no automatic execution occurred.
- `ABSTAINED`: evidence/calibration/OOD conditions do not justify a confident action.
