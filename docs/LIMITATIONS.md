# Limitations and failure cases

HydroSwarm is research decision support. It has not been validated for live utility use,
regulatory decisions, public-health advisories, or autonomous control.

## Scientific limits

- WNTR/EPANET predictions inherit network-model errors, demand uncertainty, simplified
  mixing assumptions, sensor timing errors, and missing operational controls.
- The system localizes a modeled source and does not identify contaminant chemistry,
  toxicity, pathogen viability, or laboratory confirmation requirements.
- Sparse or topologically redundant sensors can make sources non-identifiable. A calibrated
  region can remain broad after sampling.
- Synthetic training cannot establish field performance. Evaluation on held-out simulated
  networks measures generalization within the tested generator, not a real utility.
- Population exposure is a proxy unless node-level demographic and consumption inputs are
  supplied and governed correctly.

## Operational failure cases

- Unit mismatch, frozen sensors, drift, clock jitter, communication loss, or an unmodeled
  valve/pump state can produce false alarms or misleading likelihoods.
- Signature caches can be stale after topology/configuration changes; checksum failure
  forces an exact fallback and may increase latency.
- Exact simulation can time out or become numerically unstable. Such a plan is rejected,
  never promoted to verified.
- OOD/calibration invalidation intentionally suppresses confident planning. Classical-safe
  mode is less expressive and may return only sampling or no-action advice.
- The local API is not designed as an internet-exposed, authenticated multi-tenant service.

## Safety boundary

No output authorizes flushing, isolation, pump/valve changes, public notification, or any
other field action. Qualified staff must check current hydraulics, water-quality protocols,
regulations, consequences, and utility procedures. Human approval is mandatory and the
software contains no execution connector.

## Honest claims

Repository benchmark files report only runs produced by the checked-in evaluation code.
No synthetic metric is presented as field accuracy, and no `VERIFIED` label is emitted
without a complete WNTR result. Missing checkpoints, calibration, or frontend assets are
reported explicitly rather than replaced with embedded demo claims.

Native Windows is correct but not performance-equivalent to Linux/Docker for exact
simulator-heavy workloads (Windows has no `fork()` syscall, so every real WNTR/EPANET
call pays a fresh interpreter startup there) -- see
[Performance: native Windows vs. Linux/Docker](INSTALLATION.md#performance-native-windows-vs-linuxdocker).
No claim of native Windows performance parity with Linux is made.
