# Glossary

Terms used across the console, API, and documentation, for a reader unfamiliar with the
domain or this project's specific vocabulary.

## Experience / provenance modes

- **LIVE** -- real backend, real current incident, real WNTR verification.
- **REFERENCE INCIDENT** -- a deterministic, checksummed replay of the frozen golden
  scenario. The primary judge demo path; not live telemetry. See [Reference demo](REFERENCE_DEMO.md).
- **DEMO_FALLBACK** (`ILLUSTRATIVE DEMO`) -- hand-authored illustrative content, shown
  only when the backend is unavailable and no other mode was explicitly requested. No
  genuine model/checkpoint identity or runtime latency.
- **ERROR** (`INCIDENT UNAVAILABLE`) -- the state cannot be safely rendered; fails closed,
  no operational recommendations.
- **Replay** -- a workspace, not a provenance mode: inspects a completed LIVE or
  REFERENCE incident's real, hash-chained event ledger.

## Decision/authority terms

- **VERIFIED** -- a response plan completed a real WNTR/EPANET simulation and passed
  every configured hard constraint. Never emitted from a neural estimate alone.
- **REJECTED** -- a plan failed exact verification (a hard constraint, timeout,
  instability, or completeness check).
- **STALE verification** -- a plan's verification was CURRENT when computed, but the
  incident's evidence context has since changed; a stale verification cannot be approved.
- **APPROVED** -- a human operator recorded approval. HydroSwarm never executes an action
  automatically; approval is a recorded decision, not an infrastructure action.
- **Human approval boundary** -- the controller's dedicated pause before any plan can be
  approved. Distinct from every other pause (e.g. the sampling pause).
- **Abstention** -- the system explicitly declines to recommend a confident action when
  evidence, calibration, or OOD conditions don't justify one.

## Statistical / calibration terms

- **Conformal candidate set / coverage** -- a calibrated set of candidate source nodes
  sized so that, on average over held-out data, the true source falls inside the set at
  least as often as the configured target rate (e.g. 90%). This is a marginal guarantee
  over a distribution, not a per-incident probability.
- **OOD (out-of-distribution)** -- evidence that the current incident falls outside the
  conditions the calibration artifact was validated on (e.g. an unrecognized network
  topology). `CAUTION` / `OUTSIDE_VALIDATED_RANGE` suppress confident planning.
- **Disagreement (classical/neural)** -- a Jensen-Shannon divergence between the
  classical physics-derived source estimate and HydroCore's estimate; high disagreement
  reduces trust in the fused result.
- **ECE (expected calibration error)** -- how far a model's stated confidence is from its
  actual accuracy, averaged over confidence bins. Lower is better-calibrated.

## System / architecture terms

- **HydroCore-v4** -- the frozen, shipped-by-default neural architecture. See
  [Final system](FINAL_SYSTEM.md) for its exact identity.
- **Runtime-enabled output** -- a trained model head whose predictions the API/console
  actually serve to an operator. A trained head that is *not* runtime-enabled must never
  be presented as an operational prediction -- see [Final system](FINAL_SYSTEM.md#what-is-and-is-not-runtime-enabled).
- **Golden scenario** -- the one frozen, checksummed WNTR network/incident fixture used
  for regression measurement and to generate the REFERENCE INCIDENT. Regenerable with
  `python scripts/run_golden.py`; not a benchmark claim on its own.
- **Signature (classical)** -- a precomputed, checksummed hydraulic-simulation response
  (concentration at each sensor) for a candidate source location, used by the classical
  physics-based localization path.
- **Locked final evaluation** -- a held-out test set reserved for one final,
  authorization-gated evaluation pass. Not opened during ordinary development; its status
  is recorded in `reports/results/v4/architecture-freeze.json`.
