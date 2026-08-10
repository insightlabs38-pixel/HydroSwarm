# Technical brief

A one-page technical summary for a reviewer who wants real depth without reading every
doc in the repository. Each claim links to its authoritative source.

## What runs by default

`hydroswarm.api.app:app` serves HydroCore-v4 (4.18M parameters, `small` variant, frozen)
by default -- see [Final system](FINAL_SYSTEM.md) for the exact identity, hashes, and
which of its 21 trained output heads are actually runtime-enabled versus trained-but-
unpromoted versus excluded. `hydroswarm self-test` validates the same bundle the API
serves through one shared resolver (`hydroswarm.runtime.paths.resolve_v4_bundle_dir`),
so the two can never silently disagree about which weights are actually live.

## Decision pipeline

Classical, physics-derived source signatures and HydroCore-v4's neural estimate fuse
under a dynamic, disagreement-aware trust coefficient (sensor health, missingness,
residuals, entropy, five-component OOD evidence). The fused belief is calibrated with
split-conformal prediction (alpha=0.1, held-out coverage 91.4%). Deterministic Scout logic
ranks the next sample by expected information gain; deterministic Strategist logic
proposes typed response plans under a hard simulation budget. Every candidate plan is
re-simulated with real WNTR/EPANET hydraulics before it can be marked `VERIFIED` -- no
neural or classical estimate alone promotes a plan. See
[Full architecture](ARCHITECTURE.md) and [Model card](MODEL_CARD.md).

## Controller and event model

An 18-state deterministic controller drives every incident: detection, data quality,
hydraulic state estimation, source localization, evidence sufficiency check, sample
selection/receipt, plan generation, constraint checking, exact verification, plan
comparison, human approval, and completion. Every transition is a real event in an
append-only, hash-chained ledger -- replay reproduces the exact chain, including the
`previous_hash`/`event_hash` linkage, not a re-narrated summary. See
[Reference demo](REFERENCE_DEMO.md) for a concrete, generated walkthrough of this chain.

## Evaluation

`reports/results/v4/phase13-metrics-and-baselines.md` is the authoritative HydroCore-v4
evaluation: source top-1 72.1-73.3%, top-3/coverage@3 86.8-87.6%, MRR 0.811-0.817,
event-presence F1 0.895, event-cause macro F1 0.698 (on 3 supported classes). The report
surfaces two evaluation caveats up front rather than hiding them (a trivially-perfect
`sensor_fault` metric with zero true negatives in the evaluated population, and a
near-chance `ood_category` head that never received a real training gradient under the
current governed data-split design). The locked final evaluation has not been opened. See
[Evaluation protocol](EVALUATION.md) for the full methodology and required ablations.

## Safety/authority boundaries that are never weakened

No `VERIFIED` without a completed WNTR run. No `APPROVED` without a separate human event.
No execution of any kind -- HydroSwarm is decision support, not infrastructure control.
Stale verifications block approval. Invalid calibration or an unrecognized topology
suppresses planning (`CAUTION`), not just a warning label. See
[Final system: authority boundaries](FINAL_SYSTEM.md#authority-boundaries-never-weakened)
and [Limitations](LIMITATIONS.md).

## Runtime and deployment

FastAPI + SQLite + a bounded local worker, offline by default (no internet calls, no
remote map tiles, no hosted model API). Native setup scripts or a multiarch
(`linux/amd64`, `linux/arm64`) Docker image; both resolve the frozen bundle through the
same path-resolution logic. See [Installation](INSTALLATION.md).

## Reproducibility

`python scripts/run_golden.py` regenerates the frozen golden scenario end to end (real
WNTR simulation, not a cached artifact). `python -m pytest` runs the full test suite
(unit, scientific/real-simulator, integration, end-to-end). CI runs on Windows and
Ubuntu. See the top-level [README](../README.md#research-evaluation-and-historical-development)
for exact commands.
