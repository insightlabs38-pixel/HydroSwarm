# Claims and evidence ledger

This ledger gives reviewers a compact mapping from a claim to its evidence and the wording boundary that keeps the claim scientifically honest.

| Claim | Evidence | Safe wording | Do not claim |
|---|---|---|---|
| V5 is the final model | [M11.2 freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json), [runtime manifest](../models/hydrocore-v5-release/runtime_manifest.json) | “HydroCore-v5 M10 frozen release is the final serving identity” | that V4 is still current |
| Exact model identity is frozen | same + [M10.5 completion](../reports/evaluation/hydrocore-v5/m10/m10-5-completion/m10-5-completion-closure.json) | quote seed/hash/parameter count exactly | approximate/alternate checkpoint identity |
| Only Sentinel is valid trained task | finalist freeze / runtime manifest | “final trained task family is Sentinel” | “Scout/Strategist/OOD are learned authorities” |
| Five learned outputs are runtime-enabled | finalist freeze / V5 loader | list the exact five | include `next_step` |
| OOD/Scout/planning remain deterministic | finalist freeze, [safety counters](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json) | name `OODDetector`, `rank_sample_locations`, `generate_response_plans` | imply neural heads select authoritative samples/plans |
| V5 does not silently fall back to V4 | V5 loader, safety counters | “invalid V5 assets fail closed without V4 fallback” | that every degraded state remains full hybrid |
| Exact simulation is required for verification | architecture/runtime + safety counters | “configured WNTR/EPANET verification is required for `VERIFIED`” | “WNTR proves real-world safety” |
| Human approval is required | finalist freeze / safety counters | “separate human approval event required” | “approval executes infrastructure” |
| M11.6 final lock passed | [gate](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-gate.json), [closure](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-closure.json) | “locked-final and locked-topology gates passed” | “all predictive metrics passed thresholds” |
| Final population was complete | gate / post-run governance | “125/125 complete: 105 final + 20 topology” | extrapolate beyond tested population |
| Lock was opened once | [opened record](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-opened-record.json), governance | “one authorized opening; no rerun/post-lock tuning” | claim the lock remains unopened |
| All hard safety counters were zero | [safety counters](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json) | “0 violations across all 15 frozen counters in 125 locked incidents” | “guaranteed safe” |
| Aggregate applicable final coverage passed | [metrics](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-metrics.json), gate | “88.6% aggregate locked-final coverage; frozen floor 85%” | imply every condition slice passed 85% |
| Nominal predictive result | metrics | “73.3% Top-1 on nominal locked-final, n=15” | present 73.3% as all-condition final accuracy |
| Aggregate stress predictive result | metrics | “55.2% Top-1 across all locked-final, n=105” | hide stress degradation |
| Novel-topology signal exists | metrics | “55% Top-1 / 70% Top-3 descriptive on n=20 novel-topology cases” | call it calibrated topology-transfer performance |
| Novel topology failed closed | gate / metrics | “0% calibrated/actionable/approved; fail-closed gate passed” | imply 60% raw set inclusion is calibrated coverage |
| Data are synthetic | dataset artifacts | “all model/evaluation evidence is WNTR/EPANET-generated synthetic evidence” | field-validation claim |
| Larger M model was not promoted | [M9 final summary](../reports/evaluation/hydrocore-v5/m9-final/m9-final-summary.md) | “larger M did not meet predeclared meaningful capacity-gain threshold” | “small is universally superior” |
| M10.4 development trajectories passed | [M10.4 closure](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-closure.json) | “full development trajectory gate passed” | treat M10.4 as final held-out evidence |
| Reference incident demonstrates workflow | [Reference demo](REFERENCE_DEMO.md) | “checksummed WNTR-backed replay of a frozen workflow” | use it as final V5 benchmark evidence |
| Live Example demonstrates current `v0.2.1` governed runtime behavior | [Reference demo](REFERENCE_DEMO.md#reference-incident-vs-live-example), current runtime | “Current frozen runtime on bundled scenario inputs may abstain rather than reproduce the Reference replay” | “Reference Incident is a live V5 run” or “Live Example is benchmark evidence” |
| Current source app serves V5 | source runtime / finalist freeze | “current source app defaults to V5; the published `v0.2.1` release-compose image also serves V5” | say a pre-`v0.2.0` release-compose tag is V5 |

## Numeric claim rules

When quoting a result:

1. name the population;
2. include `n`;
3. state calibration applicability when reporting coverage;
4. state whether the metric was gating or descriptive when relevant;
5. avoid converting marginal conformal coverage into per-incident probability;
6. avoid comparing populations as if they were matched if their generation/protocol differs.

## Safety claim rules

Preferred:

> “All 15 preregistered hard authority/safety counters were zero across the complete 125-incident locked evaluation.”

Not supported:

> “HydroSwarm is proven safe for water utilities.”

Preferred:

> “Novel-topology incidents retained descriptive localization signal while the runtime withheld calibrated/actionable authority.”

Not supported:

> “HydroSwarm generalizes safely to unseen utility networks.”

## Historical claim rules

Historical V3/V4/M9/M10 artifacts can be cited when discussing research evolution. Their metrics, model identities, and “locked unopened” statements must be labelled with their historical generation/milestone. They do not override the current M11.6 terminal state.

## Source hierarchy

For current facts, prefer in this order:

1. frozen/generated identity/governance artifacts;
2. generated final metrics/gate/safety artifacts;
3. runtime code for actual serving/authority wiring;
4. this current documentation;
5. historical prose only for historical claims.

This prevents documentation drift from becoming scientific truth.
