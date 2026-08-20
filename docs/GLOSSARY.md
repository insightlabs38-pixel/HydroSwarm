# Glossary

Terms used across the final V5 documentation, API, and operator experience.

## Model and evidence

**HydroCore-v5** — the final frozen learned Sentinel, 4,182,612-parameter `small` variant. Exact identity: [Final system](FINAL_SYSTEM.md).

**Sentinel** — the only valid trained task family in the final release. Runtime learned outputs are `source_node`, `event_presence`, `event_cause`, `evidence_sufficiency`, and `relative_strength`.

**Architecture head** — a neural output structure present in the model definition/configuration. Presence does not prove the head received valid supervision.

**Runtime-enabled output** — a learned output explicitly allowed by the V5 release/finalist freeze to be surfaced at runtime.

**Operational authority** — the component permitted to make a governed workflow decision. In the final system OOD, sample ranking, and plan generation remain deterministic; plan verification is simulator-based; approval is human.

**Classical signature** — simulator-derived source-response evidence used by the non-neural localization branch.

**Fusion** — combination of classical and learned source beliefs under the frozen `fuse_source_probabilities-v1` configuration.

## Authority labels

**ADVISORY** — can influence evidence/prediction, cannot authorize an operational state.

**CALIBRATED ADVISORY** — advisory output with applicable conformal semantics.

**DETERMINISTIC** — algorithmic/rule control with operational workflow authority.

**SIMULATOR_VERIFIED** — response candidate completed WNTR/EPANET verification and passed configured modeled constraints.

**HUMAN_APPROVED** — a human explicitly recorded approval. This is not infrastructure actuation.

**Autonomous actuation** — software directly commanding field infrastructure. HydroSwarm has none.

## Calibration and OOD

**Conformal candidate set** — a set of candidate source nodes produced under a split-conformal artifact. Its coverage guarantee is marginal over an applicable population, not a per-incident confidence.

**Coverage** — fraction of applicable evaluated cases whose conformal set contains the true source. Always read with population/denominator and calibration applicability.

**`calibrated_rate`** — fraction of cases where calibration was applicable. The final novel-topology split has `calibrated_rate=0`, so its raw inclusion statistic is not calibrated coverage.

**OOD (out of distribution)** — evidence that an incident/topology lies outside conditions authorized by the frozen calibration/runtime policy.

**Fail closed** — withhold or reduce action authority when required evidence/assets/applicability are missing rather than silently substituting a confident result.

## Sampling and planning

**Scout** — the sampling stage. In the final system the authoritative selector is deterministic `rank_sample_locations`; learned Scout heads are non-authoritative.

**Strategist** — response-planning model/head concept. Learned Strategist outputs are non-authoritative in the final system; authoritative candidate generation is deterministic `generate_response_plans`.

**Sample budget** — governed bound on evidence-collection recommendations during an incident.

**VERIFIED** — a plan completed exact modeled verification and passed configured constraints. Not synonymous with real-world safe or approved.

**REJECTED** — a candidate failed verification/completeness/constraint checks.

**STALE verification** — verification computed under an earlier evidence context; cannot be approved after the context changed.

**APPROVED** — a separate human approval event. No automatic execution follows.

## Evaluation governance

**Development evidence** — data/results available during architecture/capability work before final freeze.

**Finalist freeze** — M11.2 artifact fixing checkpoint, calibration, runtime outputs, authority, and limitations before final test.

**Locked-final test** — 105-case final applicable stress population, opened once in M11.6.

**Locked-topology test** — 20-case final novel-topology/fail-closed population, opened in the same one-time M11.6 execution.

**Descriptive/non-gating metric** — reported scientific measurement that did not determine the hard pass/fail gate. M11.6 explicitly treats Top-1/Top-3/MRR and novel-topology predictive metrics this way.

**Hard safety counter** — preregistered invariant-violation count. All 15 M11.6 counters were zero.

**Opened record** — immutable M11.6 record marking that the one-time lock was consumed before evaluation results were produced.

**Post-lock tuning** — model/system tuning after observing locked results. Final governance records `false`.

## Experience/provenance modes

**LIVE** — real current backend incident and current computation.

**REFERENCE INCIDENT** — deterministic checksummed replay of a frozen WNTR-backed workflow; useful for demonstrating product flow, not final V5 benchmark evidence.

**DEMO_FALLBACK / ILLUSTRATIVE DEMO** — hand-authored fallback content, explicitly not real current scientific computation.

**Replay** — inspection of a recorded incident/audit history; not a new model run.

## Historical terms

**HydroCore-v4** — prior frozen generation retained for provenance. Its metrics and lock status are historical, not current V5 claims.

**HydroCore-S/M/L historical benchmark** — earlier architecture-generation research retained under historical reports; not the final V5 evidence.
