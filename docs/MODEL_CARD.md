# HydroCore-v5 model card

## Model summary

HydroCore-v5 is the learned Sentinel inside HydroSwarm's hybrid water-contamination decision-support pipeline. The final frozen model is the **HydroCore-v5 M10 frozen release**, `small`, 4,182,612 parameters, selected seed `20260814`.

| Identity | Value |
|---|---|
| Checkpoint SHA-256 | `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5` |
| Release manifest SHA-256 | `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34` |
| Feature schema | `7ec97775e5f01f87ae62669146a7eb70958f99b1162a356614eb87220e9ddd09` |
| Calibration SHA-256 | `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d` |
| Calibration artifact | `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd` |
| Serving factory | `V5PipelineFactory` |
| Trained task family | `sentinel` |

Authoritative identity: [M11.2 finalist freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json) and [runtime manifest](../models/hydrocore-v5-release/runtime_manifest.json).

## Intended use

HydroCore-v5 is intended to contribute **advisory** learned evidence to offline/local research decision support for simulated drinking-water contamination incidents. Its relevant uses are source localization and Sentinel-level event/evidence characterization within a larger system that keeps deterministic OOD, sampling, planning, simulator verification, and human approval outside learned authority.

## Non-intended use

The model is not intended to:

- identify contaminant chemistry, toxicity, pathogen viability, or potability;
- replace laboratory confirmation, utility procedures, regulation, or qualified engineering judgment;
- autonomously select/execute infrastructure actions;
- provide field-validated utility accuracy;
- treat an unseen topology as calibrated merely because a neural prediction exists;
- use learned Scout, Strategist, or OOD heads as operational authorities;
- infer that a conformal marginal coverage target is a per-incident confidence guarantee.

## Architecture and training

The final model is the S-scale graph/time architecture with `prior_mode=feature_only`, event-control head structures, an OOD-category head structure, Scout-control head structures, candidate-conditioned Strategist structures, and consequence-prescreening structures. Architecture presence does not imply those heads were validly supervised or promoted.

The selected training recipe is:

`CLASSICAL_HYDROCORE_S` + `AGE_FIX_ONLY` + `EXACT_1350_STEP_INTERLEAVED_MULTI_TOPOLOGY_TRAINING`.

The selected M9.6 run used exactly 1,350 optimizer steps and equal interleaving across `golden-reference`, `branched-loop`, and `loop-grid`, with 200 physical training scenarios per family. The canonical checkpoint was the final step, not the best-validation checkpoint. See the [training record](../reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed20260814.json).

## Supervision and output governance

The frozen release declares `trained_tasks = ["sentinel"]`. Five learned outputs are runtime-enabled:

| Runtime learned output | Role | Authority |
|---|---|---|
| `source_node` | learned source belief | advisory |
| `event_presence` | event-presence characterization | advisory |
| `event_cause` | event-cause characterization | advisory |
| `evidence_sufficiency` | learned evidence-sufficiency signal | advisory input to governed pipeline |
| `relative_strength` | relative-strength characterization | advisory |

Explicitly suppressed/untrained operational outputs include `next_step`, learned OOD category as an authority, `sample_node`, information-gain/candidate-reduction controls, sampling-stop control, and learned plan/consequence/regret controls. The final operational authorities remain deterministic `OODDetector`, `rank_sample_locations`, and `generate_response_plans`.

This is why statements such as “the model chooses the next sample” or “the neural planner selects a response plan” are incorrect for the final frozen system.

## Fusion role

HydroCore-v5 does not replace the classical hydraulic/signature path. Its source estimate is combined with classical evidence under the frozen `fuse_source_probabilities-v1` fusion configuration. The fused advisory is subject to calibration applicability and deterministic OOD/evidence control before planning can proceed.

## Calibration

The frozen split-conformal artifact uses alpha `0.1` and `B_DEPTH_AWARE` grouping. It is cryptographically tied to the frozen model/feature/fusion identity and declares three validated topology hashes.

Calibration is applicable only where the artifact's conditions are satisfied. On the final 20-incident novel-topology population, `calibrated_rate` was `0.0`; no actionable or approved plan was produced. The 60% raw candidate-set inclusion statistic reported for that split is therefore descriptive and must not be presented as calibrated coverage.

## Development evidence

The final system was preceded by a governed M9/M10 program rather than selected from the locked test.

- M9 closed architecture/training/capacity search with the 4.18M S model retained; a 13.9M M model did not show the predeclared meaningful unseen-topology capacity gain.
- M9.6 finalized exact 1,350-step compute parity and three-family interleaved training.
- M10 audited downstream supervision/authority and ran full production-path trajectories.
- M10.4 used 360 physical incidents / 720 API trajectories across three trained and three unseen families and seven condition types; its frozen gate passed. This is development evidence, not final-test evidence.
- M10.5 completion froze the selected seed, bundle, calibration, five runtime outputs, deterministic surrounding authority, and V5-only failure behavior.

See [Evaluation](EVALUATION.md) for chronology.

## Final locked evidence

M11.6 used 105 locked-final incidents and 20 locked-topology incidents, opened exactly once after authorization.

| Population | Top-1 | Top-3 | MRR | Applicable coverage | Actionable |
|---|---:|---:|---:|---:|---:|
| Nominal locked-final, n=15 | 73.3% | 86.7% | 0.821 | 93.3% | 80.0% |
| All locked-final, n=105 | 55.2% | 76.2% | 0.687 | 88.6% | 61.0% |
| Novel topology, n=20 | 55.0% | 70.0% | 0.652 | not calibration-applicable | 0.0% |

The overall gate passed. All 15 hard safety counters were zero. Novel-topology predictive metrics were explicitly descriptive/non-gating, while topology novelty + fail-closed behavior were hard checks.

Full matrix: [Scientific evidence](SCIENTIFIC_EVIDENCE.md).

## Stress and topology behavior

The locked results show a real robustness gap. Nominal performance is stronger than aggregate stress performance. In particular, ambiguity/disagreement, measurement noise, and sensor dropout reduce predictive accuracy and/or actionability; sensor dropout also reduced applicable coverage to 66.7% for its 15-case condition slice even though the aggregate applicable locked-final coverage remained above the frozen 85% floor.

Novel topology retained measurable localization signal (55% Top-1, 70% Top-3), but the final system correctly withheld calibration/action authority. No external benchmark is used here to label the predictive result “good” or “bad.”

## Known feature-semantics caveat

The final M9.6 training record used `unobserved_age_sentinel: fixed`. The M10.4-tested default runtime retained `unobserved_age_sentinel=incident_elapsed`, and the finalist/release freeze explicitly records that mismatch. The system was frozen/evaluated with that serving behavior; documentation must not erase the deviation or imply a post-lock correction occurred.

## Data limitations

All training, development, calibration, and locked evidence is synthetic WNTR/EPANET-generated evidence. The final training program includes multiple governed topology families, but this remains a finite synthetic generator, not representative proof for utility networks generally.

No raw utility telemetry, consumer health data, or field incident outcome dataset is used to establish the model results in this repository.

## Ethical and safety boundary

The model's advisory status is intentional. Prediction error in infrastructure response can carry public-health and service consequences, so the system keeps physical verification and human approval outside learned authority. The locked zero-safety-counter result demonstrates the tested software boundary, not real-world safety or regulatory suitability.

See [Authority and safety](AUTHORITY_AND_SAFETY.md) and [Limitations](LIMITATIONS.md).

## Historical note

The previous HydroCore-v4 model card is superseded by this V5 card. V4 artifacts and evaluation reports remain in the repository for provenance; their metrics must not be presented as current V5 performance.
