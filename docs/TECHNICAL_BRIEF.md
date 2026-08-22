# Technical brief

A compact technical review of the final HydroSwarm system. For exact hashes and authority, [Final system](FINAL_SYSTEM.md) is canonical.

## System in one paragraph

HydroSwarm is a local hybrid decision-support pipeline for simulated drinking-water contamination incidents. A classical hydraulic/signature branch and a learned HydroCore-v5 Sentinel produce source evidence; frozen split-conformal calibration is applied when valid; deterministic OOD/evidence control decides whether the workflow may continue; deterministic Scout and plan-generation logic choose evidence/action candidates; WNTR/EPANET is required to verify response plans; and a distinct human event is required for approval. The learned model never owns verification, approval, or actuation authority.

## Frozen V5 identity

Finalist: **HydroCore-v5 M10 frozen release**, `small`, 4,182,612 parameters, seed `20260814`.

- checkpoint: `de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`
- release manifest: `f3fb08642738128f020c50e20e6b68c417bf80703f7ef6bc8f42db2aa41f8d34`
- calibration file: `8f77f06b72316455e1f8040dbeb5907503e4eb623dd527d9ea809a56e96c046d`
- calibration artifact: `f2503e856c467eb38c6c7f6dbde679527c1921925941ec52809bd6e8e6dd16dd`
- alpha/grouping: `0.1`, `B_DEPTH_AWARE`
- serving: `V5PipelineFactory(resolve_v5_bundle_dir())`

## Learned versus deterministic authority

The model architecture includes optional specialist/control heads, but final valid training scope is `sentinel` only. Runtime learned outputs are exactly `source_node`, `event_presence`, `event_cause`, `evidence_sufficiency`, and `relative_strength`.

Operational controls are deterministic:

- OOD: `OODDetector`
- sampling: `rank_sample_locations`
- planning: `generate_response_plans`
- physical verification: WNTR/EPANET
- approval: human operator
- autonomous actuation: none

This separation is enforced by the release loader and was rechecked by M11.6 safety counters.

## Training and model selection

The selected S model used `AGE_FIX_ONLY` plus exact 1,350-step interleaved multi-topology training over `golden-reference`, `branched-loop`, and `loop-grid`. M9 tested larger capacity; the ~13.9M-parameter M model did not meet the predeclared meaningful unseen-topology gain, so S remained selected.

A frozen caveat: the M9.6 training record used a fixed unobserved-age sentinel, while M10.4-tested serving retained `incident_elapsed`. This known train/serve feature-semantics deviation was not changed after lock.

## Final evidence

M11.6 opened the final locked populations exactly once after authorization:

| Population | n | Top-1 | Top-3 | Coverage/applicability | Actionable |
|---|---:|---:|---:|---:|---:|
| nominal final | 15 | 73.3% | 86.7% | 93.3% coverage | 80.0% |
| all locked-final | 105 | 55.2% | 76.2% | 88.6% coverage | 61.0% |
| novel topology | 20 | 55.0% | 70.0% | calibration inapplicable | 0.0% |

Overall M11.6: 125/125 complete, locked-final PASS, locked-topology PASS, 15/15 hard safety counters zero, one authorized opening, no rerun, no post-lock tuning.

Topology predictive metrics are descriptive/non-gating. The operational topology result is fail-closed: 0% calibrated, 0% actionable, 0% human-approved.

## Runtime

FastAPI + SQLite + bounded local jobs expose typed local APIs and durable audit history. The current source app serves V5. `hydroswarm self-test --strict` checks the V5 bundle and a bounded real WNTR run.

For V5 Docker, either run the published release (`docker compose -f docker-compose.release.yml up`, `ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`) or build the current checkout; both resolve to the same frozen V5 identity.

## Reproducibility

Do not rerun M11.6. Verify the immutable model/calibration/release hashes plus the M11.6 opened/metrics/gate/safety/closure chain. Ordinary source checks, self-test, and non-locked tests remain reproducible.

See [Reproducibility](REPRODUCIBILITY.md) and [Claims and evidence](CLAIMS_AND_EVIDENCE.md).

## Limits

All data are synthetic. Stress performance is materially lower than nominal performance. Novel-topology prediction is not calibrated. WNTR/EPANET inherits network/model assumptions. No field/utility accuracy, chemistry determination, public-health safety, or autonomous-control claim is supported.
