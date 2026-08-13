# HydroSwarm End-to-End LIVE Robustness and Scalability Characterization

## Executive summary

Study 2 executed 264 deterministic, API-driven HydroSwarm incident trajectories against frozen HydroCore-v4. Unlike Study 1, it used the real network importer, persistence, `HybridInferencePipeline.analyze()`, dynamic fusion, conformal calibration, live OOD calculation, active-sampling endpoint, plan generator, exact WNTR verifier, and approval route. No model, calibration, schema, threshold, physics, or authority policy changed.

The API safely suppressed 252 trajectories and abstained at the analysis boundary for nine all-unusable-evidence trajectories. Three loop-grid trajectories legitimately generated and exactly verified a plan. All measured authority invariants held. Two product findings require review: repeated sampling of an already-observed node (`ROB-LIVE-01`, MEDIUM), and an unvalidated coastal topology receiving live OOD `NORMAL` rather than a caution indication despite calibration inapplicability (`ROB-LIVE-02`, HIGH).

This is characterization evidence, not a field-performance claim or a reason to tune the frozen system.

## Frozen identities and protocol

| Identity | Value |
| --- | --- |
| Product baseline | `e45f72cf730d3f12c13dbcb9403c64f185510173` |
| Runner head before report commit | `60d8fb79739413ef45af9285e448bb428be96957` |
| Model / calibration SHA-256 | `a501ad87…712d16c7` / `829c167b…402f68fa` |
| Feature schema / normalization SHA-256 | `7ec97775…20e9ddd09` / `e0808f21…db0fa1114` |
| Signature policy SHA-256 | `06e31d92…acb686811` |
| WNTR / exposed EPANET | 1.5.0 / not separately exposed |
| Python / platform | 3.12.13 / Linux 6.17.0 aarch64 |
| Logical CPUs / RAM | 16 / 62.66 GiB |
| Locked test opened | `false` before and after |

The frozen [protocol](LIVE_ROBUSTNESS_PROTOCOL.md) and `protocol.json` declare seeds, populations, metrics, timing, exclusions, and no-post-result tuning. `HARNESS_CORRECTION.md` records three transparent supersessions: a smoke row is excluded; fail-closed no-evidence 409s are `ABSTAINED`, not harness faults; and a calibration-validated branched fixture was replaced with unvalidated coastal-branch before interpretation.

| Population | Rows | Nodes / links | Role |
| --- | ---: | ---: | --- |
| golden-reference | 231 | 6 / 7 | reference perturbation matrix |
| loop-grid | 9 | 9 / 11 | calibrated scale/lifecycle control |
| coastal-branch | 24 | 8 / 8 | development-only unfamiliar topology |

Evidence was generated in memory by the governed WNTR generator with development-holdout ownership, then submitted to the product API. It was never added to train, validation, calibration, or a locked set.

## Localization results

Raw live-pipeline aggregates include weak and suppressed contexts; they are not comparable deployment benchmarks.

| Family | n | Top-1 | Top-3 | MRR | Candidate size | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal golden-reference | 12 | 0.000 | 0.750 | 0.396 | 4.000 | 0.000 |
| Missingness | 60 | 0.263 | 0.737 | 0.522 | 2.895 | 0.000 |
| Sensor coverage | 36 | 0.333 | 1.000 | 0.611 | 4.000 | 0.000 |
| Sensor health | 30 | 0.375 | 1.000 | 0.604 | 3.000 | 0.000 |
| Noise / bias | 45 | 0.667 | 1.000 | 0.822 | 4.000 | 0.000 |
| Hydraulic mismatch | 30 | 0.100 | 0.900 | 0.475 | 3.700 | 0.000 |
| Ambiguity/conflict | 18 | 0.667 | 1.000 | 0.833 | 4.000 | 0.000 |
| Unfamiliar coastal topology | 24 | 0.000 | 0.750 | 0.400 | 3.750 | 0.000 |
| Loop-grid scale control | 9 | 0.000 | 0.667 | 0.278 | 3.333 | 0.333 |

The 0%, 10%, 25%, 50%, and 75% missingness levels had top-1 of 0.250, 0.250, 0.250, 0.500, and 0.000. This small deterministic population does not justify a monotonicity claim. At 75% missingness three rows rejected analysis with HTTP 409 for no usable evidence; their unavailable localization metrics are `null`.

## Calibration, OOD, and authority

All evaluable golden-reference rows were calibration-inapplicable and planning-suppressed, mainly for `CALIBRATION_INVALID_OR_MISSING`, `CANDIDATE_REGION_TOO_BROAD`, and `MODEL_EVIDENCE_INSUFFICIENT`. This is an OBSERVATION, not a safety fail: raw localization is preserved and the authority boundary held.

Coastal-branch was calibration-inapplicable and planning-suppressed in all 24 rows, but live OOD was `NORMAL` with `network_novelty=0.0`. This is `ROB-LIVE-02` (HIGH): the runtime did not surface the expected unfamiliar-topology caution. Root-cause hypothesis: the OOD reference lacks calibration's topology-hash allow-list. The defect did not grant planning authority in any measured row.

| Class | Result |
| --- | --- |
| PASS | INV-1 through INV-10 held where applicable. Suppressed plan generation returned 409. No verified plan lacked WNTR verification. Evidence mutation caused stale approval rejection (409). A verified/current plan was approved and persisted to `CLOSED`; no actuation route exists. |
| FAIL | No measured authority-invariant failure. |
| OBSERVATION | 252/264 suppressed; 9/264 safe API-level abstentions. |
| ROB-LIVE-01 | MEDIUM. 27/120 actionable recommendations (27/333 sample records) selected a node already in current evidence after re-analysis. The harness stopped rather than insert duplicate synthetic evidence. |
| ROB-LIVE-02 | HIGH. All 24 coastal-branch rows had calibration inapplicability plus live OOD `NORMAL`; see above. |

## Actual active sampling

The real sampling endpoint was invoked 333 times: 93 recommendations acquired a simulated measurement, 213 stopped legitimately, and 27 were the repeated-observed-node finding. No posterior-source proxy was used. The product's expected-information-gain field is retained in nested raw records.

For 93 acquired rounds, median realized entropy reduction was **-0.071 bits** (range -1.814 to +0.974). The study therefore does not support a claim that every recommendation improves localization. No random-node baseline was run because an identical valid-node/budget baseline was not available throughout the configured scenarios.

## Planning, verification, and lifecycle

Four plan candidates were generated in eligible loop-grid trajectories. Three were `VERIFIED`; one was `ABSTAINED` with `SIMULATION_BUDGET_EXCEEDED`. The live verifier consumed four exact WNTR/EPANET simulations. No deterministic rejection appeared, so the deterministic-rejection rate is INCONCLUSIVE. Controlled trajectories demonstrated current verified approval (HTTP 200, final state `CLOSED`) and stale approval rejection after evidence mutation (HTTP 409).

## Performance and scalability

Local-process measurements only; not utility-scale deployment claims. `peak_process_rss_mb` is sampled process high-water RSS and p95 appears only for n >= 20.

| Stage | n | Median ms | p95 ms | Min--max ms |
| --- | ---: | ---: | ---: | ---: |
| Network import | 264 | 75.8 | 89.0 | 53.0--280.1 |
| Incident creation | 264 | 11.1 | 13.4 | 9.4--16.6 |
| Live pipeline analysis | 264 | 162.5 | 652.9 | 112.5--838.2 |
| Actual sampling recommendation | 54 | 8.3 | 16.4 | 4.6--19.1 |
| Evidence update + re-analysis | 54 | 236.6 | 398.1 | 114.1--603.6 |
| Plan generation | 255 | 1.1 | 1.5 | 1.0--11.0 |
| Exact WNTR verification | 3 | 132.2 | n < 20 | 131.9--203.4 |
| Full trajectory | 264 | 259.2 | 815.7 | 188.4--1294.8 |

Peak RSS was 798.8--1422.8 MB (median 1083.2; p95 1384.0) over 255 trajectories that reached memory finalization. The legitimate size range was only 6--9 nodes and 7--11 links, so large-network scalability is a LIMITATION. No timeout or WNTR convergence failure occurred.

## Locked-test status and reproduction

`locked_test_opened` was checked false before every resumable campaign slice and after the final slice. The runner rejects locked paths/splits and never opened or enumerated `locked_final_test` or `locked_topology_test`.

```bash
.venv/bin/python scripts/run_live_robustness_characterization.py --repetitions 3
.venv/bin/python -m pytest -q tests/evaluation/test_live_robustness_characterization.py
```

Raw nested records are `reports/evaluation/live-robustness/results.json` and CSV; `summary.json` is the deterministic aggregate. Study 1 remains [ROBUSTNESS_SCALE_EVALUATION.md](ROBUSTNESS_SCALE_EVALUATION.md) and must not be called a live workflow measurement.
