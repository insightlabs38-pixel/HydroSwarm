# HydroSwarm measured evaluation

Promotion gate: **PASS**

All values below were produced by the frozen WNTR fixture. Missing neural checkpoints are reported as not run.

| Metric | Mean | 95% normal CI |
| --- | ---: | ---: |
| localization_top1_accuracy | 1 | [1, 1] |
| true_source_probability | 0.994134 | [0.994013, 0.994255] |
| candidate_contraction | 3 | [3, 3] |
| entropy_reduction_bits | 1.94807 | [1.94718, 1.94897] |
| unsafe_plan_rejection | 1 | [1, 1] |
| safe_plan_acceptance | 1 | [1, 1] |
| approval_pause | 1 | [1, 1] |
| replay_valid | 1 | [1, 1] |
| exposure_reduction_mg | 14723.2 | [14723.2, 14723.2] |
| latency_seconds | 1.5101 | [-0.105196, 3.1254] |
| peak_python_tracemalloc_mb | 1.16555 | [0.346294, 1.98481] |
| logical_cache_hit_rate | 0.666667 | [0.0133333, 1] |

## Gate checks

- PASS: localization_top1_accuracy
- PASS: candidate_contraction
- PASS: unsafe_plan_rejection
- PASS: safe_plan_acceptance
- PASS: latency_seconds
- PASS: replay_valid
- PASS: no_cache_ablation_measured
- PASS: no_exact_verifier_fails_closed
- PASS: authoritative_hashes_repeat

## Measurement limitations

- RAM: Python allocation peak from tracemalloc; native WNTR/EPANET memory is not included.
- Confidence intervals: 95% normal-approximation interval over configured repeated seeds.
- Small, medium, and large neural variants were not run because no trained checkpoint was supplied.
- This fixture is a regression benchmark, not evidence of field performance.
