# HydroSwarm measured evaluation

Promotion gate: **PASS**

All values below were produced by the frozen WNTR fixture. Learned-model evidence is
reported separately in `learning-evaluation-final.json`.

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
| latency_seconds | 0.980088 | [-0.391434, 2.35161] |
| peak_python_tracemalloc_mb | 1.16535 | [0.346173, 1.98452] |
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
- The golden report's legacy neural matrix is not the learning benchmark. HydroCore-S,
  HydroMono-S, and the budget-complete HydroCore-M candidate were run on the governed
  held-out corpus; L remains untrained.
- This fixture is a regression benchmark, not evidence of field performance.

## Learned held-out hydraulic-shift benchmark

- Classical signature top-1: 91.5% [87.5%, 95.0%].
- HydroCore-S neural top-1: 94.5% [91.0%, 97.5%].
- HydroMono-S top-1: 94.5% [91.5%, 97.5%].
- Hybrid top-1: 96.0% [93.0%, 98.5%], +4.5 points over classical.
- Held-out conformal coverage: 91.0%; mean set size 0.92; ECE 0.0269.
- HydroCore-M neural/hybrid: 94.5% / 94.0%; M was not promoted because S hybrid reaches
  96.0% at substantially lower latency.
- M profile heads: start 27.0%, duration 35.5%, strength 45.5%; still exploratory.
- Independent topology: classical 35.7%, M neural 47.1%, M hybrid 44.3%; coverage 27.1%,
  mean set size 0.41, `CAUTION`, planning suppressed.
- Limitation: synthetic reference networks only; not field or cross-utility proof.
