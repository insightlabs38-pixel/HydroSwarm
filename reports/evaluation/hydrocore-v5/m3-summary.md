# Milestone 3 summary: causal calibration and candidate-set contraction

Frozen calibration policy: **B_DEPTH_AWARE**
Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED

## Held-out (development_holdout) comparison

| scheme | coverage | mean set size | median set size | singleton rate | planning-gate pass rate |
|---|---|---|---|---|---|
| A (network/condition) | 0.907 | 1.87 | 1.00 | 0.548 | 1.000 |
| B (network+depth) | 0.900 | 1.86 | 1.00 | 0.568 | 1.000 |

## Per-depth-bucket held-out coverage (the aggregate numbers above can mask a subgroup failure)

| scheme | EARLY coverage | MID coverage | MATURE coverage |
|---|---|---|---|
| A | 0.811 | 0.958 | 1.000 |
| B | 0.900 | 0.921 | 0.879 |

Target coverage (1-alpha): 0.90. Scheme A's worst bucket is 8.9pp under target; Scheme B's worst bucket is 2.1pp under target. Material per-bucket undercoverage in A not corrected by B: **False**. This drove the B_DEPTH_AWARE decision via the per-bucket override (Scheme A materially under-covers EARLY, the most decision-consequential bucket, while Scheme B does not; the aggregate-only comparison alone would have kept Scheme A on a false premise of parity).

APS/RAPS follow-up recommended: **False** (not run this session; Milestone 3.6 is optional and conditional on excessively broad sets persisting after depth-aware grouping).
