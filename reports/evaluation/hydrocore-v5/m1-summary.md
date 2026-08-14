# Milestone 1 summary: causal-prefix HydroCore-v5, same model size

Promotion decision: **NO_ARM_MEETS_PROMOTION_RULE**

## Arm mean top-1 by causal depth

| arm | 1 | 2 | 3 | 4 | 6 | 12 | 25 |
|---|---|---|---|---|---|---|---|
| A | 0.275 | 0.496 | 0.596 | 0.817 | 0.979 | 0.983 | 0.996 |
| B | 0.279 | 0.479 | 0.592 | 0.842 | 0.975 | 0.979 | 0.983 |
| C | 0.283 | 0.517 | 0.592 | 0.837 | 0.967 | 0.971 | 0.975 |

## Classical identifiability baseline (development_holdout, Milestone 1.2)

| depth | classical top1 | classical top3 | mean ambiguous sources | distinguishable fraction |
|---|---|---|---|---|
| 1 | 0.233 | 0.725 | 3.98 | 0.000 |
| 2 | 0.492 | 0.800 | 1.48 | 0.533 |
| 3 | 0.592 | 0.850 | 1.00 | 1.000 |
| 4 | 0.833 | 0.958 | 1.12 | 0.883 |
| 6 | 0.942 | 1.000 | 1.02 | 0.975 |
| 12 | 0.900 | 0.983 | 1.06 | 0.950 |
| 25 | 0.900 | 0.983 | 1.05 | 0.950 |

## Promotion rule detail

```json
{
  "decision": "NO_ARM_MEETS_PROMOTION_RULE",
  "per_arm": {
    "B": {
      "gain_2_3_step_pp": [
        -1.6666666666666718,
        -0.4166666666666652
      ],
      "regression_6_step_pp": 0.4166666666666652,
      "mature_regression_pp": [
        0.4166666666666763,
        1.2499999999999956
      ],
      "meets_gain_threshold": false,
      "no_material_6_step_regression": true,
      "no_mature_history_regression": true,
      "provisional_winner": false
    },
    "C": {
      "gain_2_3_step_pp": [
        2.083333333333326,
        -0.4166666666666652
      ],
      "regression_6_step_pp": 1.2499999999999956,
      "mature_regression_pp": [
        1.2500000000000067,
        2.083333333333337
      ],
      "meets_gain_threshold": false,
      "no_material_6_step_regression": true,
      "no_mature_history_regression": true,
      "provisional_winner": false
    }
  }
}
```

Scope note: corpus is a single canonical topology (golden-reference), Sentinel-task-only supervision, contamination events only (see reports/evaluation/hydrocore-v5/m1-prefix-dataset.json). Not directly comparable in absolute magnitude to the frozen v4 baseline curve (reports/evaluation/hydrocore-v5/m0-baseline.json), which used a different (historical, committed) corpus -- this milestone's conclusion is about the RELATIVE ranking of arms A/B/C trained on the same corpus, not an absolute claim against v4.
