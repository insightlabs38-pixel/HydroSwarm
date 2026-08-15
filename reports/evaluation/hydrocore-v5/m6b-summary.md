# Milestone 6B summary: cadence-diverse training

Seeds: [20260814, 31874]. Wall seconds: 4795.1. Overall decision: **KEEP_CURRENT_TEMPORAL_TRAINING**

Arm C (timestamp-conditioning ablation): SKIPPED_WITH_REASON (see JSON `arm_c_timestamp_ablation.reason`).

## Primary endpoint: post-onset fixed-report-count identical-prediction fraction

| seed | N | FIXED_CADENCE_CONTROL | CADENCE_DIVERSE | reduction (pp) |
|---|---|---|---|---|
| 20260814 | 2 | 1.000 | 1.000 | 0.0 |
| 20260814 | 3 | 0.938 | 0.938 | 0.0 |
| 31874 | 2 | 1.000 | 1.000 | 0.0 |
| 31874 | 3 | 0.938 | 0.938 | 0.0 |

## Standard regime (EARLY/MATURE top1) and matched-time spread

| seed | arm | EARLY top1 | MATURE top1 | matched-time spread (pp) | calibration coverage |
|---|---|---|---|---|---|
| 20260814 | FIXED_CADENCE_CONTROL | 0.453 | 0.992 | 0.00 | 0.903 |
| 20260814 | CADENCE_DIVERSE | 0.408 | 0.988 | 0.00 | 0.903 |
| 31874 | FIXED_CADENCE_CONTROL | 0.458 | 0.988 | 0.00 | 0.903 |
| 31874 | CADENCE_DIVERSE | 0.406 | 0.979 | 0.00 | 0.903 |

## Per-seed promotion criteria

| seed | criterion 1 (invariance) | criterion 2 (no regression) | criterion 3 (matched-time stable) | criterion 4 (calibration) | decision |
|---|---|---|---|---|---|
| 20260814 | False | True | True | True | KEEP_CURRENT_TEMPORAL_TRAINING |
| 31874 | False | False | True | True | KEEP_CURRENT_TEMPORAL_TRAINING |

