# Milestone 7 summary: topology diversity and OOD-aware fusion

Seeds: [20260814, 31874]. Wall seconds: 1986.9.

## Part A -- known-network (golden-reference) regression check

| seed | arm | bucket | neural top1 | classical top1 | hybrid top1 |
|---|---|---|---|---|---|
| 20260814 | CURRENT | EARLY | 0.500 | 0.438 | 0.500 |
| 20260814 | CURRENT | MATURE | 1.000 | 0.938 | 1.000 |
| 20260814 | EXPANDED | EARLY | 0.500 | 0.500 | 0.500 |
| 20260814 | EXPANDED | MATURE | 1.000 | 0.938 | 1.000 |
| 31874 | CURRENT | EARLY | 0.438 | 0.438 | 0.500 |
| 31874 | CURRENT | MATURE | 1.000 | 0.938 | 1.000 |
| 31874 | EXPANDED | EARLY | 0.562 | 0.500 | 0.562 |
| 31874 | EXPANDED | MATURE | 1.000 | 0.938 | 1.000 |

## Part A -- unseen dev-family generalization (MATURE bucket, hybrid top1)

| seed | family | CURRENT (unseen) | EXPANDED (unseen unless noted) |
|---|---|---|---|
| 20260814 | coastal-branch | 0.438 | 0.438 |
| 20260814 | tree-branch | 0.812 | 0.750 |
| 20260814 | dense-loop | 0.625 | 0.688 |
| 20260814 | branched-loop | 0.562 | 0.562* |
| 20260814 | loop-grid | 0.375 | 0.562* |
| 31874 | coastal-branch | 0.375 | 0.438 |
| 31874 | tree-branch | 0.875 | 0.812 |
| 31874 | dense-loop | 0.625 | 0.688 |
| 31874 | branched-loop | 0.562 | 0.562* |
| 31874 | loop-grid | 0.250 | 0.562* |

(* = family EXPANDED trained on; unmarked = unseen to that arm.)

## Part B -- OOD-aware fusion promotion decision

| seed | model | gate (classical>neural, unseen) | promoted |
|---|---|---|---|
| 20260814 | CURRENT | False | False |
| 20260814 | EXPANDED | False | False |
| 31874 | CURRENT | False | False |
| 31874 | EXPANDED | False | False |

