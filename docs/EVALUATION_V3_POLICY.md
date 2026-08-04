# v3 split and locked-final-test policy

Governs the `agent/gcp-multitopology-v3` overnight run (overnight-plan.txt Task 0.8).
This documents `configs/evaluation_policy_v3.json`, the machine-readable version, which
`hydroswarm.training.authorize_locked_final_test()` enforces before any locked-test command
may run. It is additive to, and does not replace, `docs/EVALUATION.md`'s account of the
learning-v1 evaluation protocol and results.

## Split roles

| Split | Purpose | Forbidden |
|---|---|---|
| train | optimization only | checkpoint selection, early stopping, calibration fitting |
| validation | checkpoint selection, early stopping | gradient updates, calibration fitting |
| calibration | conformal/threshold fitting, after checkpoint selection is final | gradient updates, checkpoint selection |
| development_holdout | architecture comparison, repeated iteration | gradient updates |
| ood_development | OOD threshold/behavior development | gradient updates, use as the final OOD test |
| locked_final_test | opened exactly once, after final configuration is selected | any tuning from its results, opening before `final-selection.json` exists, opening more than once |
| locked_topology_test | held-out topology generalization / safety-boundary check | fine-tuning or calibration on that topology |

## Opening the locked final test

`authorize_locked_final_test()` raises `SplitPolicyViolation` unless every one of these is
true, matching the plan's Stage 6/7 requirements exactly:

1. `reports/results/v3/final-selection.json` exists.
2. The selected configuration is recorded in the experiment registry.
3. All required tests pass (pytest, ruff, pyright, frontend lint/test/build).
4. Manifest hashes match between the registry record and the on-disk locked-test manifest.
5. Calibration was fit without any locked test data.
6. No further architecture or hyperparameter tuning is planned.

After the test runs: report results honestly (including negative ones), never retune
against them, preserve every output immutably, and record in the experiment registry that
the locked test was opened, with its run_id.

## Never permitted (all splits)

- Altering, regenerating, relabeling, or inspecting locked test outcomes to make an
  architectural decision.
- Modifying historical result files to make current performance appear better.
- Replacing a failed evaluation with a hard-coded value.
- Promoting a new default checkpoint automatically.
- Overwriting the currently promoted HydroCore-S artifact.
- Training HydroCore-L without all S/M work complete and an explicit machine-readable
  decision record authorizing it.
