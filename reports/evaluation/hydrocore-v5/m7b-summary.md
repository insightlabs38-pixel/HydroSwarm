# Milestone 7B summary: calibration under actively acquired evidence

Predictor: Milestone-1 winner (arm A); Milestone 2 decision was PCGRAD_JUSTIFIED (4182612 parameters, checkpoint sha256=44a2721394d95985...)
alpha=0.1, K=3, sampling policy=CURRENT_EIG (advisory, unchanged).
Fit incidents (pool B): 108. Evaluation incidents (pool C): 108. Disjoint: True.

## Round-wise coverage (target ~90%, material-undercoverage bar = 5.0pp)

| arm | round | n | coverage | 95% CI | materially below target | mean set size | median set size |
|---|---|---|---|---|---|---|---|
| B_DEPTH_AWARE | PASSIVE | 108 | 0.981 | [0.9560548769369134, 1.0] | False | 1.92 | 1.00 |
| B_DEPTH_AWARE | ACTIVE_ROUND_1 | 108 | 0.898 | [0.8411051100550645, 0.9551911862412318] | False | 1.52 | 1.00 |
| B_DEPTH_AWARE | ACTIVE_ROUND_2_PLUS | 108 | 0.889 | [0.8296172852363576, 0.9481604925414201] | False | 1.86 | 1.00 |
| ADAPTIVE_EVIDENCE | PASSIVE | 108 | 0.907 | [0.8527393708745209, 0.962075443940294] | False | 1.23 | 1.00 |
| ADAPTIVE_EVIDENCE | ACTIVE_ROUND_1 | 108 | 0.880 | [0.8182599414087962, 0.9409993178504631] | False | 1.31 | 1.00 |
| ADAPTIVE_EVIDENCE | ACTIVE_ROUND_2_PLUS | 108 | 0.981 | [0.9560548769369134, 1.0] | False | 2.27 | 1.00 |

## Fallback usage (ADAPTIVE_EVIDENCE)

Fit-corpus resolution level counts: {'L1_NETWORK_CONDITION_ACQSTATE_DEPTH': 324}
Eval-side fallback usage rate (fraction NOT at the most specific L1 level): 0.000

## Incident-clustered bootstrap (ADAPTIVE_EVIDENCE minus B_DEPTH_AWARE, 5000 resamples)

| scope | n incidents | coverage diff | 95% CI | candidate-size diff | 95% CI |
|---|---|---|---|---|---|
| overall | 108 | 0.000 | [-0.03857566765578635, 0.040753950483014494] | -0.160 | [-0.3178848451708286, 0.0] |
| PASSIVE | 108 | -0.074 | [-0.12962962962962954, -0.02777777777777779] | -0.685 | [-0.8888888888888888, -0.4907407407407407] |
| ACTIVE_ROUND_1 | 108 | -0.019 | [-0.08333333333333337, 0.04629629629629628] | -0.204 | [-0.39814814814814814, -0.0092592592592593] |
| ACTIVE_ROUND_2_PLUS | 72 | 0.093 | [0.027516680567139233, 0.1711711711711711] | 0.407 | [0.201788990825688, 0.6228331723804923] |

## Promotion criteria (predeclared, experiments.txt M7B.7)

1. Post-sample coverage restored (round1 & round2+ not materially below target): True
2. Passive coverage preserved (regression <= 5.0pp): False (observed regression: 7.4074074074074066)
3. Candidate-set inflation acceptable (<= 1.5x, not saturating full action space): True
4. Candidate-gate-pass not materially degraded (<= 10.0pp drop): False
5. No safety/authority threshold changed: True

**Decision: KEEP_B_DEPTH_AWARE_AND_MARK_POST_SAMPLE_UNCALIBRATED**


## Optional cross-policy diagnostic (RANDOM_VALID_UNSAMPLED, winning calibrator, not refit)

Winning calibrator: B_DEPTH_AWARE. Overall coverage on RANDOM_VALID_UNSAMPLED: 0.929. This is distribution-transfer evidence only, computed after and separately from the decision above.

active sampling authority changed: False. locked tests opened: before=False, after=False.
