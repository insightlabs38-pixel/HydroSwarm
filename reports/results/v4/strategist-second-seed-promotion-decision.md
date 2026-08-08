# Strategist second-seed promotion decision (core-issues5.txt Section 7)

Branch `agent/gcp-multitopology-v3`. This document records the required
second corrected-input Strategist seed run and the resulting promotion
decision for learned `plan_value`/`plan_validity` prescreening/ordering.

## What was run

Second seed (`20260812`), identical governed dataset/evaluation protocol
as the first (`20260807`, commit `d7e540e`):

```bash
export PYTHONPATH=src
python scripts/train_strategist_heads.py \
  --corpus-root data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected \
  --teacher-checkpoint experiments/runs/v4-stage-a-sentinel/E1-seed20260810/20260808T010104Z-4a0ea368/checkpoints/checkpoint-0016/model.safetensors \
  --run-root experiments/runs/v4-strategist-heads-v4corpus-corrected-seed2 \
  --registry experiments/registry/v4-strategist-heads-v4corpus-corrected-seed2.jsonl \
  --output reports/results/v4/strategist-heads-training-v4corpus-corrected-seed2.json \
  --seed 20260812

python scripts/run_stage_e_strategist_comparison.py \
  --corpus-root data/learning-v2/cycle-b2-trajectories-v4/strategist-tensors-normalized-corrected \
  --split validation \
  --strategist-checkpoint experiments/runs/v4-strategist-heads-v4corpus-corrected-seed2/20260808T183640Z-4b2b4500/checkpoints/checkpoint-0010/model.safetensors \
  --limit 1000 \
  --output reports/results/v4/stage-e-strategist-comparison-v4corpus-corrected-seed2.json
```

10 epochs, 1739.4s (slower than seed 1's 669.5s due to concurrent CPU load
from other work in this same pass, not a training-config difference).
`load_report` confirmed the same "genuinely fresh init, action_head
correctly dropped" shape as seed 1. Checkpoint:
`experiments/runs/v4-strategist-heads-v4corpus-corrected-seed2/20260808T183640Z-4b2b4500/checkpoints/checkpoint-0010/model.safetensors`
(gitignored/ephemeral, matches this project's established
`experiments/runs/` convention — see the `hydroswarm_checkpoint_persistence`
memory record).

## Seed-to-seed consistency

| metric | seed 1 (`20260807`) | seed 2 (`20260812`) |
|---|---|---|
| `plan_validity_f1` | 0.9972 | 0.99722 |
| `plan_value_mse` | 0.00494 | 0.004728 |
| `exposure_proxy_mse` | 0.1077 | 0.10840 |
| `containment_time_proxy_mse` | 0.0743 | 0.07504 |
| `plan_regret_proxy_mse` | 0.01466 | 0.014122 |

Every metric agrees to within ~2-3% relative — strong, genuine seed-to-seed
consistency, not a coincidence of an easy corpus (task losses above show
real, nonzero, non-degenerate variance across tasks).

## Stage E 4-policy comparison (1000 validation scenarios each)

| policy | seed | mean simulator calls | selected-valid rate | mean regret vs oracle | matched oracle-best rate |
|---|---|---|---|---|---|
| `exact_all` (oracle) | both | 9.0 | 1.000 | 0.0 | 1.000 |
| `deterministic_heuristic` | 1 | 3.0 | 1.000 | 0.00596 | 0.966 |
| `deterministic_heuristic` | 2 | 3.0 | 1.000 | 0.00596 | 0.966 |
| `learned_prescreen` | 1 | 3.0 | 1.000 | 0.00645 | 0.830 |
| `learned_prescreen` | 2 | 3.0 | 1.000 | 0.00640 | 0.830 |
| `learned_ordering` | 1 | 1.0 | 1.000 | 0.01085 | 0.800 |
| `learned_ordering` | 2 | 1.0 | 1.000 | 0.01052 | 0.801 |

`selected_valid_rate` is 1.000 for every policy in both seeds -- hydraulic/
service safety is never compromised regardless of which ranking policy
selects the candidate that actually gets exact-verified. The two seeds
agree closely with each other on every column (gate 7's literal
requirement: "outcome consistency across seeds" -- satisfied).

## Decision

**Do not promote `plan_value`/`plan_validity` to `runtime_enabled_outputs`
at this time.** Not currently enabled in any real checkpoint identity
(`scripts/build_phase15_v4_checkpoint.py`'s own `RUNTIME_ENABLED_OUTPUTS`
already excludes them, for the separate, now-resolved reason of gate 7
being unmet); this decision keeps them excluded going forward for a
substantive reason instead.

Reasoning, stated honestly per this section's own instruction ("Either
result is acceptable. Do not force promotion to make the product appear
more AI"):

- The two seeds agree closely with EACH OTHER (gate 7 is satisfied as a
  procedural matter), but what they agree ON is that `learned_prescreen`
  does not beat `deterministic_heuristic` at the SAME simulator-call budget
  (3 calls each): `deterministic_heuristic` has consistently LOWER regret
  (0.00596 vs ~0.0064) and a much higher oracle-match rate (0.966 vs 0.83)
  in both seeds. Consistency between seeds is a necessary condition for
  promotion, not a sufficient one -- it does not by itself establish that
  the learned policy is worth trusting over the deterministic baseline it
  would replace.
- `learned_ordering` (1 exact call vs deterministic's 3) is a genuinely
  different, real efficiency tradeoff -- 3x fewer EPANET executions for a
  consistent, bounded regret increase (~0.0105 vs ~0.006), still at
  selected_valid_rate=1.0. This is closer to the architecture's own stated
  intent for learned Strategist ("prescreening, ordering, simulator-budget
  efficiency" -- core-issues5.txt Section 6), but promoting a
  budget/regret tradeoff policy into default operational behavior is a
  product decision (how much regret is an operator willing to accept for
  3x fewer exact simulations?), not a decision this evidence alone
  resolves unilaterally.
- Exact WNTR/EPANET verification is completely unaffected either way --
  every policy here only decides which bounded candidate(s) get exactly
  verified, never bypasses verification itself.

## Follow-up (not blocking architecture freeze)

If a future pass wants to promote `learned_ordering` specifically for its
efficiency profile, do so as an explicit, separate, documented product
decision (a named alternate operating mode an operator can choose, e.g.
"fast/reduced-budget mode"), not a silent default -- keep
`deterministic_heuristic` as the default `runtime_enabled_outputs`-gated
behavior given it is currently both safer-equal AND lower-regret than
`learned_prescreen` at equal budget.
