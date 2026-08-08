# Sensor-Fault Head: Promotion Decision (core-issues5.txt Section 17)

## Decision: leave the learned `sensor_fault` output disabled

This matches current reality: in the real checkpoint identity
(`experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/checkpoint_identity.json`),
`sensor_fault` is present in `trained_outputs` but absent from both
`runtime_enabled_outputs` and `validated_outputs`. No code or governance
change was required to reach this state -- it already holds. Deterministic
sensor-health logic remains the sole authority at runtime. This decision
does not block architecture freeze, per Section 17's own text ("This is not
a reason to delay architecture freeze if the learned head remains
disabled").

## Evidence: the degenerate population is real, not an evaluation artifact

A direct query against the real, committed governed validation corpus
(`data/learning-v2/cycle-b2-joint-v4/tensors-normalized/validation`, via
`hydroswarm.training.ShardedScenarioDataset`, 300 real scenarios sampled)
found:

- 1100 valid sensor-bearing node evaluations (`sensor_fault_mask=True`)
- 1100 positive (`sensor_fault=1.0`)
- 0 negative (`sensor_fault=0.0`)

100% prevalence, 0% healthy negatives. Phase 13's earlier F1=1.0 finding
for this head reflects this degenerate population, not a working
classifier -- an F1 of 1.0 is trivially achievable by a constant-positive
predictor when there are no negatives to misclassify.

## Root cause: `.any()` over the full scenario window, evaluated against a fault-injection model where 3 of 4 fault types are persistent-or-monotonic by construction

The label (`src/hydroswarm/training/corpus.py`, lines 399-416) is:

```python
node_id in scenario.sensor_nodes and (
    scenario.frozen_mask[:, idx].any()
    or scenario.communication_outage_mask[:, idx].any()
    or scenario.drift_mask[:, idx].any()
    or scenario.unit_mismatch_mask[:, idx].any()
)
```

`scenario_to_example` builds with `window_steps=len(scenario.timestamps_seconds)`,
so the full scenario *is* the evidence window visible to the model here --
this is not a simple windowing bug. The real defect is in what `.any()` is
applied to, given how each of the four fault masks is generated in
`src/hydroswarm/data/scenarios.py::_degrade` (lines 306-362):

| Fault mode | Mechanism | Behavior w.r.t. scenario end |
|---|---|---|
| `frozen` (332-337) | `observed[start:, col] = observed[start-1, col]`, `frozen[start:, col] = True` | **Persistent**: once triggered at any `start`, stays `True` through the last timestep. |
| `drift` (314-327) | `drift = hours * drift_per_hour * direction`, applied to **every** sensor in **every** scenario (not probabilistic -- `directions` is drawn unconditionally), `drift_mask = abs(drift) >= quantization_step` | **Monotonic**: grows without bound as elapsed hours increase, so it mechanically crosses the quantization threshold given a long-enough scenario, independent of any random "did a fault occur" draw. |
| `unit_mismatch` (345-349) | `unit_mismatch[:, col] = True` for the whole column if triggered | **Whole-scenario**: uniform across all timesteps by construction, not localized to a sub-window at all. |
| `communication_outage` (338-344) | `mask[start:stop, col] = False`, bounded to `stop = start + shape[0]//8` | **Transient**: the only one of the four that actually recovers within the scenario. |

Because 3 of 4 mechanisms are persistent, monotonic, or scenario-wide, the
`.any()`-over-full-window aggregation is not the primary driver of the
100%/0% split -- it's a symptom of `drift` in particular being a
continuous, universally-applied instrument-drift model, not a discrete rare
anomaly. A naive point-fix (e.g., anchor the label to the last timestep
instead of `.any()`) was evaluated and rejected: it would leave `frozen`
and `unit_mismatch` unchanged (both already true-at-end whenever triggered)
and would make `drift` *more* likely to be positive at the last timestep
specifically, since drift magnitude is largest late in the scenario. Only
the `communication_outage` case would actually be rebalanced by such a
change.

## Why this is not a cheap fix

A real repair requires a semantic redesign of what "sensor fault" means,
not a mechanical change to the aggregation window:

- Deciding whether slow, bounded instrument drift belongs in the same
  binary label as discrete anomaly events (frozen/outage/unit-mismatch) at
  all, or whether it should be a separate continuous target.
- If kept, choosing a materiality threshold for drift that reflects a
  genuine fault rather than expected long-run sensor variance.
- Re-deriving real prevalence statistics, retraining, and re-evaluating
  against a corrected label before any promotion could be considered.

This is exactly the "Otherwise" branch Section 17 anticipates: the
degenerate population and its root cause are now diagnosed and recorded
honestly; the head remains disabled; deterministic sensor-health logic
remains authoritative; no promotion was attempted on the basis of the
degenerate F1=1.0 result.
