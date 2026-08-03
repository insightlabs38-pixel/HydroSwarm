# HydroCore Model Card

## Intended use

HydroCore ranks possible contamination sources, estimates evidence sufficiency and sensor
faults, reranks candidate sample locations, and scores structured response actions. It is
an experimental component inside a classical-neural decision-support pipeline. It is not
an autonomous control system and cannot approve or execute a response plan.

## Architecture

The default configuration uses 10 Transformer layers, width 384, 8 attention heads, FFN
width 1152, structural/temporal/sensor-quality encoders, and residual specialist adapters
for HydroSentinel (32), HydroScout (48), and HydroStrategist (64). The exact implemented
parameter count is **24,538,903**. `parameter_report()` exposes the count by subsystem.

The model accepts masks for missing sensor values and padded nodes. It retains node-level
representations for source, sample, fault, and action pointer-style heads. Physics masks
and exact WNTR verification remain outside and authoritative over the network.

## Calibration and abstention

Source distributions are fused with a classical signature posterior using a dynamic trust
coefficient based on sensor health, missingness, residuals, hydraulic uncertainty,
entropy, OOD score, and Jensen-Shannon disagreement. Split-conformal candidate sets target
90% marginal coverage; measured coverage and set size must be reported on held-out data.
High disagreement, OOD evidence, poor sensor health, invalid networks, or failure to find a
safe plan forces inspection or abstention.

## Limitations

- No trained checkpoint or real-world validation is bundled.
- Synthetic performance does not imply utility-network performance.
- Source strength and mass consequences depend on assumptions about contaminant behavior.
- Conformal coverage is marginal on the calibration distribution, not a per-incident or
  arbitrary-distribution guarantee.
- The default model size is an engineering target, not evidence that it is more capable.

## Promotion gate

The 24.5M model should only replace a smaller or classical fallback if held-out testing
shows a meaningful operational gain: at least five percentage points better incident
resolution, 20% fewer samples at matched localization, 30% fewer invalid plans, improved
unseen-network robustness/calibration, or lower verified-plan regret.

