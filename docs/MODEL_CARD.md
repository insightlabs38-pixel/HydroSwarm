# HydroCore Model Card

## Intended use

HydroCore ranks possible contamination sources, estimates evidence sufficiency and sensor
faults, reranks candidate sample locations, and scores structured response actions. It is
an experimental component inside a classical-neural decision-support pipeline. It is not
an autonomous control system and cannot approve or execute a response plan.

## Architecture

The large configuration uses 8 local/global hydraulic blocks, width 360, 8 attention
heads, FFN width 1080, structural/temporal/sensor-quality encoders, and residual specialist
adapters for HydroSentinel (32), HydroScout (48), and HydroStrategist (64). The exact S/M/L
parameter counts are **4,040,645 / 12,401,861 / 24,415,397**. `parameter_report()` exposes
the count by subsystem. Source logits learn a residual correction over the training-only
classical signature prior and explicitly mask reservoirs/tanks as invalid sources.

The model accepts masks for missing sensor values and padded nodes. It retains node-level
representations for source, sample, fault, and action pointer-style heads. Physics masks
and exact WNTR verification remain outside and authoritative over the network.

## Calibration and abstention

Source distributions are fused with a classical signature posterior using a dynamic trust
coefficient based on sensor health, missingness, residuals, hydraulic uncertainty,
entropy, OOD score, and Jensen-Shannon disagreement. Split-conformal candidate sets target
90% marginal coverage. On 200 held-out hydraulic-shift scenarios, coverage is 91.0%, mean
set size is 0.92, and ECE is 0.0269.
High disagreement, OOD evidence, poor sensor health, invalid networks, or failure to find a
safe plan forces inspection or abstention.

## Limitations

- A trained HydroCore-S checkpoint and equal-budget HydroMono-S checkpoint are bundled;
  HydroCore-M is a clearly labeled two-epoch partial checkpoint and L is untrained.
- Held-out evidence uses one topology under a withheld hydraulic regime, not an unseen
  topology or real utility network.
- Start-time, duration, and strength-bin heads remain weak (20.5%, 42.5%, and 34.0%).
- Synthetic performance does not imply utility-network performance.
- Source strength and mass consequences depend on assumptions about contaminant behavior.
- Conformal coverage is marginal on the calibration distribution, not a per-incident or
  arbitrary-distribution guarantee.
- The default model size is an engineering target, not evidence that it is more capable.

## Promotion gate

The 24.5M model should only replace the promoted S or classical fallback if held-out testing
shows a meaningful operational gain: at least five percentage points better incident
resolution, 20% fewer samples at matched localization, 30% fewer invalid plans, improved
unseen-network robustness/calibration, or lower verified-plan regret.
