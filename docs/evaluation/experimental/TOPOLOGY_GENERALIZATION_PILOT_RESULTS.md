# Topology-generalization pilot: results (EXPERIMENTAL, NON-RELEASE)

Branch: `exp/topology-generalization`. Protocol frozen before this run in
`docs/evaluation/experimental/TOPOLOGY_GENERALIZATION_EXPERIMENT_PLAN.md`.
This document reports what was actually measured; nothing here changes any
production default, gate, or the frozen `v0.2.1` release, and no locked
(`data/locked/m11-6/`) artifact was opened.

## What was run

Two fresh `HydroCore` ("small" variant, 4.18M-ish params, `event_control_heads=True`)
models, trained from scratch, identical in every respect (seed `20260814`,
`configs/training-v5-causal.yaml`'s optimizer/scheduler/task-weight settings,
600 real-source training examples — 200 each of golden-reference/branched-loop/
loop-grid, drawn from the already-generated, leakage-checked
`data/learning-v2/cycle-b2` corpus, 6 epochs) **except**:

- **CONTROL**: stock `HydroCore`, `node_feature_dim=19`, `edge_feature_dim=13`.
- **EXPERIMENTAL_TOPOLOGY_RELATIVE**: the same architecture, but every
  batch is passed through `hydroswarm.model.topology_normalization.augment_batch`
  first, appending a per-graph max-abs-normalized copy of every
  `FeatureScope.TOPOLOGY_RELATIVE`-tagged node/edge column (7 node + 4 edge
  columns), so `node_feature_dim=26`, `edge_feature_dim=17`.

`coastal-branch` (the corpus's `ood-UNSEEN_TOPOLOGY` split) was never used
for training or calibration fitting, for either arm — only read at
evaluation time. Both arms' calibration was fit independently, each on its
own arm's forward passes over the same 712 real-source `calibration` split
examples (alpha `0.1`, matching the frozen release), never on
`development_holdout` or the OOD split.

Scale note: this pilot's 600-example/6-epoch/1-seed budget is deliberately
much smaller than the historical M9.6 campaign (600 scenarios but full
1350-step/20-epoch/3-seed training) — see plan doc Section 4 for why. Point
estimates below should not be read as reproducing M9.6's own numbers;
the CONTROL-vs-EXPERIMENTAL **delta**, computed under identical conditions,
is the thing this pilot measures.

## Headline metrics (source localization)

| population | n | metric | CONTROL | EXPERIMENTAL | delta (EXP-CONTROL) | 90% CI (paired bootstrap, 2000 resamples) |
|---|---|---|---|---|---|---|
| validation (known) | 300 | top1 | 0.6933 | 0.7000 | +0.0067 | [-0.0068, +0.0233] |
| validation (known) | 300 | top3 | 0.8733 | 0.8700 | -0.0033 | [-0.0133, +0.0067] |
| validation (known) | 300 | MRR | 0.7962 | 0.8002 | +0.0040 | [-0.0047, +0.0123] |
| development_holdout (known) | 300 | top1 | 0.6900 | 0.6933 | +0.0033 | [-0.0067, +0.0133] |
| development_holdout (known) | 300 | top3 | 0.8800 | 0.8767 | -0.0033 | [-0.0133, +0.0067] |
| development_holdout (known) | 300 | MRR | 0.7954 | 0.7979 | +0.0025 | [-0.0029, +0.0083] |
| **ood-UNSEEN_TOPOLOGY (coastal-branch)** | 280* | **top1** | **0.3750** | **0.3750** | **+0.0000** | **[-0.0286, +0.0286]** |
| ood-UNSEEN_TOPOLOGY | 280* | top3 | 0.7571 | 0.7321 | **-0.0250** | **[-0.0464, -0.0036]** |
| ood-UNSEEN_TOPOLOGY | 280* | MRR | 0.5857 | 0.5796 | -0.0061 | [-0.0217, +0.0098] |

\* of 400 raw examples in the OOD split, 280 carry a real (unmasked)
source-node label; the other 120 are NORMAL/SENSOR_FAULT_ONLY scenarios
with no source to localize, excluded from these rows (see "Incidental
finding" below).

Raw source: `reports/evaluation/topology-generalization/pilot-results.json`,
`reports/evaluation/topology-generalization/paired-bootstrap.json`.

## Per-topology breakdown (top1, CONTROL / EXPERIMENTAL)

| family | validation n | validation top1 | development_holdout n | development_holdout top1 |
|---|---|---|---|---|
| golden-reference (known) | 103 | 0.9417 / 0.9417 | 112 | 0.9464 / 0.9464 |
| branched-loop (known) | 103 | 0.6117 / 0.6214 | 96 | 0.5521 / 0.5625 |
| loop-grid (known) | 94 | 0.5106 / 0.5213 | 92 | 0.5217 / 0.5217 |
| coastal-branch (unseen) | — | — | 280 (OOD split) | 0.3750 / 0.3750 |

No family shows a regression larger than ~1pp on top1 in either direction;
golden-reference is identical to 4 decimal places in both arms (the same
112/112 and 103/103 examples correct).

## Calibration (diagnostic, fit independently per arm, alpha=0.1, n=712)

| | CONTROL | EXPERIMENTAL |
|---|---|---|
| coverage | 0.9073 | 0.9073 (identical) |
| expected calibration error | 0.0542 | 0.0595 (+0.0053, slightly worse) |
| mean candidate-set size | 2.593 | 2.671 (+0.078, slightly larger/less precise) |

## Actionability / abstention proxy and OOD behavior

This pilot does not exercise plan generation, WNTR verification, or the
human-approval workflow, so it cannot reproduce the real, much stricter
`actionable` definition (`reports/evaluation/hydrocore-v5/m11/m11-6a/design-freeze/m11-6a-actionability-semantics.json`:
a successful `/approve` with `decision == VERIFIED`). The numbers below are
a **research-diagnostic proxy** (`calibrated AND candidate set non-empty`),
computed identically for both arms and reused unmodified from
`hydroswarm.calibration.conformal.SplitConformalCalibrator`/
`hydroswarm.inference.ood.OODDetector.topology_level` — not the production
actionability certificate, and never claimed as such.

| population | | CONTROL proxy-actionable | EXPERIMENTAL proxy-actionable | CONTROL proxy-abstention | EXPERIMENTAL proxy-abstention |
|---|---|---|---|---|---|
| validation (known) | | 0.980 | 0.977 | 0.020 | 0.023 |
| development_holdout (known) | | 0.973 | 0.967 | 0.027 | 0.033 |
| **ood-UNSEEN_TOPOLOGY** | | **0.000** | **0.000** | **1.000** | **1.000** |

`OODDetector.topology_level` (deterministic, reused unmodified,
`validated_network_hashes` populated from each arm's own training-topology
hashes) reports `CAUTION`/`OUTSIDE_VALIDATED_RANGE` for **100% of both
arms'** `ood-UNSEEN_TOPOLOGY` rows, and `NORMAL` for 100% of known-family
rows — identical behavior in both arms, exactly the categorical, hash-gated
outcome H2 predicted, unaffected by representation.

## Safety counters

Not applicable / structurally absent, not fabricated as a measured value:
this pilot is an offline predictive-quality replay (forward pass →
softmax → conformal candidate set) with no actuation, sampling, planning,
or human-approval pathway invoked at all, so none of the 15 hard safety
counters (`autonomous_actuation_detected`, `human_approval_bypassed`,
`learned_ood_overrode_deterministic`, etc.,
`reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json`)
have any code path available to trip in either arm. The rule "prediction
does not imply operational permission" is exercised structurally, not
merely stated: the OOD/calibration gate above suppresses `proxy_actionable`
to exactly 0 on `ood-UNSEEN_TOPOLOGY` for both arms regardless of raw
prediction quality.

## Hypothesis outcome

**H1 (topology-relative feature augmentation improves unseen-topology
predictive quality): NOT SUPPORTED at this pilot's scale.**

- Top-1 on the unseen topology is **bit-for-bit identical** between arms
  (0.3750 on the same 280 examples; observed delta exactly 0.0, 90% CI
  [-0.0286, +0.0286]).
- Top-3 shows a small but 90%-CI-excludes-zero **regression**
  (-0.025, CI [-0.046, -0.004]) — reported honestly rather than tuned
  away, per the task's own requirement. MRR's CI includes zero.
- Known-family metrics show small, CI-includes-zero deltas in both
  directions — no evidence of harm, but also no evidence of the benefit
  H1 predicted.
- Calibration precision (ECE, mean set size) is slightly worse in the
  EXPERIMENTAL arm, though coverage itself is identical.

**H2 (actionability stays governed at its fail-closed floor on genuinely
unseen topology regardless of representation): CONFIRMED.** Both arms:
`proxy_actionable_rate=0.0`, `proxy_abstention_rate=1.0`,
`OODDetector` reports non-`NORMAL` for 100% of unseen-topology rows. This
experiment does not count that as a defect, per the task's requirement not
to treat increased actionability as an improvement unless legitimately
earned, and does not attempt to loosen it.

## Incidental finding (not fixed, out of scope)

`hydroswarm.training.losses._cross_entropy`'s all-invalid-in-batch
fallback (`logits.sum() * 0.0`, intended to be a graph-connected zero) can
itself return `NaN` when every logit in the batch is the model's own large
masked-out sentinel value (`~-3.4e38` for non-candidate nodes): summing
several such values overflows `float32` to `-inf`, and `-inf * 0.0 = NaN`.
Reproduces identically with `augmented=False` (i.e. on completely stock
`HydroCore` + stock loss code, unrelated to this experiment's own change) —
observed when a `batch_size=2` microbatch happens to contain two
NORMAL/SENSOR_FAULT_ONLY (no-real-source) examples together, more likely
on this pilot's small 600-example subsample than on the full campaign's
larger corpus. Worked around here (not fixed in production) by restricting
every split in this pilot to real-source examples
(`run_pilot.py::has_real_source`), which is also a legitimate scope
decision given this pilot studies source localization specifically. Left
for a future, separately-scoped fix; not touched by this branch.

## Known limitations

- Single seed, 6 epochs, 600 training examples — far short of M9.6's own
  3-seed/20-epoch/full-corpus campaign; point estimates carry real sampling
  noise (see CIs above), and a true small effect at H1's scale could be
  masked by this pilot's limited power, especially for top-1 where the
  outcome is binary per example.
- The actionability/abstention numbers are a bounded research proxy, not
  the production actionability certificate (see above) — do not read them
  as `actionable_rate` in the M11.6 locked-evidence sense.
- Calibration "condition" grouping here uses the corpus's own curriculum
  stage label (CLEAN/OPERATIONAL/DEGRADED/ADVERSARIAL/SHIFT) as a Mondrian
  key, not the exact `B_DEPTH_AWARE` scheme the frozen release's own
  calibration artifact uses — a simplification appropriate for a same-arm
  paired comparison, not a claim of matching the production calibration
  policy's exact behavior.
- Only one unseen topology family (coastal-branch) was evaluated; the
  frozen locked evidence's `locked_topology_test` covers 4 topologies and
  was never opened by this work.
- `event_presence_accuracy` and `evidence_sufficiency` were recorded but
  not deeply analyzed here (both arms track closely at this scale, ~0.70
  on the unseen split, ~1.0 on known splits) — a fuller Sentinel
  event/evidence-head comparison is left to future work if this line is
  continued.

## Recommendation

**REJECT this specific representation change (per-graph relative
augmentation of `FeatureScope.TOPOLOGY_RELATIVE` columns) as a promotion
candidate** — the measured effect on the metric it targets (unseen-topology
predictive quality) is zero-to-negative at this pilot's scale, with no
compensating calibration or known-family benefit.

Whether to **continue research** on the broader hypothesis space (topology-
size/network normalization more generally) is a separate question this
pilot does not resolve: the null result could reflect (a) the hypothesis
itself being wrong -- plausible, since the production model already trains
with interleaved multi-family data and its spatial backbone already
does genuine edge-indexed message passing, so the specific gap this
experiment targeted may simply not be where the remaining headroom is: or
(b) this pilot's small scale/single seed lacking the power to detect a real
but modest effect. Distinguishing these would require, at minimum, the
full M9.6-scale campaign (3+ seeds, full epoch budget) with this exact
representation change, which this session's compute budget does not
support. Given the top-1 result's near-exact null (not just "not
significant" but bit-for-bit identical on the same 280 examples) and the
statistically real top-3 regression, this is not a promising enough signal
to recommend spending that larger budget on this specific representation
change; a different angle (e.g. genuinely graph-aware structural
descriptors for `GraphStructuralEncoder`, which today still consumes no
`edge_index` at all despite its name) is a more speculative but arguably
better-motivated next step, not attempted in this branch.

**Not a candidate for future promotion** in its current form.
