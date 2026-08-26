# candidate-conditioned-localizer-v1: plan

Branch: `exp/candidate-conditioned-localizer-v1`. Follows
`exp/source-identifiability-analysis` and the oracle audit in
`docs/evaluation/ORACLE_INFORMATION_AUDIT.md` (read that first: the fair,
nuisance-searched oracle reproduces the original 96.4% Top-1
failure-recovery figure exactly, so the representation-limited motivation
for this branch survives the audit). **Experimental, non-release.** No
change to `models/hydrocore-v5-release`, `data/locked/`, any M11.6
artifact, or any governance module.

## 1. Where candidate identity currently enters the model

`src/hydroswarm/model/core.py::HydroCore.forward`:

```python
sentinel_nodes = role_hidden["sentinel"]          # [B, N, D], post-backbone
...
source_logits = self.source_node_head(sentinel_nodes).squeeze(-1)   # RoleHead(D, 1)
```

`source_node_head` is a **shared-weight per-node linear projection** applied
independently to every node's own post-backbone hidden state -- not a fixed
global classifier over anonymous output positions, and not tied to any
node's identity/index (no `nn.Embedding` over node ID anywhere in this
path). This is more topology-agnostic than the motivating framing implies
at face value.

The actual gap is *how* `sentinel_nodes[candidate]` comes to reflect sensor
evidence at all: entirely implicitly, through `LatentHydraulicBlock`
message passing over `edge_index` (`self.backbone`, `num_layers` stacked
layers, `layers.py`). A candidate many hops from the nearest sensor only
receives that sensor's evidence after `num_layers` rounds of local
aggregation -- there is no mechanism by which a candidate's own
representation ever *directly* compares itself against a specific sensor's
reading. `source_node_head` then scores whatever the backbone happened to
mix in, with no explicit "does this candidate's structural relationship to
sensor S match what sensor S actually observed" computation anywhere.

A second, unprivileged physics signal already exists in the model
(`classical_prior`, fused via `prior_projection` as an input feature and
optionally a logit bias when `prior_mode` includes it -- see the audit doc
Section 1) but is a single scalar per node from a fixed offline signature
library, not a candidate-vs-live-evidence comparison either, and the
release config (`prior_mode: "feature_only"`) already had it available for
the 56 failures being explained.

`CandidatePlanEncoder` (`candidate_plan_encoder.py`) is the closest
existing precedent for "build a per-candidate representation from the
candidate's own identity/features rather than an anonymous learned query" --
used for the Strategist's plan-scoring pass, not localization. Its
shared-MLP-over-candidate-features pattern (no per-candidate-ID parameter)
is the template this branch's new module follows.

## 2. Why the existing head may be topology-specific

It is *not* topology-specific in the naive "fixed output width tied to
training node IDs" sense (already ruled out above). The more defensible
version of the hypothesis: **receptive-field/over-squashing distance**.
With a fixed, small `num_layers` (the `small` variant used throughout this
pilot has few backbone layers), a peripheral candidate on an unseen,
possibly larger/differently-shaped topology may simply never receive enough
rounds of message passing to incorporate distant sensor evidence at all,
regardless of how well the *shared* per-node head's weights generalize.
This predicts exactly the pattern the source-identifiability analysis found:
failures concentrated in low-centrality/far-from-sensor candidates,
attenuating once identifiability itself is controlled for.

## 3. Candidate-conditioned tensors available (label-free, inference-time)

All computed once, per topology, by
`scripts/hydrocore_v5_experimental/candidate_conditioned_localizer_v1/
candidate_sensor_features.py` (cached the same way
`exp/graph-structural-encoder-v2`'s `structural_features.py`/
`observability_features.py` cache per-topology structural computations --
this corpus reuses a handful of topologies across thousands of examples):

- `candidate_hop_distance` `[B, N, N]`: unweighted all-pairs graph hop
  distance (`UNREACHABLE_HOP_SENTINEL = -1` for no path).
- `active_sensor_mask_nodes` `[B, N]`: reduction of the existing
  per-timestep `sensor_mask` to "has ≥1 valid reading this window."
- `candidate_structural_features` `[B, N, 6]`: degree/betweenness/closeness
  centrality + hop-to-nearest/mean-hop-to-sensor/fraction-within-2-hop
  (`NODE_STRUCTURAL_COLUMNS`).
- `candidate_physics_features` `[B, N, 3]` (Arm C only,
  `physics_features.py`): nearest-sensor observed peak log-concentration,
  hop-distance-vs-peak-magnitude compatibility, hop-distance-vs-arrival-time
  compatibility -- all derived from `temporal_features[..., 0]`
  (`log1p(concentration_mg_l)`, the existing channel-0 convention) and the
  hop tensor above. See Section 4 for what this deliberately does NOT do.

`sentinel_nodes` itself (already computed by the existing backbone) is
reused as both the per-candidate query base and the per-sensor evidence
key/value source -- no new encoder duplicates what `TemporalEncoder`
already does per node.

## 4. What Arm C's physics features do NOT do (scope reduction, documented)

The oracle audit's `fair_oracle.py` establishes the *rigorous* fair
comparator: a per-candidate EPANET nuisance-grid search against the real
observation. Running that inside a training loop over thousands of pilot
examples (each with its own randomized demand/hydraulics) is not affordable
here -- it would require a fresh grid-search EPANET replay per example, not
a one-time-per-topology precompute (unlike `candidate_sensor_features.py`,
whose inputs are pure graph structure and genuinely repeat across
examples). Arm C instead uses a **zero-EPANET-call arrival-pattern proxy**
computed directly from each example's own observed readings and the
label-free hop tensor (Section 3). This is explicitly a weaker,
cheaper stand-in for the audit's own nuisance-searched residual, not a
replication of it -- reported as a limitation, not hidden.

## 5. Planned ablation arms

| arm | `localizer_mode` | structural feats | physics feats | tests |
|---|---|---|---|---|
| A_CONTROL | `default` | -- | -- | H0 baseline (`source_node_head`, byte-identical to pre-branch HydroCore-v5) |
| B_CANDIDATE_CONDITIONED | `candidate_conditioned` | yes (6-dim) | no | H1, H2, H3 |
| C_PHYSICS_INFORMED | `candidate_conditioned` | yes (6-dim) | yes (3-dim) | H4 |

`source_logits` is computed **entirely** by the new
`CandidateConditionedLocalizer` scorer when its mode is active (not summed
with `source_node_head`'s own output) so H1 is tested without confounding
the new mechanism with the old one's residual contribution.
`source_node_head`'s parameters still exist (checkpoint-shape stability,
matching the `prior_mode`/`strategist_mode` convention already used
throughout `core.py`) but receive no gradient in that mode.

Arm D (graph-native message passing beyond candidate conditioning) is
**not implemented in this pilot** -- B/C are not structurally impossible
(the opposite: they train and run cleanly, Section 7), so per the task's
own instruction ("Do not make D the primary implementation unless B/C
prove structurally impossible") it is deferred pending B/C's own pilot
result, not attempted speculatively.

## 6. Parameter-count controls

Measured directly (`HydroCore.parameter_report_dict()`, `small` variant,
`node_feature_dim=19`, `edge_feature_dim=13`, `event_control_heads=True`):

| arm | total params | delta vs A_CONTROL |
|---|---|---|
| A_CONTROL | 4,044,113 | -- |
| B_CANDIDATE_CONDITIONED | 4,231,129 | +187,016 (+4.6%) |
| C_PHYSICS_INFORMED | 4,231,897 | +187,784 (+4.6%) |

A_CONTROL's count matches `exp/graph-structural-encoder-v2`'s own
A_CONTROL exactly (4,044,113) -- both branches build the identical `small`
variant with identical kwargs, a useful cross-branch consistency check.
+4.6% is a real, non-trivial capacity delta (above GSE-v2's own ~1%
capacity-control threshold) that this pilot does **not** control for with a
parameter-matched arm (Optional Arm E) -- compute budget (Section 8) did
not extend to a fourth full training run. This is reported as a limitation
in the final report, not silently absorbed into the headline comparison.

## 7. Leakage risks and how they are closed

- **Node-ID leakage**: no `nn.Embedding`/parameter indexed by node position
  anywhere in `CandidateConditionedLocalizer` -- verified directly by the
  permutation-invariance unit tests (`tests/unit/test_candidate_localizer.py`),
  not just by code inspection.
- **Topology-ID memorization**: candidate/sensor/hop features are computed
  fresh from `edge_index` per example, never cached across topologies by
  identity, and the module has no parameter whose count depends on node/
  topology count.
- **Train/test source-signature leakage**: `candidate_physics_features`
  reads only the CURRENT example's own observed sensor channel and
  label-free hop distances -- never another example's signature, never a
  precomputed-from-test-labels library.
  `candidate_structural_features`/`candidate_hop_distance` never read
  `source_node`/`source_node_mask`.
  Every feature-computation function's docstring states this contract
  explicitly and is exercised by dedicated tests.
- **Test-derived normalization**: none of the new features are normalized
  against corpus-wide statistics computed over eval splits; all
  normalization (hop/diameter ratios, log1p) is a fixed, per-example,
  data-independent transform.
- **Nuisance-parameter leakage**: Arm C's physics features never receive
  true strength/start/duration (Section 4; they are not consumed by this
  module at all, unlike the audited oracle).

## 8. Oracle-gap metric definition

For a population where both a fair-oracle accuracy and a HydroCore-v5 (here:
A_CONTROL, this pilot's own frozen-equivalent baseline) accuracy are
defined:

```
gap_closed = (experimental_accuracy - control_accuracy) / (oracle_accuracy - control_accuracy)
```

Reported only where `oracle_accuracy - control_accuracy` is non-negligible
(avoids a divide-by-near-zero blowing up the ratio) and where the oracle
figure being referenced is the **fair** one from
`ORACLE_INFORMATION_AUDIT.md`, never the original privileged figure. This
pilot's own corpus (`data/learning-v2/cycle-b2`) is a different population
from the M11.6 confirmatory set the oracle was computed on (Section 9 notes
this explicitly), so the primary oracle-gap analysis in the final report
uses the M11.6/fair-oracle numbers as the qualitative target and this
pilot's A-vs-B/C deltas as the architectural evidence, rather than
computing one single blended ratio across two different populations.

## 9. Training/compute budget and what this pilot can and cannot claim

Mirrors `exp/graph-structural-encoder-v2` exactly for direct comparability:
`data/learning-v2/cycle-b2/tensors-normalized`, seed `20260814`, 200
examples/family x 3 known families (golden-reference/branched-loop/
loop-grid) = 600 train examples, 6 epochs, CPU, `fp32=True`,
`deterministic=True`, `configs/training-v5-causal.yaml` optimizer
settings. Evaluated on `validation` (n<=300), `development_holdout`
(n<=300), `calibration`, and `ood-UNSEEN_TOPOLOGY` (`coastal-branch`).
3 arms x ~14 min/arm (extrapolated from GSE-v2's own 6-arm/~85min run) =~
45 min total, single seed.

**What this pilot cannot claim**, stated up front rather than discovered by
a reader: it does not retrain the actual `models/hydrocore-v5-release`
checkpoint or re-run the M11.6 locked evaluation, so it cannot directly
report whether Arm B/C would have recovered the specific 56 M11.6
confirmatory failures or the specific incidents the fair oracle succeeds
on -- those are M11.6-locked-set-specific outcomes tied to that exact
frozen checkpoint's training run. What it CAN report, on its own paired
corpus: whether candidate conditioning improves Top-1/Top-3/MRR overall,
on unseen topology, and on low-centrality/far-from-sensor subgroups
relative to A_CONTROL -- the architectural-level evidence for H1/H2/H3/H4
the task calls for. The final report treats the M11.6/oracle-audit numbers
and this pilot's own numbers as two distinct, clearly-labeled evidence
sources, never blended into one misleadingly precise combined statistic.

Single seed, single pilot scale: per the task's own instruction ("If the
pilot is negative, stop rather than brute-force hyperparameters unless
diagnostics reveal a clear implementation/training issue"), this is
explicitly NOT a claim of statistical robustness at M11.6-locked-evaluation
scale -- see the final report's limitations section.
