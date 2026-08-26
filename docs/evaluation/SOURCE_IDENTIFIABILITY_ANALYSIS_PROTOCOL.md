# Source-identifiability analysis protocol

Branch: `exp/source-identifiability-analysis`. Status: experimental, post-hackathon,
analysis-only. **Trains nothing. Touches no checkpoint. Never opens, alters, or
re-executes the M11.6 locked evaluation** (`data/locked/m11-6/**`,
`reports/evaluation/hydrocore-v5/m11/m11-6*/**`) -- those are read-only
inputs. Never alters `models/hydrocore-v5-release/**`, `docs/CLAIMS_AND_EVIDENCE.md`,
or any other frozen v0.2.1 artifact.

## 0. Question

Before any graph-native/GNN localization architecture is attempted: are
HydroCore-v5's hard source-localization failures explained by (a) the
sensors genuinely being unable to distinguish the true source from a
competing candidate under the current network/sensor-layout/incident
conditions (**information-limited**), or (b) the learned representation
failing to exploit distinguishing evidence that is physically present
(**representation-limited**)?

## 1. Source-signature definition(s)

A "signature" is the vector of sensor observations HydroSwarm could in
principle see if a given candidate node were the source, holding the
network's current hydraulic state, sensor layout, timing, strength, and
duration fixed. Three defensible, complementary definitions are computed
(not one arbitrary choice):

1. **RAW** -- `log1p(concentration_mg_l)` at the incident's sensor nodes,
   at the incident's own sample timestamps. Same transform the model's own
   feature builder uses (`HydraulicFeatureBuilder`), so it is what
   HydroSwarm could "in principle observe." Entangled with source strength
   (nuisance variable) by construction.
2. **NORMALIZED (shape)** -- RAW divided by its own L2 norm (equivalently,
   peak-scaled). Marginalizes injected strength so comparisons reflect
   temporal/spatial *pattern* rather than magnitude. This is the
   strength-nuisance-controlled view.
3. **ARRIVAL-ORDER** -- per-sensor first-crossing time of a small absolute
   detection threshold (with an above-threshold-never-reached sentinel),
   giving a scale-invariant timing/activation-order fingerprint distinct
   from both RAW and NORMALIZED.

Within one incident, every candidate's signature is generated under
*exactly* the same randomized hydraulic state, timing, duration, strength,
and sensor set as the incident's real (true-source) simulation -- see
Section 6 for how that randomized state is reproduced deterministically
from the incident's own seed. This directly satisfies "identical/controlled
incident conditions" without needing a separate nuisance-marginalization
step at the pairwise-distance stage: strength/timing/demand are already
held fixed *within* an incident's own candidate pool. Cross-incident
stratification (by strength bin, condition_kind, depth bucket) is used
afterward to check whether identifiability itself depends on incident
severity.

No true-source IDs, hidden simulator internals, held-out labels, or
network-specific identifiers are used as features -- only forward
simulation of physically-realizable candidate hypotheses under already-known
incident metadata (timing/strength/sensor set), which is exactly the
counterfactual-signature-library computation the repo's own classical
localizer (`src/hydroswarm/classical/signatures.py`,
`src/hydroswarm/classical/prior.py`) already performs for real inference.

## 2. Normalization

RAW intentionally keeps strength un-normalized (so we can measure how much
strength alone helps/hurts separability). NORMALIZED intentionally removes
it. Reporting both, side by side, is how this analysis avoids begging the
question of whether strength is "part of the source's identity" or a
nuisance variable -- both readings are defensible and the task requires
checking both.

## 3. Candidate-pair distance metrics

- **Normalized RMSE / L2** on RAW and NORMALIZED signatures (matches the
  residual convention already used by `bayesian_source_posterior`).
- **Cosine distance** (`1 - cosine similarity`) on RAW -- scale-invariant
  without explicit re-normalization, cross-checks NORMALIZED+L2.
- **Correlation distance** (`1 - Pearson r`) on RAW -- invariant to both
  scale and additive offset, captures pure temporal/spatial co-pattern.
- Arrival-order distances use plain L1 distance over per-sensor delays
  (ties/sentinels handled explicitly).

DTW and Wasserstein are deliberately **not** used for signature-vs-signature
comparison: the observation windows are short, discretized, multi-sensor
tensors (not single long univariate series or two full distributions), so
neither has a physically well-motivated aggregation here, and the task
explicitly says not to add complex metrics for their own sake. Wasserstein
distance and entropy *are* reused, appropriately, in Section 7's
information-gain proxy, where the objects being compared genuinely are
distributions (posterior beliefs).

## 4. Source-level identifiability metrics (per incident, per candidate)

For the true source in each incident, against every other legal candidate
in the same incident's candidate pool:
- nearest / second-nearest competing-source distance,
- mean / median candidate separation,
- ambiguity count: candidates within (a) a physically motivated threshold
  tied to the sensor noise floor (`sensor_noise_std`, `quantization_step`
  from that incident's own generator config) and (b) a data-driven
  percentile threshold within that incident's own pairwise-distance
  distribution,
- margin = second-nearest minus nearest,
- a single **identifiability score** = nearest-competitor distance,
  normalized by that incident's own mean pairwise distance (so scores are
  comparable across networks/incidents of different scale).

These are computed per metric (RMSE/cosine/correlation) x per signature
definition (RAW/NORMALIZED/ARRIVAL-ORDER), then reported primarily under
NORMALIZED+correlation (least confounded by strength) with the others as
sensitivity checks.

A node-level (source-level, not incident-level) identifiability table is
also produced by averaging over incidents sharing that true source node,
but is explicitly flagged low-n per node given only ~125 confirmatory
incidents across 5 networks; group-level (centrality tercile, etc.)
comparisons are the statistically meaningful unit, not individual nodes.

## 5. Clean vs. noisy/stressed analyses

Two tiers, kept clearly labeled and never conflated:
- **Confirmatory (n=125, small, authoritative):** the frozen M11.6 locked
  evaluation's own `condition_kind` taxonomy (`NOMINAL`,
  `MEASUREMENT_NOISE`, `SENSOR_DROPOUT`, `LOW_COVERAGE_ACTIVE_SAMPLING`,
  `SENSOR_HEALTH_DEGRADED`, `SEVERITY_SHIFT`, `AMBIGUITY_DISAGREEMENT` for
  `locked_final_test`; `NOMINAL` only for `locked_topology_test`), joined
  against physical (clean-signature) identifiability computed here. Two of
  the seven conditions (`MEASUREMENT_NOISE` via `sensor_noise_std`,
  `SEVERITY_SHIFT` via `strength_bins`) map directly onto
  `ScenarioGenerationConfig` fields and are additionally re-degraded (not
  just clean) for a like-for-like stressed-signature comparison; the other
  four apply their perturbation through a mechanism outside
  `ScenarioGenerationConfig` (see Section 6 note) and are analyzed via
  their recorded outcome plus the condition-independent clean-signature
  identifiability, explicitly documented as such rather than silently
  approximated.
- **Exploratory (larger n):** a new, clearly-labeled-as-exploratory
  identifiability-only corpus generated on the same networks (Section 6),
  spanning many more sources/timings/strengths/seeds at both a `CLEAN` and
  a `MEASUREMENT_NOISE`/`SENSOR_DROPOUT`-style stress setting built
  directly from `WNTRScenarioGenerator`'s own native degrade parameters.
  This tier has no tied HydroCore-v5 prediction (no model is invoked) and
  is used only for Phase 3/4/7's physical-identifiability and
  oracle-vs-stress questions at higher statistical power, plus the m9-0a
  topology-generalization JSON's own richer per-candidate belief data
  (pre-final-architecture, used only as corroborating context, never as
  confirmatory HydroCore-v5 evidence).

## 6. Topology-held-out analysis

Uses the M11.6 locked split directly: `network_family="golden-reference"`
(`locked_final_test`, known/trained topology) vs.
`network_family="locked-topology-procedural"` (`locked_topology_test`,
genuinely unseen, audited-novel 9-12-junction procedural topologies). Per
incident, the exact randomized hydraulic state (demand regime, roughness
jitter, pipe outages, tank levels) and sensor subset are reproduced
deterministically by re-running `WNTRScenarioGenerator.generate_with_network`
with `ScenarioGenerationConfig(seed=<recorded seed>, source_node=<recorded
source>, **<recorded generator_config>)` against the correct base `.inp`
(`data/frozen/golden_network.inp` for golden-reference,
`data/locked/m11-6/topologies/locked-topology-{0..3}.inp` for the unseen
set) -- verified by recomputing `network_sha256` and comparing to the
recorded value before trusting any signature built from it. The returned
randomized (but source-free) WNTR model is then reused, unmodified, to
simulate every other junction as a counterfactual source under identical
conditions. This is read-only replay of already-frozen scenario specs, not
a re-run of the locked evaluation itself (no model inference, no touching
`m11-6-raw-incidents.jsonl`/gate/closure files).

## 7. Computational constraints

Per incident, building a full candidate signature library costs one
`HydraulicSimulator.simulate_incident` EPANET call per junction in that
network (4 for golden-reference, 9-12 for the locked procedural
topologies). Confirmatory tier: 125 incidents x ~6 candidates average
&asymp; under 1,000 EPANET calls. Exploratory tier is sized to stay in the
same order of magnitude (a few thousand calls total), using the existing
`SignatureCache`/on-disk caching so repeated runs of this analysis don't
re-simulate. Because `simulate_incident` returns concentration for *every*
network node (not just the incident's own sensor subset), Phase 7's
counterfactual "add one more sensor" analysis is answered by re-slicing
already-computed candidate traces -- **zero additional simulator calls**.

## 8. Leakage risks

- Never use the true source label as a feature of the signature itself
  (it is only used afterward, to label which candidate in the pool was
  correct).
- Never fit anything (no thresholds, no normalization constants) on the
  confirmatory 125-incident tier and then report metrics on that same
  tier as if independently validated; thresholds used for "ambiguity
  count" are either fixed a priori from generator-config noise parameters
  or computed per-incident from that incident's own distribution (never
  tuned against the outcome labels).
- Keep the exploratory (larger, self-generated) tier's seeds disjoint from
  every seed appearing in `data/locked/m11-6/**` and from every training
  seed family recorded in `experiments/registry/*.jsonl`, so nothing here
  can be mistaken for -- or accidentally leak into -- a future training
  corpus.
- Never treat this analysis's oracle/template numbers as a production
  safety or promotion claim; every output artifact is labeled
  `NON-PROMOTABLE / DIAGNOSTIC ONLY`, matching the existing convention in
  `m10-3b-oracle-utility.json`.

## 9. Expected deliverables

Scripts under `scripts/hydrocore_v5/source_identifiability/`, data tables
and JSON/markdown results under
`reports/evaluation/hydrocore-v5/source-identifiability/`, this protocol
document, and a final report answering every question the task poses,
plus a ranked recommendation (INFORMATION_LIMITED / REPRESENTATION_LIMITED
/ MIXED) with concrete next-step directions. All committed to
`exp/source-identifiability-analysis` only.
