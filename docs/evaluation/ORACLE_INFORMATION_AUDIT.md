# Oracle information audit (Task 1, `exp/candidate-conditioned-localizer-v1`)

Status: **experimental, post-hackathon, analysis-only**. Read-only audit of
`exp/source-identifiability-analysis` (merged into this branch's history as
its starting point). Never modifies M11.6 locked evidence, frozen model
artifacts, or that branch's own committed reports -- this document and the
new `fair_oracle.py`/`run_build_fair_oracle.py` modules it describes are
additive only.

## 0. Question

The source-identifiability analysis reported that a non-learned physics
oracle, replayed against the same real observed M11.6 incident evidence,
recovers the true source in **96.4% (54/56)** of HydroCore-v5's own Top-1
confirmatory failures -- the headline evidence for that branch's
REPRESENTATION_LIMITED conclusion. Before treating that number as
architectural motivation, this task requires verifying: **does the oracle
comparison actually use only information available to HydroCore-v5 at
inference time?**

## 1. What the oracle actually replays

`scripts/hydrocore_v5/source_identifiability/oracle.py::rank_candidates`
itself only consumes a `raw_signatures: dict[candidate, matrix]` and the
real `observed` sensor matrix -- it is a thin wrapper around the repo's
existing `hydroswarm.classical.prior.bayesian_source_posterior`, and by
itself does nothing privileged. The privilege enters one layer up, in how
`raw_signatures` is built:

- `library.py::build_incident_bundle` calls
  `common.simulate_candidate(incident, candidate)` for **every** candidate
  junction in the incident.
- `common.py::simulate_candidate`:
  ```python
  simulator = HydraulicSimulator(incident.randomized_network)
  return simulator.simulate_incident(
      candidate_node,
      strength_mg_min=incident.injection_strength_mg_min,
      start_minute=incident.start_minute,
      duration_minutes=incident.duration_minutes,
  )
  ```
  `incident.injection_strength_mg_min`, `incident.start_minute`, and
  `incident.duration_minutes` are not free parameters -- they come straight
  out of `common.py::reconstruct_incident`, which reads them from
  `incident.relative_strength`/`incident.start_minute`/`incident.duration_minutes`
  on `scenario.manifest.incident` (an `IncidentTruth`, `src/hydroswarm/
  data/scenarios.py`) -- **the frozen scenario's own ground-truth label**.
  Every candidate in an incident's pool is therefore simulated at the
  identical TRUE strength, TRUE start time, and TRUE duration -- the oracle
  never has to guess these; it only searches over candidate identity
  (location).
- `incident.randomized_network` (the same object reused, unmutated, across
  every candidate) is the post-`_randomize_hydraulics` WNTR model --
  i.e. also the incident's true drawn demand regime/roughness/tank-level
  realization, shared across every candidate.

The source-identifiability final report's own Section 2 already documents
this construction ("Every candidate ... is simulated under bit-for-bit
identical randomized hydraulics, timing, strength, demand ... nuisance
variables are held fixed by construction within each incident, not
marginalized after the fact") but never flags it as a fairness question
against HydroCore-v5's own available information, nor tests how much of the
96.4% figure depends on it. That is this task's gap to close.

### Checklist against the task's privileged-information list

| nuisance | used by the oracle? | source |
|---|---|---|
| exact true source strength | **yes** | `incident.injection_strength_mg_min` = `IncidentTruth.relative_strength` |
| exact injection start time | **yes** | `incident.start_minute` = `IncidentTruth.start_minute` |
| exact duration | **yes** | `incident.duration_minutes` = `IncidentTruth.duration_minutes` |
| exact hidden demand realization | **yes** | `incident.randomized_network`'s drawn `demand_regime`/roughness/tank state, shared unmutated across every candidate |
| simulator state unavailable to HydroCore | **yes** | a fresh full EPANET/WNTR chemical-transport solve per candidate, something HydroCore-v5 (a learned model that never calls the simulator at inference) does not do |
| any label-derived quantity | **yes** (strength/start/duration are literally `IncidentTruth` fields) | -- |

### Does HydroCore-v5 itself have access to these values?

No. `src/hydroswarm/model/core.py` predicts start time, duration, and
relative strength as **outputs**, not inputs:

```python
self.profile_heads = nn.ModuleDict({
    "start_time": RoleHead(d_model, 12),
    "duration": RoleHead(d_model, 8),
    "relative_strength": RoleHead(d_model, 4),
})
...
start_time_logits=self._profile_logits(self.profile_heads["start_time"], incident_context, 4),
duration_logits=self._profile_logits(self.profile_heads["duration"], incident_context, 3),
relative_strength_logits=self._profile_logits(self.profile_heads["relative_strength"], incident_context, 3),
```

`HydroBatch` has no field carrying true strength/start/duration into the
model. HydroCore-v5 must infer source location from the same noisy,
partial, real sensor observation the oracle also sees -- but, unlike the
oracle, without ever being told the true nuisance values, and without
running a physics simulator at inference at all.

One separate, unprivileged physics signal *is* already wired into
HydroCore-v5's own forward pass when `prior_mode` includes it: a
`classical_prior` per-node tensor, added as an input feature (`prior_mode
in ("feature_only", "feature_and_logit")`) and optionally as a logit bias.
Tracing where that number comes from (`hydroswarm.training.corpus.
model_input_classical_prior` -> `SignatureLibrary.posterior_from_observations`)
shows it is computed from the **observed sensor series only** -- a
genuinely different, non-privileged computation from `oracle.py`'s. The
released `models/hydrocore-v5-release/runtime_manifest.json` records
`prior_mode: "feature_only"`, so HydroCore-v5's M11.6 predictions already
had access to *some* fair physics signal and still produced the 56
confirmatory failures being audited here -- this is useful context for
Phase 1 (Section 3 of the companion plan doc), not itself a source of
privilege in the oracle comparison.

## 2. Classification

**PRIVILEGED ORACLE.**

The oracle's candidate replay uses the true strength, true start time, true
duration, and the true (shared) demand/hydraulic realization -- all four of
the task's listed privileged-nuisance categories, plus a simulator call
HydroCore-v5 never makes. It is not a SAME-INFORMATION comparison as
originally presented.

## 3. Fair, nuisance-searched correction

Per the task's requirement, this is corrected rather than treated as
grounds to discard the finding. New module: `scripts/hydrocore_v5/
source_identifiability/fair_oracle.py` (+ `run_build_fair_oracle.py`).

**Design.** For every candidate, instead of simulating once at the true
`(strength, start, duration)`, profile-search (minimum-residual /
maximum-likelihood) over the full cross product of

```
start_time_bins_min = (0, 60, 120, 240)      # minutes
duration_bins_min   = (30, 60, 120)          # minutes
strength_bins       = (0.5, 1.0, 2.0)        # relative strength
```

-- **verbatim copies of `ScenarioGenerationConfig`'s own field defaults**
(`src/hydroswarm/data/scenarios.py`), i.e. the population-level finite
support the generator itself draws each row's true value from. Using this
grid is not label leakage: it encodes "the source could have started at any
of these plausible times/strengths/durations," never which one actually
happened for this row, and is comparable in resolution to what
HydroCore-v5's own `start_time`/`duration`/`relative_strength` heads (4/8/4
size, i.e. actually finer) are themselves trained to discriminate among.
Every candidate is scored by its own best (lowest masked-RMSE-in-log1p-space)
grid point against the real observation -- the true value is never looked
up or used to pick the grid point.

**What remains privileged.** The candidate's own hydraulic/demand
realization (`incident.randomized_network`, produced once per incident and
shared across every candidate and grid point) is still held at its true
drawn value; marginalizing it too would require re-running
`_randomize_hydraulics` per grid point, a combinatorial EPANET-call cost
this pilot's budget could not absorb (see Section 5). This residual
privilege is documented, not hidden.

**Scope run.** All 56 confirmatory incidents where HydroCore-v5's own
recorded Top-1 was wrong -- the exact population the "96.4%" figure
describes (9,000 EPANET calls, ~892s wall-clock; see
`reports/evaluation/hydrocore-v5/source-identifiability/fair-oracle/
fair-oracle-results.jsonl` for the full per-incident record, including
which grid point won for every candidate).

## 4. Result

| | privileged oracle (original) | fair, nuisance-searched oracle |
|---|---|---|
| Top-1 (n=56, HydroCore-v5 failures) | 0.964 (54/56) | **0.964 (54/56)** |
| Top-3 | -- | 1.000 (56/56) |
| MRR | -- | 0.982 |
| failing incidents | seeds `3200843224878050951`, `3808993170199992408` (both `SENSOR_HEALTH_DEGRADED`, `golden-reference`, n_candidates=4) | **the identical two seeds** |

The fair oracle recovers the true source in **exactly the same 54 of 56**
incidents as the privileged one, and fails on **exactly the same two**. The
96.4% figure is **not an artifact of the privileged strength/start/duration
values** -- removing that privilege and searching instead reproduces the
identical outcome to three significant figures.

A secondary, unplanned finding from the per-candidate search log: the true
source's own best-fitting grid point matches the true `(strength, start,
duration)` triple in only **30.4% (17/56)** of incidents -- the other 69.6%
of the time, some other point in the plausible grid fits the true source's
signature at least as well as the true nuisance values do. Nuisance
parameters themselves are frequently NOT well pinned down by this sensor
evidence, even though **source location** is: this is itself informative
for Phase 1/Arm C (a candidate-conditioned model should not be expected to
recover exact strength/timing confidently even where it recovers location
confidently) and is consistent with HydroCore-v5's own comparatively weak
`start_time`/`duration`/`relative_strength` head performance being a
separate, harder, more genuinely information-limited problem than source
localization.

## 5. Decision

Per the task's instruction, the privileged construction does **not**
invalidate the experiment -- it is corrected, and the correction confirms
rather than overturns the original conclusion. Proceeding to Phase 1/the
candidate-conditioned architecture is justified using the **fair oracle**
(0.964 Top-1 on the failure subset, identical failing population) as the
oracle-gap denominator throughout the rest of this branch's evaluation, not
the original privileged number -- they happen to coincide here, but the
fair number is the one with a defensible claim to same-information
comparability and is what all `oracle_gap_closed` calculations downstream
reference.

**Residual limitation carried forward explicitly:** the fair oracle still
shares the true per-incident hydraulic/demand realization across
candidates (Section 3). This is a smaller, more defensible privilege than
strength/start/duration (it affects transport physics identically for
every candidate within one incident, rather than handing the oracle the
specific temporal/magnitude signature of the true injection), but it is
not zero, and no attempt was made in this pilot to quantify it directly (it
would need re-running `_randomize_hydraulics` per grid point x candidate, a
budget this pilot did not have). Any future higher-power run should budget
for that check before treating the oracle as a fully same-information
bound.

## 6. Reproducibility

```
python scripts/hydrocore_v5/source_identifiability/run_build_fair_oracle.py            # 56 failure-subset incidents, ~892s
python scripts/hydrocore_v5/source_identifiability/run_build_fair_oracle.py --all       # optional: full 125, not run for this audit
```

Output: `reports/evaluation/hydrocore-v5/source-identifiability/fair-oracle/
fair-oracle-results.jsonl` (one row per incident: fair/privileged Top-1,
rank, margin, and the full per-candidate winning grid point). New source
modules: `scripts/hydrocore_v5/source_identifiability/fair_oracle.py`.
Nothing under `data/locked/**`, `models/**`, or the
`exp/source-identifiability-analysis` branch's own committed files was
modified.
