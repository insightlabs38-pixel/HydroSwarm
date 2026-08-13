# Capability Diagnostic Protocol

Status: PRE-REGISTERED before any diagnostic results were generated.
Branch: `diag/capability-bottleneck`, based on `main` @
`f06642421f8bbeefe5615812b143d14cf10bcda8`.

## 0. Motivation and confirmed premise

The merged LIVE robustness/remediation work (`fix/live-robustness-findings`,
PR #11) fixed the ROB-LIVE-01/02 safety/authority findings. It did not
investigate *why* LIVE (real API-driven) source localization is much worse
than the frozen controlled evaluation. This diagnostic exists to answer that.

Before writing this protocol we verified the premise directly against
real, already-committed artifacts (not re-derived from memory):

| Metric | Controlled/frozen (`docs/MODEL_CARD.md`, `reports/results/v4/phase13-metrics-and-baselines.md`) | LIVE post-remediation (`reports/evaluation/live-robustness/post-remediation-summary.json`, 264 runs) |
| --- | ---: | ---: |
| top-1 | 0.7205–0.7331 | 0.306 |
| top-3 | 0.8680–0.8756 | 0.847 |
| MRR | 0.8113–0.8172 | 0.571 |
| planning-eligible rate | n/a (not a controlled-eval metric) | 0.012 |

Both `locked_test_opened=false` throughout the prior study; this diagnostic
preserves that.

One data point sharpens the puzzle further: the LIVE harness's own
**`nominal:clean_operational`** condition (n=12, the closest LIVE analog to a
clean controlled scenario) measures **top-1 = 0.000**, while the offline
`SEVERE_MISSINGNESS` development-holdout condition (a harder condition than
clean/nominal) measures **top-1 = 0.64–0.65**
(`reports/results/v4/phase13-metrics-and-baselines.md` line 76). A "clean"
LIVE incident performing far worse than a "severely degraded" offline one is
the single strongest a-priori signal that this is an evidence/input-contract
mismatch rather than a raw model-capacity problem — but this is a hypothesis
to test, not a conclusion, and Section 2 below still requires ruling out A–N
before considering O.

## 1. Hypotheses (A–P, from diagnostic.txt Section 1)

All 16 are in scope. Each will be marked SUPPORTED / PARTIALLY SUPPORTED /
NOT SUPPORTED / INCONCLUSIVE in the final report, with cited evidence file
and line/JSON-path. Do not jump to O (genuine model-capacity limitation)
before A–N are addressed.

## 2. Locked-test exclusion (hard constraint)

`reports/results/v4/architecture-freeze.json["locked_test_opened"]` is
`false` at protocol time. Every diagnostic script that touches evaluation
data MUST call the existing `locked_test_opened(repo_root)` guard (already
used by `live_robustness.py:128-130` and `robustness_scale.py:61`) and abort
if it is ever `true`. No diagnostic script will read, enumerate, or run
anything under a locked-test path. This is asserted before and after the
full diagnostic run (see `reports/evaluation/capability-diagnostic/baseline-identity.json`
for the before-value; the after-value is recorded in
`root-cause-summary.json`).

## 3. Data sources (predeclared)

To keep this diagnostic honest about compute cost, it explicitly reuses
already-collected real data wherever the existing artifact already answers
the question, and only runs new targeted experiments where no existing
artifact does. Every number in the final report is tagged with its source:
`REUSED:<path>` or `NEW:<script>`.

**Reused (already real, already committed, not regenerated):**

- `reports/evaluation/live-robustness/post-remediation-results.json` (264
  raw LIVE run records — used for Sections 14/15/16/18/19/27/28/29/37/38
  mining; every record already carries `neural_belief`, `classical_belief`,
  `fused_belief`, `suppression_reasons`, `ood_components`, `ood_level`,
  `disagreement_js`, `evidence_sufficient`, `observation_count`,
  `candidate_set_size`, `posterior_entropy`, `true_source_probability`,
  `reciprocal_rank`, `top1_correct`/`top3_correct`.)
- `reports/evaluation/live-robustness/post-remediation-summary.json` (same
  campaign, aggregated per-condition — 30 conditions across
  ambiguity/hydraulic_mismatch/measurement_bias/measurement_noise/
  missingness/nominal/scale/sensor_coverage/sensor_health/
  topology_familiarity).
- `reports/results/v4/pre-freeze-implementation-handoff.md` lines
  ~2264–2334 (real 300-scenario stride-sampled offline sampling-policy
  comparison: `classical_eig` vs `random` vs `fixed_order` vs
  `learned_scout` vs `classical_plus_residual`, both step-0
  realized-entropy-reduction and multi-step resolved-within-k). This
  already satisfies diagnostic.txt Section 31's minimum requirement
  (EIG vs random) on offline data; Section 30/33 still requires a fresh,
  LIVE-harness-specific check of whether the LIVE sampling loop's
  acquisition-time/observation-model assumptions match what this offline
  comparison assumed.
- `reports/results/v4/phase13-metrics-and-baselines.md` (topology-transfer
  and missingness-robustness numbers on the *controlled* path, used as a
  same-model, different-path comparison point against LIVE).
- `reports/results/v4/architecture-freeze.json` /
  `phase14-promotion-gates.md` (authoritative gating: source localization is
  advisory only; Scout/Strategist learned heads are disabled by measured
  evidence, not omission — directly relevant to Sections 14/23/30).

**New, generated this diagnostic (development-only, non-locked):**

- Train/serve parity tensors (Section 6): built from `GeneratedScenario`
  objects drawn from the existing `data/learning-v2/cycle-b2` **validation**
  split (never train/locked), fixed seed `20260813`, N=20 scenarios spanning
  at least 2 topology families.
- Temporal-evidence ablation (Section 8) and last-snapshot experiment
  (Section 9): same N=20 validation-split scenarios, evaluated at evidence
  depths `{full, 12, 6, 4, 3, 2, 1}` timesteps and causal prefixes
  `{1, 2, 3, ..., full}`, seed `20260813`.
- Pressure/sensor-series/network parity (Sections 11–13): deterministic
  single-incident constructions, seed `20260813`, plus a scan over the same
  N=20 parity scenarios.
- Calibration/conformal counterfactuals (Sections 24–26): recomputed
  strictly from the existing calibration-split tensors
  (`data/learning-v2/cycle-b2*/tensors-normalized/calibration`), never from
  validation or locked data.
- Confirmation holdout (Section 39): a NEW N=40 deterministic
  development-only scenario set, seed `20260899` (chosen to be visibly
  distinct from every seed already used elsewhere in this repo's frozen
  artifacts), generated by the existing `WNTRScenarioGenerator` /
  `generate_*_corpus.py` machinery, drawing only from already-governed
  training topology families (never locked, never train/calibration reused
  verbatim).
- Observability/oracle experiments (Sections 34–36): reuse the same N=20
  parity scenarios plus the existing golden-reference/loop-grid/
  branched-loop topologies already used by `live_robustness.py`.

Any experiment whose real result cannot fit in the session's compute/time
budget will be explicitly marked `NOT RUN — <reason>` in the final report.
It will never be filled with an invented number.

## 4. Production-behavior freeze (Section 4 of diagnostic.txt)

During discovery, the following are NOT modified: model weights/checkpoint,
preprocessing semantics, runtime evidence semantics, fusion policy,
calibration artifact/alpha, OOD thresholds, planning thresholds, sampling
algorithm, product APIs, WNTR behavior. All new code lives under
`scripts/capability_diagnostic/` and `reports/evaluation/capability-diagnostic/`
and is evaluation-only instrumentation. If a clear defect is found, it gets
a `CAP-XX` id, a minimized reproducer, and is left unfixed on this branch.

## 5. Metrics

Localization: top-1, top-3, MRR, posterior entropy, confidence, candidate-set
coverage/size. Decision utility: initial/1-sample/2-sample/3-sample
actionable rate, median samples-to-actionability. Sampling: EIG vs random
realized entropy reduction, rank improvement, candidate contraction. Safety:
authority violations, unsupported planning, verification bypasses (none
expected — these would trigger an immediate Section 47 stop).

## 6. Failure classification taxonomy

`CAP-DATA-XX`, `CAP-PARITY-XX`, `CAP-TEMPORAL-XX`, `CAP-FEATURE-XX`,
`CAP-CLASSICAL-XX`, `CAP-NEURAL-XX`, `CAP-FUSION-XX`, `CAP-CAL-XX`,
`CAP-OOD-XX`, `CAP-GATE-XX`, `CAP-SAMPLE-XX`, `CAP-TOPOLOGY-XX`,
`CAP-HARNESS-XX` — exactly as enumerated in diagnostic.txt Section 42. Not
every poor metric becomes a CAP id; only concrete, reproducible defects do.

## 7. Stopping rules

- If Section 5 (reproduction) fails to reproduce the documented controlled
  range, STOP, classify `CAP-EVAL-REPRODUCTION`, and do not proceed under
  the assumption the 72–73% figure is valid.
- Any Section 47 escalation condition (locked-test access, label leakage,
  permutation bug, tensor corruption, future-evidence use, wrong-network
  inference, incompatible calibration fit, safety/authority bypass,
  corrupted frozen artifact) halts and is reported immediately, not buried.
- The main experiment matrix (Section 3 above) is not changed after seeing
  results except via a documented `HARNESS_CORRECTION` entry (matching the
  existing `reports/evaluation/live-robustness/HARNESS_CORRECTION.md`
  convention already used in this repo).
- Section 40 (capacity scaling pilot) and Section 41 (data-diversity pilot)
  only run if the stated preconditions in diagnostic.txt actually hold after
  earlier sections complete — evaluated honestly, not assumed.

## 8. No-result-driven-tuning rule

No production threshold, weight, or config is changed based on diagnostic
findings during this pass. Findings feed a remediation-order recommendation
for a future, separate branch.
