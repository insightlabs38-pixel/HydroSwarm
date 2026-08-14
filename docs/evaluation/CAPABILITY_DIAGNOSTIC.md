# Capability Diagnostic — Final Report

Diagnostic branch: `diag/capability-bottleneck`. Protocol pre-registered at
`docs/evaluation/CAPABILITY_DIAGNOSTIC_PROTOCOL.md` before any results were
generated. All raw artifacts are under
`reports/evaluation/capability-diagnostic/`. Every number in this report is
either measured directly by a script committed on this branch, or cited from
an already-committed real artifact (never fabricated, never estimated
without saying so).

## BASELINE

- main SHA: `f06642421f8bbeefe5615812b143d14cf10bcda8`
- diagnostic branch SHA (at time of writing): `e684dd76fab405edf60e05b173610d93b6fd1e74`
- model SHA: `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7` (confirmed byte-identical between `models/hydrocore-v4-release/model.safetensors` and `experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810/model.safetensors`)
- calibration SHA: `829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa`
- locked test remained unopened: **YES** (`locked_test_opened` checked and recorded `false` before/after every one of the ~30 scripts on this branch; see each report's `locked_test_opened`/`locked_test_opened_after` field)

## CONTROLLED REPRODUCTION

- documented top1: 0.7205–0.7331
- reproduced top1: **0.7205** (`reports/evaluation/capability-diagnostic/reproduction.json`)
- documented top3: 0.8680–0.8756
- reproduced top3: **0.8680**
- reproduced MRR: **0.8113**
- verdict: **REPRODUCED** — measured against the exact served checkpoint (not merely a same-lineage training export; the two differ byte-for-byte, confirmed and documented), 712/1000 validation examples eligible.

## TRAIN/SERVE PARITY

- exact-equivalent tensors match: **NO** (2 of ~15 tensor keys diverge; node order, edge features, temporal_features, classical_prior, and all masks match exactly)
- first divergent stage: `quality_features` (health channel) at missing timesteps
- affected features: `quality_features` health channel (CAP-PARITY-01), `node_features` current-health snapshot column on 4/20 scenarios (downstream of the same bug), temporal tensor length/window (CAP-PARITY-02)
- logit impact: not separately isolated at the logit level for these two specific bugs (small cell counts); their primary measured effect is on the feature tensors themselves, which is the standard this diagnostic held itself to per Section 6's "do not settle for comparing final logits only"
- CAP finding: **CAP-PARITY-01** (production health channel ignores `observation.missing`, defaults to healthy) and **CAP-PARITY-02** (production omits `window_steps`, silently caps evidence at 12 timesteps vs training's full 25)

## EVIDENCE CONTRACT

- training timesteps: 25 per sensor (24-hour span, 1-hour resolution — verified by direct scenario generation, not assumed)
- LIVE initial timesteps: **1** per sensor (last valid reading only — confirmed by direct reading of `hydroswarm.evaluation.live_robustness._payloads`, lines 204–238: `position = valid[-1] if valid else ...`)
- temporal span: training 86,400s; LIVE initial 0s
- history mismatch: **YES, severe** — HydroCore is trained on full 24-hour trajectories and served, in the LIVE robustness harness, exactly one snapshot
- performance full history: top1 0.95–1.00 (temporal-ablation.json / confirmation-holdout.json)
- performance 6-step (causal prefix): top1 1.00
- performance 3-step (causal prefix): top1 0.50
- performance 2-step (causal prefix): top1 0.40
- performance 1-step (causal prefix = latest snapshot only, i.e. the real harness's actual policy): top1 0.30 (main set) / **0.05** (independent confirmation-holdout set, N=20)
- verdict: **INPUT_EVIDENCE_REGIME_PROBLEM.** Critically, LATEST-k (a sliding window ending at the incident's last observation) stays flat/noisy for k=1..12 — it does NOT recover with "slightly more of the same kind" of evidence — while CAUSAL-PREFIX (a genuinely growing trajectory from the incident's onset) reaches top1=1.0 by just 6 steps. The real harness's "send only the latest valid reading" choice is close to the least informative single point available, not a representative snapshot, because the classical physics-based signature matching this system relies on needs the contamination-onset dynamics that only an early window captures.

## INPUT / FEATURE PARITY

- pressure mismatch: tested and **ruled out** as a material driver — fixed 25m pressure (exactly matching the real LIVE harness) gives top1=0.80, identical to true-WNTR-pressure's top1=0.80 (N=15)
- timestamp mismatch: none found in the timestamp-derivation logic itself (harness's per-observation `observed_at` computation is correct); the SPARSITY of what gets sent (1 reading) is the real issue, covered above
- node-order mismatch: **none in the deployed model** — `measure_equivariance` against the real frozen checkpoint shows ~1e-6 max diff and 30/30 prediction agreement across all 3 governed topology families, after correcting a real-but-dormant tooling bug (CAP-DATA-02, in unused `training/permutation.py` mask handling, confirmed to never execute in the real training loop)
- normalization shift: none material found — train/validation/calibration feature-channel distributions closely match (largest drift <0.08 std-units)
- classical-prior shift: coastal (unseen) topology's runtime-generated prior is **not degenerate** (JS divergence from governed-topology shape 0.0089, vs. 0.11–0.196 either-vs-uniform)
- network mismatch: **YES — major finding.** See COMPONENT/CALIBRATION/OOD sections below (CAP-DATA-01)
- other: sensor-series reconstruction (multi-sensor-per-node merging, grab samples, out-of-order timestamps) all behave correctly; no spurious duplicate-series defect

## COMPONENT PERFORMANCE (real LIVE dataset, 255/264 analyzable records)

```
                TOP1     TOP3    MRR
neural:         0.318    n/a†    n/a†
classical:      0.176    n/a†    n/a†
hybrid:         0.306    0.847   0.571
```
† top3/MRR not separately broken out per-component in this pass (top1 was the diagnostic priority per the protocol's fusion-decomposition focus); hybrid's top3/MRR are the real campaign aggregate.

- neural correct/classical wrong: 66/255 (25.9%)
- classical correct/neural wrong: 30/255 (11.8%)
- fusion harms correct component: **33/255 (12.9%)** — of these, 30 are classical-was-right/fusion-wrong and only 3 are neural-was-right/fusion-wrong
- verdict: the protocol's warned-against pattern ("neural good, hybrid bad → don't retrain neural") does **not** dominate; classical being overridden hurts more, but classical itself is the weaker branch overall (0.176 vs 0.318) under the current degraded-evidence regime, so fusion is not badly miscalibrated given what it's fed. Regret vs. best single component is small (−0.012 average). This is classified TERTIARY, not primary — likely to shrink once evidence-content and network-identity issues are fixed.

## CALIBRATION

- exact pristine applicability: calibrated_rate = 1.0 (5/5 fresh pristine-config scenarios)
- same-family randomized applicability: demand-only/tank-only stay calibrated (1.0); **roughness-only and full-hydraulic-randomization break it (0.0)** — narrower than "any non-exact config" — roughness specifically is hashed, demand/tank are not
- raw accuracy when calibration invalid: **~0.6 top1, statistically indistinguishable from when calibration is valid** — calibration-applicability is not tracking real localization quality at all in this comparison
- dominant applicability blocker: **CAP-CAL-01** — golden-reference's real production network hash is never in `validated_topology_hashes` because it's built two different ways (programmatic vs. `.inp`-file-loaded) that diverge by ~1e-9 relative float noise, enough to change the SHA-256. Confirmed directly against the real 264-run LIVE campaign: golden-reference (231/264, 87.5% of all runs) is **0% calibrated**; loop-grid (9/264, no construction-path asymmetry) is **100% calibrated**.
- counterfactual family-policy coverage: a "connectivity-family" policy (accepting any same-family hydraulic variation) would recover calibration validity for demand/tank-varied scenarios but not roughness-varied or genuinely unseen-topology scenarios; the CAP-CAL-01 fix (correcting the hash mismatch itself) is a cleaner, lower-risk remediation than loosening the applicability policy
- verdict: **CAP-CAL-01 (PRIMARY, tied) + CAP-CAL-02 (SECONDARY)**. CAP-CAL-02: the calibrator's per-network/per-condition Mondrian scores are unreachable at runtime (`network_id=str(getattr(network,'name','unknown'))` never matches the artifact's clean corpus-family keys, and `condition=` is never passed) — even calibration-valid incidents silently fall back to pooled global scores.

## OOD

- nominal false caution rate: **100%** — all 12 real LIVE `nominal:clean_operational` records show `ood_level=CAUTION`, never `NORMAL` (verified directly against `post-remediation-results.json` this session)
- unseen detection rate: coastal-branch correctly detected as novel (this part of OOD is working as intended)
- dominant components: `network_novelty` — confirmed by direct source reading (`src/hydroswarm/inference/ood.py:44-49,128`) that `topology_novelty()` returns 1.0 whenever the served network's hash isn't in `validated_network_hashes` (the SAME hash set CAP-CAL-01 examines), and a hard override (`if network_novelty > 0 and level == NORMAL: level = CAUTION`) forces at least CAUTION regardless of every other OOD component
- verdict: **CAP-OOD-01 — the identical root cause as CAP-DATA-01/CAP-CAL-01**, manifesting through a second, independent suppression gate. One float-precision construction bug simultaneously defeats both the calibration gate and the OOD gate for 87.5% of real LIVE traffic.

## SUPPRESSION

- initial actionable rate: 0.012 (3/264)
- within 1/2/3 samples: flat at ~0.011–0.023 (only 39/264 records have real per-round data; among those, only 3 records ever became actionable via sampling — see `suppression-analysis.json`'s `actionable_within_n`)
- blocker frequencies (of 255 analyzable): CALIBRATION_INVALID_OR_MISSING 96.5%, OOD_CAUTION 96.5% (the **exact same** 246-record set as calibration — structurally coupled via CAP-DATA-01/CAP-OOD-01), MODEL_EVIDENCE_INSUFFICIENT 82.4%, CANDIDATE_REGION_TOO_BROAD 63.5%, HIGH_CLASSICAL_NEURAL_DISAGREEMENT 4.7%
- blocker intersections: 63.5% of incidents trip all 4 major gates simultaneously — suppression is massively over-determined
- counterfactual actionability without each gate: removing calibration alone, candidate-size alone, evidence-sufficiency alone, or OOD alone each leaves eligibility at ~1.1% (identical to observed); only removing disagreement (rare) roughly doubles it to 2.3%. The idealized inference-quality-only ceiling (suppress only when actually wrong) reaches **29.5%**.
- dominant blocker: **CALIBRATION_INVALID_OR_MISSING and OOD_CAUTION, jointly, both downstream of CAP-DATA-01.** No single gate is individually "the" bottleneck because they co-fire almost universally — but they co-fire because they share the same upstream cause, which IS individually fixable.

## SAMPLING

- EIG runs: real classical EIG evaluated on 300 stride-sampled offline scenarios (`reports/results/v4/pre-freeze-implementation-handoff.md`, already-real, cited not re-run — this diagnostic verified it satisfies diagnostic.txt's minimum EIG-vs-random requirement)
- random runs: same 300-scenario comparison
- EIG median entropy change: step-0 mean realized entropy reduction −0.210 bits (counterintuitively negative; investigated in the source report, an open question, not silently smoothed over)
- random median entropy change: +0.007 bits
- EIG rank improvement / samples-to-sufficiency: **resolved-within-3 = 69.7% (classical_eig) vs. 61.0% (random)**, never-resolved 91/300 vs. 117/300 — EIG clearly beats random on the operational metric that matters for a promotion decision
- expected/realized IG correlation: not directly recomputed this pass (cited from the existing real report); the single-step sign anomaly above is the closest existing measurement
- sampling-time mismatch: **YES, real** — `rank_sample_locations` (`src/hydroswarm/sampling/active.py`) has no time-of-observation parameter at all and assumes Gaussian sample noise (std=0.05 mg/L); the real LIVE harness always samples at the end of the 24-hour simulated horizon with zero added noise. Empirically confirmed: with a reduced sensor count, every recommended sample's real concentration came back exactly 0.0 mg/L (decayed/null), though entropy still dropped ~0.88 bits from that informative null.
- verdict: **CAP-SAMPLE-01 (SECONDARY)** — the sampling *algorithm* is not the primary problem (it beats random on a fair operational metric); its assumptions about *when* a sample is taken diverge from this specific harness's behavior.

## OBSERVABILITY

- indistinguishable source groups: **none found** on golden-reference — closest candidate pair (J1/J2) has Euclidean signature distance 108.5 with clearly distinct peak magnitudes (61.9 vs. 126.1 mg/L)
- accuracy upper bound from available evidence: oracle-perfect classical prior improves top1 from 0.90 (real full-trajectory baseline) to 1.00 — a real but modest +0.10 ceiling, showing limited additional headroom once evidence content is already good
- verdict: current localization failures are **not** explained by fundamental physical unobservability on this network.

## TOPOLOGY

- pristine known: top1 1.0 (N=8, small sample)
- randomized same-family: top1 0.875
- known transfer (branched-loop, a trained family): top1 0.625
- coastal unseen: top1 0.75
- neural/classical/hybrid decomposition: not separately re-broken-out per topology this pass (covered at the LIVE-dataset level above); structural descriptors (junction/link count, degree, cycle rank) do **not** obviously track transfer quality at this sample size — branched-loop (trained) scored worse than coastal-branch (never trained), a small-N result flagged as noisy and possibly confounded by CAP-DATA-01-style construction issues on branched-loop too (not fully characterized this pass)
- verdict: topology-family identity is not a clean, monotonic driver of transfer quality in this data; the data-diversity pilot (Section 41) was correctly skipped as its precondition ("topology transfer poor but in-topology performance good") does not hold as stated, and this comparison should be re-run after CAP-DATA-01 is fixed.

## MODEL CAPACITY

- evidence model is underfit: **NO** — validation top1 (0.72) is not modest; no train-split number exists anywhere in this repo to compute a train/validation gap directly (stated honestly, not assumed)
- evidence model is overfit: **NO** — development_holdout losses under harder populations (SEVERE_MISSINGNESS, UNSEEN_TOPOLOGY) reflect genuine population difficulty, not a train/validation gap
- evidence capacity is primary bottleneck: **NO**
- scaling pilot run: **NO** — per diagnostic.txt Section 40's own stated preconditions, zero of five were met (train/serve parity does not pass; evidence semantics are not matched; calibration/fusion clearly dominate the LIVE utility loss; the model is not underfit) — skipped with explicit justification, not run reflexively and not run just to "check"
- verdict: classified **DISTRIBUTION-SHIFTED** (train/serve construction-path + evidence-content shift), consistent with every causal experiment run this session. Multi-task interference: 9/100 task-pairs show cosine similarity below −0.1, concentrated in the already-disabled Scout-head outputs (consistent with Scout's independently-documented underperformance) — not a signal that the promoted `source_node` head is antagonized by anything currently in production.

## NEW CAP FINDINGS

| ID | Severity | Evidence | Estimated Utility Impact |
|---|---|---|---|
| CAP-TEMPORAL-01 | HIGH | Direct causal experiment (temporal ablation) + independent confirmation-holdout replication | Dominant driver of the raw localization gap |
| CAP-DATA-01 | HIGH | 3 independent confirmations (2 subagents + hand-verification) + confirmation-holdout replication + exact match to real 264-run LIVE calibration pattern | Dominant driver, jointly with CAP-CAL-01/CAP-OOD-01, of the near-zero planning-eligibility gap |
| CAP-CAL-01 | HIGH | Same root cause as CAP-DATA-01; cross-checked against real LIVE dataset (231/264 golden-reference runs, 0% calibrated) | Same as above |
| CAP-OOD-01 | HIGH | Direct source-code confirmation of the hard CAUTION override; all 12 real nominal LIVE records show CAUTION | Same as above |
| CAP-PARITY-01 | MEDIUM | Direct tensor-level diff against real production code, 17/20 scenarios affected | Modest today, compounds if evidence sparsity is fixed |
| CAP-PARITY-02 | LOW-MEDIUM | Direct tensor shape divergence | Rarely binds today given current evidence sparsity |
| CAP-CAL-02 | MEDIUM | Direct source inspection of the real runtime call site | Secondary refinement, not blocking |
| CAP-SAMPLE-01 | MEDIUM | Source-code comparison + empirical demonstration (0.0 mg/L samples) | Affects marginal sampling value, not initial localization |
| CAP-DATA-02 | LOW | Root-caused and corrected locally; confirmed dormant in real training loop | None on current production behavior |

Full detail for every finding: `reports/evaluation/capability-diagnostic/root-cause-summary.json`.

## ROOT-CAUSE RANKING

1. **PRIMARY** — Evidence sparsity / temporal-evidence mismatch (CAP-TEMPORAL-01): the LIVE harness's single-last-reading evidence policy is close to worst-case for this system's classical signature matching.
2. **PRIMARY** — Network-identity construction mismatch cascading into calibration and OOD suppression (CAP-DATA-01 / CAP-CAL-01 / CAP-OOD-01): one float-precision hash bug defeats two independent suppression gates for 87.5% of real LIVE traffic.
3. **SECONDARY** — Train/serve construction-path defects (CAP-PARITY-01, CAP-PARITY-02): modest today, will compound once evidence sparsity is fixed.
4. **SECONDARY** — Calibration Mondrian-key mismatch (CAP-CAL-02) and active-sampling time/noise-model mismatch (CAP-SAMPLE-01).
5. **TERTIARY** — Fusion policy interactions (12.9% "fusion harms" rate, small net regret): likely a downstream symptom of degraded evidence/classical quality, not an independent defect worth tuning yet.
- **NOT SUPPORTED** — Neural model capacity.

## RECOMMENDED REMEDIATION ORDER

1. Fix CAP-DATA-01 at the source: use one consistent network-construction path (the real `.inp`-file-loaded path, since that's what production already uses) for scenario generation, corpus building, AND calibration/signature fitting, so hashes agree by construction rather than by coincidence. Re-fit calibration and signature artifacts against the corrected hash.
2. Fix the LIVE robustness harness's (and any real product ingestion path's) initial-evidence policy to send real recent history instead of a single last-valid snapshot (CAP-TEMPORAL-01) — no model or calibration change required.
3. Fix CAP-PARITY-01 (`_effective_sensor_health` should check `observation.missing`) and CAP-PARITY-02 (pass `window_steps` explicitly in `HybridInferencePipeline.analyze`'s feature-building stage) together, since both are small, localized, and will matter more once (1)–(2) land.
4. Fix CAP-CAL-02 (correct `network_id`/`condition` passed into `candidate_set()` so the real per-network/per-condition Mondrian calibration is actually reachable).
5. Re-examine CAP-SAMPLE-01 (align EIG's noise/time assumptions with real harness sampling behavior, or align the harness's sampling behavior with EIG's assumptions) once (1)–(2) make initial evidence realistic enough for sampling improvements to matter.
6. Re-measure fusion regret and topology-transfer patterns AFTER (1)–(4) land, before considering any fusion-weight or topology-diversity changes — both are likely confounded by the primary causes today.
7. Do not pursue a capacity-scaling pilot; re-evaluate only if, after (1)–(6), LIVE performance still falls meaningfully short of the controlled range.

## IDENTITY

- production changed: **NO**
- model changed during main diagnostic: **NO**
- calibration changed during main diagnostic: **NO**
- locked test opened: **NO**

## FINAL CONCLUSION

**PRIMARY BOTTLENECK = MULTIPLE INTERACTING COMPONENTS**, specifically two independently-dominant, both fully non-model, both low-remediation-cost causes:

1. **EVIDENCE SPARSITY** (CAP-TEMPORAL-01) — explains the bulk of the raw localization accuracy gap (controlled top1 0.72 vs. LIVE top1 0.31).
2. **A NETWORK-IDENTITY CONSTRUCTION DEFECT** (CAP-DATA-01, cascading into CAP-CAL-01 and CAP-OOD-01) — explains the bulk of the near-zero planning-eligibility gap (0.012), independent of (1), via two separately-firing but commonly-rooted suppression gates.

Neither is a neural-model-capacity limitation. Both are supported by direct causal experiments (not merely correlational observations) on the real frozen model and real production code paths, cross-checked against the real 264-run LIVE dataset, and replicated on an independent fresh confirmation holdout. This diagnostic recommends fixing both before considering any change to HydroCore itself, fusion weights, or calibration policy design.
