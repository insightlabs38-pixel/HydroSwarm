# core-issues.txt Phase 3 (rebuild and retrain) — completion report

Branch: `agent/gcp-multitopology-v3`, starting commit `b78515e` (end of the Phase 1+2
repair pass), ending commit `f7d5a82`. All work is committed and pushed; the working
tree is clean. **The locked final test has not been opened. `final-selection.json`
does not exist.** For full narrative detail, per-milestone results tables, and exact
resume commands, see `reports/results/v3/phase3-handoff.md` (the living log this report
summarizes); this document is the structured completion record core-issues.txt's Phase
3 section asks for.

## Scope

core-issues.txt Phase 3, items 13-20: regenerate the corpus with the Phase 1+2 repair
fixes in effect, fit real normalization, run all corpus gates, run a corrected
short E0/E1/E2 screen, select and fully train the strongest two, fit calibration on the
exact deployed dynamic hybrid predictor, re-run the HydroMono/no-adapter control, and
decide on HydroCore-M. All eight items are complete.

## Commits made (19)

| Commit | Summary |
|---|---|
| `3b58fa3` | Fit normalization from sharded Cycle A/B corpora (`--train-shards`) |
| `2ebb438` | Rebuild sharded corpus tensors with fitted normalization |
| `4e6ecdd` | Fix: write normalized shards to a sibling dir, not in place |
| `63ac528` | Add `run_corpus_gates.py`, the 9-gate pre-training suite |
| `5b6d805` | Parameterize Stage 2-4 scripts to target any corpus root |
| `0730303` | Add live Phase 3 handoff report |
| `e75f6a0` | Regenerate corrected Cycle B → `data/learning-v2/cycle-b2` |
| `7f3d840` | Docs: handoff update after corpus regen + gates |
| `1fbd878` | Select cycle-b2's top two finalists (E1, E0); per-finalist runs |
| `4872a7b` | Docs: handoff update after corrected E0/E1/E2 screening |
| `8308729` | Refactor: extract `build_sensor_series` (pure, no behavior change) |
| `ee5aa65` | Add `fit_dynamic_fusion_calibration.py` |
| `d446c7b` | Docs: handoff update after full finalist training |
| `a99cdbc` | **Fix: hydraulic-state snapshot time defect in live inference** |
| `4cfa1d9` | Diagnostics: per-branch top-1 correctness in calibration output |
| `2e46d92` | **Fix: widen calibration script's signature-artifact hypothesis grid** |
| `486c685` | Data: land finalist training + real dynamic-fusion calibration results |
| `712596a` | Docs: decide against training HydroCore-M |
| `f7d5a82` | Data: land HydroMono/no-adapter control results |

Full suite: **436 passed**, 0 failed (was 420 at session start; +16 new tests across
`test_fit_normalization.py`, `test_rebuild_normalized_shards.py`,
`test_run_corpus_gates.py`, `test_fit_dynamic_fusion_calibration.py`, and
`test_hybrid_pipeline.py`'s new regression test). `ruff check` and `pyright` clean on
every file touched, and on the full `src`/`scripts`/`tests` tree at completion.

## Defects fixed

Two real, previously-undiscovered defects were found and fixed while implementing item
18 (fitting calibration against the *actual* deployed predictor, not an approximation,
is what surfaced both — neither was visible to the prior `fixed_weight_fusion`
approach, which never re-ran real feature/classical-localization code):

1. **Hydraulic-state snapshot time mismatch (`a99cdbc`, source fix).**
   `HybridInferencePipeline.analyze()` called `simulator.calculate_state()` with no
   argument, defaulting to the network's *last* simulated timestep, while every
   feature-building path used at training/verification time
   (`hydroswarm.training.corpus.build_feature_context`, `hydroswarm.cli`) explicitly
   uses a 3,600s snapshot. For an 86,400s simulation this is a large, silent
   train/serve skew. Confirmed empirically against a real trained checkpoint (true
   source's neural belief: ~0.30 undifferentiated → ~0.49 correctly ranked once fixed).
   Fixed with a single named constant (`FEATURE_SNAPSHOT_TIME_SECONDS`) used by all
   three call sites; regression test added.
2. **Signature-artifact hypothesis grid too narrow (`2e46d92`, calibration-script-only
   fix).** Copying production's (`src/hydroswarm/runtime/defaults.py`) narrow
   `SignatureBuilder` bins into the calibration script produced a classical localizer
   that cannot represent this corpus's actual incidents. Confirmed with a direct
   20-scenario A/B: 65% classical top-1 with the narrow bins vs. 95% with bins spanning
   the corpus's real generation grid. Fixed in the calibration script only — **not**
   in production's `runtime/defaults.py`, which is a live-inference latency/coverage
   tradeoff for a human to decide, documented as an explicit follow-up recommendation
   below rather than changed unilaterally.

## Corpus, topology, and normalization identity

`data/learning-v2/cycle-b2` (seed 71000, same as the preserved `data/learning-v2/cycle-b`
— directly comparable, differing only in the corrected pipeline): 12,750 scenarios,
identical split/topology structure to the old corpus (9000/1000/1000/1750 train/
validation/calibration/development_holdout; golden-reference/branched-loop/loop-grid
training + coastal-branch held out; UNSEEN_TOPOLOGY/SEVERE_MISSINGNESS OOD categories at
400 each).

- `data/learning-v2/cycle-b2/dataset-report.json` — generation report.
- `data/learning-v2/cycle-b2/label-audit.json` — raw (pre-normalization) label audit.
- `data/learning-v2/cycle-b2/corpus-gates-report.json` — **all 9 corpus gates passed.**
- Normalization (fit from all 9,000 raw train examples):
  `node_normalization_sha256=4dcf22a68839a8630e83b0e90f47ac3400b176b576e76d8bee5662221d238691`,
  `edge_normalization_sha256=3e715d707475d81eba90de6609246f51bb0eee8a94c58eab4958f4370fca514d`.
  Independently reproduced by the `normalization_ownership` gate refitting directly
  from `tensors/train` alone.
- `tensors-normalized/` holds the final, governed training tensors; raw `tensors/`
  survives untouched (needed for the ownership gate above, and for any future re-audit).

## Screening, training, and calibration results

**Corrected E0/E1/E2 screening** (`reports/results/v3/cycle-b2-stage2-screening.json`,
identical seed/epochs/batch-size to the original): E1 (0.7473) > E0 (0.7463) >
E2 (0.7426). **Selected: E1, E0** (exactly two, per item 16's instruction — the
original cycle-b run had carried all three forward; this pass does not).

**Full finalist training**, both seeds each, all 16 epochs completed with real,
non-empty checkpoints (`reports/results/v3/cycle-b2-stage3-{E1,E0}.json`):

| Config | Seed | Checkpoint sha256 (model.safetensors) | val top1 | dev-holdout top1 |
|---|---|---|---|---|
| E1 | 20260810 | `051cfd94dec4a7ec61e559a1268b66acaada2d6248bda8c976846f9064ef3a23` | 0.7247 | 0.7093 |
| E1 | 20260811 | `4ae71f3b31c3e7d4e10667126aad5343d64dad513aa48c626c4e3fa42a5dd63a` | 0.7191 | 0.7142 |
| E0 | 20260810 | `04ada898f994c8cd54e12d65a7997256d80e5d6fb4c96a003f56e3492ad43580` | 0.7163 | 0.7215 |
| E0 | 20260811 | `c8f6a5e62a09264f653eec90854ca4934581348e05aa3c86cefc65cb5eee65df` | 0.7275 | 0.7142 |

E1 and E0 are effectively tied within seed noise; no single winner is crowned here
(that is `final-selection.json`/Stage 6 territory, out of scope for this pass).

**Real dynamic-fusion calibration** (best-val-top1 seed per finalist —
`E1-seed20260810`, `E0-seed20260811` —
`reports/results/v3/cycle-b2-dynamic-fusion-calibration-{E1,E0}.json`,
`fusion_config_hash=DYNAMIC_TRUST_FUSION_CONFIG` throughout, not
`fixed_weight_fusion_config(...)`):

| Checkpoint | Calibration artifact sha256 | Coverage (target 0.90) | Mean set size | ECE |
|---|---|---|---|---|
| E1-seed20260810 | `639384e86ce3c6ad30fb73914b08b8aa302337d77feb3472481e00c6d6cf040d` | 0.9143 | 2.636 | 0.0791 |
| E0-seed20260811 | `548009981c74a1d1c66c28936c1e66d65eca670638329c737101dad3d22a922f` | 0.9143 | 2.584 | 0.0809 |

**HydroMono/no-adapter control** (`reports/results/v3/cycle-b2-stage4-controls-training.json`):
both seeds completed all 16 epochs, val top1 0.7177/0.7177 — essentially identical to
the adapter-bearing finalists. Honest finding: no measurable adapter benefit at this
budget on the corrected corpus.

**HydroCore-M: not trained.** Both finalists show the same smooth, decelerating
validation-loss convergence curve across all 16 epochs (large early gains shrinking to
a small residual improvement) — the signature of converging within current capacity,
not an early capacity-starved plateau. No competing diagnostic run this pass points the
other way. Per the explicit "do not train M without a concrete capacity-based
justification," M is not trained.

## Comparison with preserved provisional Stage 2-4 results

`data/learning-v2/cycle-b` and `reports/results/v3/{stage2-architecture-screening,
stage3-finalist-training,stage4-controls-training}.json` (pre-repair-pass, already
self-flagged provisional) are **untouched**. The corrected numbers differ meaningfully:
val top1 in the corrected pass (~0.69-0.73 across screening and full training) is
noticeably lower than what the original pipeline's numbers implied — expected and
correct, not a regression: the corrected pipeline no longer credits the model for
predictions on padded/invalid positions, no longer lets an unpopulated topology
silently simplify the task, and (as of this pass) no longer silently mis-times the
live hydraulic-state snapshot or under-specifies the classical localizer's hypothesis
grid. The corrected numbers are the trustworthy ones.

## Follow-up recommendation (not applied)

Widen `src/hydroswarm/runtime/defaults.py`'s production `SignatureArtifact` hypothesis
grid (`start_time_bins=(0,60)`, `duration_bins=(30,60)`, `strength_bins=(0.5,1.0)`) —
per Defect 2 above, this measurably caps live classical-localization accuracy for any
real incident outside that narrow range. Left to a human to weigh (added
hypothesis-search cost vs. incident-coverage), not applied unilaterally here.

## Remaining known limitations

- Item 18's calibration is fit against `E1-seed20260810` and `E0-seed20260811` only
  (the higher-val-top1 seed per finalist), not all four trained checkpoints — a
  deliberate scope bound (each fit is a real, WNTR-backed re-simulation pass over the
  full calibration split); the other two checkpoints can be calibrated with the same
  command by swapping the `--checkpoint` path.
- Every limitation already recorded in `data/learning-v2/cycle-b2/dataset-report.json`
  carries forward unchanged (OOD holdout covers 2 of ~10 governed categories; no
  hard-negative curation; no `ood_class` per-example target; Scout/Strategist labels do
  not exist in this corpus).
- Scout, Strategist, and learned-OOD target generators still do not exist (unchanged
  from the Phase 1+2 repair pass; explicitly out of scope for Phase 3, in scope for the
  separate `core-issues2.txt` expansion pass, not started).
- The production signature-artifact-grid finding above is documented, not fixed.

## Exact commands to reproduce or resume every step

See `reports/results/v3/phase3-handoff.md` for the complete, copy-pasteable command
for every item (13 through 20), including the two job-runner run directories per
long-running job (`experiments/jobs/cycle-b2-*/status.json` + `job.log` + the exact
`resume_command` each recorded). Quick index:

```bash
export PYTHONPATH=src

# 13: corpus regeneration (already complete; this is the exact command used)
python scripts/generate_cycle_b_corpus.py --output data/learning-v2/cycle-b2 --seed 71000

# 14: normalization fit + rebuild (already complete)
python scripts/fit_normalization.py --train-shards data/learning-v2/cycle-b2/tensors/train \
  --node-output data/learning-v2/cycle-b2/normalization/node-normalization.json \
  --edge-output data/learning-v2/cycle-b2/normalization/edge-normalization.json
python scripts/rebuild_normalized_shards.py --corpus-dir data/learning-v2/cycle-b2 \
  --node-normalization data/learning-v2/cycle-b2/normalization/node-normalization.json \
  --edge-normalization data/learning-v2/cycle-b2/normalization/edge-normalization.json

# 15: corpus gates (already complete, all passed)
python scripts/run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2

# 15/16: screening (already complete)
python scripts/run_stage2_architecture_screening.py \
  --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
  --experiments E0 E1 E2 --run-root experiments/runs/cycle-b2-stage2 \
  --registry experiments/registry/cycle-b2-stage2.jsonl \
  --output reports/results/v3/cycle-b2-stage2-screening.json

# 17: full finalist training (already complete)
python scripts/run_stage3_finalist_training.py \
  --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
  --run-root experiments/runs/cycle-b2-stage3 \
  --registry experiments/registry/cycle-b2-stage3.jsonl \
  --output reports/results/v3/cycle-b2-stage3-finalist-training.json

# 18: real dynamic-fusion calibration (already complete for the two checkpoints above)
python scripts/fit_dynamic_fusion_calibration.py --corpus-dir data/learning-v2/cycle-b2 \
  --checkpoint <path-to-model.safetensors> --variant small --overrides-json '<json>' \
  --node-normalization data/learning-v2/cycle-b2/normalization/node-normalization.json \
  --edge-normalization data/learning-v2/cycle-b2/normalization/edge-normalization.json \
  --signature-cache-dir experiments/cache/signatures \
  --output <report.json> --calibration-artifact-output <calibration.json>

# 19: HydroMono/no-adapter control (already complete)
python scripts/run_stage4_controls_training.py \
  --corpus-root data/learning-v2/cycle-b2 --tensors-dirname tensors-normalized \
  --run-root experiments/runs/cycle-b2-stage4 \
  --registry experiments/registry/cycle-b2-stage4.jsonl \
  --output reports/results/v3/cycle-b2-stage4-controls-training.json
```

## Restrictions honored

No work on `main`. `data/learning-v2/cycle-b`, all previously-promoted checkpoints, and
the provisional Stage 2-4 result files are untouched. Locked test data was not opened;
`final-selection.json` does not exist. No destructive git/filesystem commands were used.
No sudo, no credential exposure. All commits pushed to `origin/agent/gcp-multitopology-v3`.
