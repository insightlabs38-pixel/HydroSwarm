# Pre-freeze audit — core-issues3.txt Phase 0

Machine-readable version: `reports/results/v4/pre-freeze-audit.json`.

## Head reconciliation

core-issues3.txt was audited against `237eb70`. The actual current HEAD at the
start of this pass is `64c6f24`, three commits ahead:

- `a9e4a1c` — landed the Cycle B2 trajectory corpus (core-issues2.txt Phase 7)
- `fc6f8fb` — core-issues2.txt completion report
- `64c6f24` — Scout/Strategist training-loop design note

None of these three commits touch scenario reconstruction, Strategist plan
value, or the OOD taxonomy split, so core-issues3.txt's problem statements
remain accurate against this HEAD. No prior work is repeated by this pass.

## Environment

Arm64 GCP VM (`aarch64`, 16 vCPU, 62 GiB RAM, 212 GiB free disk), Python
3.12.13, PyTorch 2.13.0, NumPy 2.5.1, WNTR 1.5.0 with the Arm64 EPANET patch
already applied. No live training/generation process was found running
(`ps aux` clean; every `experiments/jobs/*/status.json` reports
`state: COMPLETED`) — nothing needed to be preserved or resumed before
editing began.

## Baseline gates (recorded before any source file changed)

| Gate | Result |
|---|---|
| `pytest -q` | 513 passed, 0 failed (302s) |
| `ruff check src tests scripts` | all checks passed |
| `pyright` | 0 errors, 0 warnings, 0 informations |
| `run_corpus_gates.py --corpus-dir data/learning-v2/cycle-b2` | 9/9 passed |

## Completed requirements (verified against code, not just reports)

- Full `targets_v2` schema, Sentinel label generation and validation.
- Scout/Strategist/OOD/control/auxiliary label-generation libraries — real,
  unit-tested, and wired into a real corpus builder.
- 13,150-record trajectory corpus generated with zero generation errors.
- Stage 1 smoke/failure screening passed against the real merged corpus.
- E0/E1 Stage 3 finalist training and calibration jobs completed
  (`experiments/jobs/bundle-f-stage*`, `cycle-b2-stage3-E0`/`E1`, all
  `state: COMPLETED`, exit code 0).

## Partially completed requirements

- Scout/Strategist targets exist only as JSONL; no sequence-aware training
  loop consumes them yet (documented as the largest remaining piece in the
  prior handoff, and confirmed still true by inspection).
- Only 6 of 11 governed OOD categories are reproducible from this corpus's
  generator; the learned `ood_class` head is untrained.
- `evidence_sufficiency` implements only the sensor-health + entropy +
  OOD-validity subset of its governed definition — calibrated candidate-set
  size and classical/neural disagreement need a trained+calibrated Stage-A
  checkpoint (core-issues3.txt Phase 8's ordering dependency, not an
  oversight).

## Contradictions found between reports and code

**1. The trajectory corpus's own generator reuses one pristine network per
topology, not per scenario.** `scripts/generate_trajectory_corpus.py`
constructs exactly one WNTR network object and one `FeatureContext` per
training topology family (`networks[family]`, `contexts[family]`) and passes
the *same* object into `build_incident_trajectory` for every one of that
family's scenarios — discarding each scenario's own randomized demand
regime, roughness perturbation, tank-level variation, and pipe-outage state
that Cycle B2 itself was originally simulated against. This invalidates
travel-time labels, the Strategist's WNTR verification context, and any
state-dependent Scout artifact for all 13,150 records. This is precisely
core-issues3.txt Phase 1's problem statement, and precisely what restriction
#5 requires marking provisional. `build_incident_trajectory`,
`scenario_to_example`, and `build_strategist_trajectory` themselves already
correctly *accept and use* a per-scenario `network`/`feature_context`
argument (this was fixed for the plain Sentinel tensor path back in the
original `core-issues.txt` repair pass) — the defect is isolated to the
*caller*, not the shared trajectory-assembly code. This materially narrows
Phase 1's implementation scope for this pass.

**2. Strategist `plan_value` is not governed.** It is read from
`proposal.predicted_value * proposal.predicted_validity` — the old heuristic
prescreener's own score — not from exact WNTR consequence vectors relative
to the `NO_ACTION` comparator, per core-issues3.txt Phase 3.2. Tracked for a
later stage of this pass.

**3. `action_template` vocabulary size mismatch.** The deterministic planner
produces 9 templates; `HydroCore.action_head` defaults to
`action_vocabulary_size=8`. Already known and documented (not newly found),
tracked for Phase 3.5/9.

## Provisional artifacts (do not use for final training)

- `data/learning-v2/cycle-b2-trajectories/` (all four splits + the
  `tensors-enriched/` merge) — built from the pristine-context bug above.
- `experiments/runs/event-control-smoke/` — smoke-only, gitignored, never a
  promotion candidate.

## Remediation plan for this pass

1. **Phase 1** (this pass, next): add
   `hydroswarm/training/scenario_reconstruction.py::reconstruct_scenario_network()`
   as the single canonical replay/reconstruction function; refactor
   `scripts/run_corpus_gates.py`'s `gate_deterministic_replay` to call it
   instead of its own inline copy; fix `generate_trajectory_corpus.py` to
   reconstruct per-scenario network + feature context; add regression tests;
   regenerate a corrected corpus under `data/learning-v2/cycle-b2-trajectories-v2/`
   (restriction #4 — the old directory is left untouched).
2. **Phase 2**: formalize the signature-artifact policy actually in force
   (bucketed by topology family, fit from train-split scenarios only) as a
   documented, tested, versioned policy rather than an implicit choice.
3. **Phase 3 onward**: proceed in the order `core-issues3.txt` specifies —
   Strategist label semantics, candidate-conditioned architecture, closed-loop
   Scout, OOD taxonomy split, auxiliary/regression mask fixes, architecture
   v4 contract, staged training, promotion gates, runtime integration.

## Estimated expensive jobs

Regenerating the trajectory corpus at `cycle-b2-trajectories-v2` across all
four splits (~13,150 scenarios) at the previously observed ~0.5–1s/scenario
is projected at roughly 2–4 hours total. It will run as a resumable
background job, polled at a 10-minute interval, with idempotent
per-scenario-id resume.

## Locked-test discipline

The locked final test was not opened, listed, hashed, or referenced by any
command run during this audit. `final-selection.json` does not exist.
