# Section 18: One-Retrain Path for OOD, Scout, and PCGrad

core-issues5.txt Section 18 requires the frozen architecture to permit a
later, optional improvement to the learned OOD category head, learned
Scout, or PCGrad training, without a structural redesign, and without
altering current operational authority. This is an audit-and-close-gaps
pass, not a promotion of any of these three components -- all three
remain disabled/off-by-default exactly as before.

## 18.1 Learned OOD category

Audited and confirmed already satisfied, with one real gap closed:

- **Vocabulary frozen/checkpointed**: yes. `OODCategory` (11 categories,
  `hydroswarm/training/ood_categories.py`) is pinned into every
  `CheckpointIdentity` as `ood_category_names`/`ood_category_count`/
  `ood_category_schema_hash`, and `validate_checkpoint_identity` fails
  closed if a checkpoint's taxonomy drifts from the canonical order.
- **Advisory category separate from deterministic severity**: this did
  NOT exist before this pass. Added `SemanticPredictions.ood_category`
  (`hydroswarm/inference/results.py`) and wired it in
  `HybridInferencePipeline._model_semantics` (`hydroswarm/inference/
  pipeline.py`), following the exact same pattern as `event_cause`:
  gated by `runtime_enabled_outputs`, suppressed to `None` for any of the
  4 currently-unsupported categories (no real training examples yet),
  and never read by `OODDetector`/`ood_certificate`. `IncidentAnalysisResult.
  ood_level` (the deterministic 3-level severity) remains the sole
  authoritative OOD signal -- confirmed unchanged in
  `hydroswarm/inference/authority.py::ood_certificate`, which continues
  to source its value from `ood_level` only and lists
  `LEARNED_OOD_CATEGORY_SUPPRESSED:NOT_PROMOTED` regardless of whether
  the new advisory field is populated.
- **Class weights without architecture change**: yes, already true.
  `compute_multitask_loss`'s `class_weights` param passes straight to
  `F.cross_entropy(weight=...)` for the `ood_class` task -- a loss-level
  concern only.
- **OOD training populations decoupled from runtime schema**: yes, already
  true. `ood_labels.py` assigns categories purely from information already
  recorded on `GeneratedScenario`/`ScenarioManifest`.

Real checkpoint identity confirms today's state:
`ood_category_head=true`, `ood_category_count=11`, but
`trained_ood_categories=[]`/`validated_ood_categories=[]` -- the head
exists architecturally but has never been trained/validated in the
current checkpoint. A future one-retrain OOD improvement is unblocked by
architecture; it needs real training examples for the weak/missing
categories, which is a data-generation task, not a code change.

## 18.2 Learned Scout

Audited whether the runtime/data contract can represent the 6 required
items (already_sampled, current round, budget remaining, revealed
evidence, sampling constraints, current posterior) using existing
`HydroBatch` channels, with no new HydroCore parameters.

Finding: all 6 ARE representable via existing channels
(`sensor_mask`, `role_features`, `quality_features`+`sensor_mask`,
`residual_features`, `classical_prior`) -- no new parameters needed. Two
(`already_sampled` via `sensor_mask`, `current_posterior_candidate_state`
via `classical_prior`) are already wired by real code today. The other
four are architecturally available but not populated by any current
corpus-generation or pipeline code -- an honest wiring/data gap, not a
parameter-shape gap.

Per Section 18.2's explicit instruction ("define and version that
contract now"), added `hydroswarm/training/scout_state_contract.py`:
a frozen, content-hashed `SCOUT_STATE_CONTRACT` mapping each of the 6
items to its channel, encoding, and real wiring status
(`scout_state_unwired_items()` returns the 4 gaps honestly). This makes
the "one retrain, plus a corpus/pipeline wiring pass" scope explicit and
checkable, rather than assuming a bare "one retrain" claim that the
wiring gap would have quietly falsified.

## 18.3 PCGrad

Audited against the 6-item checklist; found and fixed one real gap, and
closed one real test gap:

- **Task weighting semantics preserved**: WAS FALSE. `Trainer.
  _pcgrad_backward` was called with `result.tasks` (unweighted per-task
  losses), silently discarding any configured `task_weights` (including
  the default 0.1x auxiliary-task downweighting) the instant PCGrad was
  enabled. Fixed: now called with `result.weighted`
  (`weights[task] * tasks[task]`), so PCGrad projects/combines the
  actually-intended weighted contributions. Covered by a new test,
  `test_pcgrad_respects_configured_task_weights_instead_of_silently_
  ignoring_them`.
- **Masked/no-supervision tasks don't distort projection**: confirmed
  already true. `compute_multitask_loss` only adds a task to `losses`
  (hence `result.tasks`/`result.weighted`) when both a target and a
  matching output are present in the batch.
- **Gradient accumulation restriction explicit**: confirmed already true.
  `Trainer.__init__` raises `ValueError` if `pcgrad_enabled` and
  `gradient_accumulation_steps != 1`.
- **Deterministic resume with PCGrad enabled**: WAS UNVERIFIED (no test
  existed). Added `test_pcgrad_resume_from_the_same_checkpoint_is_itself_
  deterministic`. Note: this trainer's actual resume contract is
  "resuming from a fixed checkpoint twice reaches the same result", not
  "resume matches an uninterrupted run" -- verified empirically that the
  latter does NOT hold even with PCGrad disabled (resume restarts the
  per-epoch dataloader/curriculum state rather than continuing a
  mid-epoch iterator), so that was not a valid property to assert. The
  test asserts the property the codebase actually guarantees.
- **Gradient diagnostics remain logged**: confirmed already true,
  independent of `pcgrad_enabled` (`gradnorm_logging`/
  `gradient_conflict_logging` config flags).

PCGrad remains `pcgrad_enabled=False` by default; nothing here changes
that.

## Files changed

- `src/hydroswarm/training/trainer.py`: `_pcgrad_backward` now receives
  `result.weighted`, not `result.tasks`.
- `src/hydroswarm/inference/results.py`: added
  `SemanticPredictions.ood_category`.
- `src/hydroswarm/inference/pipeline.py`: populate `ood_category`
  advisory field, governed by `runtime_enabled_outputs` and supported-class
  filtering.
- `src/hydroswarm/training/scout_state_contract.py` (new): frozen,
  versioned Scout state-conditioning channel-binding contract.
- `src/hydroswarm/training/__init__.py`: export the new contract names.
- Tests: `tests/scientific/test_training_smoke.py` (2 new PCGrad tests),
  `tests/scientific/test_hybrid_pipeline_v4_gating.py` (ood_category
  gating tests), `tests/unit/test_scout_state_contract.py` (new, 7 tests).
