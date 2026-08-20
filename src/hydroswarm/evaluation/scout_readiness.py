"""M10.2 preflight: checkpoint-governance / training-provenance audit for
HydroCore's raw Scout heads, as consumed by the frozen M9.6 selected
predictor.

Frozen correction document: `docs/evaluation/HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`.

## The finding this module records

`scout_control_heads=True` (part of `m9_1_common.SHARED_MODEL_CONFIG`,
which every M9.6 checkpoint was constructed with) makes
`sample_node_head`/`information_gain_head`/`candidate_reduction_head`/
`should_continue_sampling_head` real, present parameters in every M9.6
checkpoint, and `configs/training-v5-causal.yaml` declares a nonzero
`task_weights` entry for all four (`sample_node=1.0`,
`information_gain=0.5`, `candidate_reduction=0.5`,
`should_continue_sampling=0.5`). Neither fact means the heads were actually
trained.

`hydroswarm.training.causal_prefix.scenario_to_prefix_example` is the sole
source of the `targets` dict for every M9.6 training example
(`CausalPrefixDatasetView.__getitem__` calls it directly;
`scripts/hydrocore_v5/run_m9_6_train_arm_b.py` uses `CausalPrefixDatasetView`
unmodified, imported from `run_m9_0_arm_b`/`run_m9_0a_arm_b2`). Its
`targets = {...}` dict literal (`causal_prefix.py` lines ~255-279) contains
exactly 9 governed tasks (`source_node`, `source_region`, `start_time`,
`duration`, `relative_strength`, `event_presence`, `event_cause`,
`evidence_sufficiency`, `sensor_fault`) and NEVER includes `sample_node`,
`information_gain`, `candidate_reduction`, or `should_continue_sampling` --
empirically confirmed by calling the real function against a real generated
scenario (`tests/scientific/test_m10_2_scout_preflight.py::
test_m9_6_training_corpus_never_included_scout_targets`), not merely by
reading the source.

`hydroswarm.training.losses.compute_multitask_loss` only computes (and
therefore only backpropagates through) a task when
`task in targets and output_name in outputs` (both the classification and
regression loops use this exact guard). Since `sample_node`/
`information_gain`/`candidate_reduction`/`should_continue_sampling` were
never present in `targets` for any M9.6 training batch, this guard silently
skipped all four Scout tasks in every batch of every epoch of every seed's
M9.6 run -- proven directly, at the loss-function level, by
`tests/unit/test_scout_evaluation_state.py::
test_compute_multitask_loss_skips_a_task_absent_from_targets`.

Net effect: `sample_node_head`, `information_gain_head`,
`candidate_reduction_head`, and `should_continue_sampling_head` hold their
random initialization in every canonical FINAL_STEP_1350 M9.6 checkpoint.
This is NOT the same defect `SCOUT_STATE_SCHEMA_VERSION =
"scout-state-v1-unbuilt"` names (that placeholder is about a training-corpus
*input*-conditioning dataset layout that does not exist yet -- see
`hydroswarm.training.scout_state_contract`); this finding is about
*training-target* wiring, upstream of and independent from that placeholder,
and is a materially stronger blocker: even a perfectly-built evaluation-time
input schema (which this preflight does build --
`hydroswarm.evaluation.scout_state`) cannot make an untrained head's output
scientifically meaningful.

## Independent corroboration (not relied upon alone, and NOT reopened here)

M10.1 is closed and its results are not altered by this module. Its already
-recorded finding is independently consistent with this one: the
structurally analogous `ood_category` head is ALSO absent from this exact
same `targets` dict (`ood_class` is not one of the 9 keys either), and
`reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json` records
`"neural_ood_category_auroc": 0.5078...` -- indistinguishable from the
`0.5` an untrained 2-way-equivalent random head would produce. This module
does not depend on that number and would report the same Scout finding if
it did not exist; it is cited here only as corroborating context.

## What this module does NOT do

It does not retrain, fine-tune, or modify any checkpoint weight. It does
not change `configs/training-v5-causal.yaml`'s task weights or
`hydroswarm.training.causal_prefix`'s target dict. Fixing the underlying gap
(wiring real Scout targets into the training corpus and re-running training)
is explicitly out of scope for this preflight -- `scripts/train_scout_heads.py`
already documents a precedented, narrow, frozen-backbone approach to a
structurally identical past gap in the legacy Stage-A/Stage-D pipeline; the
same shape of fix applied to the M9.6-selected checkpoint would be a real
retrain (even if head-only/frozen-backbone) and is out of scope both for
this task ("Do NOT: retrain HydroCore-S ... tune Scout heads") and for the
M10 protocol's Section 1 (`M10 ... never retrained without an explicit
refit amendment`).
"""

from __future__ import annotations

from dataclasses import dataclass

from hydroswarm.training.output_governance import SCOUT_OUTPUTS

#: The exact governed target names M9.6 training would have needed present
#: in `scenario_to_prefix_example`'s own `targets` dict for
#: `compute_multitask_loss`'s `if task in targets and output_name in outputs`
#: guard to ever route a real gradient into any of the four raw Scout heads.
#: Equal to `SCOUT_OUTPUTS` today (the governed target name IS the task name
#: for every Scout task) -- kept as its own named constant, not a bare reuse
#: of `SCOUT_OUTPUTS`, so a future rename of one vocabulary does not silently
#: change the other's meaning without a visible diff here.
M9_6_REQUIRED_SCOUT_TARGET_KEYS: frozenset[str] = frozenset(SCOUT_OUTPUTS)

#: The exact keys `scenario_to_prefix_example` actually produces (frozen,
#: cross-checked against the real function by
#: `tests/scientific/test_m10_2_scout_preflight.py`). Recorded here,
#: separately from the test, so a report/artifact can cite the concrete
#: observed set without re-running scenario generation.
M9_6_OBSERVED_CORPUS_TARGET_KEYS: frozenset[str] = frozenset(
    {
        "source_node",
        "source_node_mask",
        "source_region",
        "source_region_mask",
        "start_time",
        "start_time_mask",
        "duration",
        "duration_mask",
        "relative_strength",
        "relative_strength_mask",
        "event_presence",
        "event_cause",
        "evidence_sufficiency",
        "sensor_fault",
        "sensor_fault_mask",
    }
)


@dataclass(frozen=True, slots=True)
class ScoutHeadTrainingAudit:
    checkpoint_label: str
    scout_heads_present: bool
    scout_heads_trained: bool
    required_scout_target_keys: tuple[str, ...]
    observed_corpus_target_keys: tuple[str, ...]
    missing_scout_target_keys: tuple[str, ...]
    finding: str


def _finding_text() -> str:
    return (
        "hydroswarm.training.causal_prefix.scenario_to_prefix_example is the sole targets "
        "source for every M9.6 training example (via CausalPrefixDatasetView, used unmodified "
        "by scripts/hydrocore_v5/run_m9_6_train_arm_b.py); its targets dict never includes "
        "sample_node/information_gain/candidate_reduction/should_continue_sampling. "
        "hydroswarm.training.losses.compute_multitask_loss's `if task in targets and "
        "output_name in outputs:` guard therefore skipped all four Scout tasks in every "
        "training batch, so candidate_reduction_head/should_continue_sampling_head/"
        "sample_node_head/information_gain_head hold their random initialization in the "
        "canonical FINAL_STEP_1350 checkpoints, despite scout_control_heads=True constructing "
        "them and despite configs/training-v5-causal.yaml declaring nonzero task_weight "
        "entries for all four. Consistent with (but independently derived from) M10.1's "
        "already-closed finding that the structurally analogous ood_category head -- also "
        "absent from this same targets dict -- scored AUROC 0.508 (~chance) in "
        "reports/evaluation/hydrocore-v5/m10/m10-1/m10-1-closure.json."
    )


M9_6_SCOUT_HEAD_AUDIT = ScoutHeadTrainingAudit(
    checkpoint_label="ARM_B_M9_6 FINAL_STEP_1350 (the M10-selected predictor, per "
    "docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md Section 1)",
    scout_heads_present=True,
    scout_heads_trained=False,
    required_scout_target_keys=tuple(sorted(M9_6_REQUIRED_SCOUT_TARGET_KEYS)),
    observed_corpus_target_keys=tuple(sorted(M9_6_OBSERVED_CORPUS_TARGET_KEYS)),
    missing_scout_target_keys=tuple(sorted(M9_6_REQUIRED_SCOUT_TARGET_KEYS - M9_6_OBSERVED_CORPUS_TARGET_KEYS)),
    finding=_finding_text(),
)

#: M10.2 preflight readiness outcomes (task's own required exact strings).
M10_2_READY_FOR_SCIENTIFIC_EVALUATION = "M10_2_READY_FOR_SCIENTIFIC_EVALUATION"
M10_2_PREFLIGHT_BLOCKED = "M10_2_PREFLIGHT_BLOCKED"


def m10_2_readiness(audit: ScoutHeadTrainingAudit = M9_6_SCOUT_HEAD_AUDIT) -> str:
    """A fair, leakage-safe evaluation-state schema and masking contract
    being buildable (and built, by `hydroswarm.evaluation.scout_state`) is
    necessary but not sufficient for M10.2 readiness -- the checkpoint whose
    outputs that schema would decode must also actually be trained. It is
    not, per `audit`, so this always currently returns
    `M10_2_PREFLIGHT_BLOCKED` for the real M9.6 audit; parameterized (rather
    than a bare module constant) so a test can exercise the
    would-be-ready branch against a hypothetical audit with
    `scout_heads_trained=True`."""

    if audit.scout_heads_present and audit.scout_heads_trained:
        return M10_2_READY_FOR_SCIENTIFIC_EVALUATION
    return M10_2_PREFLIGHT_BLOCKED
