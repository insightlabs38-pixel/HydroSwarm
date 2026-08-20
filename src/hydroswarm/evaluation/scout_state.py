"""M10.2 preflight: versioned Scout evaluation-state schema and canonical
learned-Scout decision adapter.

Frozen correction document: `docs/evaluation/HYDROCORE_V5_M10_2_PREFLIGHT_CORRECTION.md`.

## Why this module exists

`hydroswarm.training.checkpoint_identity.SCOUT_STATE_SCHEMA_VERSION =
"scout-state-v1-unbuilt"` names a *training-corpus* dataset layout (a
sharded, multi-round Scout-state dataset core-issues3.txt Phase 10 asked
for) that genuinely still does not exist -- this module does not change
that placeholder and does not claim otherwise. What this module DOES build
is a separate, narrower, EVALUATION-side contract: given a single decision
step's already-built `HydroCore.forward()` input (a `HydroBatch`, built by
the existing, already-leakage-audited causal-prefix pipeline for a fixed
evidence depth) plus explicit already-sampled/accessible/budget state, how
to (a) know exactly which raw-head output index maps to which physical
node, (b) mask out ineligible candidates the SAME way for every caller
(one canonical helper, not per-script ad hoc masking), and (c) interpret
the four raw Scout head outputs' exact meaning for M10.2 -- all without
adding a single new `HydroCore` parameter, changing any learned
computation, or touching the frozen M9.6 checkpoint weights.

Per the M10.2 preflight's own checkpoint-governance audit
(`hydroswarm.evaluation.scout_readiness.M9_6_SCOUT_HEAD_AUDIT`), the four
raw Scout heads exist in every M9.6 checkpoint (`scout_control_heads=True`)
but never received a real training gradient there -- this module's schema
and masking machinery are real and independently useful regardless (a
future retrain-only Scout improvement needs exactly this contract, per
`hydroswarm.training.scout_state_contract`'s own "one retrain, PLUS a
corpus/pipeline wiring pass" framing), but decoding a recommendation from
today's canonical checkpoint through it does not yet reflect a trained
policy -- see `LearnedScoutRecommendation`'s own docstring below.

## Leakage isolation (task requirement: adversarial leakage audit)

`build_scout_evaluation_state` accepts ONLY a `HydroBatch` (an INPUT-shaped
mapping) plus explicit, decision-time-only scalars/masks -- it has no
parameter through which a `ScoutLabel`/`ScoutTrajectoryStep`/targets_v2
target dict (all of which legitimately carry ground-truth/future/
realized-outcome fields in the SAME Python object as legitimate input
fields, per `hydroswarm.training.scout_trajectory`'s own module docstring)
could be passed in and silently read from. `assert_no_target_only_keys`
additionally fails closed at runtime if a caller's `HydroBatch` dict
happens to carry any key from targets_v2's governed target vocabulary
(a defensive, redundant check -- `HydroBatch`'s own declared field names
are already a disjoint namespace from targets_v2's target names, so this
should never trigger from a well-formed caller, but a future accidental
merge of an inputs dict with a targets dict must not be silently
tolerated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from hydroswarm.model.core import HydroBatch, HydroOutput
from hydroswarm.training.targets_v2 import TARGETS_V2

#: Independent from (and never assigned to) checkpoint_identity.py's
#: training-corpus-shaped `SCOUT_STATE_SCHEMA_VERSION` -- see module
#: docstring. This is the M10.2 evaluation-adapter contract's own version:
#: `ScoutEvaluationState`'s field set/semantics, `apply_scout_candidate_mask`'s
#: masking convention, and `LearnedScoutRecommendation`'s output semantics.
#: A future change to any of those three changes this string.
SCOUT_EVAL_STATE_SCHEMA_VERSION = "scout-eval-state-v1"

#: targets_v2's complete governed target vocabulary, plus the `_mask`
#: companion keys `hydroswarm.training.causal_prefix.scenario_to_prefix_example`
#: and `hydroswarm.training.scout_trajectory._scout_step_targets` attach
#: alongside them -- every one of these is a training LABEL (ground truth or
#: an offline-scoring/realized-outcome value), never legitimate model INPUT
#: at decision time. See `assert_no_target_only_keys`.
#:
#: EXCLUDES any name that is also a real, declared `HydroBatch` INPUT field
#: -- `"travel_time"` is both a governed auxiliary TARGET name (the
#: predicted-value task `travel_time_prediction` is trained against) AND a
#: real deterministic INPUT feature `HydraulicFeatureBuilder` populates
#: (`HydroBatch.travel_time`, already-known network travel times, not a
#: leaked label); a blanket name-based block would incorrectly reject every
#: real feature batch that legitimately carries it. No other target name
#: collides with a declared `HydroBatch` field today (see
#: `tests/unit/test_scout_evaluation_state.py::
#: test_target_only_key_set_excludes_every_real_hydrobatch_field`), so
#: excluding by name is sufficient without needing a channel-by-channel
#: allowlist.
_TARGET_ONLY_KEYS: frozenset[str] = (
    frozenset(TARGETS_V2) | frozenset(f"{name}_mask" for name in TARGETS_V2)
) - frozenset(HydroBatch.__annotations__)


class ScoutStateLeakageError(Exception):
    """Raised when a `HydroBatch` passed to `build_scout_evaluation_state`
    (or checked directly via `assert_no_target_only_keys`) contains a
    target-only key -- see module docstring's leakage-isolation section."""


def assert_no_target_only_keys(batch: Mapping[str, object]) -> None:
    leaked = set(batch) & _TARGET_ONLY_KEYS
    if leaked:
        raise ScoutStateLeakageError(
            "batch contains target-only (ground-truth/label) key(s) that must never reach "
            f"model input at Scout decision time: {sorted(leaked)}"
        )


@dataclass(frozen=True, slots=True)
class ScoutEvaluationState:
    """One batch of decision-time Scout states, ready to feed
    `HydroCore.forward()` and to interpret its raw Scout-head outputs
    against.

    `node_ids[b][i]` is the physical node name batch item `b`'s tensor
    position `i` refers to -- REQUIRED to be the same node space/ordering
    `batch["node_features"][b, i]` (and every other per-node HydroBatch
    channel) already uses, since this schema does not itself reorder or
    re-index anything; it only records, checks, and consumes that ordering
    (task requirement: "sample_node_logits index i must map deterministically
    to the same physical node ... no silent reorder mismatch"). Different
    batch items may have genuinely different topologies/node counts, padded
    to a shared `nodes` dimension the same way
    `hydroswarm.training.variable_collate.collate_variable_topology` already
    pads variable-topology training batches -- `node_ids[b]` for a padded
    position beyond that item's real node count may be any placeholder
    string; it is never read unless `node_mask`/`already_sampled_mask`/
    `accessible_mask` at that position happen to mark it a real, eligible
    candidate, which a correctly-built state never does (padding positions
    always carry `node_mask=False`).

    `already_sampled_mask`/`accessible_mask` are the two pieces of
    decision-time state this schema adds beyond what a plain `HydroBatch`
    already carries (`sensor_mask` already IS "was this node ever sampled"
    once a sample is folded into observations, per
    `hydroswarm.training.scout_state_contract`'s own frozen mapping, but
    that fusion happens upstream of this schema, in whatever built `batch`;
    this schema needs its OWN explicit already-sampled record so
    `candidate_mask()` can exclude already-sampled nodes from selection
    regardless of how -- or whether -- the caller's `batch` already reflects
    that fact in `sensor_mask`).
    """

    node_ids: tuple[tuple[str, ...], ...]
    batch: HydroBatch
    already_sampled_mask: Tensor
    accessible_mask: Tensor
    sampling_round: Tensor
    sample_budget_remaining: Tensor

    def __post_init__(self) -> None:
        assert_no_target_only_keys(self.batch)
        node_features = self.batch.get("node_features")
        if node_features is None:
            raise ValueError("batch must include node_features")
        if node_features.ndim != 3:
            raise ValueError(f"batch['node_features'] must be [batch, nodes, features], got shape {tuple(node_features.shape)}")
        batch_size, nodes, _ = node_features.shape
        if len(self.node_ids) != batch_size:
            raise ValueError(f"node_ids has {len(self.node_ids)} batch entries, batch has {batch_size}")
        for item_index, ids in enumerate(self.node_ids):
            if len(ids) != nodes:
                raise ValueError(
                    f"node_ids[{item_index}] has {len(ids)} entries, batch node dimension is {nodes}"
                )
        for name, tensor in (
            ("already_sampled_mask", self.already_sampled_mask),
            ("accessible_mask", self.accessible_mask),
        ):
            if tensor.shape != (batch_size, nodes):
                raise ValueError(f"{name} must have shape {(batch_size, nodes)}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean tensor, got dtype {tensor.dtype}")
        for name, tensor in (
            ("sampling_round", self.sampling_round),
            ("sample_budget_remaining", self.sample_budget_remaining),
        ):
            if tensor.shape != (batch_size,):
                raise ValueError(f"{name} must have shape {(batch_size,)}, got {tuple(tensor.shape)}")
        node_mask = self.batch.get("node_mask")
        if node_mask is not None and node_mask.shape != (batch_size, nodes):
            raise ValueError(f"batch['node_mask'] must have shape {(batch_size, nodes)}, got {tuple(node_mask.shape)}")

    @property
    def batch_size(self) -> int:
        return len(self.node_ids)

    @property
    def nodes(self) -> int:
        return len(self.node_ids[0]) if self.node_ids else 0

    def candidate_mask(self) -> Tensor:
        """Task requirement 3 ("candidate eligibility"): valid AND not
        padding AND not already-sampled AND accessible. This is the ONE
        place that definition lives -- `apply_scout_candidate_mask` and
        `select_candidate_node` both consume this, rather than each
        re-deriving their own notion of "eligible"."""

        node_mask = self.batch.get("node_mask")
        base = node_mask.bool() if node_mask is not None else torch.ones_like(self.already_sampled_mask)
        return base & ~self.already_sampled_mask.bool() & self.accessible_mask.bool()


def build_scout_evaluation_state(
    *,
    node_ids: Sequence[Sequence[str]],
    batch: HydroBatch,
    already_sampled: Sequence[Sequence[str]],
    accessible: Sequence[Sequence[str]] | None = None,
    sampling_round: Sequence[int],
    sample_budget_total: Sequence[int],
) -> ScoutEvaluationState:
    """Canonical adapter: given per-item node identity/already-sampled/
    accessibility as plain node-id sequences (the natural shape a caller
    already has -- e.g. from `IncidentState.observations`/an operator's
    accessibility constraint list), build the boolean tensors
    `ScoutEvaluationState` itself needs. `accessible=None` means every real
    node is accessible (matches `hydroswarm.sampling.active.
    SamplingConstraints`'s own `accessible: Mapping[str, bool] = {}` default
    -- an empty/omitted constraint means unconstrained, per that module's
    existing convention, not "nothing is accessible")."""

    assert_no_target_only_keys(batch)
    batch_size = len(node_ids)
    if len(already_sampled) != batch_size or len(sampling_round) != batch_size or len(sample_budget_total) != batch_size:
        raise ValueError("node_ids, already_sampled, sampling_round, and sample_budget_total must all have the same batch length")
    if accessible is not None and len(accessible) != batch_size:
        raise ValueError("accessible, if provided, must have the same batch length as node_ids")

    already_sampled_mask = torch.zeros(batch_size, len(node_ids[0]) if node_ids else 0, dtype=torch.bool)
    accessible_mask = torch.ones_like(already_sampled_mask)
    for item_index, ids in enumerate(node_ids):
        sampled_set = frozenset(already_sampled[item_index])
        inaccessible_set = frozenset(ids) - frozenset(accessible[item_index]) if accessible is not None else frozenset()
        for position, node_id in enumerate(ids):
            already_sampled_mask[item_index, position] = node_id in sampled_set
            accessible_mask[item_index, position] = node_id not in inaccessible_set

    return ScoutEvaluationState(
        node_ids=tuple(tuple(ids) for ids in node_ids),
        batch=batch,
        already_sampled_mask=already_sampled_mask,
        accessible_mask=accessible_mask,
        sampling_round=torch.tensor(list(sampling_round), dtype=torch.int64),
        sample_budget_remaining=torch.tensor(
            [max(0, total - len(already_sampled[i])) for i, total in enumerate(sample_budget_total)],
            dtype=torch.int64,
        ),
    )


#: The three raw per-node Scout outputs `apply_scout_candidate_mask`
#: canonically masks, and the fill value used for each -- `-inf`-equivalent
#: (matching `HydroCore.forward()`'s own `torch.finfo(dtype).min` convention
#: for logits an argmax must never select) for the logit, `0.0` (matching
#: `HydroCore.forward()`'s own node_mask convention for the two regression
#: heads) for the two bounded-regression outputs.
_MASKED_LOGIT_OUTPUT = "sample_node_logits"
_MASKED_ZERO_OUTPUTS = ("expected_information_gain", "candidate_reduction_prediction")


def apply_scout_candidate_mask(output: HydroOutput, candidate_mask: Tensor) -> dict[str, Tensor]:
    """THE single canonical post-forward-pass masking helper (task
    requirement: "implement a single canonical helper rather than having
    each M10.2 script invent masking logic"). `HydroCore.forward()` already
    masks every one of these three outputs by `node_mask` (real vs. padding)
    internally -- this helper applies the SAME masking convention again,
    narrowed to `candidate_mask` (which additionally excludes already-sampled
    and inaccessible nodes, neither of which `node_mask` itself encodes),
    so an already-sampled or inaccessible node can never win an argmax over
    the returned tensors regardless of how large its raw logit was.

    Returns a NEW dict (never mutates `output`) containing only the keys
    that were both present in `output` and eligible for masking here.
    `should_continue_sampling_logits` is intentionally absent -- it is a
    single incident-level scalar per batch item (from `incident_context`,
    not per-node), so no per-node candidate mask applies to it.
    """

    if _MASKED_LOGIT_OUTPUT not in output:
        raise KeyError(
            f"output has no {_MASKED_LOGIT_OUTPUT!r} -- the model was not constructed with the raw "
            "Scout heads present, or this is not a real HydroCore.forward() output"
        )
    mask = candidate_mask.bool()
    masked: dict[str, Tensor] = {}
    logits = output[_MASKED_LOGIT_OUTPUT]  # type: ignore[literal-required]
    if logits.shape != mask.shape:
        raise ValueError(
            f"output[{_MASKED_LOGIT_OUTPUT!r}] has shape {tuple(logits.shape)}, candidate_mask has shape "
            f"{tuple(mask.shape)}"
        )
    masked[_MASKED_LOGIT_OUTPUT] = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    for name in _MASKED_ZERO_OUTPUTS:
        if name not in output:
            continue
        value = output[name]  # type: ignore[literal-required]
        if value.shape != mask.shape:
            raise ValueError(f"output[{name!r}] has shape {tuple(value.shape)}, candidate_mask has shape {tuple(mask.shape)}")
        masked[name] = value.masked_fill(~mask, 0.0)
    return masked


def select_candidate_node(
    masked_sample_node_logits: Tensor, candidate_mask: Tensor, node_ids: Sequence[Sequence[str]]
) -> tuple[list[int | None], list[str | None]]:
    """Fail-closed argmax: a batch item with no eligible candidate at all
    (`candidate_mask[b]` all-False -- every node padding, already sampled, or
    inaccessible) returns `(None, None)` for that item rather than an
    arbitrary node, matching `HydroScout.deterministic_fallback`'s own
    "no unsampled accessible nodes -> STOP" behavior rather than silently
    picking position 0."""

    mask = candidate_mask.bool()
    has_candidate = mask.any(dim=-1).tolist()
    indices = masked_sample_node_logits.argmax(dim=-1).tolist()
    result_index: list[int | None] = [idx if has else None for idx, has in zip(indices, has_candidate)]
    result_id: list[str | None] = [
        node_ids[item][idx] if idx is not None else None for item, idx in enumerate(result_index)
    ]
    return result_index, result_id


def assert_finite_scout_outputs(output: HydroOutput) -> None:
    """Task requirement: "outputs remain finite". Checked over the raw
    (pre-mask) tensors -- `apply_scout_candidate_mask` itself can only ever
    replace values with `torch.finfo(...).min` or `0.0`, both finite, so a
    finite raw output implies a finite masked output; the converse is not
    guaranteed, which is exactly why this checks the raw tensors."""

    names = (_MASKED_LOGIT_OUTPUT, *_MASKED_ZERO_OUTPUTS, "should_continue_sampling_logits")
    for name in names:
        if name not in output:
            continue
        value = output[name]  # type: ignore[literal-required]
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"output[{name!r}] contains a non-finite value")


@dataclass(frozen=True, slots=True)
class LearnedScoutRecommendation:
    """M10.2 output semantics (task requirement 7 -- exact meaning of each
    raw head, for this preflight and for any future scientific comparison
    built on top of it):

    - `node_id`/`node_index`: the masked-argmax candidate ranking signal --
      usable ONLY as a relative ranking among currently-eligible candidates.
      NEVER authoritative: `hydroswarm.inference.authority.scout_certificate`
      hardcodes `source="CLASSICAL_EIG"`/`authority=AuthorityLevel.DETERMINISTIC`
      and takes no learned-recommendation parameter at all (see
      `tests/unit/test_scout_evaluation_state.py::
      test_scout_certificate_signature_has_no_learned_recommendation_input`)
      -- nothing in this module changes, bypasses, or is read by that
      certificate.
    - `expected_information_gain`/`candidate_reduction_prediction`/
      `should_continue_sampling_probability`: diagnostic-only, for every
      M9.6 canonical checkpoint today. Per this preflight's checkpoint-
      governance audit (`hydroswarm.evaluation.scout_readiness.
      M9_6_SCOUT_HEAD_AUDIT`), none of the four raw Scout heads received a
      real training gradient in the M9.6 run that produced the canonical
      FINAL_STEP_1350 checkpoints -- these values reflect each head's
      random initialization, not a learned policy, in every checkpoint M10
      is restricted to using. Report them for structural completeness only;
      never interpret them as if they reflect a trained Scout policy.
    - `promotable`: always `False` from `decode_learned_scout_recommendation`
      -- a frozen dataclass field with no setter, so nothing downstream of
      this module can flip it. A real promotion decision belongs to a
      separately authorized M10.2 scientific-comparison milestone, never to
      this preflight.
    """

    node_index: int | None
    node_id: str | None
    sample_node_logits: Tensor
    expected_information_gain: float | None
    candidate_reduction_prediction: float | None
    should_continue_sampling_probability: float | None
    promotable: bool = False


def decode_learned_scout_recommendation(
    output: HydroOutput, state: ScoutEvaluationState, *, batch_index: int = 0
) -> LearnedScoutRecommendation:
    assert_finite_scout_outputs(output)
    mask = state.candidate_mask()
    masked = apply_scout_candidate_mask(output, mask)
    indices, ids = select_candidate_node(masked[_MASKED_LOGIT_OUTPUT], mask, state.node_ids)
    index = indices[batch_index]
    node_id = ids[batch_index]
    gain = (
        float(masked["expected_information_gain"][batch_index, index])
        if index is not None and "expected_information_gain" in masked
        else None
    )
    reduction = (
        float(masked["candidate_reduction_prediction"][batch_index, index])
        if index is not None and "candidate_reduction_prediction" in masked
        else None
    )
    continue_logits = output.get("should_continue_sampling_logits")
    continue_probability = (
        float(torch.sigmoid(continue_logits[batch_index])) if continue_logits is not None else None
    )
    return LearnedScoutRecommendation(
        node_index=index,
        node_id=node_id,
        sample_node_logits=masked[_MASKED_LOGIT_OUTPUT][batch_index],
        expected_information_gain=gain,
        candidate_reduction_prediction=reduction,
        should_continue_sampling_probability=continue_probability,
        promotable=False,
    )
