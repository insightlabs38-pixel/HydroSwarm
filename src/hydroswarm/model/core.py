"""HydroCore shared scientific backbone and semantic specialist heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, Required, TypedDict, get_args

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    # M9.1 preflight (experiment-scoped): TemporalDynamicsBase and its
    # GRAPH_ODE/GRAPH_CDE/GRAPH_SDE variants live in continuous_time.py, a
    # module with lazy (try/except) imports of torchdiffeq/torchcde/torchsde
    # so that normal `import hydroswarm` never requires those optional
    # research dependencies. Only imported here under TYPE_CHECKING to avoid
    # making core.py itself depend on continuous_time.py (or its optional
    # deps) at runtime -- the constructor parameter below is typed against
    # this for static analysis only; at runtime it is accepted structurally
    # (anything exposing the same forward(...) contract works).
    from .continuous_time import TemporalDynamicsBase

from .adapters import BottleneckAdapter, RoleHead
from .candidate_plan_encoder import TARGET_TYPE_CLASSES, TARGET_TYPE_INDEX, CandidatePlanEncoder
from .encoders import (
    GraphStructuralEncoder,
    QualityEncoder,
    StaticFeatureEncoder,
    TemporalEncoder,
    make_activation,
    make_norm,
)
from .layers import LatentHydraulicBlock


class HydroBatch(TypedDict, total=False):
    node_features: Required[Tensor]
    temporal_features: Required[Tensor]
    quality_features: Required[Tensor]
    travel_time: Tensor
    reservoir_reachability: Tensor
    demand_centrality: Tensor
    edge_index: Tensor
    edge_features: Tensor
    node_mask: Tensor
    sensor_mask: Tensor
    quality_mask: Tensor
    edge_mask: Tensor
    timestamps: Tensor
    role_features: Tensor
    previous_actions: Tensor
    verifier_feedback: Tensor
    residual_features: Tensor
    classical_prior: Tensor
    source_candidate_mask: Tensor
    # core-issues4.txt Section E: candidate-conditioned Strategist forward
    # (strategist_mode="candidate_conditioned") reads these directly via
    # batch[...]/batch.get(...) but they were never declared on HydroBatch
    # itself -- a real schema gap for anything (a collator, a type
    # checker, a caller) that only has the TypedDict to go by. All left
    # optional here (not Required) because HydroBatch has no way to
    # express "required only in strategist_mode=candidate_conditioned";
    # forward()'s own explicit KeyError-on-missing behavior when that mode
    # is active, exercised by test_candidate_plan_batch_validation.py,
    # remains the actual enforcement.
    plan_template_ids: Tensor
    plan_target_type: Tensor
    plan_target_node_index: Tensor
    plan_target_link_index: Tensor
    plan_features: Tensor
    plan_mask: Tensor
    # Reserved for Phase 10's Strategist collator (remaining exact-
    # simulation budget / prior verifier-history features per candidate) --
    # not read by forward() yet, declared so the eventual collator has a
    # stable field name to target rather than inventing one ad hoc.
    plan_remaining_budget: Tensor
    plan_verifier_history: Tensor


class HydroOutput(TypedDict, total=False):
    hidden_state: Tensor
    latent_state: Tensor
    sentinel: Tensor
    scout: Tensor
    strategist: Tensor
    node_mask: Tensor
    source_node_logits: Tensor
    source_region_logits: Tensor
    start_time_logits: Tensor
    duration_logits: Tensor
    relative_strength_logits: Tensor
    evidence_sufficiency: Tensor
    sensor_fault_logits: Tensor
    sample_node_logits: Tensor
    expected_information_gain: Tensor
    candidate_reduction_prediction: Tensor
    should_continue_sampling_logits: Tensor
    action_logits: Tensor
    action_pointer_logits: Tensor
    plan_value: Tensor
    plan_validity_logits: Tensor
    uncertainty: Tensor
    ood_logits: Tensor
    event_presence_logits: Tensor
    event_cause_logits: Tensor
    next_step_logits: Tensor
    sensor_reconstruction_prediction: Tensor
    future_concentration_prediction: Tensor
    travel_time_prediction: Tensor
    exposure_proxy: Tensor
    pressure_risk_proxy: Tensor
    service_loss_proxy: Tensor
    containment_time_proxy: Tensor
    plan_regret_proxy: Tensor
    ood_category_logits: Tensor


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    backbone: int
    encoders: int
    adapters: int
    heads: int


@dataclass(frozen=True, slots=True)
class ModelVariant:
    d_model: int
    nhead: int
    dim_feedforward: int
    num_layers: int
    latent_tokens: int
    modality_layers: int = 1


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "small": ModelVariant(192, 6, 576, 4, 64),
    "medium": ModelVariant(256, 8, 768, 8, 64),
    "large": ModelVariant(360, 8, 1080, 8, 96),
    # HydroCore-v5 Milestone 9.7: pure-capacity-scaled counterpart of
    # "small" (the frozen M9.6 HydroCore-S recipe), used only by the
    # governed M9.7/M9.8 capacity-preflight/experiment scripts -- never a
    # replacement for "medium" above (which independently doubles
    # num_layers, a depth/stage-count change "small_v5_capacity_m" does
    # not make). d_model/nhead/dim_feedforward are widened while
    # preserving "small"'s exact head_dim (192/6=32) and
    # dim_feedforward/d_model (576/192=3.0) ratios and its num_layers=4,
    # latent_tokens=64, modality_layers=1 verbatim -- selected
    # deterministically, by parameter count only (13,919,572, nearest to
    # the protocol's predeclared ~14M center of the 12-16M v5-M band),
    # from the unique in-range point of a mechanically enumerated
    # ratio-preserving width grid. See
    # reports/evaluation/hydrocore-v5/m9-7/m9-7-capacity-candidates.json
    # and m9-7-selected-m-architecture.json for the full enumeration and
    # selection rule; this entry does not itself authorize any training.
    "small_v5_capacity_m": ModelVariant(352, 11, 1056, 4, 64),
}

#: Bumped whenever a change could make a checkpoint silently produce
#: different results if loaded with mismatched config, even when
#: load_state_dict's shape check alone would not catch it (overnight-
#: plan.txt Task 4.0). prior_mode is the first such case: the
#: prior_projection/prior_logit_scale parameters exist regardless of mode
#: (so checkpoint tensors are unaffected and old checkpoints remain
#: loadable), but which of them the forward pass actually *uses* is
#: config-only, so a mismatched prior_mode would load successfully yet
#: silently compute different outputs than training used.
#:
#: v2 -> v3 (core-issues.txt repair item 2): source_region_head's output
#: width changed from a fixed 1 (applied per node position -- never a
#: real 3-way classification, see the head's own construction comment)
#: to SOURCE_REGION_COUNT=3 applied to incident_context. A v2 checkpoint's
#: source_region_head weights are shape-incompatible AND were never
#: trained against a real label in the first place (the same bug meant
#: the loss for this task was always optimizing a meaningless target) --
#: see load_v2_source_region_head_migration for the one-time, narrowly-
#: scoped load path that re-initializes exactly that one head rather
#: than failing every v2 checkpoint's load entirely.
ARCHITECTURE_VERSION = "hydrocore-v3"

#: The exact parameter names load_v2_source_region_head_migration is
#: allowed to drop from an old checkpoint and re-initialize fresh --
#: never any other key, so an unrelated mismatch still fails closed.
V2_TO_V3_RESHAPED_PARAMETERS = frozenset({
    "source_region_head.network.1.weight",
    "source_region_head.network.1.bias",
})

PriorMode = Literal["none", "feature_only", "logit_only", "feature_and_logit"]
PRIOR_MODES: tuple[PriorMode, ...] = get_args(PriorMode)

#: overnight-plan.txt Task 4.2. "mean" (default) matches the original
#: unconditional global mean pool, kept as the ablation baseline. "latent"
#: pools the bounded global latent tokens instead of node states.
#: "attention" learns a single query token that attention-pools over valid
#: node states. "source_conditioned" combines a source-logit-weighted pool
#: of Sentinel node states, an attention pool restricted to sensor-observed
#: nodes, and a mean-pooled global latent context, projected down to
#: d_model.
IncidentPooling = Literal["mean", "latent", "attention", "source_conditioned"]
INCIDENT_POOLING_MODES: tuple[IncidentPooling, ...] = get_args(IncidentPooling)

#: overnight-plan.txt Task 4.3. "forward_only" (default) matches the
#: original single-direction transport convolution. "dual_gated" adds a
#: separately parameterized upstream-diagnostic channel over the reversed
#: edge direction, fused with a learned gate (see layers.DualChannelGraphConv).
MessageDirection = Literal["forward_only", "dual_gated"]
MESSAGE_DIRECTIONS: tuple[MessageDirection, ...] = get_args(MessageDirection)

#: core-issues3.txt Phase 4. "anonymous_queries" (default) matches the
#: original per-plan representation: a fixed-size learned parameter
#: (plan_query_tokens) independent of any incident, plus pooled incident
#: context -- a model built this way can only learn "what's usually good
#: at query position q", never "is THIS specific candidate plan good."
#: "candidate_conditioned" replaces it with CandidatePlanEncoder's output,
#: built from the actual deterministic candidate plan's own template/
#: target/features -- every downstream head (action_head, plan_value_head,
#: plan_validity_head, consequence_proxy_heads, pointer_query) is
#: unchanged either way; only how plan_hidden itself is built differs.
StrategistMode = Literal["anonymous_queries", "candidate_conditioned"]
STRATEGIST_MODES: tuple[StrategistMode, ...] = get_args(StrategistMode)

#: overnight-plan.txt Task 4.4. Class counts and index order for the
#: event/next-step control heads must exactly match
#: hydroswarm.training.targets_v2.EventCause / NextStep's enum member
#: order (CONTAMINATION, SENSOR_FAULT, HYDRAULIC_MISMATCH, AMBIGUOUS,
#: NORMAL / COLLECT_SAMPLE, INSPECT_FAULTY_SENSOR, GENERATE_PLANS, ABSTAIN). Not
#: imported directly to avoid a model<->training import cycle. For
#: event_cause, hydroswarm.training.corpus.EVENT_CAUSE_INDEX (built via
#: enumerate(EventCause)) is the actual label encoder and this count is
#: checked against it by tests/unit/test_event_control_heads.py; next_step
#: label generation is not yet built (governed by the deterministic
#: controller policy from a later task), so only the head shape is fixed
#: here, matching len(NextStep).
EVENT_CAUSE_CLASS_COUNT = 5
NEXT_STEP_CLASS_COUNT = 4

#: core-issues3.txt Phase 6 item F / additional gap F: the governed
#: ood_class target (hydroswarm.training.targets_v2's TARGET_CLASS_COUNTS)
#: is hydroswarm.training.ood_categories.OODCategory's full 11-category
#: taxonomy, but the pre-existing `ood_head` below is a 3-logit head built
#: for an entirely different, deterministic-only concept (OODLevel's
#: NORMAL/CAUTION/OUTSIDE_VALIDATED_RANGE severity, computed by
#: hydroswarm.inference.ood.OODDetector -- which remains authoritative and
#: is not changed by this flag). Training compute_multitask_loss's
#: "ood_class" task against the old 3-logit head would raise the moment a
#: real label index >= 3 appeared (SEVERE_MISSINGNESS=7, FROZEN_DRIFTING_
#: SENSOR=8 in OODCategory's declared order) -- not yet observed in any
#: promoted run only because no training run has yet sampled enough
#: non-NONE examples to trigger it (559/13,150 ~= 4.2% of the real
#: trajectory corpus). This adds the correctly-sized head as a separate,
#: net-new, opt-in output rather than resizing ood_head in place, for the
#: same checkpoint-compatibility reason as every other Phase 4.x/6.x flag.
OOD_CATEGORY_COUNT = 11
OOD_CATEGORY_HEAD_DEFAULT = False

#: overnight-plan.txt Task 4.0's explicit compatibility requirement: "The
#: updated architecture must not silently load incompatible weights with
#: missing or randomly initialized safety-critical heads." Unlike
#: prior_mode/incident_pooling/message_direction (each already an
#: existing pathway made configurable), the Task 4.4 event/next-step
#: heads are net-new parameters with no prior existence in the promoted
#: checkpoint, so -- following the same "default reproduces the
#: checkpoint-compatible original module graph exactly" convention used
#: by every other Task 4.x flag -- they are gated behind this flag rather
#: than always constructed, and are simply absent (not randomly
#: initialized and silently unused) when disabled.
EVENT_CONTROL_HEADS_DEFAULT = False

#: overnight-plan.txt Task 4.5: exactly three auxiliary objectives --
#: masked sensor reconstruction, future concentration prediction, and
#: travel-time prediction -- "do not add all planned auxiliary heads in
#: this overnight run." Same net-new-parameters compatibility concern as
#: EVENT_CONTROL_HEADS_DEFAULT, so gated the same way. Their predictions
#: are explicitly non-authoritative training signal (see
#: AUXILIARY_TASKS/AUXILIARY_TASK_DEFAULT_WEIGHT in training.losses) and
#: must never be surfaced to a user as a product decision.
AUXILIARY_HEADS_DEFAULT = False

#: overnight-plan.txt Task 4.6: optional plan-consequence prescreening
#: proxies -- exposure, pressure-risk, service-loss, containment-time,
#: and plan-regret -- computed per candidate plan to potentially reduce
#: the number of expensive exact WNTR simulations by ranking candidates.
#: "They must never replace WNTR verification" and "not exposed as
#: verified consequences": PlanVerifier's exact simulation remains the
#: sole authoritative source for plan_validity/consequence_vector: these
#: heads produce a separate, clearly-named *_proxy output that a caller
#: could use only to prioritize which candidates to actually simulate.
#: Same net-new-parameters compatibility concern as
#: EVENT_CONTROL_HEADS_DEFAULT/AUXILIARY_HEADS_DEFAULT, so gated the same
#: way.
CONSEQUENCE_PRESCREENING_HEADS_DEFAULT = False
CONSEQUENCE_PROXY_NAMES: tuple[str, ...] = (
    "exposure_proxy",
    "pressure_risk_proxy",
    "service_loss_proxy",
    "containment_time_proxy",
    "plan_regret_proxy",
)

#: core-issues3.txt Phase 4. Width of the deterministic pre-verification
#: plan_features vector CandidatePlanEncoder consumes -- the old
#: heuristic's own predicted_value/predicted_validity, action count, and
#: three bounded action-parameter scalars (start time, duration, magnitude)
#: normalized by the caller (hydroswarm.training's future Strategist
#: collator, per Phase 10). Fixed here (not derived from a training
#: example) so the encoder's own input contract is self-documenting and
#: independently testable.
STRATEGIST_MODE_DEFAULT: StrategistMode = "anonymous_queries"
PLAN_FEATURE_DIM = 6

#: core-issues4.txt Section D item 3: two governed Scout targets
#: (hydroswarm.training.targets_v2's "candidate_reduction" -- per-node,
#: masked like information_gain -- and "should_continue_sampling" --
#: incident-level, never masked) already exist and are already generated
#: by hydroswarm.training.scout_trajectory, but had no model head to
#: receive a gradient at all (a real gap: a governed target with no
#: corresponding output can never be trained, unlike an untrained output
#: with no target, which Phase 9.2's output governance already handles).
#: Same net-new-parameters compatibility concern as every other Phase-4.x/
#: 6.x/Section-D flag above, so gated the same way -- disabled by default,
#: absent (not randomly initialized and silently unused) when disabled.
SCOUT_CONTROL_HEADS_DEFAULT = False


class ArchitectureCompatibilityError(Exception):
    """Raised when a checkpoint's recorded architecture config does not
    match the model instance it is being loaded into, for a config
    dimension that load_state_dict's tensor-shape check would not itself
    catch (e.g. prior_mode)."""


def verify_architecture_compatibility(model: "HydroCore", metadata: dict[str, object]) -> None:
    """Raise ArchitectureCompatibilityError if `metadata` (as recorded in a
    checkpoint's own metadata at export time, e.g. via
    HydroCore.architecture_config()) does not match `model`'s actual
    configuration. Call this before or after load_state_dict when loading a
    checkpoint whose provenance is not otherwise already pinned (e.g. by a
    checkpoint-hash check against a known-good promoted artifact) --
    load_state_dict's tensor-shape check alone would not catch a prior_mode
    mismatch, since prior_mode does not change any parameter's shape."""

    recorded_version = metadata.get("architecture_version")
    if recorded_version is not None and recorded_version != ARCHITECTURE_VERSION:
        raise ArchitectureCompatibilityError(
            f"checkpoint architecture_version {recorded_version!r} does not match this "
            f"build's {ARCHITECTURE_VERSION!r}; a migration path must be defined before "
            "loading a checkpoint from a different architecture version"
        )
    recorded_use_adapters = metadata.get("use_adapters")
    if recorded_use_adapters is not None and recorded_use_adapters != model.use_adapters:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with use_adapters={recorded_use_adapters!r} but this "
            f"model instance is configured with use_adapters={model.use_adapters!r}; "
            "BottleneckAdapter and nn.Identity have different parameter sets"
        )
    recorded_prior_mode = metadata.get("prior_mode")
    if recorded_prior_mode is not None and recorded_prior_mode != model.prior_mode:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with prior_mode={recorded_prior_mode!r} but this model "
            f"instance is configured with prior_mode={model.prior_mode!r}; loading it would "
            "silently compute different outputs than training used"
        )
    recorded_incident_pooling = metadata.get("incident_pooling")
    if recorded_incident_pooling is not None and recorded_incident_pooling != model.incident_pooling:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with incident_pooling={recorded_incident_pooling!r} but "
            f"this model instance is configured with incident_pooling={model.incident_pooling!r}"
        )
    recorded_message_direction = metadata.get("message_direction")
    if recorded_message_direction is not None and recorded_message_direction != model.message_direction:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with message_direction={recorded_message_direction!r} but "
            f"this model instance is configured with message_direction={model.message_direction!r}; "
            "dual_gated adds separate upstream/gate parameters that forward_only does not have"
        )
    recorded_event_control_heads = metadata.get("event_control_heads")
    if recorded_event_control_heads is not None and recorded_event_control_heads != model.event_control_heads:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with event_control_heads={recorded_event_control_heads!r} but "
            f"this model instance is configured with event_control_heads={model.event_control_heads!r}; "
            "enabling it adds event_presence/event_cause/next_step head parameters that a "
            "checkpoint trained without them does not have"
        )
    recorded_auxiliary_heads = metadata.get("auxiliary_heads")
    if recorded_auxiliary_heads is not None and recorded_auxiliary_heads != model.auxiliary_heads:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with auxiliary_heads={recorded_auxiliary_heads!r} but "
            f"this model instance is configured with auxiliary_heads={model.auxiliary_heads!r}; "
            "enabling it adds sensor_reconstruction/future_concentration/travel_time head "
            "parameters that a checkpoint trained without them does not have"
        )
    recorded_consequence_prescreening = metadata.get("consequence_prescreening_heads")
    if (
        recorded_consequence_prescreening is not None
        and recorded_consequence_prescreening != model.consequence_prescreening_heads
    ):
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with consequence_prescreening_heads="
            f"{recorded_consequence_prescreening!r} but this model instance is configured "
            f"with consequence_prescreening_heads={model.consequence_prescreening_heads!r}; "
            "enabling it adds five plan consequence-proxy head parameters that a checkpoint "
            "trained without them does not have"
        )
    recorded_strategist_mode = metadata.get("strategist_mode")
    if recorded_strategist_mode is not None and recorded_strategist_mode != model.strategist_mode:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with strategist_mode={recorded_strategist_mode!r} but this "
            f"model instance is configured with strategist_mode={model.strategist_mode!r}; "
            "candidate_conditioned adds CandidatePlanEncoder parameters that anonymous_queries "
            "does not have"
        )
    recorded_ood_category_head = metadata.get("ood_category_head")
    if recorded_ood_category_head is not None and recorded_ood_category_head != model.ood_category_head_enabled:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with ood_category_head={recorded_ood_category_head!r} but this "
            f"model instance is configured with ood_category_head={model.ood_category_head_enabled!r}; "
            "enabling it adds an 11-class ood_category_head that the old 3-logit ood_head does not have"
        )
    recorded_scout_control_heads = metadata.get("scout_control_heads")
    if recorded_scout_control_heads is not None and recorded_scout_control_heads != model.scout_control_heads:
        raise ArchitectureCompatibilityError(
            f"checkpoint was trained with scout_control_heads={recorded_scout_control_heads!r} but this "
            f"model instance is configured with scout_control_heads={model.scout_control_heads!r}; "
            "enabling it adds candidate_reduction_head/should_continue_sampling_head parameters that a "
            "checkpoint trained without them does not have"
        )


def load_state_dict_with_v2_migration(
    model: "HydroCore", state_dict: dict[str, Tensor]
) -> tuple[bool, tuple[str, ...]]:
    """Load a checkpoint, tolerating exactly one known, understood shape
    change: v2's source_region_head (repair item 2, ARCHITECTURE_VERSION
    v2 -> v3). Returns (migrated, dropped_keys) -- migrated is True only
    when the v2 shape mismatch was actually encountered and handled;
    dropped_keys names exactly which tensors were re-initialized fresh
    rather than loaded, for the caller to record/surface, never silently.

    Any mismatch outside V2_TO_V3_RESHAPED_PARAMETERS still raises --
    this is a narrow, explicit migration path for one specific, already
    audited architecture change, not a general strict=False escape hatch.
    A v2 checkpoint's source_region_head weights are not just
    shape-incompatible but were never trained against a real label in
    the first place (the same bug meant this task's loss always
    optimized a meaningless per-node target), so re-initializing them
    fresh loses nothing a v2 checkpoint ever actually had.
    """

    try:
        model.load_state_dict(state_dict, strict=True)
        return False, ()
    except RuntimeError as error:
        message = str(error)
        if "size mismatch" not in message:
            raise
        mismatched = {
            key for key in V2_TO_V3_RESHAPED_PARAMETERS if f"for {key}:" in message
        }
        remaining_mismatch_markers = message.count("size mismatch for")
        if mismatched != V2_TO_V3_RESHAPED_PARAMETERS or remaining_mismatch_markers != len(mismatched):
            raise
        filtered = {key: value for key, value in state_dict.items() if key not in mismatched}
        result = model.load_state_dict(filtered, strict=False)
        if set(result.missing_keys) != V2_TO_V3_RESHAPED_PARAMETERS or result.unexpected_keys:
            raise RuntimeError(
                "v2 migration produced unexpected missing/unexpected keys: "
                f"missing={result.missing_keys!r} unexpected={result.unexpected_keys!r}"
            ) from error
        return True, tuple(sorted(mismatched))


class HydroCore(nn.Module):
    """Edge-aware local graph model with bounded global latent attention."""

    def __init__(
        self,
        *,
        node_feature_dim: int = 19,
        edge_feature_dim: int = 13,
        temporal_feature_dim: int = 6,
        quality_feature_dim: int = 4,
        role_feature_dim: int = 8,
        action_feature_dim: int = 8,
        verifier_feature_dim: int = 8,
        residual_feature_dim: int = 4,
        d_model: int = 360,
        nhead: int = 8,
        dim_feedforward: int = 1080,
        num_layers: int = 8,
        modality_layers: int = 1,
        latent_tokens: int = 96,
        plan_queries: int = 8,
        # core-issues3.txt Phase 3.5: hydroswarm.planning.action_templates.
        # ACTION_TEMPLATE_COUNT is the canonical vocabulary size (9), one
        # more than this default. NOT changed here: the promoted checkpoint
        # does not pin action_vocabulary_size in its recorded architecture
        # config, so it relies on this constructor default even though it
        # never trains the action head -- raising the default to 9 broke
        # strict reload of that checkpoint (action_head's saved weight shape
        # is [8, ...]), caught by
        # tests/integration/test_default_pipeline_factory.py. Any future
        # Strategist-enabled training run MUST pass
        # action_vocabulary_size=ACTION_TEMPLATE_COUNT explicitly (as
        # strategist_trajectory.py's ACTION_TEMPLATES has documented since
        # before this phase) until Phase 9's architecture-v4 contract makes
        # this a strictly-validated, always-recorded config field instead of
        # a silent constructor default.
        action_vocabulary_size: int = 8,
        dropout: float = 0.1,
        normalization: str = "rmsnorm",
        activation: str = "silu",
        sentinel_output_dim: int = 2,
        scout_output_dim: int = 2,
        strategist_output_dim: int = 3,
        adapter_dims: tuple[int, int, int] = (32, 48, 64),
        use_adapters: bool = True,
        prior_mode: PriorMode = "feature_and_logit",
        incident_pooling: IncidentPooling = "mean",
        message_direction: MessageDirection = "forward_only",
        event_control_heads: bool = EVENT_CONTROL_HEADS_DEFAULT,
        auxiliary_heads: bool = AUXILIARY_HEADS_DEFAULT,
        consequence_prescreening_heads: bool = CONSEQUENCE_PRESCREENING_HEADS_DEFAULT,
        strategist_mode: StrategistMode = STRATEGIST_MODE_DEFAULT,
        plan_feature_dim: int = PLAN_FEATURE_DIM,
        ood_category_head: bool = OOD_CATEGORY_HEAD_DEFAULT,
        scout_control_heads: bool = SCOUT_CONTROL_HEADS_DEFAULT,
        # Milestone 8.7 Arm C (AGE_FIX_PLUS_RELATIVE_TIME): see
        # encoders.TemporalEncoder's own docstring/comment for what this
        # changes. Defaults to "window_relative" -- every existing
        # checkpoint/caller (Arms A/B included) is unaffected unless this
        # is explicitly overridden.
        elapsed_time_normalization: str = "window_relative",
        # HydroCore-v5 Milestone 9.1 preflight: experiment-scoped seam for
        # swapping the temporal-latent-production step (temporal_encoder +
        # quality_encoder) for a continuous-time alternative (Graph Neural
        # ODE/CDE/SDE), without touching node/graph encoders, fusion shape,
        # backbone, or any output head. Defaults to None for every existing
        # caller -- when None, HydroCore builds and uses temporal_encoder/
        # quality_encoder exactly as before (this parameter is purely
        # additive; it changes zero behavior unless explicitly supplied, so
        # no production code path or existing checkpoint is affected). Not
        # recorded in architecture_config(): a supplied temporal_dynamics
        # module is never checkpointed as part of a promoted HydroCore
        # export, only used directly inside experiment scripts, so there is
        # no checkpoint-identity gap to close here the way prior_mode/
        # incident_pooling/etc. needed (those are simple value fields on a
        # promoted checkpoint; this is a whole swapped submodule that a
        # loader would need to reconstruct out-of-band regardless).
        temporal_dynamics: "TemporalDynamicsBase | None" = None,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 64 <= latent_tokens <= 96:
            raise ValueError("latent_tokens must be between 64 and 96")
        if not 1 <= plan_queries <= 8:
            raise ValueError("plan_queries must be between 1 and 8")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}, got {prior_mode!r}")
        if incident_pooling not in INCIDENT_POOLING_MODES:
            raise ValueError(
                f"incident_pooling must be one of {INCIDENT_POOLING_MODES}, got {incident_pooling!r}"
            )
        if message_direction not in MESSAGE_DIRECTIONS:
            raise ValueError(
                f"message_direction must be one of {MESSAGE_DIRECTIONS}, got {message_direction!r}"
            )
        if strategist_mode not in STRATEGIST_MODES:
            raise ValueError(f"strategist_mode must be one of {STRATEGIST_MODES}, got {strategist_mode!r}")
        self.d_model = d_model
        self.num_layers = num_layers
        self.latent_tokens_count = latent_tokens
        # core-issues3.txt Phase 9.1: previously only d_model/num_layers/
        # latent_tokens_count/plan_feature_dim were retained as instance
        # attributes -- every other dimension/layer-count constructor
        # argument was used locally to build submodules and then
        # forgotten, so architecture_config() could not record it even
        # though it is a real architecture-identity fact a checkpoint
        # loader needs to reconstruct the model from. Pure additive
        # attribute recording; changes no forward()-visible behavior.
        #
        # core-issues4.txt Section A: the input feature-width constructor
        # arguments (node/edge/temporal/quality/role/action/verifier/
        # residual_feature_dim) were never recorded even here -- used only
        # to build the input encoders/projections, then forgotten, the same
        # gap the comment above already fixed for every other dimension.
        # training.checkpoint_identity (the v4 identity module) needs these
        # to reconstruct a model from a checkpoint's identity alone.
        # Recorded with the same pure-additive, forward()-unaffected
        # convention; `dropout_value` likewise (dropout is "behavior-
        # critical" per Section A's field list, since a mismatched dropout
        # would silently compute a different train-time forward pass, but
        # was previously visible only inside the submodules it was passed
        # to, not on the model itself).
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.temporal_feature_dim = temporal_feature_dim
        self.quality_feature_dim = quality_feature_dim
        self.role_feature_dim = role_feature_dim
        self.action_feature_dim = action_feature_dim
        self.verifier_feature_dim = verifier_feature_dim
        self.residual_feature_dim = residual_feature_dim
        self.dropout_value = dropout
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.modality_layers = modality_layers
        self.plan_queries = plan_queries
        self.action_vocabulary_size = action_vocabulary_size
        self.adapter_dims = tuple(adapter_dims)
        self.sentinel_output_dim = sentinel_output_dim
        self.scout_output_dim = scout_output_dim
        self.strategist_output_dim = strategist_output_dim
        self.normalization_kind = normalization
        self.activation_kind = activation
        self.use_adapters = use_adapters
        self.prior_mode = prior_mode
        self.incident_pooling = incident_pooling
        self.message_direction = message_direction
        self.event_control_heads = event_control_heads
        self.auxiliary_heads = auxiliary_heads
        self.consequence_prescreening_heads = consequence_prescreening_heads
        self.strategist_mode = strategist_mode
        self.plan_feature_dim = plan_feature_dim
        self.ood_category_head_enabled = ood_category_head
        self.scout_control_heads = scout_control_heads
        self.elapsed_time_normalization = elapsed_time_normalization
        # core-issues.txt repair item 9: set by from_variant() so a
        # checkpoint's own architecture_config() records which named
        # variant it was built from; a model constructed directly (e.g. the
        # tiny configurations used throughout the test suite) has no named
        # variant and stays None.
        self.variant_name: str | None = None
        self.node_encoder = StaticFeatureEncoder(
            node_feature_dim, d_model, normalization=normalization, activation=activation
        )
        self.graph_encoder = GraphStructuralEncoder(
            d_model, normalization=normalization, activation=activation
        )
        temporal_args = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=modality_layers,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
            elapsed_time_normalization=elapsed_time_normalization,
        )
        self.temporal_dynamics = temporal_dynamics
        if self.temporal_dynamics is None:
            self.temporal_encoder = TemporalEncoder(temporal_feature_dim, **temporal_args)
            self.quality_encoder = QualityEncoder(quality_feature_dim, **temporal_args)
        self.modality_fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            make_activation(activation),
            make_norm(normalization, d_model),
            nn.Dropout(dropout),
        )
        self.residual_projection = nn.Linear(residual_feature_dim, d_model)
        self.prior_projection = nn.Linear(1, d_model)
        self.role_projection = nn.Linear(role_feature_dim, d_model)
        self.action_projection = nn.Linear(action_feature_dim, d_model)
        self.verifier_projection = nn.Linear(verifier_feature_dim, d_model)
        self.global_latents = nn.Parameter(torch.empty(latent_tokens, d_model))
        self.plan_query_tokens = nn.Parameter(torch.empty(plan_queries, d_model))
        nn.init.normal_(self.global_latents, std=0.02)
        nn.init.normal_(self.plan_query_tokens, std=0.02)

        if incident_pooling in ("attention", "source_conditioned"):
            self.incident_query = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.normal_(self.incident_query, std=0.02)
            self.incident_attention = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
        if incident_pooling == "source_conditioned":
            self.incident_context_projection = nn.Linear(3 * d_model, d_model)

        self.backbone = nn.ModuleList(
            [
                LatentHydraulicBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    edge_feature_dim,
                    dropout=dropout,
                    normalization=normalization,
                    activation=activation,
                    message_direction=message_direction,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = make_norm(normalization, d_model)

        roles = ("sentinel", "scout", "strategist")
        outputs = (sentinel_output_dim, scout_output_dim, strategist_output_dim)
        self.adapters = nn.ModuleDict(
            {
                role: (
                    BottleneckAdapter(d_model, width) if use_adapters else nn.Identity()
                )
                for role, width in zip(roles, adapter_dims, strict=True)
            }
        )
        self.heads = nn.ModuleDict(
            {role: RoleHead(d_model, size) for role, size in zip(roles, outputs, strict=True)}
        )

        # Semantic heads expose the actual scientific tasks rather than anonymous widths.
        self.source_node_head = RoleHead(d_model, 1)
        self.prior_logit_scale = nn.Parameter(torch.tensor(0.54132485))
        # Exactly 3 incident-level region classes (hydroswarm.training.corpus.
        # SOURCE_REGION_COUNT) applied to incident_context, not a per-node
        # width -- core-issues.txt repair item 2: this used to be RoleHead(
        # d_model, 1) applied per node position, making "region" an
        # accidental node index rather than a real 3-way classification.
        self.source_region_head = RoleHead(d_model, 3)
        self.sensor_fault_head = RoleHead(d_model, 1)
        self.sample_node_head = RoleHead(d_model, 1)
        self.information_gain_head = nn.Sequential(make_norm(normalization, d_model), nn.Linear(d_model, 1), nn.Softplus())
        # core-issues4.txt Section D item 3: candidate_reduction is a
        # per-node fraction in [0, 1] (targets_v2's own documented unit),
        # so Sigmoid -- not Softplus, which is unbounded above like
        # information_gain_head's bits-valued target -- matches the
        # target's actual range. should_continue_sampling is an
        # incident-level raw logit (RoleHead), trained with
        # BCEWithLogitsLoss like event_presence/sensor_fault.
        if self.scout_control_heads:
            self.candidate_reduction_head = nn.Sequential(
                make_norm(normalization, d_model), nn.Linear(d_model, 1), nn.Sigmoid()
            )
            self.should_continue_sampling_head = RoleHead(d_model, 1)
        self.profile_heads = nn.ModuleDict(
            {
                "start_time": RoleHead(d_model, 12),
                "duration": RoleHead(d_model, 8),
                "relative_strength": RoleHead(d_model, 4),
            }
        )
        self.evidence_head = nn.Sequential(make_norm(normalization, d_model), nn.Linear(d_model, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(make_norm(normalization, d_model), nn.Linear(d_model, 1), nn.Softplus())
        self.ood_head = RoleHead(d_model, 3)
        # core-issues3.txt Phase 6: the correctly-sized (11-class) governed
        # ood_class head, separate from the pre-existing 3-logit ood_head
        # above (a different, deterministic-severity-adjacent concept --
        # see OOD_CATEGORY_HEAD_DEFAULT's docstring). Only constructed when
        # ood_category_head is enabled.
        if self.ood_category_head_enabled:
            self.ood_category_head = RoleHead(d_model, OOD_CATEGORY_COUNT)
        # overnight-plan.txt Task 4.4: incident-level control heads for the
        # targets_v2 event_presence/event_cause/next_step contract. Only
        # constructed when event_control_heads is enabled -- see
        # EVENT_CONTROL_HEADS_DEFAULT's docstring for why this is gated
        # rather than unconditional.
        if self.event_control_heads:
            self.event_presence_head = RoleHead(d_model, 1)
            self.event_cause_head = RoleHead(d_model, EVENT_CAUSE_CLASS_COUNT)
            self.next_step_head = RoleHead(d_model, NEXT_STEP_CLASS_COUNT)
        # overnight-plan.txt Task 4.5: optional, configuration-controlled
        # auxiliary objectives. Only constructed when auxiliary_heads is
        # enabled -- see AUXILIARY_HEADS_DEFAULT's docstring.
        if self.auxiliary_heads:
            self.sensor_reconstruction_head = RoleHead(d_model, 1)
            self.future_concentration_head = RoleHead(d_model, 1)
            self.travel_time_head = RoleHead(d_model, 1)
        # overnight-plan.txt Task 4.6: optional, non-authoritative plan
        # consequence-prescreening proxies, applied to the same per-plan
        # representation (plan_hidden) that action/plan_value/
        # plan_validity already use. Only constructed when
        # consequence_prescreening_heads is enabled -- see
        # CONSEQUENCE_PRESCREENING_HEADS_DEFAULT's docstring.
        if self.consequence_prescreening_heads:
            self.consequence_proxy_heads = nn.ModuleDict(
                {name: RoleHead(d_model, 1) for name in CONSEQUENCE_PROXY_NAMES}
            )
        # core-issues3.txt Phase 4: only constructed in candidate_conditioned
        # mode -- net-new parameters, same compatibility convention as every
        # other Phase-4.x flag above (default anonymous_queries mode has
        # zero new parameters and is byte-identical to the pre-Phase-4
        # module graph).
        if self.strategist_mode == "candidate_conditioned":
            self.candidate_plan_encoder = CandidatePlanEncoder(
                d_model,
                action_vocabulary_size=action_vocabulary_size,
                plan_feature_dim=plan_feature_dim,
                dropout=dropout,
                normalization=normalization,
                activation=activation,
            )
        self.action_head = RoleHead(d_model, action_vocabulary_size)
        self.plan_value_head = RoleHead(d_model, 1)
        self.plan_validity_head = RoleHead(d_model, 2)
        self.pointer_query = nn.Linear(d_model, d_model, bias=False)

    @classmethod
    def from_variant(cls, variant: str, **overrides: object) -> HydroCore:
        try:
            configuration = MODEL_VARIANTS[variant.lower()]
        except KeyError as error:
            raise ValueError(f"unknown model variant: {variant}") from error
        values = {
            "d_model": configuration.d_model,
            "nhead": configuration.nhead,
            "dim_feedforward": configuration.dim_feedforward,
            "num_layers": configuration.num_layers,
            "latent_tokens": configuration.latent_tokens,
            "modality_layers": configuration.modality_layers,
            **overrides,
        }
        instance = cls(**values)  # type: ignore[arg-type]
        instance.variant_name = variant.lower()
        return instance

    def architecture_config(self) -> dict[str, object]:
        """Config dimensions load_state_dict's shape check would not itself
        catch if mismatched (overnight-plan.txt Task 4.0). Callers persist
        this into checkpoint metadata (see export_model's `metadata` arg)
        so verify_architecture_compatibility() can detect a checkpoint
        being loaded into a model configured differently than it was
        trained with.

        core-issues.txt repair item 9: `variant` and `use_adapters` were
        added alongside the pre-existing dimensions above -- both are real
        identity facts a runtime loader must reconstruct the model from,
        not leave at their constructor defaults.

        core-issues3.txt Phase 9.1 (partial): added every remaining
        dimension/layer-count/output-width constructor argument that was
        previously used only to build submodules and then forgotten, not
        recorded here -- a real gap for "strictly reconstruct the model
        from its checkpoint metadata" even though load_state_dict's own
        shape check happens to catch most of these today (unlike
        prior_mode/pooling/message_direction, which change zero parameter
        shapes and need the explicit verify_architecture_compatibility
        checks above). Deliberately NOT included here: schema hashes owned
        by other layers (ACTION_TEMPLATE_SCHEMA_HASH in
        hydroswarm.planning.action_templates, targets_v2's schema version,
        the feature schema fingerprint, PlanValuePolicy/signature-artifact-
        policy versions) -- model/core.py currently has zero external
        hydroswarm-package imports, and reaching into planning/training
        modules from here would invert that layering. Assembling those
        alongside this dict into one complete checkpoint identity belongs
        in whatever layer already orchestrates checkpoint export (e.g. a
        future dedicated checkpoint-identity module), not here -- the rest
        of Phase 9.1 remains open, tracked in the handoff report."""

        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "variant": self.variant_name,
            "use_adapters": self.use_adapters,
            "prior_mode": self.prior_mode,
            "incident_pooling": self.incident_pooling,
            "message_direction": self.message_direction,
            "event_control_heads": self.event_control_heads,
            "auxiliary_heads": self.auxiliary_heads,
            "consequence_prescreening_heads": self.consequence_prescreening_heads,
            "strategist_mode": self.strategist_mode,
            "ood_category_head": self.ood_category_head_enabled,
            "scout_control_heads": self.scout_control_heads,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "num_layers": self.num_layers,
            "modality_layers": self.modality_layers,
            "latent_tokens": self.latent_tokens_count,
            "plan_queries": self.plan_queries,
            "action_vocabulary_size": self.action_vocabulary_size,
            "adapter_dims": self.adapter_dims,
            "sentinel_output_dim": self.sentinel_output_dim,
            "scout_output_dim": self.scout_output_dim,
            "strategist_output_dim": self.strategist_output_dim,
            "plan_feature_dim": self.plan_feature_dim,
            "normalization": self.normalization_kind,
            "activation": self.activation_kind,
            # core-issues4.txt Section A: input feature widths, now recorded
            # as instance attributes above.
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "temporal_feature_dim": self.temporal_feature_dim,
            "quality_feature_dim": self.quality_feature_dim,
            "role_feature_dim": self.role_feature_dim,
            "action_feature_dim": self.action_feature_dim,
            "verifier_feature_dim": self.verifier_feature_dim,
            "residual_feature_dim": self.residual_feature_dim,
            "dropout": self.dropout_value,
            "elapsed_time_normalization": self.elapsed_time_normalization,
        }

    def _attention_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Single learnable query attention-pools over positions where mask
        is True. Rows with no valid position (mask.any(dim=1) is False)
        are redirected to attend over position 0 only, matching the
        existing safe_mask convention used for the backbone, so
        MultiheadAttention never sees an all-masked row (which would
        otherwise produce NaN)."""

        batch_size = hidden.shape[0]
        safe_mask = mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        query = self.incident_query.expand(batch_size, -1, -1)
        pooled, _ = self.incident_attention(
            query, hidden, hidden, key_padding_mask=~safe_mask, need_weights=False
        )
        return pooled.squeeze(1)

    @staticmethod
    def _optional_context(
        batch: HydroBatch, key: str, projection: nn.Linear, batch_size: int, device: torch.device
    ) -> Tensor:
        value = batch.get(key)  # type: ignore[literal-required]
        if value is None:
            return torch.zeros(batch_size, projection.out_features, device=device)
        value = torch.nan_to_num(value.float())
        if value.ndim == 3:
            value = value.mean(dim=1)
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError(f"{key} must be [batch, features] or [batch, sequence, features]")
        return projection(value)

    @staticmethod
    def _profile_logits(head: nn.Module, pooled: Tensor, class_count: int) -> Tensor:
        """Mask architecturally reserved outputs that are not governed label classes."""

        logits = head(pooled)
        classes = torch.arange(logits.shape[-1], device=logits.device)
        return logits.masked_fill(classes >= class_count, torch.finfo(logits.dtype).min)

    def forward(self, batch: HydroBatch) -> HydroOutput:
        required = ("node_features", "temporal_features", "quality_features")
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"missing HydroBatch fields: {', '.join(missing)}")
        node_features = batch["node_features"]
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, features]")
        batch_size, nodes, _ = node_features.shape
        node_mask = batch.get(
            "node_mask", torch.ones(batch_size, nodes, dtype=torch.bool, device=node_features.device)
        ).bool()
        if node_mask.shape != (batch_size, nodes):
            raise ValueError("node_mask must have shape [batch, nodes]")
        safe_mask = node_mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        zeros = torch.zeros(batch_size, nodes, device=node_features.device)
        static = self.node_encoder(node_features)
        graph = self.graph_encoder(
            batch.get("travel_time", zeros),
            batch.get("reservoir_reachability", zeros),
            batch.get("demand_centrality", zeros),
        )
        if self.temporal_dynamics is None:
            temporal = self.temporal_encoder(
                batch["temporal_features"], batch.get("sensor_mask"), batch.get("timestamps")
            )
            quality = self.quality_encoder(
                batch["quality_features"], batch.get("quality_mask"), batch.get("timestamps")
            )
        else:
            # M9.1 preflight seam: the continuous-time module additionally
            # receives graph connectivity (unlike temporal_encoder/
            # quality_encoder above, which are purely per-node) so its
            # vector field can be graph-aware, per the frozen preflight
            # protocol. edge_index/edge_features/edge_mask are otherwise
            # only read later, inside the backbone loop -- reading them here
            # too is non-mutating and changes nothing about their later use.
            temporal, quality = self.temporal_dynamics(
                batch["temporal_features"],
                batch["quality_features"],
                batch.get("sensor_mask"),
                batch.get("quality_mask"),
                batch.get("timestamps"),
                node_mask,
                batch.get("edge_index"),
                batch.get("edge_features"),
                batch.get("edge_mask"),
            )
        hidden = self.modality_fusion(torch.cat((static, graph, temporal, quality), dim=-1))
        residual = batch.get("residual_features")
        if residual is not None:
            if residual.shape[:2] != (batch_size, nodes):
                raise ValueError("residual_features must have shape [batch, nodes, features]")
            hidden = hidden + self.residual_projection(torch.nan_to_num(residual.float()))
        prior = batch.get("classical_prior")
        if prior is not None:
            if prior.shape != (batch_size, nodes):
                raise ValueError("classical_prior must have shape [batch, nodes]")
            if self.prior_mode in ("feature_only", "feature_and_logit"):
                hidden = hidden + self.prior_projection(torch.nan_to_num(prior.float()).unsqueeze(-1))
        context = self._optional_context(batch, "role_features", self.role_projection, batch_size, node_features.device)
        context += self._optional_context(batch, "previous_actions", self.action_projection, batch_size, node_features.device)
        context += self._optional_context(batch, "verifier_feedback", self.verifier_projection, batch_size, node_features.device)
        hidden = hidden + context[:, None, :]
        hidden = hidden.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        latents = self.global_latents.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.backbone:
            hidden, latents = block(
                hidden,
                latents,
                safe_mask,
                batch.get("edge_index"),
                batch.get("edge_features"),
                batch.get("edge_mask"),
            )
        hidden = self.final_norm(hidden).masked_fill(~node_mask.unsqueeze(-1), 0.0)
        pooled = (hidden * node_mask.unsqueeze(-1)).sum(1) / node_mask.sum(1, keepdim=True).clamp_min(1)
        role_hidden = {role: adapter(hidden) for role, adapter in self.adapters.items()}
        role_outputs = {
            role: self.heads[role](value).masked_fill(~node_mask.unsqueeze(-1), 0.0)
            for role, value in role_hidden.items()
        }
        sentinel_nodes = role_hidden["sentinel"]
        scout_nodes = role_hidden["scout"]
        # core-issues5.txt Section 2 (P0 blocker): a candidate-conditioned
        # model must support two distinct passes. PASS 1 (incident-only
        # analysis -- Sentinel/localization/event/control outputs) runs
        # before any response plan exists; no plan tensors are available or
        # meaningful yet. PASS 2 (plan scoring) supplies real deterministic
        # candidate-plan tensors once planning is gated open. Previously
        # this branch treated any missing plan tensor as a hard KeyError
        # unconditionally, so a live incident-only forward pass on a
        # candidate_conditioned checkpoint always raised, and
        # HybridInferencePipeline.analyze()'s broad except-Exception around
        # `_run_model` silently converted that into an unannounced
        # CLASSICAL_SAFE fallback -- masking a real neural-path failure as
        # ordinary fail-closed behavior. plan_hidden stays None (no
        # fabricated plan-scoring outputs) whenever no plan tensors are
        # supplied; a partially-supplied set (some but not all four fields)
        # remains a hard failure, since that is a real caller defect rather
        # than "no plan yet".
        plan_hidden: Tensor | None = None
        if self.strategist_mode == "candidate_conditioned":
            # core-issues3.txt Phase 4: build plan_hidden from the actual
            # candidate plans (batch-provided), not anonymous learned query
            # positions. Uses `pooled` (not the fuller incident_context
            # computed further below, which itself depends on source_logits)
            # deliberately -- pooled is already available here without
            # reordering any of the surrounding forward-pass computation,
            # which must stay byte-identical for the default
            # anonymous_queries path (reordering blocks containing dropout
            # would change RNG consumption order and silently change that
            # path's own numerical output).
            # .get()-then-raise (not batch["..."]) because these fields are
            # NOT Required on HydroBatch (Section E's schema addition below
            # deliberately left them optional -- HydroBatch cannot express
            # "required only when strategist_mode=candidate_conditioned");
            # matches the codebase's own established convention for every
            # other optional-but-sometimes-mandatory field (e.g. edge_index
            # in the candidate_conditioned link-target branch below).
            plan_template_ids = batch.get("plan_template_ids")
            plan_target_type = batch.get("plan_target_type")
            plan_mask_optional = batch.get("plan_mask")
            plan_features_optional = batch.get("plan_features")
            plan_fields = (
                ("plan_template_ids", plan_template_ids),
                ("plan_target_type", plan_target_type),
                ("plan_mask", plan_mask_optional),
                ("plan_features", plan_features_optional),
            )
            provided_plan_fields = [name for name, value in plan_fields if value is not None]
            if not provided_plan_fields:
                # Incident-only inference: no candidate plans supplied at
                # all. This is the expected, valid shape of PASS 1 -- leave
                # plan_hidden None and fall through without touching any of
                # the candidate-plan-tensor code below.
                pass
            elif len(provided_plan_fields) < len(plan_fields):
                missing_plan_fields = [name for name, value in plan_fields if value is None]
                raise KeyError(
                    "missing HydroBatch fields required by strategist_mode=candidate_conditioned: "
                    f"{', '.join(missing_plan_fields)}"
                )
            else:
                # provided_plan_fields has all 4 names here (neither branch
                # above was taken), so every .get() above returned non-None
                # -- asserted explicitly so static analysis can narrow these
                # from `Tensor | None` to `Tensor` for the rest of this
                # block, matching the runtime guarantee already established
                # above.
                assert plan_template_ids is not None
                assert plan_target_type is not None
                assert plan_mask_optional is not None
                assert plan_features_optional is not None
                plan_mask_tensor = plan_mask_optional.bool()
                plan_features = plan_features_optional.float()
                plans = plan_template_ids.shape[1]
                # core-issues4.txt Section E: validate dimensions/value ranges
                # BEFORE any embedding/gather -- an out-of-range template id or
                # target index previously reached torch.gather/nn.Embedding
                # directly, which either raises an opaque low-level error or
                # (for a gather index within the tensor's storage bounds but
                # semantically wrong) silently reads a real, unrelated node's
                # embedding instead of failing closed.
                if plan_template_ids.shape != (batch_size, plans):
                    raise ValueError("plan_template_ids must have shape [batch, plans]")
                if plan_target_type.shape != (batch_size, plans):
                    raise ValueError("plan_target_type must have shape [batch, plans]")
                if plan_mask_tensor.shape != (batch_size, plans):
                    raise ValueError("plan_mask must have shape [batch, plans]")
                if plan_features.shape != (batch_size, plans, self.plan_feature_dim):
                    raise ValueError(
                        f"plan_features must have shape [batch, plans, {self.plan_feature_dim}]"
                    )
                # Padded plans (plan_mask False) may legitimately carry a
                # sentinel/garbage template id -- CandidatePlanEncoder's own
                # plan_mask handling already zeroes their contribution to
                # plan_hidden -- so range-check only positions the caller
                # marked real.
                if plan_mask_tensor.any():
                    real_template_ids = plan_template_ids[plan_mask_tensor]
                    if bool(((real_template_ids < 0) | (real_template_ids >= self.action_vocabulary_size)).any()):
                        raise ValueError(
                            f"plan_template_ids contains a value outside [0, {self.action_vocabulary_size}) "
                            "at a non-padded (plan_mask=True) position"
                        )
                    real_target_type = plan_target_type[plan_mask_tensor]
                    if bool(((real_target_type < 0) | (real_target_type >= len(TARGET_TYPE_CLASSES))).any()):
                        raise ValueError(
                            f"plan_target_type contains a value outside [0, {len(TARGET_TYPE_CLASSES)}) "
                            "at a non-padded (plan_mask=True) position"
                        )
                target_embedding = hidden.new_zeros(batch_size, plans, self.d_model)
                node_target_index = batch.get("plan_target_node_index")
                if node_target_index is not None:
                    is_node_target = (plan_target_type == TARGET_TYPE_INDEX["NODE"]) & plan_mask_tensor
                    if bool(is_node_target.any()):
                        real_node_targets = node_target_index[is_node_target]
                        if bool(((real_node_targets < 0) | (real_node_targets >= nodes)).any()):
                            raise ValueError(
                                f"plan_target_node_index contains a value outside [0, {nodes}) at a "
                                "real (plan_mask=True, target_type=NODE) position"
                            )
                    # core-issues4.txt Section E: clamp BOTH bounds, not just
                    # min=0 -- an unclamped upper bound previously let an
                    # out-of-range index reach torch.gather directly (an
                    # opaque runtime error at best). Clamping is safe here
                    # specifically because it only affects positions the
                    # `torch.where` below discards (is_node_target is False
                    # wherever the index is not a real, validated node
                    # target) -- padded/NONE-type positions never contribute
                    # their gathered value to target_embedding regardless of
                    # what clamp(...) resolves their index to.
                    safe_node_index = node_target_index.clamp(min=0, max=nodes - 1)
                    gathered_node = torch.gather(
                        role_hidden["strategist"], 1,
                        safe_node_index.unsqueeze(-1).expand(-1, -1, self.d_model),
                    )
                    target_embedding = torch.where(is_node_target.unsqueeze(-1), gathered_node, target_embedding)
                link_target_index = batch.get("plan_target_link_index")
                edge_index_for_targets = batch.get("edge_index")
                if link_target_index is not None and edge_index_for_targets is not None:
                    if edge_index_for_targets.ndim == 2:
                        edge_index_for_targets = edge_index_for_targets.unsqueeze(0).expand(batch_size, -1, -1)
                    edge_count = edge_index_for_targets.shape[-1]
                    is_link_target = (plan_target_type == TARGET_TYPE_INDEX["LINK"]) & plan_mask_tensor
                    if bool(is_link_target.any()):
                        real_link_targets = link_target_index[is_link_target]
                        if bool(((real_link_targets < 0) | (real_link_targets >= edge_count)).any()):
                            raise ValueError(
                                f"plan_target_link_index contains a value outside [0, {edge_count}) at "
                                "a real (plan_mask=True, target_type=LINK) position"
                            )
                    # Same both-bounds-clamped, where-discarded-if-invalid
                    # reasoning as safe_node_index above.
                    safe_link_index = link_target_index.clamp(min=0, max=max(edge_count - 1, 0))
                    edge_src, edge_dst = edge_index_for_targets[:, 0, :], edge_index_for_targets[:, 1, :]
                    src_node_idx = torch.gather(edge_src, 1, safe_link_index)
                    dst_node_idx = torch.gather(edge_dst, 1, safe_link_index)
                    src_embed = torch.gather(
                        role_hidden["strategist"], 1, src_node_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
                    )
                    dst_embed = torch.gather(
                        role_hidden["strategist"], 1, dst_node_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
                    )
                    link_embed = (src_embed + dst_embed) * 0.5
                    target_embedding = torch.where(is_link_target.unsqueeze(-1), link_embed, target_embedding)
                plan_hidden = self.candidate_plan_encoder(
                    template_ids=plan_template_ids,
                    target_type=plan_target_type,
                    target_embedding=target_embedding,
                    plan_features=plan_features,
                    plan_mask=plan_mask_tensor,
                    incident_context=pooled,
                )
        else:
            plan_hidden = self.plan_query_tokens.unsqueeze(0) + pooled[:, None, :]
        pointer_logits: Tensor | None = None
        if plan_hidden is not None:
            plan_hidden = self.adapters["strategist"](plan_hidden)
            pointer_logits = torch.einsum(
                "bqd,bnd->bqn", self.pointer_query(plan_hidden), role_hidden["strategist"]
            ).masked_fill(~node_mask[:, None, :], torch.finfo(hidden.dtype).min)
        source_mask = batch.get("source_candidate_mask", node_mask).bool()
        if source_mask.shape != (batch_size, nodes):
            raise ValueError("source_candidate_mask must have shape [batch, nodes]")
        if not torch.all(source_mask.any(dim=1)):
            raise ValueError("every graph requires at least one source candidate")
        source_logits = self.source_node_head(sentinel_nodes).squeeze(-1)
        if prior is not None and self.prior_mode in ("logit_only", "feature_and_logit"):
            prior_mass = prior.float().clamp_min(1e-8)
            source_logits = source_logits + torch.nn.functional.softplus(
                self.prior_logit_scale
            ) * torch.log(prior_mass)
        source_logits = source_logits.masked_fill(
            ~source_mask, torch.finfo(hidden.dtype).min
        )

        # overnight-plan.txt Task 4.2: incident-level heads (timing/duration/
        # strength/evidence-sufficiency/uncertainty/OOD) use incident_context
        # rather than always the plain masked mean pool, per incident_pooling.
        if self.incident_pooling == "mean":
            incident_context = pooled
        elif self.incident_pooling == "latent":
            incident_context = latents.mean(dim=1)
        elif self.incident_pooling == "attention":
            incident_context = self._attention_pool(hidden, node_mask)
        else:  # "source_conditioned"
            source_weights = torch.softmax(source_logits, dim=-1)
            source_context = torch.einsum("bn,bnd->bd", source_weights, sentinel_nodes)
            sensor_observed = batch.get("sensor_mask")
            sensor_node_mask = (
                sensor_observed.any(dim=1) & node_mask if sensor_observed is not None else node_mask
            )
            sensor_context = self._attention_pool(hidden, sensor_node_mask)
            global_context = latents.mean(dim=1)
            incident_context = self.incident_context_projection(
                torch.cat((source_context, sensor_context, global_context), dim=-1)
            )

        output = HydroOutput(
            hidden_state=hidden,
            latent_state=latents,
            sentinel=role_outputs["sentinel"],
            scout=role_outputs["scout"],
            strategist=role_outputs["strategist"],
            node_mask=node_mask,
            source_node_logits=source_logits,
            source_region_logits=self.source_region_head(incident_context),
            start_time_logits=self._profile_logits(self.profile_heads["start_time"], incident_context, 4),
            duration_logits=self._profile_logits(self.profile_heads["duration"], incident_context, 3),
            relative_strength_logits=self._profile_logits(
                self.profile_heads["relative_strength"], incident_context, 3
            ),
            evidence_sufficiency=self.evidence_head(incident_context).squeeze(-1),
            sensor_fault_logits=self.sensor_fault_head(sentinel_nodes).squeeze(-1),
            sample_node_logits=self.sample_node_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, torch.finfo(hidden.dtype).min),
            expected_information_gain=self.information_gain_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, 0.0),
            uncertainty=self.uncertainty_head(incident_context),
            ood_logits=self.ood_head(incident_context),
        )
        # core-issues5.txt Section 2: plan-scoring outputs are only ever
        # produced when plan_hidden exists -- either the anonymous_queries
        # path (always populated) or candidate_conditioned PASS 2 (real
        # candidate tensors supplied). Incident-only candidate_conditioned
        # inference (plan_hidden is None) must not fabricate these keys;
        # HydroOutput is `total=False`, so simply omitting them is the
        # correct "not available yet" representation, and every downstream
        # reader (SemanticPredictions._model_semantics, callers below) already
        # treats their absence via `.get(...)` as "no plan prediction", not
        # as an error.
        if plan_hidden is not None:
            output["action_logits"] = self.action_head(plan_hidden)
            assert pointer_logits is not None
            output["action_pointer_logits"] = pointer_logits
            output["plan_value"] = self.plan_value_head(plan_hidden).squeeze(-1)
            output["plan_validity_logits"] = self.plan_validity_head(plan_hidden)
        if self.ood_category_head_enabled:
            output["ood_category_logits"] = self.ood_category_head(incident_context)
        if self.scout_control_heads:
            output["candidate_reduction_prediction"] = (
                self.candidate_reduction_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, 0.0)
            )
            output["should_continue_sampling_logits"] = self.should_continue_sampling_head(
                incident_context
            ).squeeze(-1)
        if self.event_control_heads:
            output["event_presence_logits"] = self.event_presence_head(incident_context).squeeze(-1)
            output["event_cause_logits"] = self.event_cause_head(incident_context)
            output["next_step_logits"] = self.next_step_head(incident_context)
        if self.auxiliary_heads:
            output["sensor_reconstruction_prediction"] = (
                self.sensor_reconstruction_head(sentinel_nodes).squeeze(-1).masked_fill(~node_mask, 0.0)
            )
            output["future_concentration_prediction"] = (
                self.future_concentration_head(sentinel_nodes).squeeze(-1).masked_fill(~node_mask, 0.0)
            )
            output["travel_time_prediction"] = (
                self.travel_time_head(sentinel_nodes).squeeze(-1).masked_fill(~node_mask, 0.0)
            )
        if self.consequence_prescreening_heads and plan_hidden is not None:
            for name, head in self.consequence_proxy_heads.items():
                output[name] = head(plan_hidden).squeeze(-1)  # type: ignore[literal-required]
        return output

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def parameter_report(self) -> ParameterReport:
        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())
        temporal_modules = (
            (self.temporal_encoder, self.quality_encoder)
            if self.temporal_dynamics is None
            else (self.temporal_dynamics,)
        )
        encoders = sum(
            count(module)
            for module in (
                self.node_encoder,
                self.graph_encoder,
                *temporal_modules,
                self.modality_fusion,
                self.residual_projection,
                self.prior_projection,
                self.role_projection,
                self.action_projection,
                self.verifier_projection,
            )
        )
        heads = count(self.heads) + sum(
            count(module)
            for module in (
                self.source_node_head,
                self.source_region_head,
                self.sensor_fault_head,
                self.sample_node_head,
                self.information_gain_head,
                self.profile_heads,
                self.evidence_head,
                self.uncertainty_head,
                self.ood_head,
                self.action_head,
                self.plan_value_head,
                self.plan_validity_head,
                self.pointer_query,
            )
        )
        if self.scout_control_heads:
            heads += count(self.candidate_reduction_head) + count(self.should_continue_sampling_head)
        if self.event_control_heads:
            heads += count(self.event_presence_head) + count(self.event_cause_head) + count(self.next_step_head)
        if self.auxiliary_heads:
            heads += (
                count(self.sensor_reconstruction_head)
                + count(self.future_concentration_head)
                + count(self.travel_time_head)
            )
        if self.consequence_prescreening_heads:
            heads += count(self.consequence_proxy_heads)
        if self.strategist_mode == "candidate_conditioned":
            heads += count(self.candidate_plan_encoder)
        if self.ood_category_head_enabled:
            heads += count(self.ood_category_head)
        return ParameterReport(
            total=self.parameter_count(),
            trainable=self.parameter_count(trainable_only=True),
            backbone=count(self.backbone) + self.global_latents.numel() + self.plan_query_tokens.numel() + count(self.final_norm),
            encoders=encoders,
            adapters=count(self.adapters),
            heads=heads,
        )

    def parameter_report_dict(self) -> dict[str, int]:
        return asdict(self.parameter_report())


class NoAdapterHydroCore(HydroCore):
    """Architecture ablation retaining the shared core but removing adapters."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(use_adapters=False, **kwargs)  # type: ignore[arg-type]


class HydroMono(NoAdapterHydroCore):
    """One-shot equal-backbone baseline without specialist adaptation."""
