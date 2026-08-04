"""HydroCore shared scientific backbone and semantic specialist heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Required, TypedDict, get_args

import torch
from torch import Tensor, nn

from .adapters import BottleneckAdapter, RoleHead
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
ARCHITECTURE_VERSION = "hydrocore-v2"

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

#: overnight-plan.txt Task 4.4. Class counts and index order for the
#: event/next-step control heads must exactly match
#: hydroswarm.training.targets_v2.EventCause / NextStep's enum member
#: order (CONTAMINATION, SENSOR_FAULT, HYDRAULIC_MISMATCH, AMBIGUOUS,
#: NORMAL / COLLECT_SAMPLE, INSPECT_SENSOR, GENERATE_PLANS, ABSTAIN). Not
#: imported directly to avoid a model<->training import cycle. For
#: event_cause, hydroswarm.training.corpus.EVENT_CAUSE_INDEX (built via
#: enumerate(EventCause)) is the actual label encoder and this count is
#: checked against it by tests/unit/test_event_control_heads.py; next_step
#: label generation is not yet built (governed by the deterministic
#: controller policy from a later task), so only the head shape is fixed
#: here, matching len(NextStep).
EVENT_CAUSE_CLASS_COUNT = 5
NEXT_STEP_CLASS_COUNT = 4

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
        self.d_model = d_model
        self.num_layers = num_layers
        self.latent_tokens_count = latent_tokens
        self.use_adapters = use_adapters
        self.prior_mode = prior_mode
        self.incident_pooling = incident_pooling
        self.message_direction = message_direction
        self.event_control_heads = event_control_heads
        self.auxiliary_heads = auxiliary_heads
        self.consequence_prescreening_heads = consequence_prescreening_heads
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
        )
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
        self.source_region_head = RoleHead(d_model, 1)
        self.sensor_fault_head = RoleHead(d_model, 1)
        self.sample_node_head = RoleHead(d_model, 1)
        self.information_gain_head = nn.Sequential(make_norm(normalization, d_model), nn.Linear(d_model, 1), nn.Softplus())
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
        return cls(**values)  # type: ignore[arg-type]

    def architecture_config(self) -> dict[str, object]:
        """Config dimensions load_state_dict's shape check would not itself
        catch if mismatched (overnight-plan.txt Task 4.0). Callers persist
        this into checkpoint metadata (see export_model's `metadata` arg)
        so verify_architecture_compatibility() can detect a checkpoint
        being loaded into a model configured differently than it was
        trained with."""

        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "prior_mode": self.prior_mode,
            "incident_pooling": self.incident_pooling,
            "message_direction": self.message_direction,
            "event_control_heads": self.event_control_heads,
            "auxiliary_heads": self.auxiliary_heads,
            "consequence_prescreening_heads": self.consequence_prescreening_heads,
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
        temporal = self.temporal_encoder(
            batch["temporal_features"], batch.get("sensor_mask"), batch.get("timestamps")
        )
        quality = self.quality_encoder(
            batch["quality_features"], batch.get("quality_mask"), batch.get("timestamps")
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
        plan_hidden = self.plan_query_tokens.unsqueeze(0) + pooled[:, None, :]
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
            source_region_logits=self.source_region_head(sentinel_nodes).squeeze(-1).masked_fill(~node_mask, torch.finfo(hidden.dtype).min),
            start_time_logits=self._profile_logits(self.profile_heads["start_time"], incident_context, 4),
            duration_logits=self._profile_logits(self.profile_heads["duration"], incident_context, 3),
            relative_strength_logits=self._profile_logits(
                self.profile_heads["relative_strength"], incident_context, 3
            ),
            evidence_sufficiency=self.evidence_head(incident_context).squeeze(-1),
            sensor_fault_logits=self.sensor_fault_head(sentinel_nodes).squeeze(-1),
            sample_node_logits=self.sample_node_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, torch.finfo(hidden.dtype).min),
            expected_information_gain=self.information_gain_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, 0.0),
            action_logits=self.action_head(plan_hidden),
            action_pointer_logits=pointer_logits,
            plan_value=self.plan_value_head(plan_hidden).squeeze(-1),
            plan_validity_logits=self.plan_validity_head(plan_hidden),
            uncertainty=self.uncertainty_head(incident_context),
            ood_logits=self.ood_head(incident_context),
        )
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
        if self.consequence_prescreening_heads:
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
        encoders = sum(
            count(module)
            for module in (
                self.node_encoder,
                self.graph_encoder,
                self.temporal_encoder,
                self.quality_encoder,
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
