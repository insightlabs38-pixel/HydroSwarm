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
        self.d_model = d_model
        self.num_layers = num_layers
        self.latent_tokens_count = latent_tokens
        self.use_adapters = use_adapters
        self.prior_mode = prior_mode
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
        }

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
        return HydroOutput(
            hidden_state=hidden,
            latent_state=latents,
            sentinel=role_outputs["sentinel"],
            scout=role_outputs["scout"],
            strategist=role_outputs["strategist"],
            node_mask=node_mask,
            source_node_logits=source_logits,
            source_region_logits=self.source_region_head(sentinel_nodes).squeeze(-1).masked_fill(~node_mask, torch.finfo(hidden.dtype).min),
            start_time_logits=self._profile_logits(self.profile_heads["start_time"], pooled, 4),
            duration_logits=self._profile_logits(self.profile_heads["duration"], pooled, 3),
            relative_strength_logits=self._profile_logits(
                self.profile_heads["relative_strength"], pooled, 3
            ),
            evidence_sufficiency=self.evidence_head(pooled),
            sensor_fault_logits=self.sensor_fault_head(sentinel_nodes).squeeze(-1),
            sample_node_logits=self.sample_node_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, torch.finfo(hidden.dtype).min),
            expected_information_gain=self.information_gain_head(scout_nodes).squeeze(-1).masked_fill(~node_mask, 0.0),
            action_logits=self.action_head(plan_hidden),
            action_pointer_logits=pointer_logits,
            plan_value=self.plan_value_head(plan_hidden).squeeze(-1),
            plan_validity_logits=self.plan_validity_head(plan_hidden),
            uncertainty=self.uncertainty_head(pooled),
            ood_logits=self.ood_head(pooled),
        )

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
