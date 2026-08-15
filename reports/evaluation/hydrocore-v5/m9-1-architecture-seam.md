# Milestone 9.1 preflight: architecture-seam report

Maps the exact current HydroCore dataflow (as of `exp/hydrocore-v5-causal`, starting SHA `c7f7bddba9513e748185cd53fde6c003e7213c79`) and identifies the smallest clean seam at which a continuous-time temporal latent evolution mechanism (Graph Neural ODE / CDE / Stable SDE) can replace the current temporal pathway while preserving as much of HydroCore as possible, per the frozen protocol `docs/evaluation/HYDROCORE_V5_M9_1_PREFLIGHT_PROTOCOL.md` Section 2. Produced BEFORE any candidate architecture was trained or evaluated for predictive performance.

## 1. Current dataflow, in forward-pass order

`HydroCore.forward(batch: HydroBatch)` (`src/hydroswarm/model/core.py`):

1. **`static = self.node_encoder(node_features)`** -- `StaticFeatureEncoder` (`src/hydroswarm/model/encoders.py`), a per-node MLP (`Linear -> activation -> norm`) over `node_features: [batch, nodes, node_feature_dim=19]`. No time, no graph.
2. **`graph = self.graph_encoder(travel_time, reservoir_reachability, demand_centrality)`** -- `GraphStructuralEncoder`, a per-node MLP over 3 scalar graph-*position* features (each `[batch, nodes]`), scale-normalized then projected. **Does not consume `edge_index` at all** -- despite the name, this is a per-node static-feature encoder, not a message-passing step.
3. **`temporal = self.temporal_encoder(temporal_features, sensor_mask, timestamps)`** -- `TemporalEncoder`. Input `temporal_features: [batch, steps, nodes, temporal_feature_dim=6]`, `sensor_mask: [batch, steps, nodes]` (or matching feature shape), `timestamps: [batch, steps]`. Reshapes to `[batch*nodes, steps, d_model]` (per-node, independent across nodes), adds a sinusoidal position embedding derived from `elapsed = timestamps - timestamps[:, :1]` (then `window_relative` or `fixed_scale` normalization -- see Section 3), runs an `nn.TransformerEncoder` over the step axis with `src_key_padding_mask`, masked-mean-pools over valid steps back to `[batch, nodes, d_model]`. **No cross-node interaction anywhere in this module.**
4. **`quality = self.quality_encoder(quality_features, quality_mask, timestamps)`** -- `QualityEncoder(TemporalEncoder)`, structurally identical to step 3, independent weights, over `quality_features: [batch, steps, nodes, quality_feature_dim=4]`.
5. **`hidden = self.modality_fusion(cat(static, graph, temporal, quality, dim=-1))`** -- `Linear(4*d_model, d_model) -> activation -> norm -> dropout`. This is the fusion/readout point where the four modality latents combine into one per-node hidden state, still purely per-node (no cross-node mixing yet).
6. `hidden = hidden + residual_projection(residual_features)` (if present); `hidden = hidden + prior_projection(classical_prior)` (if `prior_mode` includes `feature_only`/`feature_and_logit`); `hidden = hidden + context` (role/action/verifier-feedback pooled context, broadcast to all nodes).
7. `hidden = hidden.masked_fill(~node_mask.unsqueeze(-1), 0.0)`.
8. **`for block in self.backbone: hidden, latents = block(hidden, latents, safe_mask, edge_index, edge_features, edge_mask)`** -- `self.backbone: nn.ModuleList[LatentHydraulicBlock]`, `num_layers=4` for the "small" variant. **This is the first and only point in the forward pass where `edge_index`/`edge_features`/`edge_mask` are consumed at all**, and the first point where any node's representation depends on any other node's. Per `LatentHydraulicBlock` (`src/hydroswarm/model/layers.py`), each of the 4 layers: (a) `EdgeAwareGraphConv` local graph message-passing with residual (sparse, `edge_index`-gathered, mean-aggregated via `index_add_`, matching a PyG-style convention, not a dense adjacency); (b) node-to-latent cross-attention update into `latents`; (c) latent self-attention; (d) latent-to-node cross-attention update back into `hidden`; (e) a feed-forward residual on `hidden`. **No time axis exists anywhere inside this loop** -- it is purely spatial/graph + bounded global-latent-token attention, run once per example (not once per timestep).
9. `hidden = self.final_norm(hidden)`, masked; `pooled = masked-mean(hidden, node_mask)`; per-role adapter/head application (`sentinel`/`scout`/`strategist`), producing `source_node_logits = source_node_head(role_hidden["sentinel"])` (per-node) and the other task heads from `pooled`/`incident_context`.

## 2. Key finding: HydroCore has no per-timestep recurrence today

Steps 3-4 consume the ENTIRE temporal window (`[batch, steps, nodes, features]`) in one non-causal-masked transformer pass (it attends freely over all valid steps within the pre-truncated causal prefix, then mean-pools), collapsing it to a single per-node vector BEFORE any graph reasoning happens. The spatial backbone (step 8) then runs `num_layers` rounds of graph-aware computation exactly once per example, with zero notion of elapsed time. "Temporal latent evolution" in the current architecture is therefore entirely the mechanism in steps 3-4: turning a per-node measurement history into one latent vector. This is exactly the mechanism a Graph ODE/CDE/SDE should replace -- not the spatial backbone, which already does the graph-aware reasoning the milestone instructions want the new temporal mechanism to ALSO gain access to (Section 6 of the milestone instructions: "The vector field MUST be graph-aware... do not flatten the whole graph into one generic MLP").

## 3. Shapes and locations (as required by the milestone instructions, Section 3)

| Item | Shape / location |
|---|---|
| Node feature dim | `node_feature_dim=19` (`node_features: [batch, nodes, 19]`) |
| Edge feature dim | `edge_feature_dim=13` (`edge_features: [batch, edges, 13]`) |
| Temporal feature dim | `temporal_feature_dim=6` (default; 7 under M8.7 Arm C, not promoted) |
| Quality feature dim | `quality_feature_dim=4` (default; 5 under M8.7 Arm C, not promoted) |
| Timestamp tensor | `timestamps: [batch, steps]`, elapsed incident-relative seconds (already origin-relative at the `HydraulicFeatureBuilder` level; see Section 5) |
| Sensor/history tensors | `temporal_features: [batch, steps, nodes, 6]`, `quality_features: [batch, steps, nodes, 4]` |
| Masks / quality tensors | `node_mask: [batch, nodes]`, `sensor_mask: [batch, steps, nodes]` (or feature-matching), `quality_mask` same shape as `quality_features`' mask, `edge_mask: [batch, edges]` |
| Graph connectivity | `edge_index: [batch, 2, edges]` (source row, target row) -- sparse index list, gathered via `EdgeAwareGraphConv._aggregate`'s per-batch-item loop + `index_add_` scatter-mean; NOT a dense adjacency matrix |
| Node encoder location | `HydroCore.node_encoder` = `StaticFeatureEncoder` (`encoders.py`) |
| Graph/message-passing location | `HydroCore.backbone` = `nn.ModuleList[LatentHydraulicBlock]` (`layers.py`), the ONLY place `edge_index` is consumed |
| Current temporal encoder location | `HydroCore.temporal_encoder` / `HydroCore.quality_encoder` = `TemporalEncoder` / `QualityEncoder` (`encoders.py`) -- this is the seam |
| Fusion/readout point | `HydroCore.modality_fusion` (`nn.Sequential(Linear(4*d_model, d_model), activation, norm, dropout)`) |
| `source_node_logits` head input | `role_hidden["sentinel"]`, i.e. `self.adapters["sentinel"]` applied to the post-backbone per-node `hidden` state, via `self.heads["sentinel"]` / `self.source_node_head` |
| Other task-head inputs | `pooled` (masked-mean over nodes) or `incident_context` (mode-dependent per `incident_pooling`), post-backbone |
| Parameter count by module | `HydroCore.parameter_report() -> ParameterReport(total, trainable, backbone, encoders, adapters, heads)` already exists and is reused directly for M9.1 parameter accounting (see `m9-1-parameter-matching.json`) |

## 4. Proposed seam (frozen before implementation)

Replace **only** `self.temporal_encoder` / `self.quality_encoder`'s role in producing the `(temporal, quality)` inputs to `modality_fusion`. Everything else -- `node_encoder`, `graph_encoder`, `modality_fusion`'s own signature, `backbone`, `final_norm`, pooling, every adapter/head, the multitask objective, classical-prior/fusion/OOD/planning semantics -- is untouched, matching the milestone's own stated preference:

> SAME: preprocessing, node/static encoders, graph inputs, output heads, multitask objective, safety interfaces. DIFFERENT: temporal latent evolution mechanism.

Mechanism: `HydroCore.__init__` gains one new, strictly additive constructor argument, `temporal_dynamics: TemporalDynamicsBase | None = None`. Default `None` preserves today's exact behavior (`temporal_encoder`/`quality_encoder` built and used exactly as before -- zero change for every existing caller/checkpoint). When provided, `forward()` calls `self.temporal_dynamics(temporal_features, quality_features, sensor_mask, quality_mask, timestamps, node_mask, edge_index, edge_features, edge_mask)` once, receiving `(temporal_latent, quality_latent)` (each `[batch, nodes, d_model]`), fed into the identical, unmodified `modality_fusion` call. This is why the new mechanism needs `edge_index`/`edge_features`/`edge_mask` routed to it (they are already available in `batch` at the top of `forward()`, just not currently passed this early) -- it is what makes the Graph ODE/CDE/SDE vector fields graph-aware in a way the current `TemporalEncoder` structurally is not.

No architecture-specific output heads are introduced. `TemporalDynamicsBase` and its four implementations (`CurrentTemporalDynamics`/`GraphODEDynamics`/`GraphCDEDynamics`/`GraphSDEDynamics`) live in the new, experiment-scoped `src/hydroswarm/model/continuous_time.py`, with lazy `torchdiffeq`/`torchcde`/`torchsde` imports so normal production `import hydroswarm` is unaffected by their absence. No experimental architecture is wired into any production factory/default path; no production model is promoted by this milestone.

locked tests opened: before=False, after=False (checked via `hydroswarm.evaluation.live_robustness.locked_test_opened`, reading `reports/results/v4/architecture-freeze.json["locked_test_opened"]`, unaffected by any change in this report). No development-holdout or locked data was read to produce this report -- it describes only architecture code already committed to `exp/hydrocore-v5-causal`.
