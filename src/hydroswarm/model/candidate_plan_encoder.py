"""Candidate-conditioned Strategist representation (core-issues3.txt Phase 4).

The audited HydroCore constructs its per-plan representation from anonymous
learned plan-query tokens (`self.plan_query_tokens`, a fixed-size parameter
independent of any incident) plus the pooled incident context -- a model
built this way can only learn "what's usually good at query position q",
never "is THIS specific candidate plan good." CandidatePlanEncoder replaces
that anonymous representation with one built from the actual deterministic
candidate plan's own identity and features, while leaving every downstream
head (action_head, plan_value_head, plan_validity_head,
consequence_proxy_heads, pointer_query) completely unchanged -- they already
operate on a generic `[B, P, D]` "one row per plan" tensor; this module only
changes how that tensor is built.

Critical leakage rule (Phase 4's own "Critical leakage rule"): the features
this module accepts are restricted, by contract, to information available
BEFORE exact WNTR verification -- action-template identity, target
identity, deterministic proposal features (the old heuristic's own
predicted_value/predicted_validity, action count, and similar
pre-verification signals), incident context, and remaining budgets. It has
no parameter for a consequence vector, exact validity, or any other
verification-derived value; there is no way to leak the label into the
input by construction, not just by caller discipline.
"""

from __future__ import annotations

from torch import Tensor, nn

from .encoders import make_activation, make_norm

#: NONE/NODE/LINK, matching hydroswarm.training.strategist_labels.TargetType
#: in fixed order -- 0 must stay NONE (padded/no-target plans embed as the
#: same class the "no target" case already needs).
TARGET_TYPE_CLASSES = ("NONE", "NODE", "LINK")
TARGET_TYPE_INDEX = {name: index for index, name in enumerate(TARGET_TYPE_CLASSES)}


class CandidatePlanEncoder(nn.Module):
    """Builds one `[B, P, D]` row per actual candidate plan.

    Inputs (all pre-verification, see module docstring's leakage rule):

    - `template_ids`: `[B, P]` long, indexes the canonical action-template
      vocabulary (hydroswarm.planning.action_templates.ACTION_TEMPLATES).
    - `target_type`: `[B, P]` long in `{0, 1, 2}` (NONE/NODE/LINK).
    - `target_embedding`: `[B, P, D]` float, the target node/link's own
      representation -- gathered by the caller (HydroCore has the node
      hidden states and edge structure this module deliberately does not
      depend on, keeping this module graph-agnostic and unit-testable with
      synthetic embeddings); zero wherever target_type is NONE.
    - `plan_features`: `[B, P, F]` float, deterministic proposal features
      (e.g. the classical prescreener's own predicted_value/
      predicted_validity, action count, start/duration/magnitude,
      source-region overlap, critical-demand relevance, downstream
      relationship) -- never a verification output.
    - `plan_mask`: `[B, P]` bool, True for real (non-padding) candidates.
    - `incident_context`: `[B, D]` float, the same incident-level context
      other incident-conditioned heads use.
    """

    def __init__(
        self,
        d_model: int,
        *,
        action_vocabulary_size: int,
        plan_feature_dim: int,
        dropout: float = 0.1,
        normalization: str = "rmsnorm",
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.template_embedding = nn.Embedding(action_vocabulary_size, d_model)
        self.target_type_embedding = nn.Embedding(len(TARGET_TYPE_CLASSES), d_model)
        self.feature_projection = nn.Linear(plan_feature_dim, d_model)
        self.incident_projection = nn.Linear(d_model, d_model)
        self.norm = make_norm(normalization, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.output_norm = make_norm(normalization, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        template_ids: Tensor,
        target_type: Tensor,
        target_embedding: Tensor,
        plan_features: Tensor,
        plan_mask: Tensor,
        incident_context: Tensor,
    ) -> Tensor:
        batch, plans = template_ids.shape
        if target_type.shape != (batch, plans):
            raise ValueError("target_type must have shape [batch, plans]")
        if target_embedding.shape[:2] != (batch, plans):
            raise ValueError("target_embedding must have shape [batch, plans, d_model]")
        if plan_mask.shape != (batch, plans):
            raise ValueError("plan_mask must have shape [batch, plans]")

        hidden = (
            self.template_embedding(template_ids)
            + self.target_type_embedding(target_type)
            + target_embedding
            + self.feature_projection(plan_features)
            + self.incident_projection(incident_context)[:, None, :]
        )
        hidden = self.dropout(self.norm(hidden))
        hidden = hidden + self.ffn(hidden)
        hidden = self.output_norm(hidden)
        return hidden.masked_fill(~plan_mask.unsqueeze(-1), 0.0)
