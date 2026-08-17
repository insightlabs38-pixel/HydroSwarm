"""M10.3A Strategist candidate-schema/supervision/representation refit --
frozen protocol constants, hashed BEFORE any Level-A training/validation
result is inspected.

Frozen document: `docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`.
Imported by the readiness audit, training, gate, and decide scripts so all
four read the exact same frozen values -- no value here may be changed
after any Level-A result is inspected.

Governs Level A (frozen backbone, Strategist-specialist-side training) and,
only if Level A's frozen gate legitimately fails for a representation-
capacity reason, Level B (one predeclared, bounded partial shared-backbone
unfreeze). Does NOT govern the true M10.3 learned-vs-deterministic
Strategist scientific comparison -- that is a separately authorized later
task, exactly like the M10.2 Scout refit precedent this protocol mirrors.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# Section 1: teacher checkpoints (frozen, immutable). Every Level-A/B run
# starts from one of the three canonical M9.6 checkpoints -- NEVER the
# M10.2 Scout-refit checkpoints (the authorizing task's own explicit
# instruction: Scout's Level-A refit modified Scout-specific support
# pathways [role_projection/residual_projection] that are irrelevant, and
# potentially confounding, to an independent Strategist characterization).
# ---------------------------------------------------------------------------

PARENT_M9_6_TEACHER_SHA256: dict[int, str] = {
    20260814: "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5",
    31874: "527e707b988870dff323b6a3e0f1bde36a152b7bbe7e4c7f4bf0e59cdaf19332",
    20260815: "b45f5afeabba820270130595dffb44152ec12fad6fa383cce67013f14524113c",
}

# ---------------------------------------------------------------------------
# Section 2: populations (development-only, disjoint from every locked
# split, from every M10.1/M10.2 seed range, and from the future true-M10.3
# policy-comparison population). Family scope: golden-reference only --
# matches the M10.2 Scout refit's own precedent (a deliberately bounded,
# single-family pilot scope, not a result-driven narrowing since it is
# frozen here before any Level-A result exists). Seed namespace: role
# "strategist_refit_m10_3", base 1_300_000_000 -- continues the M10.1
# (1_100_000_000) / M10.2 (1_200_000_000) per-milestone seed-block
# convention; verified disjoint by grep over 1_300_000_000..1_399_999_999
# (zero prior hits) before this document was frozen.
# ---------------------------------------------------------------------------

FAMILY = "golden-reference"
DEPTH = 25  # MATURE bucket -- matches the M10.2 Scout refit's own frozen depth.
MAXIMUM_PLAN_COUNT = 9  # ACTION_TEMPLATE_COUNT; the governed candidate-count upper bound, never invented.

TRAIN_SEED_BASE = 1_300_000_000
TRAIN_COUNT = 250  # matches the M10.2 Scout refit's own frozen TRAIN_COUNT.
VALIDATION_SEED_BASE = 1_300_100_000
#: Set directly to 300 (not 100) from the start: the M10.2 Scout refit's own
#: history shows a 100-scenario validation pool can produce too few
#: real-recommendation-bearing examples for its own frozen GATE_MIN_SUPPORT,
#: requiring a same-day, pre-metric amendment. Strategist candidates are
#: denser per scenario (up to 9 real candidates/incident vs Scout's
#: at-most-one recommendation/step), so 300 is expected to be generously
#: sufficient, not merely "enough" -- a resourcing choice informed by prior
#: programmatic experience, frozen here BEFORE any Level-A result exists,
#: not a result-driven change within this protocol's own execution.
VALIDATION_COUNT = 300
SOURCE_ROUND_ROBIN = True  # matches the M10.2 Scout refit's own convention.

# ---------------------------------------------------------------------------
# Section 3: candidate-plan training schema (frozen, versioned; see
# `hydroswarm.training.strategist_candidate_corpus` for the full channel-
# wiring documentation).
# ---------------------------------------------------------------------------

CANDIDATE_TRAINING_SCHEMA_VERSION = "strategist-candidate-training-v1"

# ---------------------------------------------------------------------------
# Section 4: canonical trainable target set (frozen). action_template/
# target_pointer are DELIBERATELY EXCLUDED -- repository evidence
# (configs/training-v5-causal.yaml's own comment: "v3-legacy head; still
# trained by the unmodified default v3 model") establishes these are not
# part of this experiment's canonical scope; broadening scope to include
# them would be a material, undisclosed design change this protocol does
# not authorize.
# ---------------------------------------------------------------------------

STRATEGIST_TARGET_KEYS: tuple[str, ...] = (
    "plan_validity",
    "plan_value",
    "exposure_proxy",
    "pressure_risk_proxy",
    "service_loss_proxy",
    "containment_time_proxy",
    "plan_regret_proxy",
)

#: Frozen task weights -- copied verbatim from `configs/training-v5-causal.yaml`
#: (the repository's own existing, already-governed values), never re-derived
#: or tuned for this protocol.
TASK_WEIGHTS: dict[str, float] = {
    "plan_validity": 1.0,
    "plan_value": 0.5,
    "exposure_proxy": 0.3,
    "pressure_risk_proxy": 0.3,
    "service_loss_proxy": 0.3,
    "containment_time_proxy": 0.3,
    "plan_regret_proxy": 0.3,
}

# ---------------------------------------------------------------------------
# Section 5: Level-A trainable parameter allowlist (frozen, exact, forward-
# graph-traced -- mechanically verified via a real forward+backward pass
# before this protocol was frozen: every one of these 40 parameters
# receives real, nonzero, finite gradient; `action_head`/`pointer_query`
# (the two OTHER modules `plan_hidden` feeds) receive exactly zero
# gradient and are correctly excluded; `adapters["strategist"]` is
# `nn.Identity()` under `use_adapters=False` [M9.6's own construction],
# zero parameters, nothing to freeze/unfreeze). No shared backbone
# component is needed at all -- unlike Scout, no new backbone-injection
# layer exists on this path (`plan_hidden` is built entirely from
# `CandidatePlanEncoder`, which consumes the ALREADY-COMPUTED `pooled`
# incident context; no analogue of Scout's role_projection/
# residual_projection is required).
# ---------------------------------------------------------------------------

LEVEL_A_PARAMETER_ALLOWLIST: tuple[str, ...] = (
    "candidate_plan_encoder.template_embedding.weight",
    "candidate_plan_encoder.target_type_embedding.weight",
    "candidate_plan_encoder.feature_projection.weight",
    "candidate_plan_encoder.feature_projection.bias",
    "candidate_plan_encoder.incident_projection.weight",
    "candidate_plan_encoder.incident_projection.bias",
    "candidate_plan_encoder.norm.weight",
    "candidate_plan_encoder.ffn.0.weight",
    "candidate_plan_encoder.ffn.0.bias",
    "candidate_plan_encoder.ffn.3.weight",
    "candidate_plan_encoder.ffn.3.bias",
    "candidate_plan_encoder.output_norm.weight",
    "plan_value_head.network.0.weight",
    "plan_value_head.network.0.bias",
    "plan_value_head.network.1.weight",
    "plan_value_head.network.1.bias",
    "plan_validity_head.network.0.weight",
    "plan_validity_head.network.0.bias",
    "plan_validity_head.network.1.weight",
    "plan_validity_head.network.1.bias",
    "consequence_proxy_heads.exposure_proxy.network.0.weight",
    "consequence_proxy_heads.exposure_proxy.network.0.bias",
    "consequence_proxy_heads.exposure_proxy.network.1.weight",
    "consequence_proxy_heads.exposure_proxy.network.1.bias",
    "consequence_proxy_heads.pressure_risk_proxy.network.0.weight",
    "consequence_proxy_heads.pressure_risk_proxy.network.0.bias",
    "consequence_proxy_heads.pressure_risk_proxy.network.1.weight",
    "consequence_proxy_heads.pressure_risk_proxy.network.1.bias",
    "consequence_proxy_heads.service_loss_proxy.network.0.weight",
    "consequence_proxy_heads.service_loss_proxy.network.0.bias",
    "consequence_proxy_heads.service_loss_proxy.network.1.weight",
    "consequence_proxy_heads.service_loss_proxy.network.1.bias",
    "consequence_proxy_heads.containment_time_proxy.network.0.weight",
    "consequence_proxy_heads.containment_time_proxy.network.0.bias",
    "consequence_proxy_heads.containment_time_proxy.network.1.weight",
    "consequence_proxy_heads.containment_time_proxy.network.1.bias",
    "consequence_proxy_heads.plan_regret_proxy.network.0.weight",
    "consequence_proxy_heads.plan_regret_proxy.network.0.bias",
    "consequence_proxy_heads.plan_regret_proxy.network.1.weight",
    "consequence_proxy_heads.plan_regret_proxy.network.1.bias",
)

# ---------------------------------------------------------------------------
# Section 6: optimizer / schedule (frozen, not tuned, not swept -- reuses
# the M10.2 Scout refit's own exact values for cross-milestone consistency,
# never re-derived).
# ---------------------------------------------------------------------------

OPTIMIZER = "Adam"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 8
EPOCHS = 20
CHECKPOINT_SELECTION = "FINAL_EPOCH"

# ---------------------------------------------------------------------------
# Section 7: statistics (frozen; reused unmodified from the M9/M10 cross-
# milestone convention).
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.90
BOOTSTRAP_SEED = 20260819
GATE_MIN_SUPPORT = 20

# ---------------------------------------------------------------------------
# Section 8: Level B (frozen scope, defined here BEFORE Level-A results,
# run only if Section 9's escalation condition fires). Matches the M10.2
# Scout refit's own precedent exactly: Level-A parameters plus the LAST
# shared backbone block (`backbone[3]`, the 4th of 4 `LatentHydraulicBlock`
# modules in the `small` variant) plus its associated `final_norm.weight`.
# ---------------------------------------------------------------------------

#: Every `backbone[3]` (the LAST of the 4 `LatentHydraulicBlock` modules in
#: the `small` variant's backbone -- `len(model.backbone) == 4`, confirmed
#: against a real model instance before this protocol was frozen) parameter,
#: by name, plus `final_norm.weight` -- 25 parameters total. Matches the
#: M10.2 Scout refit protocol's own Level-B scope definition exactly (same
#: block index, same rationale: one fixed, tightly bounded tail, never
#: progressively widened).
LEVEL_B_EXTRA_PARAMETER_ALLOWLIST: tuple[str, ...] = (
    "backbone.3.local.source.weight",
    "backbone.3.local.edge.weight",
    "backbone.3.local.output.weight",
    "backbone.3.local.output.bias",
    "backbone.3.local.norm.weight",
    "backbone.3.node_norm.weight",
    "backbone.3.latent_norm.weight",
    "backbone.3.to_latent.in_proj_weight",
    "backbone.3.to_latent.in_proj_bias",
    "backbone.3.to_latent.out_proj.weight",
    "backbone.3.to_latent.out_proj.bias",
    "backbone.3.latent_self.in_proj_weight",
    "backbone.3.latent_self.in_proj_bias",
    "backbone.3.latent_self.out_proj.weight",
    "backbone.3.latent_self.out_proj.bias",
    "backbone.3.to_node.in_proj_weight",
    "backbone.3.to_node.in_proj_bias",
    "backbone.3.to_node.out_proj.weight",
    "backbone.3.to_node.out_proj.bias",
    "backbone.3.feed_forward.0.weight",
    "backbone.3.feed_forward.1.weight",
    "backbone.3.feed_forward.1.bias",
    "backbone.3.feed_forward.4.weight",
    "backbone.3.feed_forward.4.bias",
    "final_norm.weight",
)
LEVEL_B_BACKBONE_BLOCK_INDEX = 3
LEVEL_B_INCLUDES_FINAL_NORM = True

# ---------------------------------------------------------------------------
# Section 9: promotion/escalation rule (frozen BEFORE any Level-A result is
# inspected).
# ---------------------------------------------------------------------------


def payload() -> dict[str, object]:
    return {
        "parent_m9_6_teacher_sha256": PARENT_M9_6_TEACHER_SHA256,
        "family": FAMILY,
        "depth": DEPTH,
        "maximum_plan_count": MAXIMUM_PLAN_COUNT,
        "train_seed_base": TRAIN_SEED_BASE,
        "train_count": TRAIN_COUNT,
        "validation_seed_base": VALIDATION_SEED_BASE,
        "validation_count": VALIDATION_COUNT,
        "source_round_robin": SOURCE_ROUND_ROBIN,
        "candidate_training_schema_version": CANDIDATE_TRAINING_SCHEMA_VERSION,
        "strategist_target_keys": list(STRATEGIST_TARGET_KEYS),
        "task_weights": TASK_WEIGHTS,
        "level_a_parameter_allowlist": list(LEVEL_A_PARAMETER_ALLOWLIST),
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_ci": BOOTSTRAP_CI,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate_min_support": GATE_MIN_SUPPORT,
        "level_b_backbone_block_index": LEVEL_B_BACKBONE_BLOCK_INDEX,
        "level_b_includes_final_norm": LEVEL_B_INCLUDES_FINAL_NORM,
        "level_b_extra_parameter_allowlist": list(LEVEL_B_EXTRA_PARAMETER_ALLOWLIST),
    }


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(payload(), sort_keys=True, default=str).encode()).hexdigest()


def to_json_doc() -> dict[str, object]:
    return {"kind": "M10_3_REFIT_PROTOCOL", "milestone": "M10.3A-refit", **payload(), "protocol_hash": protocol_hash()}
