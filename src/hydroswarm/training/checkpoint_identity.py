"""HydroCore-v4 executable checkpoint identity (core-issues4.txt Section A/B).

## Why this exists

`hydroswarm.model.core.HydroCore.architecture_config()` plus
`verify_architecture_compatibility()` (the v3 mechanism) catch a
mismatched `prior_mode`/`incident_pooling`/`message_direction`/per-flag
head-enablement -- config facts that change zero parameter *shapes*, so
`load_state_dict(strict=True)` alone would not catch them. What v3 does
NOT catch, and what motivates this module:

- activation/normalization *kind* (a checkpoint trained with `silu` loaded
  into a model built with `gelu` reloads every tensor at the right shape
  and silently computes different numbers);
- semantic vocabulary ordering (the 9-template action vocabulary, the
  11-category OOD taxonomy, the 5-class EventCause / 4-class NextStep
  order) -- a checkpoint's logits are only meaningful under the exact
  ordering they were trained against;
- scientific-policy identity (PlanValuePolicy version, the signature-
  artifact bucketing policy, the travel-time transform) -- these determine
  what a target VALUE means, not what shape it has;
- which individual learned outputs are actually safe to consult at
  runtime (`hydroswarm.training.output_governance`'s trained/validated/
  runtime-enabled sets), replacing v3's role-only `trained_tasks` gating.

## Design rule this module follows (core-issues4.txt Section A, verbatim
intent)

`model/core.py` stays a leaf module with zero external `hydroswarm`-package
imports (a real, tested invariant -- see `architecture_config()`'s own
docstring). This module is intentionally the layer ABOVE it that reaches
into `planning`/`classical`/`calibration`/`preprocessing` to assemble one
complete identity; it is not a layering violation for THIS module to do
that, only for `model/core.py` to do it.

`HydroCore.from_checkpoint_identity` is attached to the class from here
(at the bottom of this module), not defined inside `core.py` itself, for
exactly that reason: it needs to know about `CheckpointIdentity`, which
needs to know about `core.py`'s own construction arguments, which would be
circular if `core.py` imported this module directly. Attaching a
classmethod from an external module that already imports the class is a
standard way to add extension behavor to a leaf class from a higher layer
without an import cycle. `HydroCore.from_checkpoint_identity(identity)` is
fully drop-in equivalent to a "real" classmethod when called this way.

## v3/v4 separation (the PRIMARY DESIGN RULE)

Nothing here changes `hydroswarm.model.core.ARCHITECTURE_VERSION`
(`"hydrocore-v3"`), `verify_architecture_compatibility`, or
`load_state_dict_with_v2_migration` -- the existing v3 identity/loading
path is untouched. `ARCHITECTURE_VERSION_V4` below is a new, separate
constant this module owns. `save_v4_checkpoint`/`load_v4_checkpoint` write
and require `checkpoint_identity.json`; `hydroswarm.training.checkpoint`'s
existing `load_checkpoint` (the v3 loader) now explicitly refuses to load
any directory that contains that file (see its own updated docstring/
guard), so a v4 checkpoint can never silently load through the legacy
path. A v3 checkpoint directory has no `checkpoint_identity.json`, so
`load_v4_checkpoint` on it raises `NotAV4CheckpointError` rather than
guessing.

## Section D: retain / demote / remove decisions for every existing head

`hydroswarm.training.output_governance.CANONICAL_OUTPUT_NAMES` is the
complete v4 output vocabulary. Several outputs `HydroCore.forward()`
still produces are DELIBERATELY absent from that vocabulary -- by
omission, not oversight -- which means `validate_output_governance`
structurally refuses to let them ever appear in any v4 identity's
trained/validated/runtime_enabled/diagnostic_only/training_only sets.
Recorded here so the reasoning lives next to the mechanism it constrains,
not only in a point-in-time handoff report:

1. **Old 3-logit `ood_head` (OODLevel severity-adjacent)**: no governed
   target exists for it (`hydroswarm.inference.ood.OODDetector`'s
   deterministic severity remains authoritative and does not consume this
   head at all). Absent from the vocabulary rather than demoted to
   diagnostic-only, per Section D item 8 ("Do not present the old
   three-logit learned ood_head as governed severity unless it receives a
   genuine, separately defined target").
2. **`uncertainty` output**: same treatment as (1) -- no defined target,
   no calibration evaluation exists for it (Section D item 10).
3. **`action_logits`/`action_pointer_logits`**: Section D item 6's
   preferred decision -- deterministic candidate plans own action-template
   and target identity; the learned model only ranks/validates/prescreens
   the candidates it is actually given. Both heads remain PHYSICALLY
   constructed in `HydroCore` (removing them would be a breaking, v3-
   incompatible change to shared parameters with no gating flag, not a
   pure-additive one) but are absent from the v4 output vocabulary, so
   they can never be trained/validated/runtime-enabled under v4
   governance regardless of what a caller passes.
4. **Anonymous `heads["sentinel"/"scout"/"strategist"]` RoleHead
   outputs** (the raw, un-interpreted `sentinel_output_dim`/
   `scout_output_dim`/`strategist_output_dim` vectors): same treatment --
   no concrete governed target exists for these generic widths (Section D
   item 9).

Outputs that ARE in the vocabulary but need a caller-level (not
code-level) constraint enforced here:

5. **`future_concentration`**: kept IN the vocabulary (a real target
   concept a correct future implementation could fill in), but
   `build_checkpoint_identity` unconditionally REJECTS it from
   `trained_outputs` -- `future_concentration_target` always returns an
   all-masked placeholder today (core-issues3.txt Phase 7.4's leakage
   fix), so no checkpoint has ever actually trained it; claiming otherwise
   in an identity would be a defect the identity itself should catch, not
   something a downstream consumer has to notice on its own (Section D
   item 11).

Outputs already correctly retained with no change needed:

6. Every governed Sentinel head (Section D item 1), `sample_node`/
   `information_gain` (item 4), candidate-conditioned Strategist as the v4
   Strategist mode (item 5), and the 11-class `ood_category` head as an
   advisory-only signal alongside (never overriding) deterministic
   severity (item 7) -- all already implemented exactly this way; nothing
   to demote.

Section D item 3's two previously-missing Scout heads
(`candidate_reduction_prediction`/`should_continue_sampling_logits`, gated
behind `HydroCore`'s new `scout_control_heads` flag) are real code
additions, not a vocabulary-only decision -- see `model/core.py`'s
`SCOUT_CONTROL_HEADS_DEFAULT` docstring and
`tests/unit/test_scout_control_heads.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from hydroswarm.calibration.conformal import CALIBRATION_SCHEMA_VERSION
from hydroswarm.classical.signature_registry import (
    SCHEMA_VERSION as SIGNATURE_REGISTRY_SCHEMA_VERSION,
    TOPOLOGY_WIDE_REGIME_HASH,
)
from hydroswarm.model.core import (
    ArchitectureCompatibilityError,
    HydroCore,
    INCIDENT_POOLING_MODES,
    MESSAGE_DIRECTIONS,
    OOD_CATEGORY_COUNT,
    PRIOR_MODES,
    STRATEGIST_MODES,
)
from hydroswarm.planning.action_templates import (
    ACTION_TEMPLATE_COUNT,
    ACTION_TEMPLATE_SCHEMA_HASH,
    ACTION_TEMPLATES,
)
from hydroswarm.planning.plan_value_policy import PLAN_VALUE_POLICY_VERSION
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION
from hydroswarm.training.auxiliary_labels import TRAVEL_TIME_TRANSFORM
from hydroswarm.training.corpus import SUPPORTED_EVENT_CAUSES
from hydroswarm.training.ood_categories import OODCategory
from hydroswarm.training.ood_labels import SUPPORTED_OOD_CATEGORIES
from hydroswarm.training.output_governance import (
    OutputGovernanceError,
    validate_output_governance,
)
from hydroswarm.training.scenario_reconstruction import RECONSTRUCTION_POLICY_VERSION
from hydroswarm.training.targets_v2 import TARGETS_V2, TARGETS_V2_SCHEMA_VERSION, EventCause, NextStep

#: New, v4-only architecture-family identity. Deliberately NOT assigned to
#: (or compared against) hydroswarm.model.core.ARCHITECTURE_VERSION, which
#: stays "hydrocore-v3" for the existing checkpoint/runtime path.
ARCHITECTURE_VERSION_V4 = "hydrocore-v4"

#: Filenames a v4 checkpoint directory must contain (Section B).
MODEL_WEIGHTS_FILENAME = "model.safetensors"
OPTIMIZER_STATE_FILENAME = "optimizer_state.pt"
TRAINER_STATE_FILENAME = "trainer_state.json"
IDENTITY_FILENAME = "checkpoint_identity.json"
RESOLVED_CONFIG_FILENAME = "resolved_training_config.json"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"


def _ordered_hash(values: tuple[str, ...]) -> str:
    """Same convention as ACTION_TEMPLATE_SCHEMA_HASH: a fingerprint over
    an ORDERED sequence, so a reordering (not just a membership change) is
    detected -- ordering is semantically load-bearing everywhere this is
    used (a logit vector's index IS the class it names)."""

    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def _target_schema_hash() -> str:
    """Structural fingerprint over the governed targets_v2 contract: which
    targets exist, their category, and whether they are maskable. Does NOT
    hash free-text `definition`/`unit`/`masking_rule`/`source_of_truth`
    prose -- wording clarifications should not force every v4 identity to
    be rebuilt, only changes that affect what data a row must actually
    contain."""

    structural = sorted(
        (spec.name, spec.category, spec.maskable) for spec in TARGETS_V2.values()
    )
    return hashlib.sha256(json.dumps(structural, separators=(",", ":")).encode()).hexdigest()


#: Ordered class vocabularies a v4 identity pins by NAME, not just count --
#: recomputed fresh (not cached at import time from some other module) so a
#: future reordering in ood_categories.py/targets_v2.py is guaranteed to be
#: reflected here without a separate manual update.
OOD_CATEGORY_NAMES: tuple[str, ...] = tuple(category.value for category in OODCategory)
EVENT_CAUSE_NAMES: tuple[str, ...] = tuple(cause.value for cause in EventCause)
NEXT_STEP_NAMES: tuple[str, ...] = tuple(step.value for step in NextStep)

OOD_CATEGORY_SCHEMA_HASH = _ordered_hash(OOD_CATEGORY_NAMES)
EVENT_CAUSE_SCHEMA_HASH = _ordered_hash(EVENT_CAUSE_NAMES)
NEXT_STEP_SCHEMA_HASH = _ordered_hash(NEXT_STEP_NAMES)
TARGET_SCHEMA_HASH = _target_schema_hash()

#: Reserved, forward-declared schema versions for the Scout-state and
#: Strategist-candidate SHARDED dataset layouts core-issues3.txt Phase 10 /
#: core-issues4.txt Section I specify -- those datasets/collators do not
#: exist yet (tracked in this pass's handoff as not-yet-started Phase 10
#: work), so there is no structural schema to hash here yet. Recorded as
#: plain version strings now so a v4 identity has a stable field to bump
#: the day that dataset layout is actually built, rather than adding a new
#: field to the identity contract retroactively.
SCOUT_STATE_SCHEMA_VERSION = "scout-state-v1-unbuilt"
STRATEGIST_CANDIDATE_SCHEMA_VERSION = "strategist-candidate-v1-unbuilt"

#: hydroswarm.model.core.CandidatePlanEncoder / candidate-conditioned
#: forward's plan_features vector width and meaning (core-issues3.txt
#: Phase 4's docstring: predicted_value, predicted_validity, action count,
#: 3 bounded action-parameter scalars). Named here so a future change to
#: that vector's layout is a schema-hash change, not a silent width match.
PLAN_FEATURE_SCHEMA_VERSION = "plan-features-v1"


class CheckpointIdentityError(Exception):
    """Raised when a CheckpointIdentity fails internal validation, or a
    live model does not match one, for a reason not already covered by
    hydroswarm.model.core.ArchitectureCompatibilityError or
    hydroswarm.training.output_governance.OutputGovernanceError."""


class NotAV4CheckpointError(Exception):
    """Raised by load_v4_checkpoint when the target directory has no
    checkpoint_identity.json -- distinguishes "this is a v3 checkpoint (or
    not a checkpoint at all)" from a real identity-validation failure."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    # --- Model construction -------------------------------------------
    architecture_version: str
    variant: str
    node_feature_dim: int
    edge_feature_dim: int
    temporal_feature_dim: int
    quality_feature_dim: int
    role_feature_dim: int
    action_feature_dim: int
    verifier_feature_dim: int
    residual_feature_dim: int
    d_model: int
    nhead: int
    dim_feedforward: int
    num_layers: int
    modality_layers: int
    latent_tokens: int
    plan_queries: int
    use_adapters: bool
    adapter_dims: tuple[int, int, int]
    prior_mode: str
    incident_pooling: str
    message_direction: str
    normalization: str
    activation: str
    dropout: float
    event_control_heads: bool
    auxiliary_heads: bool
    consequence_prescreening_heads: bool
    strategist_mode: str
    plan_feature_dim: int
    ood_category_head: bool
    scout_control_heads: bool
    action_vocabulary_size: int
    sentinel_output_dim: int
    scout_output_dim: int
    strategist_output_dim: int

    # --- Semantic schemas -----------------------------------------------
    action_template_names: tuple[str, ...]
    action_template_count: int
    action_template_schema_hash: str
    ood_category_names: tuple[str, ...]
    ood_category_count: int
    ood_category_schema_hash: str
    supported_ood_categories: tuple[str, ...]
    trained_ood_categories: tuple[str, ...]
    validated_ood_categories: tuple[str, ...]
    event_cause_names: tuple[str, ...]
    supported_event_causes: tuple[str, ...]
    next_step_names: tuple[str, ...]
    next_step_schema_hash: str
    target_schema_version: str
    target_schema_hash: str
    feature_schema_version: str
    feature_schema_hash: str
    plan_feature_schema_version: str
    scout_state_schema_version: str
    strategist_candidate_schema_version: str

    # --- Scientific policies and artifacts -------------------------------
    plan_value_policy_version: str
    signature_artifact_policy_version: str
    signature_registry_schema_version: int
    reconstruction_policy_version: str
    travel_time_transform: str
    normalization_hash: str
    fusion_policy_hash: str
    calibration_policy_version: str
    source_corpus_manifest_hashes: tuple[str, ...]
    teacher_checkpoint_hash: str | None = None

    # --- Output governance (core-issues4.txt Section C) ------------------
    trained_outputs: frozenset[str] = field(default_factory=frozenset)
    validated_outputs: frozenset[str] = field(default_factory=frozenset)
    runtime_enabled_outputs: frozenset[str] = field(default_factory=frozenset)
    diagnostic_only_outputs: frozenset[str] = field(default_factory=frozenset)
    training_only_outputs: frozenset[str] = field(default_factory=frozenset)

    def to_canonical_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Fields whose ORDER is semantically meaningful
        (action_template_names, ood_category_names, event_cause_names,
        next_step_names) are kept in their original order; the output-
        governance sets (genuinely unordered) are sorted for a stable,
        reproducible fingerprint."""

        ordered: dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, frozenset):
                ordered[key] = sorted(value)
            elif isinstance(value, tuple):
                ordered[key] = list(value)
            else:
                ordered[key] = value
        return ordered

    def fingerprint(self) -> str:
        """One deterministic identity fingerprint over the complete
        canonical serialization (Section A's explicit requirement) -- any
        field change, including an output-governance-set change, changes
        this value."""

        canonical = json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def build_checkpoint_identity(
    model: HydroCore,
    *,
    normalization_hash: str,
    fusion_policy_hash: str,
    source_corpus_manifest_hashes: tuple[str, ...],
    trained_outputs: frozenset[str],
    validated_outputs: frozenset[str],
    runtime_enabled_outputs: frozenset[str],
    diagnostic_only_outputs: frozenset[str] = frozenset(),
    training_only_outputs: frozenset[str] = frozenset(),
    teacher_checkpoint_hash: str | None = None,
) -> CheckpointIdentity:
    """Assemble a complete v4 identity from a live model instance plus the
    per-run provenance values that only the caller (a training script)
    knows. Fails closed (raises CheckpointIdentityError) if the model has
    no named variant, does not use the canonical action vocabulary, or if
    the output-governance sets violate output_governance's invariants."""

    config = model.architecture_config()
    variant = config["variant"]
    if not isinstance(variant, str) or not variant:
        raise CheckpointIdentityError(
            "a v4 checkpoint identity requires a named model.variant_name; construct the model "
            "via HydroCore.from_variant(...) or set variant_name explicitly before building an "
            "identity"
        )
    if config["action_vocabulary_size"] != ACTION_TEMPLATE_COUNT:
        raise CheckpointIdentityError(
            f"v4 requires the canonical {ACTION_TEMPLATE_COUNT}-template action vocabulary; "
            f"model was constructed with action_vocabulary_size={config['action_vocabulary_size']!r}"
        )
    try:
        validate_output_governance(
            trained_outputs=trained_outputs,
            validated_outputs=validated_outputs,
            runtime_enabled_outputs=runtime_enabled_outputs,
            diagnostic_only_outputs=diagnostic_only_outputs,
            training_only_outputs=training_only_outputs,
        )
    except OutputGovernanceError as error:
        raise CheckpointIdentityError(f"output governance invalid: {error}") from error
    if "future_concentration" in trained_outputs:
        # core-issues4.txt Section D item 11 / core-issues3.txt Phase 7.4:
        # future_concentration_target always returns an all-masked
        # placeholder today (the disable IS the fix -- the target
        # timestamp leaks directly into the model's own visible input
        # window, proven with a real regression test, not merely
        # inspected). Training "against" an always-masked target trains
        # nothing meaningful, so a v4 identity that claims it as trained is
        # itself a defect, not a fact worth recording. Remove this guard
        # only alongside a real cutoff-aware fix (Section D item 11's other
        # branch), not by silencing it here.
        raise CheckpointIdentityError(
            "future_concentration cannot appear in trained_outputs: its target generator "
            "(core-issues3.txt Phase 7.4) always returns an all-masked placeholder, so no v4 "
            "checkpoint has ever actually trained it"
        )

    # core-issues4.txt Section D item 8: which OOD categories a checkpoint
    # actually trained is corpus-run-specific information this function
    # cannot see (it only has the live model + output names, not the
    # corpus that produced them) -- the honest, currently-knowable fact is
    # only "did this checkpoint train the ood_category head at all", in
    # which case the best available answer is every category the current
    # generator can reproduce (SUPPORTED_OOD_CATEGORIES). A caller with
    # real per-category sample counts from its own training corpus should
    # prefer recording a narrower set directly rather than relying on this
    # default.
    trained_ood = SUPPORTED_OOD_CATEGORIES if "ood_category" in trained_outputs else frozenset()
    return CheckpointIdentity(
        architecture_version=ARCHITECTURE_VERSION_V4,
        variant=variant,
        node_feature_dim=config["node_feature_dim"],
        edge_feature_dim=config["edge_feature_dim"],
        temporal_feature_dim=config["temporal_feature_dim"],
        quality_feature_dim=config["quality_feature_dim"],
        role_feature_dim=config["role_feature_dim"],
        action_feature_dim=config["action_feature_dim"],
        verifier_feature_dim=config["verifier_feature_dim"],
        residual_feature_dim=config["residual_feature_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        num_layers=config["num_layers"],
        modality_layers=config["modality_layers"],
        latent_tokens=config["latent_tokens"],
        plan_queries=config["plan_queries"],
        use_adapters=config["use_adapters"],
        adapter_dims=tuple(config["adapter_dims"]),
        prior_mode=config["prior_mode"],
        incident_pooling=config["incident_pooling"],
        message_direction=config["message_direction"],
        normalization=config["normalization"],
        activation=config["activation"],
        dropout=config["dropout"],
        event_control_heads=config["event_control_heads"],
        auxiliary_heads=config["auxiliary_heads"],
        consequence_prescreening_heads=config["consequence_prescreening_heads"],
        strategist_mode=config["strategist_mode"],
        plan_feature_dim=config["plan_feature_dim"],
        ood_category_head=config["ood_category_head"],
        scout_control_heads=config["scout_control_heads"],
        action_vocabulary_size=config["action_vocabulary_size"],
        sentinel_output_dim=config["sentinel_output_dim"],
        scout_output_dim=config["scout_output_dim"],
        strategist_output_dim=config["strategist_output_dim"],
        action_template_names=ACTION_TEMPLATES,
        action_template_count=ACTION_TEMPLATE_COUNT,
        action_template_schema_hash=ACTION_TEMPLATE_SCHEMA_HASH,
        ood_category_names=OOD_CATEGORY_NAMES,
        ood_category_count=OOD_CATEGORY_COUNT,
        ood_category_schema_hash=OOD_CATEGORY_SCHEMA_HASH,
        supported_ood_categories=tuple(sorted(category.value for category in SUPPORTED_OOD_CATEGORIES)),
        trained_ood_categories=tuple(sorted(category.value for category in trained_ood)),
        validated_ood_categories=(),
        event_cause_names=EVENT_CAUSE_NAMES,
        supported_event_causes=tuple(sorted(cause.value for cause in SUPPORTED_EVENT_CAUSES)),
        next_step_names=NEXT_STEP_NAMES,
        next_step_schema_hash=NEXT_STEP_SCHEMA_HASH,
        target_schema_version=TARGETS_V2_SCHEMA_VERSION,
        target_schema_hash=TARGET_SCHEMA_HASH,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_hash=DEFAULT_FEATURE_SCHEMA.fingerprint,
        plan_feature_schema_version=PLAN_FEATURE_SCHEMA_VERSION,
        scout_state_schema_version=SCOUT_STATE_SCHEMA_VERSION,
        strategist_candidate_schema_version=STRATEGIST_CANDIDATE_SCHEMA_VERSION,
        plan_value_policy_version=PLAN_VALUE_POLICY_VERSION,
        signature_artifact_policy_version=TOPOLOGY_WIDE_REGIME_HASH,
        signature_registry_schema_version=SIGNATURE_REGISTRY_SCHEMA_VERSION,
        reconstruction_policy_version=RECONSTRUCTION_POLICY_VERSION,
        travel_time_transform=TRAVEL_TIME_TRANSFORM,
        normalization_hash=normalization_hash,
        fusion_policy_hash=fusion_policy_hash,
        calibration_policy_version=CALIBRATION_SCHEMA_VERSION,
        source_corpus_manifest_hashes=tuple(source_corpus_manifest_hashes),
        teacher_checkpoint_hash=teacher_checkpoint_hash,
        trained_outputs=frozenset(trained_outputs),
        validated_outputs=frozenset(validated_outputs),
        runtime_enabled_outputs=frozenset(runtime_enabled_outputs),
        diagnostic_only_outputs=frozenset(diagnostic_only_outputs),
        training_only_outputs=frozenset(training_only_outputs),
    )


def validate_checkpoint_identity(identity: CheckpointIdentity) -> None:
    """Fail closed if `identity` does not match what THIS running build's
    canonical schema sources actually produce. Protects against an old,
    stale identity (built against an earlier action-template ordering, OOD
    taxonomy, or target schema) being silently accepted as still valid --
    the whole reason these hashes exist."""

    if identity.architecture_version != ARCHITECTURE_VERSION_V4:
        raise CheckpointIdentityError(
            f"identity.architecture_version={identity.architecture_version!r} is not "
            f"{ARCHITECTURE_VERSION_V4!r}"
        )
    if identity.action_template_names != ACTION_TEMPLATES:
        raise CheckpointIdentityError("identity action_template_names do not match the canonical vocabulary")
    if identity.action_template_schema_hash != ACTION_TEMPLATE_SCHEMA_HASH:
        raise CheckpointIdentityError("identity action_template_schema_hash is stale")
    if identity.action_vocabulary_size != ACTION_TEMPLATE_COUNT:
        raise CheckpointIdentityError("identity action_vocabulary_size is not the canonical count")
    if identity.ood_category_names != OOD_CATEGORY_NAMES:
        raise CheckpointIdentityError("identity ood_category_names do not match the canonical taxonomy order")
    if identity.ood_category_schema_hash != OOD_CATEGORY_SCHEMA_HASH:
        raise CheckpointIdentityError("identity ood_category_schema_hash is stale")
    if identity.event_cause_names != EVENT_CAUSE_NAMES:
        raise CheckpointIdentityError("identity event_cause_names do not match the canonical order")
    if identity.next_step_names != NEXT_STEP_NAMES:
        raise CheckpointIdentityError("identity next_step_names do not match the canonical order")
    if identity.next_step_schema_hash != NEXT_STEP_SCHEMA_HASH:
        raise CheckpointIdentityError("identity next_step_schema_hash is stale")
    if identity.target_schema_hash != TARGET_SCHEMA_HASH:
        raise CheckpointIdentityError("identity target_schema_hash is stale")
    if identity.plan_value_policy_version != PLAN_VALUE_POLICY_VERSION:
        raise CheckpointIdentityError("identity plan_value_policy_version is stale")
    if identity.reconstruction_policy_version != RECONSTRUCTION_POLICY_VERSION:
        raise CheckpointIdentityError("identity reconstruction_policy_version is stale")
    if identity.signature_artifact_policy_version != TOPOLOGY_WIDE_REGIME_HASH:
        raise CheckpointIdentityError("identity signature_artifact_policy_version is stale")
    if identity.travel_time_transform != TRAVEL_TIME_TRANSFORM:
        raise CheckpointIdentityError("identity travel_time_transform is stale")
    if identity.prior_mode not in PRIOR_MODES:
        raise CheckpointIdentityError(f"identity prior_mode {identity.prior_mode!r} is not a known PriorMode")
    if identity.incident_pooling not in INCIDENT_POOLING_MODES:
        raise CheckpointIdentityError(
            f"identity incident_pooling {identity.incident_pooling!r} is not a known IncidentPooling"
        )
    if identity.message_direction not in MESSAGE_DIRECTIONS:
        raise CheckpointIdentityError(
            f"identity message_direction {identity.message_direction!r} is not a known MessageDirection"
        )
    if identity.strategist_mode not in STRATEGIST_MODES:
        raise CheckpointIdentityError(
            f"identity strategist_mode {identity.strategist_mode!r} is not a known StrategistMode"
        )
    unknown_supported = {
        name for name in identity.supported_ood_categories if name not in identity.ood_category_names
    }
    if unknown_supported:
        raise CheckpointIdentityError(f"identity supported_ood_categories has unknown names: {sorted(unknown_supported)}")
    try:
        validate_output_governance(
            trained_outputs=identity.trained_outputs,
            validated_outputs=identity.validated_outputs,
            runtime_enabled_outputs=identity.runtime_enabled_outputs,
            diagnostic_only_outputs=identity.diagnostic_only_outputs,
            training_only_outputs=identity.training_only_outputs,
        )
    except OutputGovernanceError as error:
        raise CheckpointIdentityError(f"output governance invalid: {error}") from error


def _model_construction_kwargs(identity: CheckpointIdentity) -> dict[str, Any]:
    return {
        "node_feature_dim": identity.node_feature_dim,
        "edge_feature_dim": identity.edge_feature_dim,
        "temporal_feature_dim": identity.temporal_feature_dim,
        "quality_feature_dim": identity.quality_feature_dim,
        "role_feature_dim": identity.role_feature_dim,
        "action_feature_dim": identity.action_feature_dim,
        "verifier_feature_dim": identity.verifier_feature_dim,
        "residual_feature_dim": identity.residual_feature_dim,
        "d_model": identity.d_model,
        "nhead": identity.nhead,
        "dim_feedforward": identity.dim_feedforward,
        "num_layers": identity.num_layers,
        "modality_layers": identity.modality_layers,
        "latent_tokens": identity.latent_tokens,
        "plan_queries": identity.plan_queries,
        "action_vocabulary_size": identity.action_vocabulary_size,
        "dropout": identity.dropout,
        "normalization": identity.normalization,
        "activation": identity.activation,
        "sentinel_output_dim": identity.sentinel_output_dim,
        "scout_output_dim": identity.scout_output_dim,
        "strategist_output_dim": identity.strategist_output_dim,
        "adapter_dims": identity.adapter_dims,
        "use_adapters": identity.use_adapters,
        "prior_mode": identity.prior_mode,
        "incident_pooling": identity.incident_pooling,
        "message_direction": identity.message_direction,
        "event_control_heads": identity.event_control_heads,
        "auxiliary_heads": identity.auxiliary_heads,
        "consequence_prescreening_heads": identity.consequence_prescreening_heads,
        "strategist_mode": identity.strategist_mode,
        "plan_feature_dim": identity.plan_feature_dim,
        "ood_category_head": identity.ood_category_head,
        "scout_control_heads": identity.scout_control_heads,
    }


def build_model_from_identity(identity: CheckpointIdentity) -> HydroCore:
    """Reconstruct a HydroCore instance EXCLUSIVELY from an identity's own
    construction fields -- no default-constructor value is ever relied on
    (Section A's explicit requirement). Validates the identity itself
    first (fail closed on a stale/tampered identity before ever
    instantiating a model from it)."""

    validate_checkpoint_identity(identity)
    model = HydroCore(**_model_construction_kwargs(identity))
    model.variant_name = identity.variant
    return model


def verify_model_matches_identity(model: HydroCore, identity: CheckpointIdentity) -> None:
    """Verify every behavior-critical field of a LIVE model instance
    against an identity, beyond what load_state_dict(strict=True) alone
    would catch (Section A's explicit requirement: activation,
    normalization mode, policy hashes, schema ordering, output
    enablement)."""

    validate_checkpoint_identity(identity)
    live = build_checkpoint_identity(
        model,
        normalization_hash=identity.normalization_hash,
        fusion_policy_hash=identity.fusion_policy_hash,
        source_corpus_manifest_hashes=identity.source_corpus_manifest_hashes,
        trained_outputs=identity.trained_outputs,
        validated_outputs=identity.validated_outputs,
        runtime_enabled_outputs=identity.runtime_enabled_outputs,
        diagnostic_only_outputs=identity.diagnostic_only_outputs,
        training_only_outputs=identity.training_only_outputs,
        teacher_checkpoint_hash=identity.teacher_checkpoint_hash,
    )
    # variant is bookkeeping-only (see build_model_from_identity) and is
    # deliberately excluded from this comparison; every other field is
    # behavior-critical and must match exactly.
    mismatched = [
        key
        for key in identity.__dataclass_fields__
        if key != "variant" and getattr(live, key) != getattr(identity, key)
    ]
    if mismatched:
        raise ArchitectureCompatibilityError(
            f"live model does not match checkpoint identity for field(s): {sorted(mismatched)}"
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def save_v4_checkpoint(
    directory: str | Path,
    *,
    model: HydroCore,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    identity: CheckpointIdentity,
    resolved_training_config: dict[str, Any],
    dataset_manifest_hashes: dict[str, str],
    transform_hashes: dict[str, str] | None = None,
    calibration_hash: str | None = None,
) -> Path:
    """Write a complete v4 checkpoint directory (Section B). Verifies the
    live model matches `identity` BEFORE writing anything, so a mismatched
    save is never silently produced."""

    verify_model_matches_identity(model, identity)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)

    tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(tensors, path / MODEL_WEIGHTS_FILENAME)
    model_hash = hashlib.sha256((path / MODEL_WEIGHTS_FILENAME).read_bytes()).hexdigest()

    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        path / OPTIMIZER_STATE_FILENAME,
    )
    _write_json(
        path / TRAINER_STATE_FILENAME,
        {"epoch": epoch, "global_step": global_step, "best_validation_loss": best_validation_loss},
    )
    _write_json(path / IDENTITY_FILENAME, identity.to_canonical_dict())
    _write_json(path / RESOLVED_CONFIG_FILENAME, resolved_training_config)

    identity_bytes = (path / IDENTITY_FILENAME).read_bytes()
    manifest = {
        "model": model_hash,
        "checkpoint_identity": hashlib.sha256(identity_bytes).hexdigest(),
        "checkpoint_identity_fingerprint": identity.fingerprint(),
        "normalization": identity.normalization_hash,
        "transforms": dict(transform_hashes or {}),
        "dataset_manifests": dict(dataset_manifest_hashes),
        "feature_schema": identity.feature_schema_hash,
        "target_schema": identity.target_schema_hash,
    }
    if calibration_hash is not None:
        manifest["calibration"] = calibration_hash
    _write_json(path / ARTIFACT_MANIFEST_FILENAME, manifest)
    return path


def load_v4_checkpoint(
    directory: str | Path,
    *,
    model: HydroCore | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[HydroCore, CheckpointIdentity, dict[str, Any]]:
    """Load and validate a v4 checkpoint (Section B's exact required
    order):

    1. Load and validate checkpoint_identity.json.
    2. Reconstruct the model exclusively from that identity (or verify a
       caller-supplied model against it -- never silently prefer the
       caller's own config over the recorded identity).
    3. Verify every behavior-critical field/schema/policy hash (done as
       part of 1-2, via validate_checkpoint_identity/
       verify_model_matches_identity).
    4. Strict-load weights.
    5. Verify trained/validated/runtime output invariants (done as part of
       1, via validate_output_governance inside validate_checkpoint_identity).
    6. Load optimizer/scheduler state only after the model identity passes.
    7. Fail closed on missing or mismatched v4 metadata.
    """

    path = Path(directory)
    identity_path = path / IDENTITY_FILENAME
    if not identity_path.exists():
        raise NotAV4CheckpointError(
            f"{path} has no {IDENTITY_FILENAME} -- not a v4 checkpoint (use "
            "hydroswarm.training.checkpoint.load_checkpoint for a legacy v3 checkpoint)"
        )
    raw_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity = CheckpointIdentity(
        **{
            key: (frozenset(value) if key.endswith("_outputs") else tuple(value) if isinstance(value, list) else value)
            for key, value in raw_identity.items()
        }
    )
    validate_checkpoint_identity(identity)

    manifest_path = path / ARTIFACT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise CheckpointIdentityError(f"{path} has {IDENTITY_FILENAME} but no {ARTIFACT_MANIFEST_FILENAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_fingerprint = manifest.get("checkpoint_identity_fingerprint")
    if recorded_fingerprint != identity.fingerprint():
        # Either file was hand-edited/corrupted independently of the other
        # after save_v4_checkpoint wrote them together -- a real, detected
        # tamper/corruption signal, not merely a stale-schema mismatch
        # (validate_checkpoint_identity above already covers that case).
        raise CheckpointIdentityError(
            f"{IDENTITY_FILENAME}'s recomputed fingerprint does not match "
            f"{ARTIFACT_MANIFEST_FILENAME}'s recorded checkpoint_identity_fingerprint -- one of "
            "the two files was modified independently of the other after saving"
        )

    if model is None:
        model = build_model_from_identity(identity)
    else:
        verify_model_matches_identity(model, identity)

    state_dict = load_file(path / MODEL_WEIGHTS_FILENAME, device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    trainer_state = json.loads((path / TRAINER_STATE_FILENAME).read_text(encoding="utf-8"))
    if optimizer is not None or scheduler is not None:
        resume = torch.load(path / OPTIMIZER_STATE_FILENAME, map_location="cpu", weights_only=True)
        if optimizer is not None:
            optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resume["scheduler"])
    return model, identity, trainer_state


# See the module docstring's "Design rule this module follows" section for
# why this is attached here rather than defined as a classmethod inside
# model/core.py.
HydroCore.from_checkpoint_identity = classmethod(  # type: ignore[attr-defined]
    lambda cls, identity: build_model_from_identity(identity)
)
