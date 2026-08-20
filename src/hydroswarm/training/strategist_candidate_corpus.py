"""M10.3A Strategist candidate-schema/supervision/representation refit
amendment, Parts 2-4: real, leakage-safe (INPUT, TARGET) pairing for
genuine supervised candidate-conditioned Strategist training.

Frozen audit/protocol document:
`docs/evaluation/HYDROCORE_V5_M10_3_STRATEGIST_PREFLIGHT_REFIT.md`.

## Why this module exists

`hydroswarm.training.checkpoint_identity.STRATEGIST_CANDIDATE_SCHEMA_VERSION
= "strategist-candidate-v1-unbuilt"` names a *training-corpus* dataset
layout that genuinely still does not exist -- this module does not change
that placeholder and does not claim otherwise (same convention the M10.2
Scout refit amendment already established for
`SCOUT_STATE_SCHEMA_VERSION`). What this module builds is the missing
INPUT-tensor-construction glue connecting two ALREADY-REAL, ALREADY-GOVERNED
pieces that were never previously wired together for training:

- `hydroswarm.planning.candidate_tensorizer.plan_proposals_to_candidate_tensors`
  -- the ONE canonical, already-live-production-used (`HybridInferencePipeline.
  _score_candidate_plans`) converter from a real, bounded, deterministic
  `PlanProposal` set to `HydroBatch`'s candidate-conditioned INPUT fields.
  Reused here UNMODIFIED -- this module does not reimplement it.
- `hydroswarm.training.strategist_trajectory.build_strategist_trajectory` --
  the already-real, already-leakage-audited offline TARGET generator
  (exact WNTR verification via `PlanVerifier`, exact plan_value/regret/
  proxy computation via the governed `plan_value_policy.evaluate_plan_value`).
  Reused here UNMODIFIED for the TARGET side.

Neither piece was ever exercised during M9.6 training (Part 1's readiness
audit): `strategist_trajectory.py`'s own docstring already notes it wires
`generate_strategist_labels` into `TrajectoryState`, but nothing in the
repository ever converted its `labels`/`targets` output into `HydroBatch`'s
`plan_template_ids`/`plan_target_type`/`plan_target_node_index`/
`plan_target_link_index`/`plan_features`/`plan_mask` INPUT fields for a real
training run -- `scripts/build_strategist_candidate_dataset.py` (a legacy,
now-superseded v4/cycle-b2 script operating on already-serialized trajectory
JSONL with no live generation-time context) attempted a weaker version of
exactly this, using a structurally-derived `plan_features` fallback because
it could not reconstruct the real `PlanProposal` objects from persisted
JSONL alone -- see its own docstring's "plan_features scoping decision"
section. This module has live generation-time context (it builds the
corpus directly, not from a serialized intermediate), so it can and does use
the SAME canonical `plan_proposals_to_candidate_tensors`/`PLAN_FEATURE_NAMES`
the live PASS-2 runtime scoring path itself uses -- one shared definition,
train and serve, per `candidate_tensorizer.py`'s own explicit governance
statement ("any future training-side consumer must import this rather than
reimplementing the mapping").

## INPUT vs TARGET (explicit, per this amendment's Part 4 requirement)

INPUT (what this module's `batch` may contain): the scenario's own
current-evidence Sentinel batch (`scenario_to_prefix_example`, unchanged --
Strategist supervision is a SINGLE decision point per incident, per
`strategist_trajectory.py`'s own docstring, so there is no multi-round
"already revealed" state to layer on top the way Scout needed) PLUS the
bounded deterministic candidate set's own structural identity/features
(`plan_proposals_to_candidate_tensors`'s output) -- built from
`PlanGenerationContext`'s CURRENT-EVIDENCE-derived fields (probable source
nodes from the classical localizer, static network structure: isolatable
links, downstream-flush nodes, critical-demand nodes) and each candidate's
own deterministic, unverified prescreen heuristic
(`PlanProposal.predicted_value`/`predicted_validity`, folded into
`plan_features` by the SAME live tensorizer production PASS-2 scoring uses).
Never anything from exact WNTR verification.

TARGET (what this module's `targets` dict contains): `generate_strategist_
labels`'s exact-WNTR-verified `plan_validity`/`plan_value`/five consequence
proxies -- computed against the SAME bounded candidate set, using the
scenario's own exact ground-truth `IncidentTruth` ONLY for offline exposure-
consequence evaluation (never as a model input; see leakage tests). Stacked
across the (padded, masked) candidate/"plans" axis to match
`hydroswarm.training.targets_v2.PLAN_DIMENSION_TARGETS`'s own per-example
`[plans]`-leading-dimension convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

import torch

from hydroswarm.classical.signatures import SignatureArtifact, localize_with_signatures
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.inference.pipeline import HybridInferencePipeline
from hydroswarm.planning.action_templates import ACTION_TEMPLATE_COUNT
from hydroswarm.planning.candidate_tensorizer import plan_proposals_to_candidate_tensors
from hydroswarm.planning.response import PlanProposal, generate_response_plans
from hydroswarm.training.causal_prefix import FeatureContext
from hydroswarm.training.corpus import build_sensor_series
from hydroswarm.training.strategist_labels import StrategistLabel
from hydroswarm.training.strategist_trajectory import StrategistTrajectory, build_strategist_trajectory

#: Distinct from `checkpoint_identity.STRATEGIST_CANDIDATE_SCHEMA_VERSION`
#: (still `"strategist-candidate-v1-unbuilt"`, unchanged by this module --
#: that placeholder names the ORIGINAL M9.6 checkpoint's own [never-real]
#: training-corpus claim). This is the M10.3A refit's own training-corpus
#: schema identity: the exact channel wiring/padding/masking convention
#: this module implements.
STRATEGIST_CANDIDATE_TRAINING_SCHEMA_VERSION = "strategist-candidate-training-v1"

#: Governed upper bound on real candidates per incident -- the canonical
#: bounded template count (`generate_response_plans` never returns more),
#: reused unmodified as the fixed padded "plans" axis width so every
#: incident's tensors share one shape regardless of how many templates a
#: given network/incident actually instantiates (some templates are
#: conditionally skipped, e.g. ISOLATE_SOURCE requires `isolatable_links`).
MAXIMUM_PLAN_COUNT = ACTION_TEMPLATE_COUNT

#: Governed offline-target keys this module populates, in the exact
#: canonical order Part 1's readiness audit confirms is the intended
#: trainable set -- action_template/target_pointer are deliberately EXCLUDED
#: (repository evidence: configs/training-v5-causal.yaml's own comment,
#: "v3-legacy head; still trained by the unmodified default v3 model" --
#: not part of this experiment's canonical scope).
STRATEGIST_TARGET_KEYS: tuple[str, ...] = (
    "plan_validity",
    "plan_value",
    "exposure_proxy",
    "pressure_risk_proxy",
    "service_loss_proxy",
    "containment_time_proxy",
    "plan_regret_proxy",
)


class StrategistCandidateAlignmentError(Exception):
    """Raised when the independently-reconstructed `PlanProposal` list used
    to build INPUT tensors does not align 1:1, in order, with
    `generate_strategist_labels`' own internally-generated proposal list --
    both call `generate_response_plans` with the SAME `PlanGenerationContext`
    and `maximum_plans`, which is a pure deterministic function of its
    arguments, so a mismatch here indicates a real construction-order bug,
    never expected data variance."""


@dataclass(frozen=True, slots=True)
class StrategistCandidateExample:
    scenario_id: str
    batch: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    labels: tuple[StrategistLabel, ...]
    proposals: tuple[PlanProposal, ...]
    real_plan_count: int
    schema_version: str = STRATEGIST_CANDIDATE_TRAINING_SCHEMA_VERSION


def _reconstruct_context_and_proposals(
    scenario: GeneratedScenario, network: Any, feature_context: FeatureContext, artifact: SignatureArtifact,
    *, maximum_plans: int,
) -> tuple[tuple[PlanProposal, ...], Any]:
    """Mirrors `build_strategist_trajectory`'s own internal context-
    construction sequence EXACTLY (same functions, same order, same
    current-evidence-only inputs) so the `PlanProposal` list this returns is
    guaranteed identical to the one `generate_strategist_labels` builds
    internally for the SAME scenario -- verified, not merely assumed, by
    `StrategistCandidateAlignmentError`'s fail-closed check in
    `build_strategist_candidate_example` below. Does not reimplement
    `plan_proposals_to_candidate_tensors`/`generate_response_plans`/
    `generate_strategist_labels` themselves -- only re-derives the
    `PlanGenerationContext` object those need, since `build_strategist_
    trajectory` does not expose it externally."""

    scenario_id = str(scenario.manifest.scenario_id)
    incident_id = uuid5(NAMESPACE_URL, f"https://hydroswarm.local/incidents/strategist/{scenario_id}")
    series = build_sensor_series(scenario, feature_context)
    observations, mask = HybridInferencePipeline._signature_observations(series, artifact)
    localization = localize_with_signatures(observations, artifact, observation_mask=mask)
    probable_nodes = HybridInferencePipeline._credible_nodes(localization.source_probabilities)
    context = HybridInferencePipeline._planning_context(
        incident_id, network, feature_context.graph, probable_nodes, frozenset()
    )
    proposals = generate_response_plans(context, maximum_plans=maximum_plans)
    return proposals, context


def _pad_plan_dimension(tensors: dict[str, torch.Tensor], *, real_count: int, maximum: int) -> dict[str, torch.Tensor]:
    """Pads every `[1, plans, ...]` INPUT tensor from `real_count` up to
    `maximum` along the plans axis, `plan_mask=False` at padded positions --
    same fill convention `CandidatePlanEncoder.forward`'s own
    `masked_fill(~plan_mask, 0.0)` already tolerates (padded template_ids/
    target_type values are discarded by that masking regardless of value,
    but kept at the safe sentinel 0 here for clarity, matching
    `_masked_placeholder`'s established convention in the legacy corpus
    script)."""

    if real_count > maximum:
        raise ValueError(f"real_count ({real_count}) exceeds maximum plan count ({maximum})")
    pad = maximum - real_count
    if pad == 0:
        return dict(tensors)
    padded: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key == "plan_mask":
            filler = torch.zeros(1, pad, dtype=value.dtype)
        elif key in ("plan_target_node_index", "plan_target_link_index"):
            filler = torch.full((1, pad), -1, dtype=value.dtype)
        elif value.dim() == 3:
            filler = torch.zeros(1, pad, value.shape[-1], dtype=value.dtype)
        else:
            filler = torch.zeros(1, pad, dtype=value.dtype)
        padded[key] = torch.cat([value, filler], dim=1)
    return padded


def _stack_targets(
    labels: Sequence[StrategistLabel], *, real_count: int, maximum: int,
) -> dict[str, torch.Tensor]:
    """Builds `[plans]`-shaped (unbatched, per `targets_v2.PLAN_DIMENSION_
    TARGETS`' own single-example convention) target tensors for
    `STRATEGIST_TARGET_KEYS`, stacked in the SAME order as `labels`
    (== the SAME order as the INPUT tensors' plans axis, by the alignment
    invariant `build_strategist_candidate_example` enforces), padded with
    `<key>_mask=False` up to `maximum`. `plan_validity` is int64 (matches
    `TARGET_CLASS_COUNTS["plan_validity"] = 2`, a 2-class classification
    target); every other key is float32 (regression targets, `masked_
    regression`'s own required dtype)."""

    if real_count > maximum:
        raise ValueError(f"real_count ({real_count}) exceeds maximum plan count ({maximum})")
    pad = maximum - real_count

    def _value_and_mask(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        values: list[float] = []
        masks: list[bool] = []
        for label in labels:
            raw = getattr(label, name)
            if name == "plan_validity":
                values.append(float(raw))
                masks.append(True)
            else:
                values.append(float(raw) if raw is not None else 0.0)
                masks.append(raw is not None)
        if name == "plan_validity":
            value_tensor = torch.tensor(values + [0.0] * pad, dtype=torch.int64)
        else:
            value_tensor = torch.tensor(values + [0.0] * pad, dtype=torch.float32)
        mask_tensor = torch.tensor(masks + [False] * pad, dtype=torch.bool)
        return value_tensor, mask_tensor

    targets: dict[str, torch.Tensor] = {}
    for name in STRATEGIST_TARGET_KEYS:
        value, mask = _value_and_mask(name)
        targets[name] = value
        targets[f"{name}_mask"] = mask
    return targets


def build_strategist_candidate_example(
    scenario: GeneratedScenario,
    network: Any,
    feature_context: FeatureContext,
    artifact: SignatureArtifact,
    node_ids: Sequence[str],
    edge_ids: Sequence[tuple[str, str]],
    *,
    maximum_plans: int = MAXIMUM_PLAN_COUNT,
) -> StrategistCandidateExample | None:
    """Builds one incident's real, leakage-safe candidate-conditioned
    Strategist (INPUT, TARGET) pair. Returns `None` when
    `build_strategist_trajectory` produces zero usable candidates for this
    scenario (mirrors the legacy corpus script's own masked-not-dropped
    convention would apply at the CALLER level -- this function itself
    reports "no usable step" rather than fabricating a placeholder, since a
    placeholder policy belongs to the caller's own corpus-assembly
    decision, not this single-example builder)."""

    trajectory: StrategistTrajectory = build_strategist_trajectory(
        scenario, network, feature_context, artifact, node_ids, edge_ids, maximum_plans=maximum_plans,
    )
    if not trajectory.steps:
        return None
    step = trajectory.steps[0]
    labels = step.labels
    if not labels:
        return None

    proposals, _context = _reconstruct_context_and_proposals(
        scenario, network, feature_context, artifact, maximum_plans=maximum_plans,
    )
    proposal_templates = [proposal.template for proposal in proposals]
    label_templates = [label.action_template for label in labels]
    if proposal_templates != label_templates:
        raise StrategistCandidateAlignmentError(
            f"proposal/label misalignment for scenario {scenario.manifest.scenario_id}: "
            f"proposal templates {proposal_templates} != label templates {label_templates}"
        )

    real_count = len(proposals)
    input_tensors = plan_proposals_to_candidate_tensors(proposals, node_ids=node_ids, graph=feature_context.graph)
    input_tensors = _pad_plan_dimension(input_tensors, real_count=real_count, maximum=maximum_plans)
    target_tensors = _stack_targets(labels, real_count=real_count, maximum=maximum_plans)

    return StrategistCandidateExample(
        scenario_id=str(scenario.manifest.scenario_id),
        batch=input_tensors,
        targets=target_tensors,
        labels=labels,
        proposals=proposals,
        real_plan_count=real_count,
    )
