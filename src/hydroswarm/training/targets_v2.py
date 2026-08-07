"""Governed targets_v2 contract (overnight-plan.txt Task 2.1).

The current model exposes more semantic heads (event presence/cause,
next-step advisory, Scout/Strategist heads) than the learning-v1 corpus
supervises. This module is the versioned contract those labels must satisfy
before Task 2.2-2.6 generate them: every target has a definition, a unit
where applicable, an explicit masking rule, and a stated source of truth,
so "missing" is never silently confused with "false"/"zero"/"normal".

Masking convention: a target named ``X`` that may be legitimately absent
for some examples (e.g. Strategist targets on an example with no candidate
plans yet) is paired with a boolean companion target ``f"{X}_mask"``
(``True`` where ``X`` is valid/observed), following the same ``_mask``
suffix convention already used for inputs (node_mask, edge_mask,
quality_mask, sensor_mask). A target with no mask is required whenever it
appears at all -- there is no third "unknown" state smuggled into the value
itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from .data import TopologyMetadata
from .ood_categories import OODCategory

TARGETS_V2_SCHEMA_VERSION = "targets_v2"


class TargetSchemaError(Exception):
    """Raised when target data violates the targets_v2 contract."""


class EventCause(str, Enum):
    CONTAMINATION = "CONTAMINATION"
    SENSOR_FAULT = "SENSOR_FAULT"
    HYDRAULIC_MISMATCH = "HYDRAULIC_MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"
    NORMAL = "NORMAL"


class NextStep(str, Enum):
    COLLECT_SAMPLE = "COLLECT_SAMPLE"
    INSPECT_SENSOR = "INSPECT_SENSOR"
    GENERATE_PLANS = "GENERATE_PLANS"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    category: str  # "sentinel" | "scout" | "strategist" | "control"
    name: str
    definition: str
    unit: str
    masking_rule: str
    source_of_truth: str
    maskable: bool = True

    def __post_init__(self) -> None:
        if not all((self.category, self.name, self.definition, self.unit, self.masking_rule, self.source_of_truth)):
            raise ValueError(f"target spec {self.name!r} is missing a required governance field")

    @property
    def mask_key(self) -> str:
        return f"{self.name}_mask"


_spec = TargetSpec


TARGETS_V2: dict[str, TargetSpec] = {
    spec.name: spec
    for spec in (
        _spec(
            category="sentinel",
            name="source_node",
            definition="Graph-local index (into TopologyMetadata.node_ids) of the true "
            "contamination source node.",
            unit="local node index",
            masking_rule="Required whenever event_presence is true and event_cause is "
            "CONTAMINATION; masked otherwise (no true source exists for a normal or "
            "sensor-fault-only scenario).",
            source_of_truth="Exact simulation ground truth (the injected source node).",
            maskable=True,
        ),
        _spec(
            category="sentinel",
            name="source_region",
            definition="Index of the credible-region cluster (a connected set of nodes) "
            "containing the true source, per the topology's region partition.",
            unit="local region index",
            masking_rule="Same as source_node.",
            source_of_truth="Derived deterministically from source_node and the topology's "
            "precomputed region partition.",
        ),
        _spec(
            category="sentinel",
            name="event_presence",
            definition="Whether a contamination event is occurring in this scenario at all "
            "(as opposed to normal operation or a sensor-only fault).",
            unit="boolean",
            masking_rule="Never masked; always defined by construction of the scenario.",
            source_of_truth="Scenario generator ground truth.",
            maskable=False,
        ),
        _spec(
            category="sentinel",
            name="event_cause",
            definition="One of CONTAMINATION, SENSOR_FAULT, HYDRAULIC_MISMATCH, AMBIGUOUS, "
            "NORMAL -- see hydroswarm.training.targets_v2.EventCause.",
            unit="categorical (5 classes)",
            masking_rule="Never masked; always defined by construction of the scenario.",
            source_of_truth="Scenario generator ground truth.",
            maskable=False,
        ),
        _spec(
            category="sentinel",
            name="start_time",
            definition="Binned start time of the contamination event, relative to the "
            "observation window start.",
            unit="ordinal bin index (minutes, generator-defined bin edges)",
            masking_rule="Masked when event_presence is false or event_cause != CONTAMINATION.",
            source_of_truth="Exact simulation ground truth (the injected start time), binned "
            "by the corpus generator's documented bin edges.",
        ),
        _spec(
            category="sentinel",
            name="duration",
            definition="Binned duration of the contamination injection.",
            unit="ordinal bin index (minutes, generator-defined bin edges)",
            masking_rule="Masked when event_presence is false or event_cause != CONTAMINATION.",
            source_of_truth="Exact simulation ground truth, binned by the corpus generator's "
            "documented bin edges.",
        ),
        _spec(
            category="sentinel",
            name="relative_strength",
            definition="Binned source strength relative to the generator's reference "
            "strength for that topology/regime.",
            unit="ordinal bin index (dimensionless ratio, generator-defined bin edges)",
            masking_rule="Masked when event_presence is false or event_cause != CONTAMINATION.",
            source_of_truth="Exact simulation ground truth, binned by the corpus generator's "
            "documented bin edges.",
        ),
        _spec(
            category="sentinel",
            name="sensor_fault",
            definition="Per-node indicator of whether that node's sensor is faulty "
            "(frozen, drifting, in communication outage, or unit-mismatched) in this scenario.",
            unit="boolean per node, shape [node_count]",
            masking_rule="Masked (sensor_fault_mask=False) for nodes with no sensor -- the "
            "placeholder value at those positions is never a real 'healthy' observation "
            "and must be excluded from loss, prevalence statistics, and audits alike.",
            source_of_truth="Scenario generator ground truth (which sensors were corrupted "
            "and how).",
            maskable=True,
        ),
        _spec(
            category="sentinel",
            name="evidence_sufficiency",
            definition="Whether the current evidence is sufficient to proceed to planning, "
            "per the documented deterministic rule combining calibrated candidate-set size, "
            "posterior entropy, disagreement, sensor-health threshold, and OOD state.",
            unit="boolean",
            masking_rule="Never masked; computed deterministically, not hand-labeled.",
            source_of_truth="Deterministic rule over calibration/OOD/controller state (Task "
            "2.2), not an arbitrary hand label.",
            maskable=False,
        ),
        _spec(
            category="scout",
            name="sample_node",
            definition="Graph-local index of the best next sample location.",
            unit="local node index",
            masking_rule="Masked when should_continue_sampling is false (no further sample "
            "is recommended) or when no accessible candidate exists.",
            source_of_truth="Argmax of expected information gain over accessible candidates, "
            "computed from training-only simulation/signature assets (Task 2.3).",
        ),
        _spec(
            category="scout",
            name="information_gain",
            definition="Per-node expected posterior-entropy reduction (bits) from sampling "
            "that node -- shape [node_count], matching HydroCore's expected_information_gain "
            "output (core-issues3.txt Phase 7.2: previously a scalar for only the single "
            "selected sample_node, which silently broadcast against the model's per-node "
            "prediction instead of aligning to it).",
            unit="bits, per node, shape [node_count]",
            masking_rule="Masked (information_gain_mask=False) for every node that was not "
            "an accessible, unsampled candidate this step actually scored -- an unranked node "
            "has no computed expected-gain estimate, not a true value of zero.",
            source_of_truth="Computed from training-only simulation/signature assets (Task 2.3).",
        ),
        _spec(
            category="scout",
            name="candidate_reduction",
            definition="Per-node expected fractional reduction in the calibrated candidate "
            "set size after sampling that node -- shape [node_count], same masking convention "
            "as information_gain (core-issues3.txt Phase 7.2).",
            unit="fraction in [0, 1], per node, shape [node_count]",
            masking_rule="Same as information_gain.",
            source_of_truth="Computed from training-only simulation/signature assets (Task 2.3).",
        ),
        _spec(
            category="scout",
            name="should_continue_sampling",
            definition="Whether another sample is worthwhile under the documented sampling "
            "budget policy.",
            unit="boolean",
            masking_rule="Never masked; always defined by the budget policy.",
            source_of_truth="The same documented budget policy used by evaluation (Task 2.3), "
            "not an arbitrary cutoff.",
            maskable=False,
        ),
        _spec(
            category="strategist",
            name="action_template",
            definition="Which of the bounded deterministic plan templates this action list "
            "instantiates (e.g. monitor, isolate probable zone, flush node).",
            unit="categorical (bounded template set, Task 2.4)",
            masking_rule="Masked for incidents with zero generated plan candidates.",
            source_of_truth="Bounded deterministic template generator (Task 2.4).",
        ),
        _spec(
            category="strategist",
            name="target_pointer",
            definition="Graph-local index of the plan's primary target node/link.",
            unit="local node or edge index",
            masking_rule="Same as action_template.",
            source_of_truth="Bounded deterministic template generator (Task 2.4).",
        ),
        _spec(
            category="strategist",
            name="plan_validity",
            definition="Whether WNTR verification accepted (VERIFIED) or rejected this plan.",
            unit="boolean",
            masking_rule="Same as action_template.",
            source_of_truth="Exact WNTR/EPANET simulation (PlanVerifier) -- WNTR remains "
            "authoritative; this target may never be set from a neural prediction.",
        ),
        _spec(
            category="strategist",
            name="plan_value",
            definition="Scalar value combining exposure reduction, service availability, "
            "and containment time, minus regret against the best bounded valid plan.",
            unit="dimensionless value score (generator-defined scale)",
            masking_rule="Same as action_template; undefined for invalid plans without a "
            "computed consequence vector.",
            source_of_truth="Computed from PlanVerifier consequence output (Task 2.4).",
        ),
        _spec(
            category="strategist",
            name="consequence_vector",
            definition="[exposure_mg, pressure_violation_minutes, service_availability, "
            "containment_time_s] from exact WNTR verification.",
            unit="mixed units per component, see hydroswarm.simulation.consequences",
            masking_rule="Masked for plans that were never exactly simulated.",
            source_of_truth="Exact WNTR/EPANET simulation (PlanVerifier).",
        ),
        _spec(
            category="strategist",
            name="exposure_proxy",
            definition="Non-authoritative pre-simulation estimate of consequence_vector's "
            "exposure_mg component, used only to rank candidate plans before deciding which "
            "ones are worth the cost of exact WNTR verification (Task 4.6). Never itself a "
            "verified consequence.",
            unit="mg (ranking proxy, calibrated or clearly labeled as such)",
            masking_rule="Same as action_template.",
            source_of_truth="Supervised from PlanVerifier's exact consequence_vector output "
            "(Task 2.4) -- the proxy learns to approximate the exact simulation, never "
            "replaces it.",
        ),
        _spec(
            category="strategist",
            name="pressure_risk_proxy",
            definition="Non-authoritative pre-simulation estimate of "
            "consequence_vector's pressure_violation_minutes component (Task 4.6).",
            unit="pressure-violation minutes (ranking proxy)",
            masking_rule="Same as action_template.",
            source_of_truth="Supervised from PlanVerifier's exact consequence_vector output "
            "(Task 2.4).",
        ),
        _spec(
            category="strategist",
            name="service_loss_proxy",
            definition="Non-authoritative pre-simulation estimate of "
            "consequence_vector's service_availability component, expressed as loss (1 - "
            "availability) (Task 4.6).",
            unit="fraction in [0, 1] (ranking proxy)",
            masking_rule="Same as action_template.",
            source_of_truth="Supervised from PlanVerifier's exact consequence_vector output "
            "(Task 2.4).",
        ),
        _spec(
            category="strategist",
            name="containment_time_proxy",
            definition="Non-authoritative pre-simulation estimate of "
            "consequence_vector's containment_time_s component (Task 4.6).",
            unit="seconds (ranking proxy)",
            masking_rule="Same as action_template.",
            source_of_truth="Supervised from PlanVerifier's exact consequence_vector output "
            "(Task 2.4).",
        ),
        _spec(
            category="strategist",
            name="plan_regret_proxy",
            definition="Non-authoritative pre-simulation estimate of plan_value's regret "
            "term against the best bounded valid plan (Task 4.6).",
            unit="dimensionless value-scale proxy, same scale as plan_value",
            masking_rule="Same as action_template; undefined wherever plan_value itself is "
            "undefined.",
            source_of_truth="Supervised from PlanVerifier-derived plan_value (Task 2.4).",
        ),
        _spec(
            category="control",
            name="ood_class",
            definition="OOD category for this example, or in-distribution. See Task 2.5's "
            "governed category list (unseen topology, unseen sensor layout, extreme demand, "
            "tank-state shift, valve/pump mismatch, roughness mismatch, severe missingness, "
            "frozen/drifting sensor, timing outside training range, unsupported network "
            "element or invalid calibration, or NONE).",
            unit="categorical",
            masking_rule="Never masked; every example is classified, including in-distribution.",
            source_of_truth="Governed OOD category assignment (Task 2.5), not a model prediction.",
            maskable=False,
        ),
        _spec(
            category="control",
            name="next_step",
            definition="One of COLLECT_SAMPLE, INSPECT_SENSOR, GENERATE_PLANS, ABSTAIN -- see "
            "hydroswarm.training.targets_v2.NextStep.",
            unit="categorical (4 classes)",
            masking_rule="Never masked; always defined by the deterministic controller policy "
            "for this state.",
            source_of_truth="Deterministic controller policy, not a hand label.",
            maskable=False,
        ),
        _spec(
            category="auxiliary",
            name="sensor_reconstruction",
            definition="Reconstructed node quality/concentration signal at nodes whose sensor "
            "reading was masked as missing, frozen, or dropped in the input -- a "
            "self-supervised denoising objective (Task 4.5), never an authoritative product "
            "output.",
            unit="mg/L, per node, shape [node_count]",
            masking_rule="Masked for any node that was never actually observed by the corpus "
            "generator, AND for any sensor node whose own reading was NOT degraded at the "
            "reference instant (core-issues3.txt Phase 7.3: a healthy sensor's input already "
            "equals its own truth value, so 'reconstructing' it is a trivial identity mapping, "
            "not a real denoising objective) -- only genuinely missing/frozen/communication-"
            "outage positions are unmasked.",
            source_of_truth="The corpus generator's own unmasked simulation output, used only "
            "as training supervision -- never available at inference/production time.",
        ),
        _spec(
            category="auxiliary",
            name="future_concentration",
            definition="Concentration at a fixed future horizon beyond the observation "
            "window, per node -- an auxiliary forecasting objective (Task 4.5), never an "
            "authoritative product output.",
            unit="mg/L, per node, shape [node_count]",
            masking_rule="core-issues3.txt Phase 7.4 / item H: ALWAYS masked (disabled) as of "
            "this schema version. The base ScenarioExample's temporal input features are built "
            "from the scenario's entire simulated window, not a bounded lookback from a "
            "defined 'now' -- so the target instant this auxiliary was meant to supervise is "
            "already directly visible as a raw input feature (exact leakage, not "
            "approximate). Re-enabling requires a genuinely cutoff-aware feature "
            "representation first; see auxiliary_labels.future_concentration_target's "
            "docstring.",
            source_of_truth="Exact WNTR/EPANET simulation continued past the observation "
            "window (training-only future truth).",
        ),
        _spec(
            category="auxiliary",
            name="travel_time",
            definition="Hydraulic travel time from the true contamination source to each "
            "node -- an auxiliary structural-awareness objective (Task 4.5), never an "
            "authoritative product output.",
            unit="log1p(seconds), per node, shape [node_count] (core-issues3.txt Phase 7.5: "
            "auxiliary_labels.TRAVEL_TIME_TRANSFORM -- invert with expm1 for a physical-unit "
            "metric; raw seconds would otherwise dominate a multitask MSE next to "
            "[0, 1]-ish regression targets).",
            masking_rule="Masked for nodes hydraulically unreachable from the source, and for "
            "every node on NORMAL/SENSOR_FAULT_ONLY (non-contamination) scenarios where no "
            "source exists.",
            source_of_truth="Exact WNTR/EPANET hydraulic simulation trace.",
        ),
    )
}

TARGETS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    category: tuple(spec.name for spec in TARGETS_V2.values() if spec.category == category)
    for category in ("sentinel", "scout", "strategist", "control", "auxiliary")
}


#: Fixed-cardinality categorical targets. Mirrors the corresponding
#: RoleHead output sizes in hydroswarm.model.core (source_region_head=3,
#: plan_validity_head=2, EVENT_CAUSE_CLASS_COUNT, NEXT_STEP_CLASS_COUNT) and
#: losses.py's PROFILE_CLASS_COUNTS -- kept as separately-documented
#: constants here rather than importing model.core, matching this module's
#: existing zero-coupling convention (EventCause/NextStep are likewise
#: defined locally, not imported). action_template's cardinality is a
#: configurable model constructor argument (action_vocabulary_size), not a
#: fixed constant, so it is deliberately excluded here rather than risk a
#: false-positive range rejection -- see strategist_trajectory.py's
#: ACTION_TEMPLATES for the real 9-vs-8 mismatch this exclusion protects
#: against.
#:
#: ood_class is intentionally NOT hydroswarm.model.core's ood_head width
#: (3): this target's own governed definition (see its TargetSpec above)
#: is the full OODCategory taxonomy, not a 3-way severity level (that's a
#: distinct concept -- hydroswarm.inference.ood.OODLevel, which the
#: ood_class TargetSpec text does not reference). An earlier version of
#: this table wrongly used 3 here (matching the head's current width
#: instead of the target's actual definition), which would have silently
#: accepted only 3 of the 11 governed categories and rejected the rest as
#: "invalid" -- exactly backwards. The target's own cardinality is the
#: correct value to validate against regardless of whether the current
#: model architecture has enough head capacity to learn all of it (that is
#: a model-architecture gap to size correctly when the ood_head is
#: eventually trained, not a reason to under-specify the label's own valid
#: range now).
TARGET_CLASS_COUNTS: dict[str, int] = {
    "source_region": 3,
    "start_time": 4,
    "duration": 3,
    "relative_strength": 3,
    "event_cause": len(EventCause),
    "next_step": len(NextStep),
    "ood_class": len(OODCategory),
    "plan_validity": 2,
}

#: Scalar graph-local node indices: must satisfy 0 <= value < topology.node_count.
NODE_INDEX_TARGETS = frozenset({"source_node", "sample_node"})

#: target_pointer may index either a node or a link (action_template-dependent,
#: see its TargetSpec.definition), so it gets a looser bound than a pure node
#: index: 0 <= value < max(node_count, edge_count).
DUAL_INDEX_TARGETS = frozenset({"target_pointer"})

#: Per-node arrays whose trailing dimension must equal topology.node_count.
NODE_ARRAY_TARGETS = frozenset({
    "sensor_fault", "sensor_reconstruction", "future_concentration", "travel_time",
    # core-issues3.txt Phase 7.2: converted from a scalar (the single
    # selected sample_node's value) to a full per-node array.
    "information_gain", "candidate_reduction",
})

#: Strategist targets that describe one entry per generated plan candidate;
#: whichever of these are present in the same example must agree on their
#: leading ("number of plans") dimension.
PLAN_DIMENSION_TARGETS = frozenset({
    "action_template", "target_pointer", "plan_validity", "plan_value", "consequence_vector",
    "exposure_proxy", "pressure_risk_proxy", "service_loss_proxy", "containment_time_proxy",
    "plan_regret_proxy",
})


def check_schema_version(version: str) -> None:
    """Fail clearly when loading targets written under an incompatible
    schema, per the plan's explicit requirement -- never silently reinterpret
    a different version's target encoding."""

    if version != TARGETS_V2_SCHEMA_VERSION:
        raise TargetSchemaError(
            f"incompatible target schema version {version!r}; this build expects "
            f"{TARGETS_V2_SCHEMA_VERSION!r}"
        )


def validate_targets_v2(targets: Mapping[str, Tensor], *, topology: TopologyMetadata | None = None) -> None:
    """Raise TargetSchemaError if `targets` violates the contract (core-issues2.txt
    Phase 1 item 3 -- fail closed, do not silently accept a malformed row):

    - unknown target keys;
    - a mask key without a corresponding value key, or on a non-maskable target;
    - a maskable target present without its required mask (the reverse of the
      previous check -- every present maskable target must state its validity,
      not just every present mask must be well-formed);
    - a non-boolean mask;
    - a fixed-cardinality categorical target (TARGET_CLASS_COUNTS) whose
      value at a valid (mask=True or unmaskable) position falls outside
      [0, class_count);
    - a graph-local index target (NODE_INDEX_TARGETS/DUAL_INDEX_TARGETS)
      whose value at a valid position falls outside the topology's node (or
      node-or-edge) index range -- only checked when `topology` is given;
    - a per-node array target (NODE_ARRAY_TARGETS) whose trailing dimension
      disagrees with `topology.node_count` -- only checked when `topology`
      is given (this is the "disagreement between target metadata and
      topology metadata" check);
    - two or more present PLAN_DIMENSION_TARGETS whose leading dimension
      ("number of plan candidates") disagree with each other;
    - a non-finite (NaN/Inf) value at a valid position in any floating-point
      target.
    """

    mask_suffix = "_mask"
    value_keys = {key for key in targets if not key.endswith(mask_suffix)}
    mask_keys = {key for key in targets if key.endswith(mask_suffix)}

    unknown_values = value_keys - set(TARGETS_V2)
    if unknown_values:
        raise TargetSchemaError(f"unknown targets_v2 keys: {sorted(unknown_values)}")

    for mask_key in mask_keys:
        base_name = mask_key[: -len(mask_suffix)]
        if base_name not in TARGETS_V2:
            raise TargetSchemaError(f"mask key {mask_key!r} has no matching target {base_name!r}")
        if base_name not in targets:
            raise TargetSchemaError(f"mask key {mask_key!r} present without its value key {base_name!r}")
        if not TARGETS_V2[base_name].maskable:
            raise TargetSchemaError(
                f"target {base_name!r} is not maskable but a mask key was provided; "
                f"{TARGETS_V2[base_name].masking_rule}"
            )
        if targets[mask_key].dtype != torch.bool:
            raise TargetSchemaError(f"mask key {mask_key!r} must be boolean")

    for name in value_keys:
        if TARGETS_V2[name].maskable and f"{name}_mask" not in targets:
            raise TargetSchemaError(
                f"target {name!r} is maskable and present but its required mask "
                f"{name}_mask is missing"
            )

    def _valid_positions(name: str) -> Tensor:
        mask = targets.get(f"{name}_mask")
        value = targets[name]
        return mask.bool() if mask is not None else torch.ones_like(value, dtype=torch.bool)

    for name, class_count in TARGET_CLASS_COUNTS.items():
        if name not in targets:
            continue
        value = targets[name]
        valid = _valid_positions(name)
        checked = value[valid] if valid.shape == value.shape else value
        if checked.numel() and ((checked < 0) | (checked >= class_count)).any():
            raise TargetSchemaError(
                f"target {name!r} has a value outside its valid class range [0, {class_count})"
            )

    if topology is not None:
        for name in NODE_INDEX_TARGETS:
            if name not in targets:
                continue
            value = targets[name]
            valid = _valid_positions(name)
            checked = value[valid] if valid.shape == value.shape else value
            if checked.numel() and ((checked < 0) | (checked >= topology.node_count)).any():
                raise TargetSchemaError(
                    f"target {name!r} has a graph-local index outside [0, {topology.node_count}) "
                    "for this example's topology"
                )
        upper_bound = max(topology.node_count, len(topology.edge_ids))
        for name in DUAL_INDEX_TARGETS:
            if name not in targets:
                continue
            value = targets[name]
            valid = _valid_positions(name)
            checked = value[valid] if valid.shape == value.shape else value
            if checked.numel() and ((checked < 0) | (checked >= upper_bound)).any():
                raise TargetSchemaError(
                    f"target {name!r} has a graph-local index outside [0, {upper_bound}) "
                    "for this example's topology (node-or-edge bound)"
                )
        for name in NODE_ARRAY_TARGETS:
            if name not in targets:
                continue
            value = targets[name]
            if value.dim() >= 1 and value.shape[-1] != topology.node_count:
                raise TargetSchemaError(
                    f"target {name!r} has trailing dimension {value.shape[-1]}, disagreeing with "
                    f"this example's topology node_count {topology.node_count}"
                )

    plan_dims: dict[str, int] = {}
    for name in PLAN_DIMENSION_TARGETS:
        if name in targets and targets[name].dim() >= 1:
            plan_dims[name] = targets[name].shape[0]
    if len(set(plan_dims.values())) > 1:
        raise TargetSchemaError(f"strategist plan-dimension targets disagree on plan count: {plan_dims}")

    for name in value_keys:
        value = targets[name]
        if not torch.is_floating_point(value):
            continue
        valid = _valid_positions(name)
        checked = value[valid] if valid.shape == value.shape else value
        if checked.numel() and not torch.isfinite(checked).all():
            raise TargetSchemaError(f"target {name!r} has a non-finite value at a valid position")
