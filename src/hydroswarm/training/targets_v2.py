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
            "(frozen, drifting, or in communication outage) in this scenario.",
            unit="boolean per node, shape [node_count]",
            masking_rule="Never masked for nodes with a sensor_presence=1 input feature; "
            "undefined (and should be excluded from loss) for nodes with no sensor.",
            source_of_truth="Scenario generator ground truth (which sensors were corrupted "
            "and how).",
            maskable=False,
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
            definition="Expected posterior-entropy reduction (bits) from sampling sample_node.",
            unit="bits",
            masking_rule="Same as sample_node.",
            source_of_truth="Computed from training-only simulation/signature assets (Task 2.3).",
        ),
        _spec(
            category="scout",
            name="candidate_reduction",
            definition="Expected fractional reduction in the calibrated candidate set size "
            "after sampling sample_node.",
            unit="fraction in [0, 1]",
            masking_rule="Same as sample_node.",
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
            masking_rule="Masked (no ground truth) for any node that was never actually "
            "observed by the corpus generator at any point in the scenario.",
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
            masking_rule="Masked for scenarios/nodes where the corpus generator's simulation "
            "horizon does not extend far enough past the observation window to supply ground "
            "truth.",
            source_of_truth="Exact WNTR/EPANET simulation continued past the observation "
            "window (training-only future truth).",
        ),
        _spec(
            category="auxiliary",
            name="travel_time",
            definition="Hydraulic travel time from the true contamination source to each "
            "node -- an auxiliary structural-awareness objective (Task 4.5), never an "
            "authoritative product output.",
            unit="seconds, per node, shape [node_count]",
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


def check_schema_version(version: str) -> None:
    """Fail clearly when loading targets written under an incompatible
    schema, per the plan's explicit requirement -- never silently reinterpret
    a different version's target encoding."""

    if version != TARGETS_V2_SCHEMA_VERSION:
        raise TargetSchemaError(
            f"incompatible target schema version {version!r}; this build expects "
            f"{TARGETS_V2_SCHEMA_VERSION!r}"
        )


def validate_targets_v2(targets: Mapping[str, Tensor]) -> None:
    """Raise TargetSchemaError if `targets` violates the contract:
    unknown target keys, an unmaskable target missing its required value
    combined with a present mask claiming otherwise, or a mask key without
    a corresponding value key.
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
