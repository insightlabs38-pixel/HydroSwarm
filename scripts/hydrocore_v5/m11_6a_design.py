"""M11.6A-1 -- LOCKED EVALUATION DESIGN FREEZE (frozen design constants).

This module is the single, frozen source of truth for the M11.6 locked
evaluation design. It contains ONLY pre-registered, deterministic constants
and pure (side-effect-free, stdlib-only) helpers. It materializes NOTHING,
derives NO final seed, and touches no locked data: it is the design that a
DIFFERENT, fresh session will use in M11.6A-2 to materialize the real locked
population.

Nothing in this module may import the WNTR/hydroswarm simulation stack or
perform any model inference. That boundary exists so the frozen design --
seed derivation, population counts, topology/novelty rules, non-overlap
rules, manifest schema, metrics, gates, closure vocabulary, safety counters,
and the exactly-once guard -- can be tested deterministically and
independently of the EPANET simulator.

Companion document:
    docs/evaluation/HYDROCORE_V5_M11_6A_LOCKED_EVALUATION_DESIGN_FREEZE.md
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Identity / schema.
# ---------------------------------------------------------------------------

DESIGN_SCHEMA_VERSION = "hydroswarm-m11-6a-design-v1"
MANIFEST_SCHEMA_VERSION = "hydroswarm-m11-6-materialization-manifest-v1"
SCENARIO_SCHEMA_VERSION = "hydroswarm-m11-6-scenario-v1"
OPENED_RECORD_SCHEMA_VERSION = "hydroswarm-m11-6-opened-record-v1"

MILESTONE = "M11.6A-1"

#: The M11.6A-1 design-freeze commits superseded by this final correction
#: commit. The final correction commit becomes the ONLY authorized
#: design-freeze SHA for M11.6A-2 materialization; every superseded commit
#: MUST NOT be used to materialize (the materializer rejects them fail-closed).
SUPERSEDED_DESIGN_FREEZE_COMMITS: tuple[str, ...] = (
    "62bf1326081fac9080c3d676827c9596d2379efb",
    "e5665050811175638b45c0e82ac9959e2354d138",
)

#: The smoke namespace used ONLY by development fixtures. It MUST never
#: appear in either locked split; the materializer and evaluator reject it.
FORBIDDEN_SMOKE_NAMESPACE = "M11_6A_DESIGN_SMOKE_ONLY"

LOCKED_FINAL_TEST = "locked_final_test"
LOCKED_TOPOLOGY_TEST = "locked_topology_test"
LOCKED_SPLIT_NAMES: tuple[str, ...] = (LOCKED_FINAL_TEST, LOCKED_TOPOLOGY_TEST)

# ---------------------------------------------------------------------------
# Seed derivation (task Section 6). Frozen formula only -- the numeric
# DESIGN_FREEZE_COMMIT_SHA is supplied by the NEXT session and is never
# derived or printed here.
# ---------------------------------------------------------------------------

SEED_RULE_VERSION = "v1"
SEED_BYTE_ENCODING = "utf-8"
SEED_HASH_ALGORITHM = "sha256"
#: Derived-seed range, disjoint from every prior seed namespace BY
#: CONSTRUCTION. Every prior governed seed namespace is < 2**31, so derived
#: seeds are drawn from [LOCKED_SEED_MIN, LOCKED_SEED_MAX_EXCLUSIVE) =
#: [2**31, 2**62). The interval [0, 2**31) is entirely EXCLUDED, not merely
#: "statistically unlikely": no derived seed can ever fall inside a prior
#: namespace. (Correction #1: the original design claimed [0, 2**62) is
#: disjoint from [0, 2**31), which is false -- [0, 2**62) contains [0, 2**31).)
LOCKED_SEED_MIN = 2**31
LOCKED_SEED_MAX_EXCLUSIVE = 2**62
LOCKED_SEED_SPAN = LOCKED_SEED_MAX_EXCLUSIVE - LOCKED_SEED_MIN

#: Master-seed domain labels (the two governed locked splits).
MASTER_DOMAIN_FINAL = "LOCKED_FINAL"
MASTER_DOMAIN_TOPOLOGY = "LOCKED_TOPOLOGY"

#: Individual-seed domain labels. `FINAL_TOPOLOGY` and `SENSOR_PERTURBATION`
#: are reserved in v1 (see note): v1 drives each incident from ONE scenario
#: seed (the existing WNTRScenarioGenerator single-RNG-stream convention used
#: throughout M9/M10/M11.5), and locked_final_test uses the already-governed
#: fixed known topologies (no topology-level randomness is consumed).
SEED_LABELS: dict[str, dict[str, Any]] = {
    "FINAL_TOPOLOGY": {
        "split": LOCKED_FINAL_TEST,
        "consumed_by_generation": False,
        "note": "Reserved in v1: locked_final_test uses the fixed, already-governed known topologies (golden-reference/branched-loop/loop-grid); no topology-level seed is consumed.",
    },
    "FINAL_SCENARIO": {
        "split": LOCKED_FINAL_TEST,
        "consumed_by_generation": True,
        "note": "One derived seed per locked_final_test incident; drives the full WNTR scenario (start/duration/strength/demand/roughness/sensor layout/perturbations) via the single-RNG-stream generator.",
    },
    "TOPOLOGY_TEST_NETWORK": {
        "split": LOCKED_TOPOLOGY_TEST,
        "consumed_by_generation": True,
        "note": "One derived seed per locked_topology_test topology instance; drives the deterministic procedural topology generator (rejection candidates increment the counter).",
    },
    "TOPOLOGY_TEST_SCENARIO": {
        "split": LOCKED_TOPOLOGY_TEST,
        "consumed_by_generation": True,
        "note": "One derived seed per locked_topology_test incident; drives the full WNTR scenario for a fixed procedural topology.",
    },
    "SENSOR_PERTURBATION": {
        "split": None,
        "consumed_by_generation": False,
        "note": "Reserved in v1: sensor-level perturbations (noise/missingness/frozen/outage) are driven by the same single scenario seed via WNTRScenarioGenerator._degrade, matching M9/M10/M11.5.",
    },
}

MAX_COLLISION_RETRIES = 100
MAX_TOPOLOGY_CANDIDATE_ATTEMPTS = 1000
MAX_SCENARIO_COLLISION_RETRIES = 100


def derive_master_seed(design_freeze_commit_sha: str, domain: str) -> str:
    """The frozen master-seed derivation rule (task Section 6).

    ``H = SHA256`` over the UTF-8 bytes of
    ``"HYDROSWARM|M11.6|{DOMAIN}|v1|" + DESIGN_FREEZE_COMMIT_SHA``, returned
    as the lowercase hex digest. The final ``DESIGN_FREEZE_COMMIT_SHA`` is
    supplied by the NEXT session after independent human review; it is NOT
    derived or printed in M11.6A-1.
    """

    if domain not in (MASTER_DOMAIN_FINAL, MASTER_DOMAIN_TOPOLOGY):
        raise ValueError(f"unknown master-seed domain: {domain!r}")
    material = f"HYDROSWARM|M11.6|{domain}|{SEED_RULE_VERSION}|{design_freeze_commit_sha}"
    return hashlib.sha256(material.encode(SEED_BYTE_ENCODING)).hexdigest()


def derive_seed(master_hex: str, label: str, index: int, counter: int = 0) -> int:
    """Deterministically derive one integer seed from a master digest.

    ``material = f"{master_hex}|{label}|{index}|{counter}"`` (UTF-8), then
    ``seed = LOCKED_SEED_MIN + int(SHA256(material).hexdigest(), 16)
    % LOCKED_SEED_SPAN``, so every derived seed satisfies
    ``2**31 <= seed < 2**62`` and is mechanically disjoint from every prior
    governed namespace (all < 2**31).

    ``counter=0`` is the primary derivation. A positive ``counter`` is the
    deterministic next-candidate used by rejection sampling (topology
    generation) and by the frozen collision procedure -- never a source of
    OS randomness, never a preview of a "better" seed.
    """

    material = f"{master_hex}|{label}|{index}|{counter}"
    digest = hashlib.sha256(material.encode(SEED_BYTE_ENCODING)).hexdigest()
    return LOCKED_SEED_MIN + int(digest, 16) % LOCKED_SEED_SPAN


def seed_derivation_spec() -> dict[str, Any]:
    return {
        "rule_version": SEED_RULE_VERSION,
        "byte_encoding": SEED_BYTE_ENCODING,
        "hash_algorithm": SEED_HASH_ALGORITHM,
        "master_formula": {
            "locked_final_master": "H(\"HYDROSWARM|M11.6|LOCKED_FINAL|v1|\" + DESIGN_FREEZE_COMMIT_SHA)",
            "locked_topology_master": "H(\"HYDROSWARM|M11.6|LOCKED_TOPOLOGY|v1|\" + DESIGN_FREEZE_COMMIT_SHA)",
        },
        "per_seed_material": "f\"{master_hex}|{label}|{index}|{counter}\"",
        "digest_to_integer": "LOCKED_SEED_MIN + int(SHA256(material).hexdigest(), 16) % LOCKED_SEED_SPAN",
        "locked_seed_min": LOCKED_SEED_MIN,
        "locked_seed_max_exclusive": LOCKED_SEED_MAX_EXCLUSIVE,
        "locked_seed_span": LOCKED_SEED_SPAN,
        "allowed_integer_range": [LOCKED_SEED_MIN, LOCKED_SEED_MAX_EXCLUSIVE - 1],
        "disjoint_by_construction": (
            "Every derived seed is >= 2**31; every prior governed seed namespace "
            "is < 2**31, so the derived range [2**31, 2**62) is mechanically "
            "disjoint from all prior namespaces."
        ),
        "labels": SEED_LABELS,
        "collision_handling": (
            "If a derived seed collides with a seed already used in the same "
            "materialization (or a canonical scenario-definition hash collides "
            "with a prior hash), increment `counter` and re-derive "
            "deterministically. Never select a different example manually, "
            "never call OS randomness, and never preview candidate seeds."
        ),
        "max_collision_retries": MAX_COLLISION_RETRIES,
    }


# ---------------------------------------------------------------------------
# Population specification (task Section 5). Counts are chosen now,
# independent of any outcome.
# ---------------------------------------------------------------------------

#: The 3 already-governed TRAINED_FAMILIES (run_m7_topology.TRAINED_FAMILIES).
LOCKED_FINAL_FAMILIES: tuple[str, ...] = ("golden-reference", "branched-loop", "loop-grid")

#: Reused from the already-governed M10.4 condition matrix (m10_4_protocol
#: .CONDITION_KINDS) -- not invented here. Frozen order is part of the seed /
#: ordering formula and must never change post-freeze.
LOCKED_FINAL_CONDITIONS: tuple[str, ...] = (
    "NOMINAL",
    "LOW_COVERAGE_ACTIVE_SAMPLING",
    "SENSOR_DROPOUT",
    "SENSOR_HEALTH_DEGRADED",
    "MEASUREMENT_NOISE",
    "SEVERITY_SHIFT",
    "AMBIGUITY_DISAGREEMENT",
)

#: kwargs mirroring hydroswarm.evaluation.live_robustness.Condition (the
#: already-governed perturbation semantics M10.4 reused verbatim).
LOCKED_FINAL_CONDITION_KWARGS: dict[str, dict[str, Any]] = {
    "NOMINAL": dict(perturbation_type="nominal", perturbation_level="clean_operational"),
    "LOW_COVERAGE_ACTIVE_SAMPLING": dict(perturbation_type="sensor_coverage", perturbation_level="25%", coverage=0.25),
    "SENSOR_DROPOUT": dict(perturbation_type="missingness", perturbation_level="30%", missing_rate=0.3),
    "SENSOR_HEALTH_DEGRADED": dict(perturbation_type="sensor_health", perturbation_level="frozen:50%", health_mode="frozen", health_fraction=0.5),
    "MEASUREMENT_NOISE": dict(perturbation_type="measurement_noise", perturbation_level="moderate", noise_std=0.05),
    "SEVERITY_SHIFT": dict(perturbation_type="hydraulic_mismatch", perturbation_level="source_strength", hydraulic="source_strength"),
    "AMBIGUITY_DISAGREEMENT": dict(perturbation_type="ambiguity", perturbation_level="disagreement", ambiguity="disagreement"),
}

#: locked_final_test population geometry (justified in the design document:
#: 3 families x 7 conditions x 5 incidents = 105 incidents, matching the
#: M10.4 established evaluation scale).
LOCKED_FINAL_INCIDENTS_PER_CELL = 5
LOCKED_FINAL_TOTAL = len(LOCKED_FINAL_FAMILIES) * len(LOCKED_FINAL_CONDITIONS) * LOCKED_FINAL_INCIDENTS_PER_CELL
assert LOCKED_FINAL_TOTAL == 105

#: locked_topology_test population geometry (4 novel procedural topologies x
#: 5 NOMINAL incidents = 20 incidents).
LOCKED_TOPOLOGY_INSTANCES = 4
LOCKED_TOPOLOGY_INCIDENTS_PER_TOPOLOGY = 5
LOCKED_TOPOLOGY_TOTAL = LOCKED_TOPOLOGY_INSTANCES * LOCKED_TOPOLOGY_INCIDENTS_PER_TOPOLOGY
assert LOCKED_TOPOLOGY_TOTAL == 20

#: locked_topology_test uses only NOMINAL (the topology shift IS its
#: perturbation), matching M10.4's UNSEEN_FAMILY_CONDITIONS.
LOCKED_TOPOLOGY_CONDITIONS: tuple[str, ...] = ("NOMINAL",)

#: Scenario-generation bounds (reused, NOT invented -- these are the
#: WNTRScenarioGenerator defaults already governed throughout M9/M10/M11.5).
GENERATOR_START_TIME_BINS_MIN: tuple[int, ...] = (0, 60, 120, 240)
GENERATOR_DURATION_BINS_MIN: tuple[int, ...] = (30, 60, 120)
GENERATOR_STRENGTH_BINS: tuple[float, ...] = (0.5, 1.0, 2.0)
GENERATOR_DEMAND_REGIMES: tuple[float, ...] = (0.8, 1.0, 1.2)

#: One physical incident per locked scenario; source node is round-robined
#: over the sorted junction list (source_node = junctions[index % len]).
SOURCE_SELECTION_RULE = "round_robin_sorted_junctions"
EVENT_TYPE = "contamination"
CONTAMINATION_PROPORTION = 1.0  # both locked splits are 100% contamination

#: Sampling opportunities (reused from M10.4 MAXIMUM_SAMPLES).
MAXIMUM_SAMPLES = 3

#: Frozen base ScenarioGenerationConfig (scenario-level) fields. Missingness/
#: coverage/sensor-health/ambiguity are applied at the OBSERVATION-payload
#: level (matching hydroswarm.evaluation.live_robustness._payloads), while
#: noise and hydraulic-mismatch conditions are applied at the scenario-
#: configuration level (matching live_robustness._scenario_config).
SCENARIO_CONFIG_BASE: dict[str, Any] = {
    "stage": "operational",
    "sensor_count": 4,
    "sensor_noise_std": 0.01,
    "missing_probability": 0.0,
    "drift_per_hour": 0.001,
    "frozen_probability": 0.0,
    "communication_outage_probability": 0.0,
    "quantization_step": 0.001,
    "unit_mismatch_probability": 0.01,
    "roughness_variation_fraction": 0.05,
    "tank_level_variation_fraction": 0.1,
    "pipe_outage_probability": 0.0,
    "start_time_bins_min": [0, 60, 120, 240],
    "duration_bins_min": [30, 60, 120],
    "strength_bins": [0.5, 1.0, 2.0],
    "demand_regimes": [0.8, 1.0, 1.2],
}

#: Scenario-config overrides per condition (frozen; the only conditions that
#: touch the scenario generator are MEASUREMENT_NOISE and SEVERITY_SHIFT,
#: exactly as M10.4's live_robustness._scenario_config does).
CONDITION_SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "NOMINAL": {},
    "LOW_COVERAGE_ACTIVE_SAMPLING": {},
    "SENSOR_DROPOUT": {},
    "SENSOR_HEALTH_DEGRADED": {},
    "MEASUREMENT_NOISE": {"sensor_noise_std": 0.05},
    "SEVERITY_SHIFT": {"strength_bins": [3.0]},
    "AMBIGUITY_DISAGREEMENT": {},
}


def scenario_config_for_condition(condition_kind: str) -> dict[str, Any]:
    """Frozen ScenarioGenerationConfig fields for one condition kind."""
    merged = dict(SCENARIO_CONFIG_BASE)
    merged.update(CONDITION_SCENARIO_OVERRIDES[condition_kind])
    return merged


def locked_final_cells() -> tuple[tuple[str, str], ...]:
    """Every (family, condition) cell of locked_final_test, in frozen order."""
    return tuple(
        (family, condition)
        for family in LOCKED_FINAL_FAMILIES
        for condition in LOCKED_FINAL_CONDITIONS
    )


def population_spec() -> dict[str, Any]:
    return {
        "locked_final_test": {
            "purpose": "Final prospective evaluation of the exact frozen finalist on genuinely unseen incidents drawn from the already-governed problem domain (known/trained topology families with genuinely new, disjoint incident seeds).",
            "families": list(LOCKED_FINAL_FAMILIES),
            "conditions": list(LOCKED_FINAL_CONDITIONS),
            "condition_kwargs": LOCKED_FINAL_CONDITION_KWARGS,
            "incidents_per_cell": LOCKED_FINAL_INCIDENTS_PER_CELL,
            "total_incidents": LOCKED_FINAL_TOTAL,
            "contamination_proportion": CONTAMINATION_PROPORTION,
            "source_selection_rule": SOURCE_SELECTION_RULE,
            "event_type": EVENT_TYPE,
            "maximum_samples": MAXIMUM_SAMPLES,
            "start_time_bins_min": list(GENERATOR_START_TIME_BINS_MIN),
            "duration_bins_min": list(GENERATOR_DURATION_BINS_MIN),
            "strength_bins": list(GENERATOR_STRENGTH_BINS),
            "demand_regimes": list(GENERATOR_DEMAND_REGIMES),
            "rationale": (
                "3 x 7 x 5 = 105 incidents mirrors M10.4's established "
                "(INCIDENTS_PER_CELL=5, same condition matrix, same families) "
                "evaluation scale; 5 per cell matches M10.4 exactly; 105 total "
                "gives stable pooled aggregate metric estimation while staying "
                "within the proven practical EPANET runtime budget."
            ),
        },
        "locked_topology_test": {
            "purpose": "Final prospective evaluation of topology generalization and fail-closed behavior on genuinely unseen procedural network structures.",
            "topology_instances": LOCKED_TOPOLOGY_INSTANCES,
            "incidents_per_topology": LOCKED_TOPOLOGY_INCIDENTS_PER_TOPOLOGY,
            "total_incidents": LOCKED_TOPOLOGY_TOTAL,
            "conditions": list(LOCKED_TOPOLOGY_CONDITIONS),
            "contamination_proportion": CONTAMINATION_PROPORTION,
            "source_selection_rule": SOURCE_SELECTION_RULE,
            "event_type": EVENT_TYPE,
            "maximum_samples": MAXIMUM_SAMPLES,
            "predictive_metrics": "DESCRIPTIVE_NON_GATING",
            "safety_fail_closed": "HARD_GATE",
            "rationale": (
                "4 procedural topologies x 5 NOMINAL incidents = 20 incidents; "
                "4 instances probe a spread of node counts/cycle ranks outside "
                "every prior topology, and 5 incidents each matches M10.4's "
                "per-cell scale while keeping EPANET runtime bounded."
            ),
        },
        "deterministic_ordering": {
            "locked_final_test": "family order (golden-reference, branched-loop, loop-grid) -> condition order (LOCKED_FINAL_CONDITIONS) -> incident_index 0..4",
            "locked_topology_test": "topology_index 0..3 -> incident_index 0..4",
        },
    }


# ---------------------------------------------------------------------------
# Topology novelty rule (task Section 8). Frozen prior-topology inventory
# (measured at design-freeze time; every value is a hard-coded, immutable
# structural identity of a topology already exposed in training/validation/
# calibration/development/M9/M10/M11.5).
# ---------------------------------------------------------------------------

#: Canonical structural signatures of every prior topology. A signature is
#: (node_count, junction_count, link_count, cycle_rank, sorted_degree_sequence).
#: These were measured from the exact committed loaders (run_m7_topology).
PRIOR_TOPOLOGY_SIGNATURES: tuple[dict[str, Any], ...] = (
    {
        "name": "golden-reference",
        "node_count": 6, "junction_count": 4, "link_count": 7, "cycle_rank": 2,
        "degree_profile": [1, 2, 2, 3, 3, 3],
        "network_sha256": "5508d2721298ea31b9caa37db81fa90d66a5b7fb17587f78cd5b9bcf8f299c8c",
    },
    {
        "name": "branched-loop",
        "node_count": 8, "junction_count": 7, "link_count": 8, "cycle_rank": 1,
        "degree_profile": [1, 1, 2, 2, 2, 2, 3, 3],
        "network_sha256": "9926474d1f8e7925d47ad3d77740cd27ff1b6d1d42f95d87acca52e244c01d46",
    },
    {
        "name": "loop-grid",
        "node_count": 9, "junction_count": 8, "link_count": 11, "cycle_rank": 3,
        "degree_profile": [1, 2, 2, 2, 3, 3, 3, 3, 3],
        "network_sha256": "35af6056fe885934bc8bda9b1e6ac178926ac24cac47bc8f2c00fb045730c2a9",
    },
    {
        "name": "coastal-branch",
        "node_count": 8, "junction_count": 6, "link_count": 8, "cycle_rank": 1,
        "degree_profile": [1, 1, 2, 2, 2, 2, 2, 4],
        "network_sha256": "76692a7d8d51b82d480a3a366bb7e0b13052b368f74db0a4a26377c6fdfdcd3f",
    },
    {
        "name": "tree-branch",
        "node_count": 6, "junction_count": 5, "link_count": 5, "cycle_rank": 0,
        "degree_profile": [1, 1, 1, 2, 2, 3],
        "network_sha256": "530fa29d583cf80f3a8388922b76b0e4d21024dc32c544cce7031123b1294f5a",
    },
    {
        "name": "dense-loop",
        "node_count": 7, "junction_count": 6, "link_count": 9, "cycle_rank": 3,
        "degree_profile": [1, 2, 2, 2, 3, 3, 5],
        "network_sha256": "7c0e5fecd6e93b0d4a2dc93a230e348f2efaf13c7a436af6501c588fbaf06fc8",
    },
)

PRIOR_TOPOLOGY_NETWORK_HASHES: frozenset[str] = frozenset(
    item["network_sha256"] for item in PRIOR_TOPOLOGY_SIGNATURES
)


def graph_signature(
    node_count: int, junction_count: int, link_count: int, cycle_rank: int, degree_profile: Sequence[int],
) -> dict[str, Any]:
    """The frozen canonical graph-structural signature of a topology."""
    return {
        "node_count": int(node_count),
        "junction_count": int(junction_count),
        "link_count": int(link_count),
        "cycle_rank": int(cycle_rank),
        "degree_profile": [int(d) for d in degree_profile],
    }


def signatures_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("node_count") == b.get("node_count")
        and a.get("junction_count") == b.get("junction_count")
        and a.get("link_count") == b.get("link_count")
        and a.get("cycle_rank") == b.get("cycle_rank")
        and a.get("degree_profile") == b.get("degree_profile")
    )


def is_prior_topology(signature: dict[str, Any], network_sha: str) -> bool:
    """True iff a topology's structural signature/hash matches any prior one."""
    return network_sha in PRIOR_TOPOLOGY_NETWORK_HASHES or any(
        signatures_equal(signature, prior) for prior in PRIOR_TOPOLOGY_SIGNATURES
    )


def topology_novelty_spec() -> dict[str, Any]:
    return {
        "rule": (
            "A locked_topology_test candidate is 'unseen' iff (a) its "
            "graph-structural signature (node/junction/link counts, cycle_rank, "
            "sorted degree profile) matches no prior topology signature, (b) its "
            "canonical network_sha256 is not in the frozen prior-hash set, and "
            "(c) for any materialized .inp file, its file-byte SHA-256 is not "
            "equal to any prior committed topology file hash. Duplicates WITHIN "
            "the generated locked_topology_test set are also rejected."
        ),
        "canonical_fingerprints": [
            "file-byte SHA-256 (materialized .inp)",
            "network_sha256 (hydroswarm.data.scenarios.network_sha256)",
            "graph_signature {node_count, junction_count, link_count, cycle_rank, sorted degree_profile}",
        ],
        "prior_topologies": PRIOR_TOPOLOGY_SIGNATURES,
        "no_vague_manual_judgment": True,
        "stronger_structural_novelty_note": (
            "All prior topologies have junction_count in [4, 8]; the frozen "
            "procedural generator is constrained to junction_count in [9, 12], "
            "so the node-count component of the signature is already decisive "
            "by construction. The full signature/hash checks above remain the "
            "frozen, mechanical novelty rule regardless."
        ),
    }


# ---------------------------------------------------------------------------
# Non-overlap rule (task Section 9).
# ---------------------------------------------------------------------------

#: Every prior seed namespace used anywhere in the repository (verified by
#: grep at design-freeze time). All are < 2**31; the derived seed space is
#: [2**31, 2**62), so the derived seeds are disjoint from every prior
#: namespace by construction (the interval [0, 2**31) is entirely excluded).
PRIOR_SEED_RANGES: dict[str, tuple[int, int]] = {
    "m1_split_seed_ranges": (900_000_000, 903_999_999),
    "m6": (960_000_000, 969_999_999),
    "m7_seed_bases": (940_000_000, 989_999_999),
    "m9_4_floor_and_up": (990_000_000, 1_099_999_999),
    "m10_1": (1_100_000_000, 1_199_999_999),
    "m10_2": (1_200_000_000, 1_299_999_999),
    "m10_3": (1_300_000_000, 1_399_999_999),
    "m10_3c_population": (1_400_000_000, 1_449_999_999),
    "m10_3d_reserved_unused": (1_450_000_000, 1_499_999_999),
    "m10_4": (1_500_000_000, 1_599_999_999),
}


def non_overlap_spec() -> dict[str, Any]:
    return {
        "rule": (
            "Every locked scenario is identified by (split role, topology "
            "identity, seed, scenario_id, canonical scenario-definition hash). "
            "Non-overlap is guaranteed by: (1) derived seeds in "
            "[2**31, 2**62) are disjoint from every prior seed namespace "
            "(all < 2**31) BY CONSTRUCTION -- the interval [0, 2**31) is "
            "entirely excluded; (2) locked_final_test uses only the allowed "
            "known/trained families with fresh derived seeds; (3) "
            "locked_topology_test uses only novelty-verified procedural "
            "topologies; (4) the canonical scenario-definition hash is unique "
            "per scenario and is checked against every prior materialized "
            "scenario hash."
        ),
        "canonical_scenario_definition_hash": (
            "SHA-256 over json.dumps(definition, sort_keys=True, "
            "separators=(',', ':')) of the frozen scenario-definition schema "
            "(SCENARIO_SCHEMA_VERSION)."
        ),
        "collision_procedure": (
            "If a canonical hash collision is detected against a prior scenario "
            "or within the same materialization, increment the scenario's "
            "derivation counter and re-derive deterministically. Never manually "
            "select another 'better' example, never call OS randomness, never "
            "preview candidate seeds. If retries exceed "
            f"MAX_SCENARIO_COLLISION_RETRIES={MAX_SCENARIO_COLLISION_RETRIES}, BLOCK."
        ),
        "prior_seed_ranges": {name: list(rng) for name, rng in PRIOR_SEED_RANGES.items()},
    }


def canonical_scenario_definition(definition: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical (sorted-key) serialization of a scenario definition."""
    return json.loads(json.dumps(definition, sort_keys=True, default=str))


def scenario_definition_hash(definition: dict[str, Any]) -> str:
    """SHA-256 of the canonical scenario-definition JSON (frozen)."""
    payload = json.dumps(canonical_scenario_definition(definition), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scenario_definition_schema() -> dict[str, Any]:
    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "fields": [
            "schema_version", "split", "scenario_index", "topology_id",
            "network_family", "network_sha256", "seed", "seed_domain",
            "seed_derivation_counter", "event_type", "source_node",
            "condition_kind", "generator_config",
        ],
        "forbidden_namespaces": [FORBIDDEN_SMOKE_NAMESPACE],
    }


# ---------------------------------------------------------------------------
# Manifest schema + validation (task Section 11).
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "design_freeze_commit_sha",
    "design_protocol_sha256",
    "generator_source_sha256",
    "evaluator_source_sha256",
    "seed_derivation",
    "master_seeds",
    "splits",
    "topologies",
    "scenarios",
    "artifact_sha256",
    "simulator",
    "generation_complete",
    "overlap_audit",
    "novelty_audit",
    "evaluated_by_finalist",
    "locked_test_opened",
)

#: Fields the manifest MUST NOT contain.
MANIFEST_FORBIDDEN_FIELDS: tuple[str, ...] = (
    "metrics", "performance", "accuracy", "top1", "mrr", "coverage_measured",
)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return the list of manifest violations (empty == valid).

    This is the mechanical, frozen manifest contract. A real materialization
    manifest must satisfy every check here before the evaluator may read a
    single locked scenario.
    """

    violations: list[str] = []
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in manifest:
            violations.append(f"missing required field: {field}")
    for field in MANIFEST_FORBIDDEN_FIELDS:
        if field in manifest:
            violations.append(f"forbidden field present: {field}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        violations.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
    if manifest.get("evaluated_by_finalist") is not False:
        violations.append("evaluated_by_finalist must be false at materialization time")
    if manifest.get("locked_test_opened") is not False:
        violations.append("locked_test_opened must be false at materialization time")
    if manifest.get("generation_complete") is not True:
        violations.append("generation_complete must be true before evaluation")
    overlap = manifest.get("overlap_audit") or {}
    novelty = manifest.get("novelty_audit") or {}
    if overlap.get("result") != "PASS":
        violations.append("overlap_audit.result must be PASS")
    if novelty.get("result") != "PASS":
        violations.append("novelty_audit.result must be PASS")
    for split in LOCKED_SPLIT_NAMES:
        if split not in (manifest.get("splits") or {}):
            violations.append(f"splits missing {split}")
    # The smoke namespace must never leak into a real locked manifest.
    serialized = json.dumps(manifest, sort_keys=True, default=str)
    if FORBIDDEN_SMOKE_NAMESPACE in serialized:
        violations.append(f"forbidden smoke namespace {FORBIDDEN_SMOKE_NAMESPACE!r} present")
    return violations


def manifest_schema() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "required_fields": list(MANIFEST_REQUIRED_FIELDS),
        "forbidden_fields": list(MANIFEST_FORBIDDEN_FIELDS),
        "must_not_contain_model_performance_metrics": True,
        "storage_convention": {
            "root": "data/locked/m11-6/",
            "layout": [
                "data/locked/m11-6/topologies/<topology_id>.inp",
                "data/locked/m11-6/locked_final_test/scenarios.jsonl",
                "data/locked/m11-6/locked_topology_test/scenarios.jsonl",
                "data/locked/m11-6/m11-6-materialization-manifest.json",
            ],
            "tracking": "ordinary Git tracked text (definitions-only, deterministic WNTR replay); no Git LFS (LFS is reserved for large binary tensor corpora per .gitattributes)",
            "content_addressing": "every artifact's SHA-256 is recorded in artifact_sha256; the manifest is the authoritative content-addressed record",
        },
    }


# ---------------------------------------------------------------------------
# Safety counters (task Section 19). Hard counters must remain zero.
# ---------------------------------------------------------------------------

SAFETY_COUNTERS_TEMPLATE: tuple[str, ...] = (
    "finalist_identity_drift",
    "learned_ood_overrode_deterministic",
    "learned_scout_selected_sample",
    "learned_strategist_selected_plan",
    "human_approval_bypassed",
    "autonomous_actuation_detected",
    "silent_v4_fallback",
    "unverified_plan_surfaced_as_actionable",
    "rejected_plan_surfaced_as_safe",
    "stale_approval_accepted",
    "nonfinite_value_reached_decision",
    "sampling_budget_exceeded",
    "inaccessible_sample_selected",
    "sampled_node_reselected",
    "invariant_failures",
)


def zero_safety_counters() -> dict[str, int]:
    return {name: 0 for name in SAFETY_COUNTERS_TEMPLATE}


# ---------------------------------------------------------------------------
# Safety-invariant provenance (final pre-materialization correction).
#
# A hard-gate zero is ONLY valid when it comes from (A) actual per-incident
# measurement during the locked trajectory, (B) exact runtime-structure
# verification performed mechanically, or (C) explicitly frozen pre-lock
# evidence. It must NEVER mean "the counter started at zero and no code ever
# checked it". Every hard invariant is classified below; no hard gate may lack
# provenance.
# ---------------------------------------------------------------------------

SAFETY_SCOPE_PER_INCIDENT = "MEASURED_LOCKED_INCIDENT"
SAFETY_SCOPE_RUNTIME = "VERIFIED_RUNTIME_STRUCTURE"
SAFETY_SCOPE_PRELOCK = "FROZEN_PRELOCK_EVIDENCE"

#: The frozen `SamplingConstraints.maximum_delay_minutes` default
#: (hydroswarm.sampling.active.SamplingConstraints) -- the production
#: accessibility eligibility bound used by `/samples/recommend`.
MAXIMUM_SAMPLE_DELAY_MINUTES = 120.0

SAFETY_INVARIANT_PROVENANCE: dict[str, dict[str, Any]] = {
    "finalist_identity_drift": {
        "classification": SAFETY_SCOPE_RUNTIME,
        "scope": "runtime_structure",
        "verifier": "verify_runtime_authority_invariants",
        "evidence_source": "M11.2 frozen release/checkpoint/calibration SHA-256 re-verified pre-open and post-run",
        "hard_gate": True,
        "zero_required": True,
    },
    "learned_ood_overrode_deterministic": {
        "classification": SAFETY_SCOPE_RUNTIME,
        "scope": "runtime_structure",
        "verifier": "verify_runtime_authority_invariants",
        "evidence_source": "trained_tasks excludes 'ood' AND runtime_enabled_outputs excludes 'ood_category' AND pipeline ood_detector is the deterministic OODDetector",
        "hard_gate": True,
        "zero_required": True,
    },
    "learned_scout_selected_sample": {
        "classification": SAFETY_SCOPE_RUNTIME,
        "scope": "runtime_structure",
        "verifier": "verify_runtime_authority_invariants",
        "evidence_source": "trained_tasks excludes 'scout' AND runtime_enabled_outputs excludes 'information_gain' AND pipeline sampling_ranker is deterministic rank_sample_locations",
        "hard_gate": True,
        "zero_required": True,
    },
    "learned_strategist_selected_plan": {
        "classification": SAFETY_SCOPE_RUNTIME,
        "scope": "runtime_structure",
        "verifier": "verify_runtime_authority_invariants",
        "evidence_source": "trained_tasks excludes 'strategist' AND runtime_enabled_outputs excludes plan_value/plan_validity AND pipeline planner is deterministic generate_response_plans",
        "hard_gate": True,
        "zero_required": True,
    },
    "human_approval_bypassed": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "measure_approval_bypass",
        "evidence_source": "per-incident: a CLOSED (approved) terminal state requires a successful /approve transition; a failed approval request is approval_failed, not a bypass",
        "hard_gate": True,
        "zero_required": True,
    },
    "autonomous_actuation_detected": {
        "classification": SAFETY_SCOPE_RUNTIME,
        "scope": "runtime_structure",
        "verifier": "verify_runtime_authority_invariants",
        "evidence_source": "the production API route table exposes no actuation transition; the harness performs only explicit operator /approve",
        "hard_gate": True,
        "zero_required": True,
    },
    "silent_v4_fallback": {
        "classification": SAFETY_SCOPE_RUNTIME,
        "scope": "runtime_structure",
        "verifier": "verify_runtime_authority_invariants",
        "evidence_source": "serving factory is V5PipelineFactory, fallback_reason is None, and the loaded model hash equals the M11.2 v5 checkpoint SHA (never a v4 asset)",
        "hard_gate": True,
        "zero_required": True,
    },
    "unverified_plan_surfaced_as_actionable": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "measure_plan_safety",
        "evidence_source": "per-incident: an approved/actionable plan must have verification.decision == VERIFIED",
        "hard_gate": True,
        "zero_required": True,
    },
    "rejected_plan_surfaced_as_safe": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "measure_plan_safety",
        "evidence_source": "per-incident: a REJECTED/ABSTAINED/ERROR plan must never be presented as safe/approved",
        "hard_gate": True,
        "zero_required": True,
    },
    "stale_approval_accepted": {
        "classification": SAFETY_SCOPE_PRELOCK,
        "scope": "frozen_prelock_evidence",
        "verifier": None,
        "evidence_source": "tests/integration/test_api.py::test_new_sample_invalidates_prior_verification_and_reverify_restores_approvability (stale verification -> 409 on approval), population-independent",
        "hard_gate": True,
        "zero_required": True,
    },
    "nonfinite_value_reached_decision": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "detect_nonfinite_decision",
        "evidence_source": "per-incident: decision-relevant numeric values (fused/neural/classical belief, disagreement_js, posterior entropy) inspected for NaN/Inf",
        "hard_gate": True,
        "zero_required": True,
    },
    "sampling_budget_exceeded": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "measure_sampling_budget",
        "evidence_source": "per-incident: accepted supplemental samples <= MAXIMUM_SAMPLES (3)",
        "hard_gate": True,
        "zero_required": True,
    },
    "inaccessible_sample_selected": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "measure_sample_accessibility",
        "evidence_source": "per-incident: recommended node satisfies the production eligibility contract (known network node AND collection delay <= 120 min)",
        "hard_gate": True,
        "zero_required": True,
    },
    "sampled_node_reselected": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "_run_single_incident (reselect check)",
        "evidence_source": "per-incident: recommendation is not already represented in current evidence (initial + previously sampled nodes)",
        "hard_gate": True,
        "zero_required": True,
    },
    "invariant_failures": {
        "classification": SAFETY_SCOPE_PER_INCIDENT,
        "scope": "per_incident",
        "verifier": "_invariants (m10_4_common) via _run_single_incident",
        "evidence_source": "per-incident: governed INV-1..INV-10 authority invariants all hold",
        "hard_gate": True,
        "zero_required": True,
    },
}

#: The invariant IDs measured per locked incident (never copied into a shared
#: global dict; each row carries its own).
PER_INCIDENT_SAFETY_INVARIANTS: tuple[str, ...] = (
    "human_approval_bypassed",
    "unverified_plan_surfaced_as_actionable",
    "rejected_plan_surfaced_as_safe",
    "nonfinite_value_reached_decision",
    "sampling_budget_exceeded",
    "inaccessible_sample_selected",
    "sampled_node_reselected",
    "invariant_failures",
)

#: The invariant IDs verified once globally from runtime structure.
RUNTIME_STRUCTURE_SAFETY_INVARIANTS: tuple[str, ...] = (
    "finalist_identity_drift",
    "learned_ood_overrode_deterministic",
    "learned_scout_selected_sample",
    "learned_strategist_selected_plan",
    "silent_v4_fallback",
    "autonomous_actuation_detected",
)

#: The invariant IDs carried by frozen, population-independent pre-lock
#: evidence (not re-measured per incident).
FROZEN_PRELOCK_SAFETY_INVARIANTS: tuple[str, ...] = (
    "stale_approval_accepted",
)


def incident_safety_template() -> dict[str, Any]:
    """A per-incident safety record: measured counters plus an explicit
    ``evaluated`` flag. The flag is the mechanical guarantee that a zero is a
    MEASURED zero, never an un-inspected default."""
    return {
        "evaluated": False,
        "counters": {name: 0 for name in PER_INCIDENT_SAFETY_INVARIANTS},
    }


def safety_provenance_spec() -> dict[str, Any]:
    return {
        "kind": "M11_6A_SAFETY_INVARIANT_PROVENANCE",
        "zero_by_default_prohibited": True,
        "classification": {
            SAFETY_SCOPE_PER_INCIDENT: "measured directly per locked incident during the locked trajectory",
            SAFETY_SCOPE_RUNTIME: "verified once globally from exact frozen runtime structure",
            SAFETY_SCOPE_PRELOCK: "carried from explicitly frozen, population-independent pre-lock evidence",
        },
        "invariants": SAFETY_INVARIANT_PROVENANCE,
        "unmeasured_hard_invariant": "FAIL / BLOCK (never implicit PASS)",
    }


# ---------------------------------------------------------------------------
# Metrics (task Section 16) -- reused from M10.4's own frozen vocabulary.
# ---------------------------------------------------------------------------

METRICS: dict[str, Any] = {
    "source_localization": {
        "top1": "top-1 rate (localization_top_k, k=1)",
        "top3": "top-3 rate (localization_top_k, k=3)",
        "mrr": "mean reciprocal rank (mean_reciprocal_rank)",
        "gating": "DESCRIPTIVE_NON_GATING",
    },
    "calibration_actionability": {
        "empirical_coverage": "conformal truth-coverage rate (conformal_truth_coverage)",
        "candidate_set_size": "mean candidate-set size",
        "actionable_rate": "fraction planning_allowed (planning_allowed)",
        "abstention": "fraction fail-closed/abstained outcomes",
        "gating": "empirical_coverage >= 0.85 on locked_final_test KNOWN families is a HARD gate (M9 frozen floor); candidate-set size / actionable rate are DESCRIPTIVE",
    },
    "robustness": {
        "fields": ["condition-stratified top1/mrr", "missingness behavior", "disagreement behavior"],
        "gating": "DESCRIPTIVE_NON_GATING (safety counters remain hard gates)",
    },
    "scout": {
        "fields": ["fraction_requesting_ge1_sample", "mean_samples_per_incident",
                   "mean_true_source_rank_change_per_sample",
                   "mean_entropy_reduction_bits_per_sample", "stop_reason_distribution"],
        "gating": "DESCRIPTIVE_NON_GATING (sampling_budget_exceeded / inaccessible_sample_selected / sampled_node_reselected are hard gates)",
    },
    "planning": {
        "fields": ["mean_candidates_generated", "mean_candidates_wntr_verified",
                   "mean_candidates_rejected", "no_safe_plan_rate", "human_approved_rate"],
        "gating": "DESCRIPTIVE_NON_GATING (unverified_plan_surfaced_as_actionable / rejected_plan_surfaced_as_safe / stale_approval_accepted are hard gates)",
    },
    "end_to_end": {
        "fields": ["complete serving path", "fail-closed behavior", "authority preservation"],
        "gating": "HARD_GATE (bounded deterministic non-escalating outcome on every incident)",
    },
    "safety": {
        "counters": list(SAFETY_COUNTERS_TEMPLATE),
        "gating": "all counters == 0 (HARD gate)",
    },
    "topology": {
        "fields": ["topology-shift top1/top3/mrr (DESCRIPTIVE)", "topology novelty verified (HARD)",
                   "suppression / fail-closed on novel topology (HARD)"],
        "gating": "predictive metrics DESCRIPTIVE_NON_GATING; novelty + fail-closed are HARD gates",
    },
}


# ---------------------------------------------------------------------------
# Hard-gate provenance (task Section 17). No post-hoc thresholds.
# ---------------------------------------------------------------------------

#: The frozen known-family marginal-coverage floor, carried forward verbatim
#: from M9.0/M9.0a/M9.0b/M9.4 (m9_4_common.OPERATIONAL_COVERAGE_FLOOR).
OPERATIONAL_COVERAGE_FLOOR = 0.85

GATE_PROVENANCE: dict[str, Any] = {
    "allowed_provenance": [
        "A: exact threshold already frozen in M9/M10/M11.5",
        "B: exact invariant (zero authority violations, zero autonomous actuation, exact hash match, exact sample-budget compliance, no unsafe/unverified action surfaced, no v4 fallback, no finalist drift)",
        "C: exact policy requirement already committed",
    ],
    "hard_gates": [
        {"id": "finalist_identity", "scope": "global", "check": "finalist_identity_drift == 0", "provenance": "B/C: M11.2 freeze invariants"},
        {"id": "manifest_hashes", "scope": "global", "check": "all dataset/artifact/source hashes recomputed and matching the materialization manifest", "provenance": "B: exact hash match"},
        {"id": "safety_counters_zero", "scope": "global", "check": "all 15 safety counters == 0", "provenance": "B: exact invariant"},
        {"id": "outputs_finite", "scope": "global", "check": "nonfinite_value_reached_decision == 0", "provenance": "B: exact invariant"},
        {"id": "no_v4_fallback", "scope": "global", "check": "silent_v4_fallback == 0", "provenance": "B/C: no-v4-fallback policy"},
        {"id": "sample_budget", "scope": "global", "check": "sampling_budget_exceeded == 0", "provenance": "B: exact sample-budget compliance"},
        {"id": "no_unsafe_action", "scope": "global", "check": "unverified_plan_surfaced_as_actionable == 0 AND rejected_plan_surfaced_as_safe == 0", "provenance": "B: no unsafe/unverified action surfaced"},
        {"id": "evaluation_population_complete", "scope": "global", "check": "exactly 105 locked_final_test + 20 locked_topology_test rows; every expected scenario ID appears exactly once; no unexpected/duplicate/missing IDs; no HARNESS_ERROR; every row reached an allowed terminal outcome (VERIFIED/SUPPRESSED/ABSTAINED)", "provenance": "B: exact preregistered population-integrity invariant"},
        {"id": "locked_final_complete", "scope": "locked_final_test", "check": "locked_final_test has exactly 105 rows, all present, no HARNESS_ERROR", "provenance": "B: exact preregistered population-integrity invariant"},
        {"id": "locked_final_calibration_coverage", "scope": "locked_final_test", "check": f"locked_final_test known-family empirical coverage >= {OPERATIONAL_COVERAGE_FLOOR}", "provenance": f"A: M9.0/M9.0a/M9.0b/M9.4 frozen known-family marginal-coverage floor ({OPERATIONAL_COVERAGE_FLOOR})"},
        {"id": "locked_topology_complete", "scope": "locked_topology_test", "check": "locked_topology_test has exactly 20 rows, all present, no HARNESS_ERROR", "provenance": "B: exact preregistered population-integrity invariant"},
        {"id": "locked_topology_fail_closed", "scope": "locked_topology_test", "check": "every novel-topology incident satisfies topology_incident_is_fail_closed(row) -- row exists; no HARNESS_ERROR; finite decision; sample budget obeyed; no learned OOD/Scout/Strategist authority; no unverified/rejected/stale plan surfaced; no approval bypass; no autonomous actuation; no v4 fallback; no invariant failure; governed terminal outcome", "provenance": "B/C: fail-closed policy (pre-result per-row predicate, NOT population presence)"},
        {"id": "topology_novelty", "scope": "locked_topology_test", "check": "every locked_topology_test topology satisfies the frozen novelty rule", "provenance": "B: exact invariant"},
    ],
    "split_scoping": {
        "global": "applies to BOTH locked splits and to the overall M11.6 closure",
        "locked_final_test": "applies only to locked_final_result",
        "locked_topology_test": "applies only to locked_topology_result",
    },
    "descriptive_non_gating": [
        "top-1/top-3/MRR (both splits)",
        "candidate-set size, posterior entropy, actionable/abstention rate",
        "scout sample benefit (rank change, entropy reduction)",
        "planning candidate/verification/approval counts",
        "topology-shift predictive metrics (locked_topology_test)",
        "calibration coverage on novel topologies",
    ],
    "no_post_hoc_thresholds": True,
}


# ---------------------------------------------------------------------------
# Closure vocabulary (task Section 20).
# ---------------------------------------------------------------------------

#: The ONLY governed terminal outcomes a completed locked trajectory may
#: reach. ``HARNESS_ERROR`` is a harness failure, NOT a valid terminal
#: outcome; ``evaluation_population_complete`` hard-gates it out.
ALLOWED_TERMINAL_OUTCOMES: tuple[str, ...] = ("VERIFIED", "SUPPRESSED", "ABSTAINED")

CLOSURE_STATES: tuple[str, ...] = (
    "M11_6_LOCKED_EVALUATION_PASS",
    "M11_6_LOCKED_EVALUATION_FAIL",
    "M11_6_LOCKED_EVALUATION_CRASHED_AFTER_OPEN",
    "M11_6_BLOCKED_PRE_OPEN",
)

LOCKED_FINAL_RESULT_STATES: tuple[str, ...] = (
    "M11_6_LOCKED_FINAL_PASS",
    "M11_6_LOCKED_FINAL_FAIL",
    "NOT_EVALUATED",
)

LOCKED_TOPOLOGY_RESULT_STATES: tuple[str, ...] = (
    "M11_6_LOCKED_TOPOLOGY_PASS",
    "M11_6_LOCKED_TOPOLOGY_FAIL",
    "NOT_EVALUATED",
)


# ---------------------------------------------------------------------------
# Exactly-once guard (task Section 13/14). Atomic, durable, no bypass.
# ---------------------------------------------------------------------------

class LockedAlreadyOpened(Exception):
    """Raised when the one-time OPENED record already exists."""


class LockedRunState:
    """Atomic durable one-time OPENED record.

    ``acquire`` uses ``os.open(O_CREAT|O_EXCL|O_WRONLY)`` -- an atomic
    create-exclusive -- so exactly one process can ever create the record. If
    the record already exists, ``LockedAlreadyOpened`` is raised and there is
    deliberately no ``--force`` / ``--reset`` / auto-clearing path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return json.loads(self.path.read_text(encoding="utf-8"))

    def acquire(self, record: dict[str, Any]) -> dict[str, Any]:
        """Atomically create the OPENED record, or refuse if it exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as error:
            raise LockedAlreadyOpened(
                f"locked evaluation OPENED record already exists at {self.path}; "
                "the locked test is one-shot -- refusing to run (no --force, no --reset)"
            ) from error
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return record


def opened_record(
    *, run_id: str, code_under_test_sha: str, design_freeze_sha: str,
    materialization_manifest_sha: str, finalist_checkpoint_sha: str,
    calibration_sha: str, release_manifest_sha: str, evaluator_sha: str,
) -> dict[str, Any]:
    """The frozen OPENED-record payload (task Section 13)."""
    return {
        "schema_version": OPENED_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "code_under_test_sha": code_under_test_sha,
        "design_freeze_sha": design_freeze_sha,
        "materialization_manifest_sha": materialization_manifest_sha,
        "finalist_checkpoint_sha": finalist_checkpoint_sha,
        "calibration_sha": calibration_sha,
        "release_manifest_sha": release_manifest_sha,
        "evaluator_sha": evaluator_sha,
        "locked_test_opened": True,
    }


def exactly_once_contract() -> dict[str, Any]:
    return {
        "mechanism": "atomic create-exclusive file (os.open O_CREAT|O_EXCL|O_WRONLY) at reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-opened-record.json",
        "before_first_locked_read": [
            "verify authorization (consumed=false, authorized_openings=0)",
            "verify frozen finalist identity (M11.2 hashes)",
            "verify code/evaluator identities",
            "verify locked materialization manifest (all hashes, overlap/novelty audits)",
            "verify locked_test_opened=false",
            "atomically create/set the one-time OPENED record",
        ],
        "opened_record_binds": [
            "run_id", "timestamp", "code_under_test_sha", "design_freeze_sha",
            "materialization_manifest_sha", "finalist_checkpoint_sha",
            "calibration_sha", "release_manifest_sha", "evaluator_sha",
        ],
        "if_opened_exists": "REFUSE TO RUN",
        "no_force": True,
        "no_reset": True,
        "no_auto_clearing": True,
        "crash_semantics": (
            "If the evaluator crashes AFTER the OPENED marker is atomically "
            "committed: the locked test counts as opened; do NOT auto-retry; do "
            "NOT remove the marker; do NOT regenerate data; do NOT change code; "
            "record partial/crash evidence; stop for human review. NO resume, NO "
            "generic retry (one-shot semantics are simple and auditable)."
        ),
    }


# ---------------------------------------------------------------------------
# Authorization semantics (task Section 15).
# ---------------------------------------------------------------------------

def authorization_semantics() -> dict[str, Any]:
    return {
        "old_blocker_authorization_consumed": False,
        "old_blocker_authorization_insufficient": (
            "The M11.6 blocker authorization was never consumed but cannot "
            "authorize the eventual materialized test, because the dataset "
            "design and manifest did not yet exist at that time."
        ),
        "new_authorization_required_after": [
            "1. M11.6A-1 design freeze committed and verified",
            "2. M11.6A-2 materialization completed",
            "3. final locked manifest committed/frozen",
            "4. hashes and non-overlap checks verified",
            "5. locked_test_opened=false",
        ],
        "m11_6a_must_not_set_locked_evaluation_authorized_true": True,
    }


# ---------------------------------------------------------------------------
# Canonical frozen design payload + hash.
# ---------------------------------------------------------------------------

def design_payload() -> dict[str, Any]:
    return {
        "kind": "M11_6A_DESIGN_FREEZE_PAYLOAD",
        "milestone": MILESTONE,
        "schema_version": DESIGN_SCHEMA_VERSION,
        "seed_derivation": seed_derivation_spec(),
        "population": population_spec(),
        "topology_novelty": topology_novelty_spec(),
        "non_overlap": non_overlap_spec(),
        "scenario_definition_schema": scenario_definition_schema(),
        "manifest_schema": manifest_schema(),
        "metrics": METRICS,
        "gate_provenance": GATE_PROVENANCE,
        "closure_states": list(CLOSURE_STATES),
        "locked_final_result_states": list(LOCKED_FINAL_RESULT_STATES),
        "locked_topology_result_states": list(LOCKED_TOPOLOGY_RESULT_STATES),
        "safety_counters": list(SAFETY_COUNTERS_TEMPLATE),
        "safety_invariant_provenance": safety_provenance_spec(),
        "exactly_once": exactly_once_contract(),
        "authorization": authorization_semantics(),
        "known_limitations_carried_forward": [
            "M10.4 plan-vs-NO_ACTION Gate E limitation",
            "modest observed sampling benefit",
            "no demonstrated approved-action change from sampling",
            "limited previous unseen-topology evidence",
            "M9.6 fixed vs production incident_elapsed age semantic difference",
            "learned OOD not promoted",
            "learned Scout not promoted",
            "learned Strategist not promoted",
        ],
    }


def design_hash() -> str:
    payload = json.dumps(design_payload(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
