"""Canonical scenario-specific hydraulic-state reconstruction.

core-issues3.txt Phase 1: ``scripts/generate_trajectory_corpus.py`` built one
pristine WNTR network and one ``FeatureContext`` per topology family and
reused both for every scenario in that family, discarding each scenario's
own randomized demand regime, roughness perturbation, tank-level variation,
and pipe-outage state that the scenario was actually simulated against.
That silently invalidates travel-time labels, the Strategist's WNTR
verification context, and any state-dependent Scout artifact.

This module factors out the *one* canonical function that replays a stored
``ScenarioManifest`` against its pristine topology and returns the exact
randomized WNTR model plus its derived ``FeatureContext`` -- the same
replay logic ``scripts/run_corpus_gates.py``'s ``deterministic_replay`` gate
already implements for a different purpose (a determinism check, not a
reconstruction API). Per Phase 1 item 3 ("do not maintain two independent
implementations of replay reconstruction"), both the corpus gate and the
trajectory generator now call this one function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from hydroswarm.data.scenarios import (
    NEGLIGIBLE_STRENGTH_MG_MIN,
    CurriculumStage,
    EventType,
    GeneratedScenario,
    ScenarioGenerationConfig,
    ScenarioManifest,
    WNTRScenarioGenerator,
    network_sha256,
)
from hydroswarm.simulation.wrapper import HydraulicSimulator

from .corpus import FeatureContext, _hydraulic_state_hash, build_feature_context

#: A corpus generator's own degradation-probability policy, e.g.
#: scripts/generate_cycle_b_corpus.py's ``_degradation_probabilities``.
#: Deliberately not imported from any one script here -- every corpus
#: generator owns its own policy and must pass it in explicitly, so this
#: module carries no hidden dependency on one generator's defaults.
DegradationPolicy = Callable[[CurriculumStage], Mapping[str, float]]

#: Bumped whenever reconstruct_scenario_network's own replay contract
#: changes (which config fields it derives from a manifest, or how). Not
#: the same as generator_version (hydroswarm.data.scenarios), which
#: versions scenario *generation* itself.
RECONSTRUCTION_POLICY_VERSION = "scenario-reconstruction-v1"


class ScenarioReconstructionError(RuntimeError):
    """Raised when a reconstructed scenario does not semantically match its
    stored manifest -- fail closed rather than silently handing a caller a
    hydraulic context that does not correspond to the scenario it claims
    to represent."""


@dataclass(frozen=True, slots=True)
class ReconstructedScenarioContext:
    """The exact per-scenario replay of a stored ``ScenarioManifest``
    against its pristine topology: the randomized WNTR model actually
    simulated, its derived ``FeatureContext``, and the identity hashes
    every downstream consumer (trajectory generation, Strategist
    planning/verification, travel-time targets, Scout sample truth) must
    use instead of a topology-family-shared pristine context."""

    scenario: GeneratedScenario
    network: Any
    feature_context: FeatureContext
    #: Stable across every scenario sharing this topology family --
    #: hashes the pristine network, not this scenario's randomized state.
    topology_hash: str
    #: This scenario's own exact randomized network configuration --
    #: demand pattern, roughness, tank levels, pipe status.
    network_state_hash: str
    hydraulic_state_hash: str
    replay_matched: bool
    #: True only when ``original`` was supplied, its artifact_sha256 differed
    #: from the reconstruction's own, but the semantic array-equality
    #: fallback still matched (the cross-architecture signed-zero case).
    artifact_hash_drifted: bool
    reconstruction_policy_hash: str


def reconstruct_scenario_network(
    pristine_network: Any,
    manifest: ScenarioManifest,
    *,
    degradation_policy: DegradationPolicy,
    original: GeneratedScenario | None = None,
    generator: WNTRScenarioGenerator | None = None,
) -> ReconstructedScenarioContext:
    """Exactly regenerate ``manifest``'s scenario against
    ``pristine_network``, returning the randomized WNTR model actually
    simulated (demand regime, roughness variation, tank levels, pipe
    outages) and its derived ``FeatureContext`` -- never the pristine
    network alone.

    Uses the manifest's own recorded seed, source, event type, and stage;
    the degradation-probability policy must be supplied by the caller
    (Phase 1 item 2: "use the same seed, event type, source, start,
    duration, demand regime, degradation probabilities, and topology
    loader as the original generator").

    Fails closed (raises ``ScenarioReconstructionError``) if the replay
    does not match the stored manifest. When ``original`` (the scenario's
    own previously-generated ``GeneratedScenario``) is supplied, mismatch
    detection falls through to a semantic array-equality check on
    ``observed_concentration``/``observation_mask``/``truth_concentration``
    before failing -- the same cross-architecture-tolerant check
    ``run_corpus_gates.py`` already uses, since IEEE-754 leaves signed-zero
    results implementation-defined across CPU/SIMD backends, so a
    hash-only mismatch alone is inconclusive.
    """

    generator = generator or WNTRScenarioGenerator()
    config = ScenarioGenerationConfig(
        seed=manifest.seed,
        network_id=manifest.network_id,
        network_family=manifest.network_family,
        split=manifest.split,
        stage=manifest.stage,
        event_type=EventType(manifest.event_type),
        source_node=manifest.incident.source_nodes[0],
        sensor_count=len(manifest.sensor_nodes),
        # Matches every corpus generator's own convention (generate_cycle_b_corpus.py,
        # run_corpus_gates.py): pipe outages are never enabled for this corpus
        # family. rng.random() is still drawn unconditionally inside
        # _randomize_hydraulics regardless of this value, so the rng stream
        # position is identical either way -- only the branch outcome differs.
        pipe_outage_probability=0.0,
        **degradation_policy(manifest.stage),
    )
    scenario, randomized_network = generator.generate_with_network(pristine_network, config)
    replay_matched, artifact_hash_drifted = _verify_replay(scenario, manifest, original)
    feature_context = build_feature_context(randomized_network)
    return ReconstructedScenarioContext(
        scenario=scenario,
        network=randomized_network,
        feature_context=feature_context,
        topology_hash=network_sha256(pristine_network),
        network_state_hash=HydraulicSimulator(randomized_network).state_hash(),
        hydraulic_state_hash=_hydraulic_state_hash(feature_context.state),
        replay_matched=replay_matched,
        artifact_hash_drifted=artifact_hash_drifted,
        reconstruction_policy_hash=RECONSTRUCTION_POLICY_VERSION,
    )


def _verify_replay(
    regenerated: GeneratedScenario,
    manifest: ScenarioManifest,
    original: GeneratedScenario | None,
) -> tuple[bool, bool]:
    if regenerated.manifest.replay_sha256 != manifest.replay_sha256:
        raise ScenarioReconstructionError(
            f"replay_sha256 differs for scenario {manifest.scenario_id}: "
            f"reconstruction does not match the stored manifest's own generation parameters"
        )
    if original is None:
        return True, False
    # artifact_sha256 alone must never gate whether the array-level check
    # runs: a tampered .npz whose containing manifest record was left
    # untouched has a *matching* recorded artifact_sha256 (the old,
    # untampered value) even though the actual arrays differ. The hash is
    # only informational bookkeeping (hash_only_drift, for the documented
    # cross-architecture signed-zero case); the array comparison below is
    # the actual, unconditional determinism criterion.
    hash_drifted = regenerated.manifest.artifact_sha256 != original.manifest.artifact_sha256
    for key in ("observed_concentration", "observation_mask", "truth_concentration"):
        left, right = getattr(regenerated, key), getattr(original, key)
        if np.array_equal(left, right, equal_nan=True):
            continue
        # NORMAL/SENSOR_FAULT_ONLY scenarios inject NEGLIGIBLE_STRENGTH_MG_MIN
        # (1e-9 mg/min) rather than exactly zero, so their concentration
        # arrays sit at ~1e-8 mg/L -- far below quantization_step (1e-3
        # mg/L, the smallest physically meaningful resolution anywhere else
        # in this system) but still large enough to occasionally land on a
        # float32 rounding boundary two independently-deterministic
        # reconstructions agree on with each other but not with a corpus
        # generated under different BLAS/thread conditions (same class of
        # cross-environment nondeterminism as the documented signed-zero
        # case, just below exact-equality resolution instead of at it).
        # 1e-6 mg/L is three orders of magnitude below the smallest
        # meaningful signal and cannot mask a real reconstruction defect,
        # which would produce differences at the strength/travel-time
        # scale (>= NEGLIGIBLE_STRENGTH_MG_MIN), not at float rounding
        # noise.
        if not np.allclose(left, right, atol=1e-6, rtol=0.0, equal_nan=True):
            raise ScenarioReconstructionError(
                f"{key} array differs on reconstruction for scenario {manifest.scenario_id}"
            )
    return True, hash_drifted


def simulate_all_node_truth(
    reconstruction: ReconstructedScenarioContext,
    *,
    base_strength_mg_min: float = 10.0,
) -> pd.DataFrame:
    """Re-run the incident simulation against the reconstructed scenario's
    exact randomized network, returning concentration at EVERY node --
    not just the scenario's own originally-chosen sensor subset.

    core-issues3.txt Phase 5 (item Q: "All-node Scout sample truth must
    come from the exact randomized scenario, not a pristine topology or a
    generic signature prediction"). WNTRScenarioGenerator.generate_with_
    network already computes exactly this full-node frame internally
    (HydraulicSimulator.simulate_incident's own return value) but only
    keeps the sensor-subset columns before discarding it
    (`frame = simulation.concentration_mg_l.loc[:, list(sensors)]`) --
    this function reruns that same simulate_incident call against the
    SAME already-reconstructed network and the manifest's own recorded
    incident parameters (source/start/duration/strength), so no fresh RNG
    draws are needed and the result is deterministic. Callers that need to
    verify this reproduces the original scenario's own stored sensor
    columns should compare against `reconstruction.scenario.
    truth_concentration` at `reconstruction.scenario.manifest.
    sensor_nodes` (see the regression test for this function).

    `base_strength_mg_min` must match the value the corpus's own generator
    used (ScenarioGenerationConfig.base_strength_mg_min, default 10.0 --
    no corpus generator in this repository overrides it; reconstruct_
    scenario_network's own config construction likewise never does).
    """

    manifest = reconstruction.scenario.manifest
    incident = manifest.incident
    is_contamination = manifest.event_type == EventType.CONTAMINATION.value
    injection_strength = (
        incident.relative_strength if is_contamination else NEGLIGIBLE_STRENGTH_MG_MIN / base_strength_mg_min
    )
    simulator = HydraulicSimulator(reconstruction.network)
    simulation = simulator.simulate_incident(
        incident.source_nodes[0],
        strength_mg_min=base_strength_mg_min * injection_strength,
        start_minute=incident.start_minute,
        duration_minutes=incident.duration_minutes,
    )
    return simulation.concentration_mg_l
