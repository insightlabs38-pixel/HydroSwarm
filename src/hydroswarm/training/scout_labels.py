"""Scout label generation (overnight-plan.txt Task 2.3).

Builds on already-implemented, already-tested classical/sampling
infrastructure rather than reimplementing it:

- hydroswarm.classical.signatures.SignatureBuilder/SignatureCache builds a
  full-hypothesis-space signature artifact from *exact simulation*, keyed
  and cached by network/state (Task 1.2's SignatureRegistry indexes these).
- hydroswarm.classical.prior.bayesian_source_posterior turns a scenario's
  own partial observations into a posterior over source hypotheses.
- hydroswarm.sampling.active.rank_sample_locations already implements every
  piece Task 2.3 asks for: enumerate accessible candidates, estimate
  outcomes from the signature library, compute expected information gain
  and candidate reduction, account for delay/cost/redundancy/accessibility,
  identify the best candidate, and decide whether another sample is
  worthwhile.

This module is the missing glue: turning one GeneratedScenario's own
observations into a governed ScoutLabel. It deliberately does not force
Scout labels into ScenarioExample.targets (a single-snapshot-per-incident
model) -- Scout supervision is inherently sequential ("each base incident
may produce multiple sequential Scout states", Task 2.3). The natural home
for that sequence is TrajectoryState/FullTrajectory (Task 2.6); a future
corpus builder wires generate_scout_label() into that trajectory structure,
called once per sampling step with an updated `already_sampled` set, once
real multi-topology scenario generation (Phase 5) exists. Calling it once
with already_sampled=() (the default) produces the first Scout decision at
incident start, which is what this module's tests exercise.

No leakage risk: computing a Scout label for a scenario uses only that
scenario's own observations plus a signature artifact built from exact
network simulation (not fit from other scenarios' incident outcomes), the
same governance shape as the existing classical_prior Sentinel input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from hydroswarm.classical.prior import bayesian_source_posterior
from hydroswarm.classical.signatures import SignatureArtifact, SignatureBuilder, SignatureCache, SignatureCacheKey
from hydroswarm.data.scenarios import GeneratedScenario
from hydroswarm.sampling.active import SampleCandidate, SamplingConstraints, rank_sample_locations
from hydroswarm.simulation.wrapper import HydraulicSimulator

from .corpus import aligned_observations


def build_signature_artifact_for_network(
    network: Any,
    cache: SignatureCache,
    *,
    key: SignatureCacheKey,
    source_nodes: Sequence[str] | None = None,
    sensor_nodes: Sequence[str] | None = None,
    sample_times_seconds: Sequence[int] = (0, 3600, 7200, 10800, 14400, 18000, 21600),
    start_time_bins: Sequence[int] = (0, 60, 120, 240),
    duration_bins: Sequence[int] = (30, 60, 120),
    strength_bins: Sequence[float] = (0.5, 1.0, 2.0),
    demand_regimes: Sequence[str] = ("nominal",),
) -> SignatureArtifact:
    """Fit (or load, if cached) the full-hypothesis-space signature artifact
    Scout label generation needs. Every candidate junction is both a
    possible source and a possible sample location, matching how
    scenario_to_example already treats every junction as a source
    candidate."""

    junctions = tuple(sorted(network.junction_name_list))
    simulator = HydraulicSimulator(network)
    builder = SignatureBuilder(simulator, cache)
    return builder.build_or_load(
        key=key,
        source_nodes=source_nodes or junctions,
        start_time_bins=start_time_bins,
        duration_bins=duration_bins,
        strength_bins=strength_bins,
        demand_regimes=demand_regimes,
        sensor_nodes=sensor_nodes or junctions,
        sample_times_seconds=sample_times_seconds,
    )


def reindex_to_signature_grid(
    scenario: GeneratedScenario, node_ids: Sequence[str], sample_times_seconds: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Align a scenario's own (sensor-aligned) observations onto the
    signature artifact's fixed time grid, matching the nearest-neighbor
    reindexing SignatureBuilder itself uses when fitting signatures.

    Public (not a module-private leading-underscore helper) because a
    second real caller needs it: core-issues3.txt Phase 12 Stage D's
    Scout-policy comparison calls
    hydroswarm.classical.signatures.localize_with_signatures directly (for
    per-node marginal source probabilities and a candidate set -- a
    different question from generate_scout_label's own "which node to
    sample next"), which needs these exact reindexed observation arrays as
    input. Reused here rather than duplicated so both callers stay
    provably consistent with the same reindexing convention."""

    observed, valid = aligned_observations(scenario, node_ids)
    frame = pd.DataFrame(observed, index=scenario.timestamps_seconds, columns=node_ids)
    mask_frame = pd.DataFrame(valid, index=scenario.timestamps_seconds, columns=node_ids)
    reindexed = frame.reindex(index=list(sample_times_seconds), method="nearest")
    mask_reindexed = mask_frame.reindex(index=list(sample_times_seconds), method="nearest")
    return reindexed.to_numpy(dtype=np.float32), mask_reindexed.to_numpy(dtype=bool)


@dataclass(frozen=True, slots=True)
class ScoutLabel:
    sample_node_id: str | None
    information_gain_bits: float
    candidate_reduction_fraction: float
    should_continue_sampling: bool
    stop_reason: str | None
    posterior_entropy_bits: float
    alternatives: tuple[str, ...]
    #: Every candidate rank_sample_locations evaluated (accessible or not,
    #: up to its top_k), exposed so a trajectory builder can record
    #: core-issues2.txt Phase 2's "per-node expected information gain where
    #: supported" and accessibility, without recomputing the posterior/
    #: signature-matching work this module already did.
    candidates: tuple[SampleCandidate, ...] = ()
    #: The same denominator candidate_reduction_fraction was normalized
    #: against (distinct source nodes in the signature artifact's
    #: hypothesis space), exposed so a caller building per-node reduction
    #: targets for every candidate (core-issues3.txt Phase 7.2) can apply
    #: the identical normalization instead of recomputing/duplicating it.
    total_candidate_count: int = 1


def generate_scout_label(
    scenario: GeneratedScenario,
    artifact: SignatureArtifact,
    *,
    already_sampled: Sequence[str] = (),
    constraints: SamplingConstraints | None = None,
    noise_scale_mg_l: float = 0.5,
    revealed_samples: Mapping[str, float] | None = None,
) -> ScoutLabel:
    """`revealed_samples` (core-issues3.txt Phase 5): node_id -> a single
    degraded measurement value revealed by an EARLIER Scout step in the
    same trajectory (see scout_trajectory.build_scout_trajectory), for
    nodes outside the scenario's own original sensor set. Broadcast across
    every time-grid column for that node -- a documented simplification
    (a Scout grab sample is one point-in-time reading, not a continuous
    sensor feed like the scenario's own original sensors; the signature-
    matching grid this module already uses has no representation for
    "a single instant" observation finer than that). Without this
    parameter, behavior is identical to before Phase 5 (the scenario's own
    fixed original observations only) -- this is what made every prior
    Scout "trajectory" re-rank the same fixed evidence at every step."""

    observed, valid = reindex_to_signature_grid(scenario, artifact.sensor_nodes, artifact.sample_times_seconds)
    if revealed_samples:
        # pandas' reindex().to_numpy() can return a read-only view sharing
        # memory with its source frame -- copy before mutating regardless
        # of whether that view happens to be writable for any given pandas
        # version, rather than relying on an implementation detail.
        observed = observed.copy()
        valid = valid.copy()
        node_positions = {node_id: index for index, node_id in enumerate(artifact.sensor_nodes)}
        for node_id, value in revealed_samples.items():
            column = node_positions.get(node_id)
            if column is None:
                continue
            observed[:, column] = value
            valid[:, column] = True
    posterior = bayesian_source_posterior(observed, artifact.library, observation_mask=valid, noise_scale=noise_scale_mg_l)

    base_constraints = constraints or SamplingConstraints()
    effective_constraints = SamplingConstraints(
        collection_time_minutes=base_constraints.collection_time_minutes,
        operational_cost=base_constraints.operational_cost,
        accessible=base_constraints.accessible,
        already_sampled=frozenset(already_sampled) | base_constraints.already_sampled,
        maximum_delay_minutes=base_constraints.maximum_delay_minutes,
        minimum_information_gain_bits=base_constraints.minimum_information_gain_bits,
    )
    remaining_candidates = [
        node for node in artifact.sensor_nodes if node not in effective_constraints.already_sampled
    ]
    result = rank_sample_locations(
        artifact,
        posterior.probabilities,
        constraints=effective_constraints,
        candidate_nodes=remaining_candidates,
        noise_scale_mg_l=noise_scale_mg_l,
    )

    total_candidates = max(1, len({hypothesis.source_node for hypothesis in artifact.hypotheses}))
    if result.recommended_node is None:
        return ScoutLabel(
            sample_node_id=None,
            information_gain_bits=0.0,
            candidate_reduction_fraction=0.0,
            should_continue_sampling=False,
            stop_reason=result.stop_reason,
            posterior_entropy_bits=result.posterior_entropy_bits,
            alternatives=(),
            candidates=result.ranked,
            total_candidate_count=total_candidates,
        )
    top = result.ranked[0]
    reduction_fraction = float(min(1.0, max(0.0, top.expected_candidate_reduction / total_candidates)))
    alternatives = tuple(candidate.node_id for candidate in result.ranked[1:3])
    return ScoutLabel(
        sample_node_id=result.recommended_node,
        information_gain_bits=top.expected_information_gain_bits,
        candidate_reduction_fraction=reduction_fraction,
        should_continue_sampling=not result.stop,
        stop_reason=result.stop_reason,
        posterior_entropy_bits=result.posterior_entropy_bits,
        alternatives=alternatives,
        candidates=result.ranked,
        total_candidate_count=total_candidates,
    )
