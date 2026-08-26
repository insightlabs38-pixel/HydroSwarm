"""Fair, nuisance-searched oracle comparator (Task 1 audit correction).

`oracle.py`'s `rank_candidates` (reused unmodified for backward
compatibility -- nothing in the frozen `exp/source-identifiability-analysis`
branch is edited by this module) replays every candidate junction under
`common.simulate_candidate`, which hard-codes the TRUE incident's exact
`start_minute`, `duration_minutes`, and `injection_strength_mg_min`
(`common.reconstruct_incident`) identically for every candidate. Those three
quantities are drawn straight from the frozen scenario's own ground-truth
`IncidentTruth`/`ScenarioManifest` (see `data/locked/m11-6/**/scenarios.jsonl`
rows' `start_minute`/`duration_minutes`/`relative_strength` fields, which
`run_build_confirmatory.py` copies verbatim from `bundle.incident.*`) -- i.e.
label-derived quantities. HydroCore-v5 never receives them as input: it
predicts them itself via `start_time_head`/`duration_head`/
`relative_strength_head` classification heads
(`src/hydroswarm/model/core.py`). See `docs/evaluation/
ORACLE_INFORMATION_AUDIT.md` for the full classification
(PRIVILEGED_ORACLE) and reasoning.

This module builds the fairest practical correction identified there: a
NUISANCE-SEARCHED oracle. For every candidate junction, it profile-searches
(maximum-likelihood / minimum-residual) over the SAME finite bins
`hydroswarm.data.scenarios.ScenarioGenerationConfig` itself draws the true
row values from --

    start_time_bins_min = (0, 60, 120, 240)   minutes
    duration_bins_min   = (30, 60, 120)       minutes
    strength_bins       = (0.5, 1.0, 2.0)     relative strength

-- a population-level generator hyperparameter, not this incident's own
label, so using it is not label leakage: it encodes "the source could have
started at any of these plausible times/strengths/durations", exactly the
support a real physics-based investigator would search without knowing
which one actually happened, and exactly the support HydroCore-v5's own
`start_time`/`duration`/`relative_strength` classification heads are
themselves trained to discriminate among (4/8/4-way heads over comparable
bins -- see `RoleHead(d_model, 12/8/4)` sizes in `core.py`, i.e. HydroCore-v5
is not being compared against a search space strictly finer than what it
itself is asked to resolve).

The candidate's own hydraulic/demand realization
(`incident.randomized_network`, produced once per incident by
`WNTRScenarioGenerator._randomize_hydraulics` and shared, unmutated, across
every candidate and every grid point) is still held at its TRUE drawn value.
Marginalizing that too would require re-running `_randomize_hydraulics` per
grid point (a combinatorial blow-up of EPANET calls well beyond this pilot's
simulator budget) -- this is a DOCUMENTED RESIDUAL PRIVILEGE of the fair
oracle, not a silent one; see the audit doc's limitations section for the
bounded secondary check quantifying it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import common, library, oracle, signatures
from hydroswarm.simulation.wrapper import HydraulicSimulator

#: Verbatim copy of `ScenarioGenerationConfig`'s own field defaults
#: (`src/hydroswarm/data/scenarios.py`) -- the population-level support the
#: generator draws each row's true start/duration/strength from. Not tuned,
#: not row-specific, not label-derived.
START_TIME_BINS_MIN: tuple[int, ...] = (0, 60, 120, 240)
DURATION_BINS_MIN: tuple[int, ...] = (30, 60, 120)
STRENGTH_BINS: tuple[float, ...] = (0.5, 1.0, 2.0)
BASE_STRENGTH_MG_MIN_DEFAULT: float = 10.0


@dataclass(frozen=True, slots=True)
class NuisanceGridPoint:
    start_minute: int
    duration_minutes: int
    relative_strength: float


def default_nuisance_grid() -> tuple[NuisanceGridPoint, ...]:
    return tuple(
        NuisanceGridPoint(start_minute=s, duration_minutes=d, relative_strength=r)
        for s, d, r in itertools.product(START_TIME_BINS_MIN, DURATION_BINS_MIN, STRENGTH_BINS)
    )


@dataclass(frozen=True, slots=True)
class CandidateSearchResult:
    candidate: str
    best_point: NuisanceGridPoint
    best_residual_rmse: float
    best_raw_signature: np.ndarray  # (n_times, n_sensors) log1p(mg/L), sensor columns only
    grid_size: int


def _base_strength_mg_min(row: dict[str, Any]) -> float:
    generator_config = row.get("generator_config") or {}
    return float(generator_config.get("base_strength_mg_min", BASE_STRENGTH_MG_MIN_DEFAULT))


def search_candidate(
    incident: common.ReconstructedIncident,
    candidate_node: str,
    *,
    observed_log1p: np.ndarray,
    observed_mask: np.ndarray,
    base_strength_mg_min: float,
    grid: Sequence[NuisanceGridPoint],
) -> CandidateSearchResult:
    """Profile-search one candidate over the nuisance grid, scoring each
    point by masked RMSE (log1p scale) against the real observation --
    never against the true simulated signature, so this never uses the
    true strength/start/duration even implicitly through the scoring
    target. Returns the minimum-residual (best-fitting) grid point.
    """

    simulator = HydraulicSimulator(incident.randomized_network)
    mask = np.asarray(observed_mask, dtype=bool)
    best: CandidateSearchResult | None = None
    for point in grid:
        result = simulator.simulate_incident(
            candidate_node,
            strength_mg_min=base_strength_mg_min * point.relative_strength,
            start_minute=point.start_minute,
            duration_minutes=point.duration_minutes,
        )
        frame = result.concentration_mg_l.loc[:, list(incident.sensor_nodes)]
        raw = np.log1p(frame.to_numpy(dtype=np.float64))
        if raw.shape != observed_log1p.shape:
            raise ValueError(
                f"candidate {candidate_node!r} grid point produced shape {raw.shape}, "
                f"expected {observed_log1p.shape}"
            )
        if mask.any():
            residual = raw[mask] - observed_log1p[mask]
        else:
            residual = raw.ravel() - observed_log1p.ravel()
        rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan")
        if best is None or rmse < best.best_residual_rmse:
            best = CandidateSearchResult(
                candidate=candidate_node,
                best_point=point,
                best_residual_rmse=rmse,
                best_raw_signature=raw,
                grid_size=len(grid),
            )
    assert best is not None
    return best


@dataclass(frozen=True, slots=True)
class FairOracleResult:
    result: oracle.OracleResult
    per_candidate_search: dict[str, CandidateSearchResult]
    grid_size: int


def rank_candidates_fair(
    bundle: library.IncidentBundle,
    row: dict[str, Any],
    *,
    grid: Sequence[NuisanceGridPoint] | None = None,
) -> FairOracleResult:
    """Fair-oracle counterpart of `oracle.rank_candidates`: identical
    downstream ranking/probability machinery (`bayesian_source_posterior`),
    but every candidate's signature is its OWN best (minimum-residual)
    nuisance-grid fit against the real observation, rather than a signature
    simulated at the true incident's exact nuisance values.
    """

    grid = tuple(grid) if grid is not None else default_nuisance_grid()
    incident = bundle.incident
    base_strength = _base_strength_mg_min(row)
    observed = bundle.observed_sensor_matrix
    mask = bundle.observed_mask

    searches: dict[str, CandidateSearchResult] = {}
    for candidate in incident.junctions:
        searches[candidate] = search_candidate(
            incident,
            candidate,
            observed_log1p=observed,
            observed_mask=mask,
            base_strength_mg_min=base_strength,
            grid=grid,
        )

    raw_signatures = {c: s.best_raw_signature for c, s in searches.items()}
    ranked = oracle.rank_candidates(
        raw_signatures,
        observed,
        true_source=row["source_node"],
        observation_mask=mask,
    )
    return FairOracleResult(result=ranked, per_candidate_search=searches, grid_size=len(grid))


__all__ = [
    "START_TIME_BINS_MIN",
    "DURATION_BINS_MIN",
    "STRENGTH_BINS",
    "BASE_STRENGTH_MG_MIN_DEFAULT",
    "NuisanceGridPoint",
    "default_nuisance_grid",
    "CandidateSearchResult",
    "FairOracleResult",
    "search_candidate",
    "rank_candidates_fair",
]
