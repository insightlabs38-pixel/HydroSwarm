"""Milestone 9.0b: four Mondrian calibration-grouping schemes over the
UNMODIFIED `hydroswarm.calibration.conformal.SplitConformalCalibrator`
(docs/evaluation/HYDROCORE_V5_M9_0B_PROTOCOL.md, Section 5).

Every scheme reuses the SAME calibrator class -- only the STRING passed as
`network_id` (the Mondrian group key) differs between CURRENT_FAMILY_DEPTH/
POOLED_DEPTH_AWARE/BROAD_FALLBACK_CONTROL. HIERARCHICAL_CONSERVATIVE
combines two of those fitted calibrators (family:depth and pooled-depth)
via `HierarchicalCalibrator`, a small experimental wrapper -- NOT wired into
`SplitConformalCalibrator` itself, per the protocol's "experiment-scoped
code only, no production runtime change" scope (Section 15/19 of the
milestone instructions).

Experiment-scoped only: nothing here is imported by production inference
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hydroswarm.calibration.conformal import (
    CalibrationExample,
    SplitConformalCalibrator,
    _quantile,
)

SCHEME_NAMES: tuple[str, ...] = (
    "CURRENT_FAMILY_DEPTH", "POOLED_DEPTH_AWARE", "BROAD_FALLBACK_CONTROL", "HIERARCHICAL_CONSERVATIVE",
)
ALPHA = 0.1
#: SplitConformalCalibrator.fit's own default, unchanged/not tuned anywhere
#: in M9.0b (protocol Section 5/7).
MINIMUM_GROUP_SIZE = 10


@dataclass(frozen=True, slots=True)
class SchemeRow:
    """One calibration or evaluation row, family/depth_bucket kept SEPARATE
    (not pre-baked into one network_id string) so every scheme's grouping
    can be built from the SAME underlying rows without re-running
    inference per scheme (protocol Section 2)."""

    probabilities: tuple[float, ...]
    true_index: int
    condition: str
    family: str
    depth_bucket: str


def _network_id_current_family_depth(row: SchemeRow) -> str:
    return f"{row.family}:{row.depth_bucket}"


def _network_id_pooled_depth(row: SchemeRow) -> str:
    return row.depth_bucket


def _network_id_broad_fallback(_row: SchemeRow) -> None:
    return None


#: The three schemes expressible purely as "which network_id string do we
#: pass" -- HIERARCHICAL_CONSERVATIVE (below) is not, since it needs BOTH
#: groupings simultaneously per row.
NETWORK_ID_BUILDERS = {
    "CURRENT_FAMILY_DEPTH": _network_id_current_family_depth,
    "POOLED_DEPTH_AWARE": _network_id_pooled_depth,
    "BROAD_FALLBACK_CONTROL": _network_id_broad_fallback,
}


def fit_scheme(
    scheme: str, rows: Sequence[SchemeRow], *, alpha: float = ALPHA, model_hash: str,
    minimum_group_size: int = MINIMUM_GROUP_SIZE,
) -> SplitConformalCalibrator:
    network_id_fn = NETWORK_ID_BUILDERS[scheme]
    examples = [
        CalibrationExample(probabilities=row.probabilities, true_index=row.true_index, condition=row.condition, network_id=network_id_fn(row))
        for row in rows
    ]
    return SplitConformalCalibrator.fit(
        examples, alpha=alpha, model_hash=model_hash, feature_schema_hash="n/a",
        dataset_manifest_hash=f"m9-0b-{scheme}", minimum_group_size=minimum_group_size,
    )


def candidate_set_for_scheme(
    scheme: str, calibrator: SplitConformalCalibrator, row: SchemeRow,
) -> tuple[tuple[int, ...], str, str]:
    """Returns (candidate_set, selection_source, selection_group_key). Never
    reads `row.true_index` -- candidate-set selection uses only
    probabilities/condition/network_id, matching runtime's own no-label
    guarantee (protocol Section 5, Section 22 test 5)."""
    network_id_fn = NETWORK_ID_BUILDERS[scheme]
    network_id = network_id_fn(row)
    source, group_key, _scores = calibrator.selection(condition=row.condition, network_id=network_id)
    candidate = calibrator.candidate_set(row.probabilities, condition=row.condition, network_id=network_id)
    return candidate, source, group_key


@dataclass(frozen=True, slots=True)
class HierarchicalCalibrator:
    """Scheme D (protocol Section 5). `q_used = max(q_family_depth,
    q_pooled_depth)` when the family:depth group has >= minimum_group_size
    calibration examples; otherwise `q_used = q_pooled_depth` (which itself
    may fall further to CONDITION_SPECIFIC/GLOBAL via `pooled_depth`'s own
    built-in fallback). By construction `q_used >= q_pooled_depth` in EVERY
    case -- this scheme can only be as-or-more conservative (wider, safer
    candidate sets) than POOLED_DEPTH_AWARE alone, never less."""

    family_depth: SplitConformalCalibrator
    pooled_depth: SplitConformalCalibrator
    alpha: float

    def quantile_and_source(self, row: SchemeRow) -> tuple[float, str, str]:
        pooled_source, pooled_group, pooled_scores = self.pooled_depth.selection(
            condition=row.condition, network_id=row.depth_bucket,
        )
        q_pooled = _quantile(pooled_scores, self.alpha)
        family_key = f"{row.family}:{row.depth_bucket}"
        if family_key in self.family_depth.artifact.network_scores:
            family_scores = self.family_depth.artifact.network_scores[family_key]
            q_family = _quantile(family_scores, self.alpha)
            return max(q_family, q_pooled), "HIERARCHICAL_FAMILY_DEPTH_MAX_POOLED", family_key
        return q_pooled, f"HIERARCHICAL_POOLED_FALLBACK:{pooled_source}", pooled_group

    def candidate_set(self, row: SchemeRow) -> tuple[tuple[int, ...], float, str, str]:
        """Returns (candidate_set, quantile_used, source, group_key). Never
        reads `row.true_index`."""
        q_used, source, group = self.quantile_and_source(row)
        values = np.asarray(row.probabilities, dtype=float)
        values = values / values.sum()
        candidate = tuple(int(index) for index in np.flatnonzero(1.0 - values <= q_used))
        return candidate, q_used, source, group


def fit_hierarchical(
    rows: Sequence[SchemeRow], *, alpha: float = ALPHA, model_hash: str, minimum_group_size: int = MINIMUM_GROUP_SIZE,
) -> HierarchicalCalibrator:
    family_depth = fit_scheme("CURRENT_FAMILY_DEPTH", rows, alpha=alpha, model_hash=model_hash, minimum_group_size=minimum_group_size)
    pooled_depth = fit_scheme("POOLED_DEPTH_AWARE", rows, alpha=alpha, model_hash=model_hash, minimum_group_size=minimum_group_size)
    return HierarchicalCalibrator(family_depth=family_depth, pooled_depth=pooled_depth, alpha=alpha)


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Standard Wilson score interval, no scipy dependency. `confidence`
    must be 0.95 (the only frozen value M9.0b uses -- protocol Section 11);
    the z constant below is the two-sided 95% normal quantile."""
    if n == 0:
        return (float("nan"), float("nan"))
    if confidence != 0.95:
        raise ValueError("only the frozen 95% Wilson interval is supported")
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))
