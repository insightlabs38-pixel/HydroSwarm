"""Statistics helpers for the source-identifiability analysis.

`unpaired_bootstrap_diff` is new (there is no existing unpaired two-group
bootstrap utility in the repo -- `m9_1_common.paired_bootstrap`, reused
verbatim wherever this analysis compares the SAME incidents under two
systems/conditions, assumes equal-length paired sequences, which does not
hold for a between-group comparison like "low-centrality incidents vs.
high-centrality incidents"). Parameterized identically to
`m9_1_common.paired_bootstrap` (2,000 resamples, seed 20260815, 90%
percentile interval) for consistency, even though the resampling scheme
itself must differ.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_INTERVAL = 0.90


def unpaired_bootstrap_diff(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    interval: float = BOOTSTRAP_INTERVAL,
) -> dict:
    """Percentile bootstrap CI for mean(group_a) - mean(group_b), independent samples."""

    a = np.asarray(list(group_a), dtype=np.float64)
    b = np.asarray(list(group_b), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {
            "n_a": int(a.size),
            "n_b": int(b.size),
            "observed_mean_diff": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_entirely_positive": None,
            "ci_entirely_non_positive": None,
        }
    observed = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for i in range(resamples):
        resample_a = a[rng.integers(0, a.size, size=a.size)]
        resample_b = b[rng.integers(0, b.size, size=b.size)]
        diffs[i] = resample_a.mean() - resample_b.mean()
    lower_pct = (1 - interval) / 2 * 100
    upper_pct = (1 - (1 - interval) / 2) * 100
    ci_lower = float(np.percentile(diffs, lower_pct))
    ci_upper = float(np.percentile(diffs, upper_pct))
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "observed_mean_diff": observed,
        "resamples": resamples,
        "bootstrap_seed": seed,
        "interval": interval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_entirely_positive": bool(ci_lower > 0),
        "ci_entirely_non_positive": bool(ci_upper <= 0),
    }


def tercile_labels(values: Sequence[float]) -> list[str]:
    """Assign 'T1' (lowest third) / 'T2' / 'T3' (highest third) by rank, ties broken by index order."""

    order = np.argsort(np.asarray(values), kind="stable")
    n = len(values)
    labels = [""] * n
    for rank, index in enumerate(order):
        third = min(2, (rank * 3) // n)
        labels[index] = f"T{third + 1}"
    return labels


def median_split(values: Sequence[float]) -> list[str]:
    values_arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(values_arr))
    return ["HIGH" if v >= median else "LOW" for v in values_arr]


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_INTERVAL",
    "unpaired_bootstrap_diff",
    "tercile_labels",
    "median_split",
]
