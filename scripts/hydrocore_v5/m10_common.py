"""Shared, governed helpers for Milestone 10 (system-level downstream
validation). Frozen protocol: `docs/evaluation/HYDROCORE_V5_M10_PROTOCOL.md`.

M10 takes the M9-selected HydroCore-S predictor (frozen M9.6 canonical
FINAL_STEP_1350 checkpoints, M9.7A-authoritative policy) as-is -- this
module trains NOTHING and re-registers no new model variant. It reuses
UNMODIFIED re-exports from `m9_8_common`/`m9_6_common`/`m9_4_common` for
every governance primitive, depth grid, family list, and threshold M10 does
not redefine -- only the M10-specific seed namespace (Section 3 of the
protocol) and OOD-condition constructors are new here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import m9_4_common as m4  # noqa: E402
import m9_6_common as m6  # noqa: E402
import m9_8_common as m8  # noqa: E402

# ---------------------------------------------------------------------------
# Re-exported, UNMODIFIED governance/utility primitives.
# ---------------------------------------------------------------------------

current_commit = m4.current_commit
current_branch = m4.current_branch
assert_locked_test_closed = m4.assert_locked_test_closed
checkpoint_sha256 = m4.checkpoint_sha256
wilson_interval_90 = m4.wilson_interval_90
full_junction_list = m4.full_junction_list
environment_info = m6.environment_info
depth_bucket_of = m6.depth_bucket_of

ROOT_PATH = m4.ROOT
ALL_FAMILY_LOADERS = m4.ALL_FAMILY_LOADERS
FROZEN_BRANCH = m4.FROZEN_BRANCH

SEEDS: tuple[int, ...] = m4.SEEDS
assert SEEDS == (20260814, 31874, 20260815)
DEPTHS: tuple[int, ...] = m4.DEPTHS
EARLY_DEPTHS = m4.EARLY_DEPTHS
MID_DEPTHS = m4.MID_DEPTHS
MATURE_DEPTHS = m4.MATURE_DEPTHS

ALL_FAMILIES: tuple[str, ...] = m8.ALL_FAMILIES
TRAINED_FAMILIES: tuple[str, ...] = m8.TRAINED_FAMILIES
UNSEEN_FAMILIES: tuple[str, ...] = tuple(f for f in ALL_FAMILIES if f not in TRAINED_FAMILIES)

# ---------------------------------------------------------------------------
# Section 1 of the M10 protocol: selected predictor identity (frozen,
# taken as-is -- M9.7A-authoritative FINAL_STEP_1350 policy, M9.6's own
# canonical export, the SAME checkpoints M9.8 reused).
# ---------------------------------------------------------------------------

S_VARIANT = "small"
CANONICAL_CHECKPOINT_POLICY = m6.CANONICAL_CHECKPOINT_POLICY
assert CANONICAL_CHECKPOINT_POLICY == "FINAL_STEP_1350"
M9_6_TRAINING_RUNS_DIR = m6.M9_6_TRAINING_RUNS_DIR
SELECTED_PARAMETER_COUNT = 4182612

M10_DIR = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m10"
M10_0_DIR = M10_DIR / "m10-0"
M10_1_DIR = M10_DIR / "m10-1"
M10_2_PREFLIGHT_DIR = M10_DIR / "m10-2-preflight"


def canonical_s_checkpoint(seed: int) -> dict:
    import json

    record = json.loads((M9_6_TRAINING_RUNS_DIR / f"ARM_B_M9_6-seed{seed}.json").read_text())
    assert record["canonical_checkpoint_policy"] == CANONICAL_CHECKPOINT_POLICY
    return record


# ---------------------------------------------------------------------------
# Section 3 of the M10 protocol: fresh, disjoint development-only OOD seed
# namespace. base=1_100_000_000, source_stride=10_000, role
# "ood_development_m10_1" -- verified disjoint from every existing seed base
# in the repository at protocol-freeze time (grep over
# 1_100_000_000..1_199_999_999 found zero prior hits).
# ---------------------------------------------------------------------------

M10_1_SEED_BASE = 1_100_000_000
M10_1_SOURCE_STRIDE = 10_000
#: Per-family offset, spaced 10_000_000 apart (matching the M9.8 convention
#: of >= 100_000 spacing between family blocks, generously widened here
#: since M10.1 uses fewer repeats/source than M9.8).
M10_1_FAMILY_OFFSET: dict[str, int] = {family: 10_000_000 * index for index, family in enumerate(ALL_FAMILIES)}


def m10_1_seed_base(family: str, condition: str, condition_index: int) -> int:
    """Deterministic per-(family, condition) seed base, frozen before
    execution. condition_index is the condition's fixed position in
    M10_1_CONDITIONS (Section 5 of the protocol) -- never reassigned."""

    return M10_1_SEED_BASE + M10_1_FAMILY_OFFSET[family] + condition_index * 1_000_000


#: Section 5 of the M10 protocol, frozen order (index is part of the seed
#: formula above, so this tuple's order must never change post-freeze).
#: Topology shift is NOT a separate generated condition here -- it is the
#: existing TRAINED_FAMILIES-vs-UNSEEN_FAMILIES cross-cut, evaluated on the
#: IN_DISTRIBUTION population across all six families exactly as M9.8 did
#: (Section 5 of the M10 protocol table). SENSOR_DROPOUT/SEVERITY_SHIFT are
#: genuine per-scenario perturbations, generated within every family.
M10_1_CONDITIONS: tuple[str, ...] = (
    "IN_DISTRIBUTION",
    "SENSOR_DROPOUT",
    "SEVERITY_SHIFT",
)
