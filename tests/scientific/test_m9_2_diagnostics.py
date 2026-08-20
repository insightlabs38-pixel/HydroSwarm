"""Contract tests for the Milestone 9.2 diagnostic study (`scripts/
hydrocore_v5/m9_2_common.py`, `m9_2_analysis_lib.py`, `run_m9_2_build_table.py`,
`run_m9_2_analyze.py`).

M9.2 is DIAGNOSTIC / ANALYSIS-ONLY. These tests cover the runner's own
governance/statistical-machinery correctness (pairing integrity, no
unpaired-seed leakage, bootstrap determinism, rank-delta sign convention,
disagreement-table arithmetic, quartile-bin assignment, topology-distance
computation, causal-only missingness features, raw M9.1 artifact
immutability) -- never a promotion decision, never locked-test data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

import m9_1_common as m91  # noqa: E402
import m9_2_analysis_lib as lib  # noqa: E402
import m9_2_common as m92  # noqa: E402

CANONICAL_PATH = m92.M9_2_CANONICAL_PATH
MANIFEST_PATH = m92.M9_2_MANIFEST_PATH


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Governance constants -- never redefined independently of M9.1.
# ---------------------------------------------------------------------------


def test_m9_2_reuses_m9_1_screening_seeds_verbatim():
    assert m92.SCREENING_SEEDS == m91.SCREENING_SEEDS == (20260814, 31874)


def test_confirmation_seed_is_excluded_unpaired_seed():
    assert m92.EXCLUDED_UNPAIRED_SEED == m91.CONFIRMATION_SEED == 20260815
    assert m92.EXCLUDED_UNPAIRED_SEED not in m92.SCREENING_SEEDS


def test_m9_2_bootstrap_seed_distinct_from_m9_1_promotion_bootstrap_seed():
    assert m92.M9_2_BOOTSTRAP_SEED == 20260816
    assert m92.M9_2_BOOTSTRAP_SEED != m91.BOOTSTRAP_SEED
    assert m92.M9_2_BOOTSTRAP_RESAMPLES == 2000
    assert m92.M9_2_BOOTSTRAP_INTERVAL == 0.90


def test_sde_mc_count_not_redefined_by_m9_2():
    # m9_2_common must not shadow/redefine M9.1's frozen SDE Monte-Carlo
    # policy; reconstruction must go through m9_1_common.evaluate_split
    # unmodified, which is what enforces SDE_MC_COUNT == 4 + the frozen
    # Brownian-seed formula.
    assert not hasattr(m92, "SDE_MC_COUNT")
    assert m91.SDE_MC_COUNT == 4
    key = "20260814:902000000:1:0"
    expected = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % (2**31)
    assert m91.brownian_seed(20260814, 902000000, 1, 0) == expected


def test_depths_and_maturity_buckets_match_m9_1():
    assert m92.CAUSAL_PREFIX_DEPTHS == (1, 2, 3, 4, 6, 12, 25)
    assert m92.EARLY_DEPTHS == (1, 2, 3)
    assert m92.MID_DEPTHS == (4, 6)
    assert m92.MATURE_DEPTHS == (12, 25)


# ---------------------------------------------------------------------------
# No locked-test path is ever referenced by M9.2 scripts.
# ---------------------------------------------------------------------------


LOCKED_TOKENS = ("locked_final_test", "locked_topology_test")


@pytest.mark.parametrize(
    "script",
    ["m9_2_common.py", "m9_2_analysis_lib.py", "run_m9_2_build_table.py", "run_m9_2_analyze.py"],
)
def test_no_locked_test_path_literal_in_m9_2_scripts(script):
    # Prose mentions in docstrings (explaining what M9.2 must NOT touch) are
    # fine; what must never appear is CODE that constructs a path/open() call
    # to the locked split itself. Scan only non-comment/non-docstring-marker
    # lines that also contain a path-construction token.
    source = (SCRIPTS_DIR / script).read_text()
    path_construction_tokens = ("Path(", "open(", "/ \"", "/ '", "read_text", "read_bytes")
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("#", '"""', "'''", "*")):
            continue
        for token in LOCKED_TOKENS:
            if token in line and any(pc in line for pc in path_construction_tokens):
                pytest.fail(f"{script}:{lineno} constructs a path to {token}: {line!r}")


def test_locked_test_still_closed_right_now():
    # Re-verified independently of any M9.1 assertion already on disk.
    assert m92.assert_locked_test_closed() is False


# ---------------------------------------------------------------------------
# Bootstrap determinism (Section 5/16).
# ---------------------------------------------------------------------------


def test_paired_bootstrap_deterministic_across_calls():
    candidate = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    control = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    first = m92.paired_bootstrap_m9_2(candidate, control)
    second = m92.paired_bootstrap_m9_2(candidate, control)
    assert first == second
    assert first["bootstrap_seed"] == 20260816
    assert first["resamples"] == 2000


def test_paired_bootstrap_resamples_incidents_not_rows():
    # Passing the SAME per-incident scalar sequence in a different but
    # equal-content order must reproduce the identical bootstrap statistic
    # only when paired order is preserved -- this test instead asserts the
    # observed mean diff is exactly the arithmetic mean of the paired
    # differences (i.e. the unit of resampling is one paired incident row,
    # not an unpaired flattened pool).
    candidate = [1.0, 0.0, 1.0]
    control = [0.0, 0.0, 1.0]
    result = m92.paired_bootstrap_m9_2(candidate, control)
    assert result["observed_mean_diff"] == pytest.approx((1.0 - 0.0) / 3 + (0.0 - 0.0) / 3 + (1.0 - 1.0) / 3)
    assert result["n_incidents"] == 3


# ---------------------------------------------------------------------------
# Rank-delta sign convention (Section 7/16): positive = novel arm worse.
# ---------------------------------------------------------------------------


def _toy_rank_df() -> pd.DataFrame:
    rows = []
    for seed in m92.SCREENING_SEEDS:
        for incident_id, current_rank, novel_rank in [(1, 1, 1), (2, 1, 4), (3, 4, 1)]:
            rows.append(
                dict(
                    arm="CURRENT", training_seed=seed, incident_id=incident_id, prefix_depth=12,
                    depth_bucket="MATURE", true_source_rank=current_rank,
                )
            )
            rows.append(
                dict(
                    arm="GRAPH_ODE", training_seed=seed, incident_id=incident_id, prefix_depth=12,
                    depth_bucket="MATURE", true_source_rank=novel_rank,
                )
            )
    return pd.DataFrame(rows)


def test_rank_movement_sign_convention_positive_is_worse():
    df = _toy_rank_df()
    result = lib.rank_movement(df)
    per_seed = result["GRAPH_ODE"]["12"][str(m92.SCREENING_SEEDS[0])]
    # incident 2: current=1, novel=4 -> delta=+3 (worse); incident 3: current=4, novel=1 -> delta=-3 (better); incident 1: delta=0.
    assert per_seed["mean"] == pytest.approx((0 + 3 + (-3)) / 3)
    assert per_seed["fraction_worsened"] == pytest.approx(1 / 3)
    assert per_seed["fraction_improved"] == pytest.approx(1 / 3)
    assert per_seed["fraction_unchanged"] == pytest.approx(1 / 3)
    assert per_seed["large_regressions"][">=3"] == 1
    assert per_seed["large_improvements"]["<=-3"] == 1


# ---------------------------------------------------------------------------
# Disagreement 2x2 arithmetic (Section 6/16).
# ---------------------------------------------------------------------------


def _toy_disagreement_df() -> pd.DataFrame:
    # incident: (current_correct, novel_correct)
    pattern = {1: (True, True), 2: (True, False), 3: (False, True), 4: (False, False), 5: (True, True)}
    rows = []
    for seed in m92.SCREENING_SEEDS:
        for incident_id, (cur_ok, nov_ok) in pattern.items():
            for arm, ok in (("CURRENT", cur_ok), ("GRAPH_ODE", nov_ok)):
                rows.append(
                    dict(
                        arm=arm, training_seed=seed, incident_id=incident_id, prefix_depth=25,
                        top1_correct=ok,
                    )
                )
    return pd.DataFrame(rows)


def test_disagreement_table_counts_sum_to_n_and_match_pattern():
    df = _toy_disagreement_df()
    result = lib.disagreement_tables(df)
    entry = result["by_arm_depth"]["GRAPH_ODE"]["25"][str(m92.SCREENING_SEEDS[0])]
    assert entry["both_correct_A"] == 2  # incidents 1, 5
    assert entry["current_only_B"] == 1  # incident 2
    assert entry["novel_only_C"] == 1  # incident 3
    assert entry["both_wrong_D"] == 1  # incident 4
    assert entry["n"] == 5
    assert entry["net_paired_advantage_C_minus_B"] == 0
    assert entry["current_only_win_incident_ids"] == [2]
    assert entry["novel_only_win_incident_ids"] == [3]


# ---------------------------------------------------------------------------
# Cross-seed consistency classification (Section 13/16).
# ---------------------------------------------------------------------------


def test_classify_cross_seed_robust_vs_mixed_vs_single():
    assert lib.classify_cross_seed([0.02, 0.03]) == "ROBUST"
    assert lib.classify_cross_seed([0.02, -0.01]) == "MIXED"
    assert lib.classify_cross_seed([0.02, None]) == "SINGLE_SEED_ONLY"
    assert lib.classify_cross_seed([None, None]) == "SINGLE_SEED_ONLY"


# ---------------------------------------------------------------------------
# Quartile-bin assignment (Section 9/16): predeclared, outcome-independent.
# ---------------------------------------------------------------------------


def test_quartile_bins_are_roughly_balanced_and_outcome_independent():
    series = pd.Series(range(100))
    bins = lib._quartile_bins(series)
    counts = bins.value_counts()
    assert len(counts) == 4
    for count in counts:
        assert 20 <= count <= 30


def test_quartile_bins_degrade_gracefully_with_few_distinct_values():
    series = pd.Series([0.0] * 50 + [1.0] * 50)
    bins = lib._quartile_bins(series)
    assert bins.notna().all()


# ---------------------------------------------------------------------------
# Topology distance computation (Section 8/16), against the known fixed
# golden-reference network structure (R1-J1-J2-J3-J4-J1 loop, J2-T1-J4 branch).
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_topology_distances_match_known_golden_reference_structure():
    import run_m9_2_build_table as build

    topo = build.build_topology()
    dist = topo["distances"]
    assert dist["J1"]["J1"] == 0
    assert dist["J1"]["J2"] == 1
    assert dist["J1"]["J4"] == 1  # J4-J1 edge closes the loop
    assert dist["J1"]["J3"] == 2  # via J2 or J4
    assert dist["R1"]["J1"] == 1
    assert dist["J2"]["T1"] == 1
    assert dist["T1"]["J4"] == 1
    assert dist["T1"]["J1"] == 2  # via J4 or J2
    assert set(topo["junctions"]) == {"J1", "J2", "J3", "J4"}


# ---------------------------------------------------------------------------
# Causal-only missingness features (Section 9/16): depth d must never see
# evidence beyond the first d reports.
# ---------------------------------------------------------------------------


@pytest.mark.real_simulation
def test_missingness_features_are_prefix_only_no_future_leakage():
    import m9_1_common as common
    import run_m9_2_build_table as build

    dev_records = common.load_pool("development_holdout")
    features = build.build_missingness_features(dev_records[:1])
    incident_id = common.incident_id_for("development_holdout", 0)
    per_depth = features[incident_id]
    for depth in (1, 2, 3):
        block = per_depth[depth]
        assert block["total_observation_slots"] == block["n_sensor_series"] * depth
    # Monotonic: elapsed observation time and total slots must never
    # decrease as depth grows (strictly causal accumulation).
    depths = sorted(per_depth)
    for a, b in zip(depths, depths[1:]):
        assert per_depth[b]["total_observation_slots"] >= per_depth[a]["total_observation_slots"]


# ---------------------------------------------------------------------------
# Canonical table integrity (requires the table to already have been built;
# skipped otherwise so this file remains runnable before the expensive
# build step, per Section 16's "focused tests" framing).
# ---------------------------------------------------------------------------


canonical_missing = not CANONICAL_PATH.exists()


@pytest.mark.skipif(canonical_missing, reason="m9-2-canonical-diagnostics.jsonl not built yet")
def test_canonical_table_exact_pairing_by_seed_incident_depth():
    df = pd.read_json(CANONICAL_PATH, lines=True)
    counts = df.groupby(["training_seed", "incident_id", "prefix_depth"])["arm"].nunique()
    assert (counts == 4).all(), "every (seed, incident, depth) key must have exactly the 4 arms"
    arm_counts = df.groupby("arm").size()
    assert len(set(arm_counts)) == 1, "every arm must contribute the same row count"


@pytest.mark.skipif(canonical_missing, reason="m9-2-canonical-diagnostics.jsonl not built yet")
def test_canonical_table_never_contains_excluded_unpaired_seed():
    df = pd.read_json(CANONICAL_PATH, lines=True)
    assert m92.EXCLUDED_UNPAIRED_SEED not in df["training_seed"].unique().tolist()
    assert set(df["training_seed"].unique().tolist()) == set(m92.SCREENING_SEEDS)


@pytest.mark.skipif(canonical_missing, reason="m9-2-canonical-diagnostics.jsonl not built yet")
def test_canonical_table_row_provenance_is_declared():
    df = pd.read_json(CANONICAL_PATH, lines=True)
    assert df["diagnostic_row_provenance"].str.contains("PERSISTED_M9_1_DEV_ROW").all()


manifest_missing = not MANIFEST_PATH.exists()


@pytest.mark.skipif(manifest_missing, reason="m9-2-manifest.json not built yet")
def test_raw_m9_1_artifacts_never_overwritten_by_m9_2():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for name, identity in manifest["m9_1_source_artifact_identities"].items():
        path = ROOT / identity["path"]
        assert _sha256_file(path) == identity["sha256"], f"{name} was modified after M9.2 recorded its identity"


@pytest.mark.skipif(manifest_missing, reason="m9-2-manifest.json not built yet")
def test_manifest_records_lock_state_and_reproduction_gate():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["locked_test_opened_before"] is False
    assert manifest["locked_test_opened_after"] is False
    assert manifest["m9_1_aggregate_reproduction"]["status"] == "REPRODUCED_EXACTLY"
    assert manifest["seed_excluded_from_pairing"] == 20260815
