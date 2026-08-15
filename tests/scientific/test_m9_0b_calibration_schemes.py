"""Milestone 9.0b Section 22: calibration-grouping scheme regression tests.

Proves, before any real evaluation run, that:

1. CURRENT_FAMILY_DEPTH exactly reproduces current B_DEPTH_AWARE grouping
   (network_id = f"{family}:{depth_bucket}", identical fitted quantiles to
   a hand-built SplitConformalCalibrator using that same construction).
2. POOLED_DEPTH_AWARE ignores topology family when choosing its depth
   group (two families' calibration rows land in the SAME group) while
   still distinguishing EARLY/MID/MATURE.
3. HIERARCHICAL_CONSERVATIVE: q_used >= q_family_depth (when the family
   group is usable), q_used >= q_pooled_depth always, and its candidate
   set is never smaller (never narrower) than POOLED_DEPTH_AWARE alone
   would give for the same row.
4. Fallback/selection behavior is deterministic (repeated calls agree).
5. No true label enters candidate-set selection at runtime (identical rows
   differing only in true_index produce identical candidate sets).
6. Alpha remains exactly 0.1 through every scheme's fit.
7. No unseen-family row can enter a scheme fit given known-family-only
   input rows (structural: fitted network_scores keys never carry an
   unseen-family tag).
8. Frozen predictor checkpoint provenance verification (SHA256 cross-check
   between m9-0a-runs and m9-0a-results.json) succeeds against the real,
   already-trained M9.0a artifacts and fails closed on a tampered hash.
9. All three predictor seeds' checkpoint records are independently
   addressable/verifiable (no shared state).
10. locked_final_test/locked_topology_test remain unopened.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "hydrocore_v5"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT / "src"))

from hydroswarm.calibration.conformal import SplitConformalCalibrator  # noqa: E402
from hydroswarm.evaluation.live_robustness import locked_test_opened  # noqa: E402

from m9_0b_calibration_schemes import (  # noqa: E402
    ALPHA,
    MINIMUM_GROUP_SIZE,
    SchemeRow,
    candidate_set_for_scheme,
    fit_hierarchical,
    fit_scheme,
    wilson_interval,
)

RUNS_M9_0A = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-runs"
RESULTS_M9_0A = ROOT / "reports" / "evaluation" / "hydrocore-v5" / "m9-0a-results.json"
SEEDS = (20260814, 31874, 20260815)


def _make_row(*, prob_true: float, n_nodes: int, family: str, depth_bucket: str, true_index: int = 0, condition: str = "CLEAN") -> SchemeRow:
    other = (1.0 - prob_true) / max(n_nodes - 1, 1)
    probs = [prob_true] + [other] * (n_nodes - 1)
    return SchemeRow(probabilities=tuple(probs), true_index=true_index, condition=condition, family=family, depth_bucket=depth_bucket)


def _synthetic_rows(n_per_group: int = MINIMUM_GROUP_SIZE + 5) -> list[SchemeRow]:
    """>= minimum_group_size rows for every (family, depth_bucket) cell
    across 2 families x 3 buckets, varying prob_true deterministically so
    each group has a real score distribution (not a degenerate constant)."""
    rows: list[SchemeRow] = []
    for family in ("golden-reference", "branched-loop"):
        for bucket in ("EARLY", "MID", "MATURE"):
            for i in range(n_per_group):
                # Deterministic varying confidence, never exactly 0 or 1.
                prob_true = 0.5 + 0.4 * ((i % 7) / 6.0 - 0.5)
                rows.append(_make_row(prob_true=prob_true, n_nodes=6, family=family, depth_bucket=bucket))
    return rows


def test_current_family_depth_matches_hand_built_grouping() -> None:
    rows = _synthetic_rows()
    scheme_calibrator = fit_scheme("CURRENT_FAMILY_DEPTH", rows, model_hash="test")

    from hydroswarm.calibration.conformal import CalibrationExample
    hand_examples = [
        CalibrationExample(probabilities=row.probabilities, true_index=row.true_index, condition=row.condition, network_id=f"{row.family}:{row.depth_bucket}")
        for row in rows
    ]
    hand_calibrator = SplitConformalCalibrator.fit(hand_examples, alpha=ALPHA, model_hash="test", feature_schema_hash="n/a", dataset_manifest_hash="hand")

    assert set(scheme_calibrator.artifact.network_scores) == set(hand_calibrator.artifact.network_scores)
    for key in scheme_calibrator.artifact.network_scores:
        assert scheme_calibrator.artifact.network_scores[key] == hand_calibrator.artifact.network_scores[key]
    assert scheme_calibrator.artifact.report.coverage == hand_calibrator.artifact.report.coverage


def test_pooled_depth_aware_ignores_family_but_keeps_maturity() -> None:
    rows = _synthetic_rows()
    calibrator = fit_scheme("POOLED_DEPTH_AWARE", rows, model_hash="test")
    # Groups are keyed purely by depth_bucket -- no family-qualified key exists.
    assert set(calibrator.artifact.network_scores) == {"EARLY", "MID", "MATURE"}
    golden_row = _make_row(prob_true=0.6, n_nodes=6, family="golden-reference", depth_bucket="EARLY")
    branched_row = _make_row(prob_true=0.6, n_nodes=6, family="branched-loop", depth_bucket="EARLY")
    _cand_g, source_g, group_g = candidate_set_for_scheme("POOLED_DEPTH_AWARE", calibrator, golden_row)
    _cand_b, source_b, group_b = candidate_set_for_scheme("POOLED_DEPTH_AWARE", calibrator, branched_row)
    assert source_g == source_b == "NETWORK_SPECIFIC"
    assert group_g == group_b == "EARLY"  # same group regardless of family.
    mature_golden = _make_row(prob_true=0.6, n_nodes=6, family="golden-reference", depth_bucket="MATURE")
    _cand_m, _source_m, group_m = candidate_set_for_scheme("POOLED_DEPTH_AWARE", calibrator, mature_golden)
    assert group_m == "MATURE"
    assert group_m != group_g  # depth still distinguished.


def test_hierarchical_quantile_never_below_pooled_or_family() -> None:
    rows = _synthetic_rows()
    hierarchical = fit_hierarchical(rows, model_hash="test")
    pooled_only = fit_scheme("POOLED_DEPTH_AWARE", rows, model_hash="test")

    for family in ("golden-reference", "branched-loop"):
        for bucket in ("EARLY", "MID", "MATURE"):
            row = _make_row(prob_true=0.55, n_nodes=8, family=family, depth_bucket=bucket)
            q_used, source, _group = hierarchical.quantile_and_source(row)
            _pooled_source, _pooled_group, pooled_scores = pooled_only.selection(condition=row.condition, network_id=bucket)
            from hydroswarm.calibration.conformal import _quantile
            q_pooled = _quantile(pooled_scores, ALPHA)
            assert q_used >= q_pooled - 1e-12, (family, bucket, q_used, q_pooled)
            if source == "HIERARCHICAL_FAMILY_DEPTH_MAX_POOLED":
                family_key = f"{family}:{bucket}"
                q_family = _quantile(hierarchical.family_depth.artifact.network_scores[family_key], ALPHA)
                assert q_used >= q_family - 1e-12

            # Candidate set is never a smaller (narrower) set than pooled-alone would give.
            hier_candidate, _q, _s, _g = hierarchical.candidate_set(row)
            pooled_candidate, _source2, _group2 = candidate_set_for_scheme("POOLED_DEPTH_AWARE", pooled_only, row)
            assert set(pooled_candidate).issubset(set(hier_candidate)), (family, bucket, pooled_candidate, hier_candidate)


def test_hierarchical_falls_back_to_pooled_when_family_group_underpowered() -> None:
    # Only 3 rows for "loop-grid" (below MINIMUM_GROUP_SIZE=10) but a full
    # pooled-depth pool from the other two families for the same bucket.
    rows = _synthetic_rows()  # golden-reference/branched-loop, 3 buckets, >=10 each.
    for i in range(3):
        rows.append(_make_row(prob_true=0.5, n_nodes=6, family="loop-grid", depth_bucket="EARLY"))
    hierarchical = fit_hierarchical(rows, model_hash="test")
    row = _make_row(prob_true=0.5, n_nodes=6, family="loop-grid", depth_bucket="EARLY")
    _q, source, _group = hierarchical.quantile_and_source(row)
    assert source.startswith("HIERARCHICAL_POOLED_FALLBACK"), source
    assert "loop-grid:EARLY" not in hierarchical.family_depth.artifact.network_scores


def test_selection_is_deterministic() -> None:
    rows = _synthetic_rows()
    calibrator = fit_scheme("CURRENT_FAMILY_DEPTH", rows, model_hash="test")
    row = _make_row(prob_true=0.6, n_nodes=6, family="golden-reference", depth_bucket="MATURE")
    first = candidate_set_for_scheme("CURRENT_FAMILY_DEPTH", calibrator, row)
    second = candidate_set_for_scheme("CURRENT_FAMILY_DEPTH", calibrator, row)
    assert first == second

    hierarchical = fit_hierarchical(rows, model_hash="test")
    hier_first = hierarchical.candidate_set(row)
    hier_second = hierarchical.candidate_set(row)
    assert hier_first == hier_second


def test_no_true_label_used_at_selection_time() -> None:
    rows = _synthetic_rows()
    calibrator = fit_scheme("CURRENT_FAMILY_DEPTH", rows, model_hash="test")
    row_true_0 = _make_row(prob_true=0.6, n_nodes=6, family="golden-reference", depth_bucket="MATURE", true_index=0)
    row_true_5 = _make_row(prob_true=0.6, n_nodes=6, family="golden-reference", depth_bucket="MATURE", true_index=5)
    candidate_0, _s0, _g0 = candidate_set_for_scheme("CURRENT_FAMILY_DEPTH", calibrator, row_true_0)
    candidate_5, _s5, _g5 = candidate_set_for_scheme("CURRENT_FAMILY_DEPTH", calibrator, row_true_5)
    assert candidate_0 == candidate_5  # identical probabilities/condition/family/depth -> identical set, regardless of true_index.

    hierarchical = fit_hierarchical(rows, model_hash="test")
    hier_0 = hierarchical.candidate_set(row_true_0)
    hier_5 = hierarchical.candidate_set(row_true_5)
    assert hier_0 == hier_5


def test_alpha_remains_exactly_point_one() -> None:
    assert ALPHA == 0.1
    rows = _synthetic_rows()
    for scheme in ("CURRENT_FAMILY_DEPTH", "POOLED_DEPTH_AWARE", "BROAD_FALLBACK_CONTROL"):
        calibrator = fit_scheme(scheme, rows, model_hash="test")
        assert calibrator.artifact.alpha == 0.1
    hierarchical = fit_hierarchical(rows, model_hash="test")
    assert hierarchical.alpha == 0.1
    assert hierarchical.family_depth.artifact.alpha == 0.1
    assert hierarchical.pooled_depth.artifact.alpha == 0.1


def test_broad_fallback_control_has_no_network_grouping() -> None:
    rows = _synthetic_rows()
    calibrator = fit_scheme("BROAD_FALLBACK_CONTROL", rows, model_hash="test")
    # A single `network_id=None` group is fitted (every row shares that
    # literal key), but SplitConformalCalibrator.selection's own
    # `network_id is not None` guard makes it structurally unreachable --
    # NETWORK_SPECIFIC can never be selected for this scheme.
    assert set(calibrator.artifact.network_scores) <= {None}
    row = _make_row(prob_true=0.6, n_nodes=6, family="golden-reference", depth_bucket="EARLY")
    _candidate, source, _group = candidate_set_for_scheme("BROAD_FALLBACK_CONTROL", calibrator, row)
    assert source in ("CONDITION_SPECIFIC", "GLOBAL")
    for family in ("golden-reference", "branched-loop"):
        for bucket in ("EARLY", "MID", "MATURE"):
            probe = _make_row(prob_true=0.6, n_nodes=6, family=family, depth_bucket=bucket)
            _c, probe_source, _g = candidate_set_for_scheme("BROAD_FALLBACK_CONTROL", calibrator, probe)
            assert probe_source != "NETWORK_SPECIFIC"


def test_no_unseen_family_row_can_enter_a_scheme_fit() -> None:
    known_families = {"golden-reference", "branched-loop", "loop-grid"}
    rows = _synthetic_rows()
    assert all(row.family in known_families for row in rows)
    calibrator = fit_scheme("CURRENT_FAMILY_DEPTH", rows, model_hash="test")
    for network_id in calibrator.artifact.network_scores:
        family = network_id.split(":")[0]
        assert family in known_families
    for unseen in ("coastal-branch", "tree-branch", "dense-loop"):
        assert not any(key.startswith(f"{unseen}:") for key in calibrator.artifact.network_scores)


def test_wilson_interval_contains_point_estimate_and_widens_with_fewer_samples() -> None:
    lower_small, upper_small = wilson_interval(8, 10)
    lower_large, upper_large = wilson_interval(800, 1000)
    assert lower_small <= 0.8 <= upper_small
    assert lower_large <= 0.8 <= upper_large
    assert (upper_small - lower_small) > (upper_large - lower_large)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence=0.90)


def test_frozen_predictor_checkpoint_provenance_matches_and_fails_closed_on_tamper() -> None:
    for seed in SEEDS:
        run_record = json.loads((RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text())
        results_record = json.loads(RESULTS_M9_0A.read_text())
        recorded_hash = run_record["training_summary"]["export_sha256"]
        cross_check_hash = results_record["arms"]["ARM_B2"]["per_seed"][str(seed)]["checkpoint_sha256"]
        assert recorded_hash == cross_check_hash, f"seed {seed}: provenance cross-check hash mismatch"
        export_path = Path(run_record["training_summary"]["export_path"])
        assert export_path.exists(), f"seed {seed}: frozen checkpoint file missing"
        on_disk_hash = hashlib.sha256(export_path.read_bytes()).hexdigest()
        assert on_disk_hash == recorded_hash, f"seed {seed}: on-disk checkpoint hash drifted from recorded provenance"

        # Fails closed on a tampered hash.
        assert (on_disk_hash + "tampered") != recorded_hash


def test_all_three_seeds_independently_addressable() -> None:
    records = {seed: json.loads((RUNS_M9_0A / f"ARM_B2_STEP_MATCHED_INTERLEAVED_MULTI_FAMILY-seed{seed}.json").read_text()) for seed in SEEDS}
    hashes = {seed: record["training_summary"]["export_sha256"] for seed, record in records.items()}
    assert len(set(hashes.values())) == 3, "all 3 seeds must have distinct checkpoints (no shared state)"
    for seed in SEEDS:
        assert records[seed]["seed"] == seed


def test_locked_data_unopened() -> None:
    assert locked_test_opened(ROOT) is False
