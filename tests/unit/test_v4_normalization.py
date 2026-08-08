"""core-issues5.txt Section 3 (P0 blocker): governed runtime normalization
bundle loading/integrity-verification for V4 inference.

`hydroswarm.runtime.v4_normalization.load_runtime_normalization_bundle`
must load and hash-verify the real, committed train-owned node/edge
NormalizationStats artifacts and expose a fingerprint directly comparable
against CheckpointIdentity.normalization_hash -- failing closed (raising
NormalizationBundleError, never silently substituting an unnormalized
builder) on any missing/stale/corrupted/schema-incompatible artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hydroswarm.preprocessing.builder import HydraulicFeatureBuilder
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats
from hydroswarm.runtime.v4_normalization import (
    NormalizationBundleError,
    load_runtime_normalization_bundle,
)


def _write_bundle(directory: Path, *, schema_version: str = DEFAULT_FEATURE_SCHEMA.version) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    node_stats = NormalizationStats.fit(
        np.random.default_rng(0).normal(size=(20, len(DEFAULT_FEATURE_SCHEMA.node_features))),
        DEFAULT_FEATURE_SCHEMA.node_features,
        schema_version=schema_version,
    )
    edge_stats = NormalizationStats.fit(
        np.random.default_rng(1).normal(size=(10, len(DEFAULT_FEATURE_SCHEMA.edge_features))),
        DEFAULT_FEATURE_SCHEMA.edge_features,
        schema_version=schema_version,
    )
    node_stats.save(directory / "node-normalization.json")
    edge_stats.save(directory / "edge-normalization.json")


def test_valid_bundle_loads_and_fingerprint_matches_feature_builder(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    bundle = load_runtime_normalization_bundle(tmp_path)

    expected = HydraulicFeatureBuilder(
        node_normalization=bundle.node_normalization,
        edge_normalization=bundle.edge_normalization,
    ).normalization_fingerprint
    assert bundle.fingerprint == expected

    # A feature builder built from the bundle must transform identically to
    # one built directly from the same underlying stats -- the whole point
    # of exposing feature_builder() rather than making callers reconstruct
    # HydraulicFeatureBuilder(...) themselves.
    built = bundle.feature_builder()
    assert built.normalization_fingerprint == bundle.fingerprint


def test_missing_directory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(NormalizationBundleError, match="missing normalization artifact"):
        load_runtime_normalization_bundle(tmp_path / "does-not-exist")


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "node-normalization.json.sha256").unlink()
    with pytest.raises(NormalizationBundleError, match="sha256"):
        load_runtime_normalization_bundle(tmp_path)


def test_stale_artifact_content_fails_closed(tmp_path: Path) -> None:
    """The artifact file was modified after its .sha256 sidecar was
    written -- a real staleness/corruption signal, not merely a schema
    mismatch."""

    _write_bundle(tmp_path)
    path = tmp_path / "node-normalization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NormalizationBundleError, match="stale or corrupted"):
        load_runtime_normalization_bundle(tmp_path)


def test_schema_incompatible_artifact_fails_closed(tmp_path: Path) -> None:
    _write_bundle(tmp_path, schema_version="some-older-feature-schema-v0")
    with pytest.raises(NormalizationBundleError, match="schema_version"):
        load_runtime_normalization_bundle(tmp_path)


def test_real_committed_cycle_b2_artifact_loads_and_verifies() -> None:
    """The actual, real, committed artifact Stage F's training corpus was
    built from must load and integrity-verify cleanly -- not just a
    synthetic fixture."""

    repo_root = Path(__file__).resolve().parents[2]
    directory = repo_root / "data" / "learning-v2" / "cycle-b2" / "normalization"
    bundle = load_runtime_normalization_bundle(directory)
    assert bundle.node_normalization.schema_version == DEFAULT_FEATURE_SCHEMA.version
    assert bundle.edge_normalization.schema_version == DEFAULT_FEATURE_SCHEMA.version
    assert len(bundle.fingerprint) == 64
