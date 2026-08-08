"""Governed runtime normalization bundle for V4 inference (core-issues5.txt
Section 3, P0 blocker).

Stage-F training built `data/learning-v2/cycle-b2-joint-v4` by merging
`data/learning-v2/cycle-b2/tensors-normalized` -- node/edge features that
`scripts/rebuild_normalized_shards.py` transformed with the exact
train-split-fit `NormalizationStats` committed at
`data/learning-v2/cycle-b2/normalization/{node,edge}-normalization.json`
(see that script's own module docstring: it applies
`NormalizationStats.transform()`, "the exact same ... call
HydraulicFeatureBuilder.build() applies at live-inference time"). Live V4
serving must apply that SAME artifact to raw runtime features -- treating
"training tensors were already normalized" as "serving needs no
normalization" would silently retrain a different effective feature
distribution than the one actually used to fit the model's weights.

`scripts/build_phase15_v4_checkpoint.py` previously recorded
`normalization_hash="none"` for exactly this reason (a real defect, not a
documented design choice -- its own comment said "pre-normalized at
corpus-build time" as if that made a runtime artifact unnecessary). This
module is the fix's runtime half: given a directory containing the real
committed node/edge normalization artifacts (with their `.sha256`
sidecars -- the same convention `NormalizationStats.save()` already
writes), load and integrity-verify them, and expose a fingerprint
comparable against `CheckpointIdentity.normalization_hash`
(`HydraulicFeatureBuilder.normalization_fingerprint`'s own formula, so the
two are directly comparable without reimplementing it here).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from hydroswarm.preprocessing.builder import NO_NORMALIZATION_SENTINEL, HydraulicFeatureBuilder
from hydroswarm.preprocessing.schema import DEFAULT_FEATURE_SCHEMA, NormalizationStats

__all__ = [
    "NO_NORMALIZATION_SENTINEL",
    "NormalizationBundleError",
    "RuntimeNormalizationBundle",
    "load_runtime_normalization_bundle",
]

NODE_NORMALIZATION_FILENAME = "node-normalization.json"
EDGE_NORMALIZATION_FILENAME = "edge-normalization.json"


class NormalizationBundleError(Exception):
    """Raised when a runtime normalization bundle is missing, stale,
    corrupted, or schema-incompatible. Callers must fail closed (no neural
    branch on unnormalized inputs) rather than fall back to an
    unnormalized `HydraulicFeatureBuilder` when a checkpoint declares
    normalized training input."""


@dataclass(frozen=True, slots=True)
class RuntimeNormalizationBundle:
    node_normalization: NormalizationStats
    edge_normalization: NormalizationStats
    #: HydraulicFeatureBuilder.normalization_fingerprint's own formula over
    #: this exact (node, edge) pair -- directly comparable against
    #: CheckpointIdentity.normalization_hash.
    fingerprint: str
    source_dir: Path

    def feature_builder(self, **kwargs: object) -> HydraulicFeatureBuilder:
        """Build the live feature builder this bundle governs. Extra
        kwargs (device/dtype) pass through to HydraulicFeatureBuilder."""

        return HydraulicFeatureBuilder(
            node_normalization=self.node_normalization,
            edge_normalization=self.edge_normalization,
            **kwargs,  # type: ignore[arg-type]
        )


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists():
        raise NormalizationBundleError(f"missing normalization artifact: {path}")
    if not sidecar.exists():
        raise NormalizationBundleError(f"missing normalization artifact checksum sidecar: {sidecar}")
    recorded = sidecar.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if recorded != actual:
        raise NormalizationBundleError(
            f"{path} content does not match its committed .sha256 sidecar -- stale or corrupted "
            f"(recorded={recorded}, actual={actual})"
        )


def load_runtime_normalization_bundle(directory: str | Path) -> RuntimeNormalizationBundle:
    """Load and integrity-verify the governed train-owned node/edge
    normalization artifacts for live V4 inference.

    Fails closed (raises `NormalizationBundleError`) when:

    - `directory` does not contain both artifact files and their
      `.sha256` sidecars;
    - a file's content does not match its own committed sidecar (stale or
      corrupted -- the same integrity contract sharded tensor shards use
      elsewhere in this project);
    - either artifact's `schema_version` does not match the feature
      schema this runtime build actually uses.

    Never silently substitutes an unnormalized `HydraulicFeatureBuilder`
    for a caller that needs a real, verified artifact -- that decision
    (whether normalization applies at all for a given checkpoint) belongs
    to the caller, keyed off `CheckpointIdentity.normalization_hash ==
    NO_NORMALIZATION_SENTINEL`, not to this function.
    """

    directory = Path(directory)
    node_path = directory / NODE_NORMALIZATION_FILENAME
    edge_path = directory / EDGE_NORMALIZATION_FILENAME
    _verify_sidecar(node_path)
    _verify_sidecar(edge_path)

    node_stats = NormalizationStats.load(node_path)
    edge_stats = NormalizationStats.load(edge_path)
    for stats, name in ((node_stats, "node"), (edge_stats, "edge")):
        if stats.schema_version != DEFAULT_FEATURE_SCHEMA.version:
            raise NormalizationBundleError(
                f"{name} normalization schema_version {stats.schema_version!r} is incompatible "
                f"with the current feature schema {DEFAULT_FEATURE_SCHEMA.version!r}"
            )

    fingerprint = HydraulicFeatureBuilder(
        node_normalization=node_stats, edge_normalization=edge_stats
    ).normalization_fingerprint
    return RuntimeNormalizationBundle(
        node_normalization=node_stats,
        edge_normalization=edge_stats,
        fingerprint=fingerprint,
        source_dir=directory,
    )
