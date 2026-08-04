"""Per-topology signature registry (overnight-plan.txt Task 1.2).

The existing SignatureCache already keys every stored artifact by a full
SignatureCacheKey digest that includes network_hash and
hydraulic_state_hash, so two different topologies or hydraulic states can
never collide in storage or be loaded as each other's signatures -- that
part of Task 1.2 was already correct.

What was missing is the registry layer the plan asks for::

    SignatureRegistry
    └── topology_hash
        └── hydraulic_regime_hash
            └── SignatureLibrary

i.e. a way to look up "the signatures for this topology and hydraulic
regime" without already knowing the full SignatureCacheKey (which also
needs simulator_version, configuration_hash, and sensor_layout_hash), plus
the governance rule that signature fitting may only be registered from
training-split data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydroswarm.classical.signatures import SignatureArtifact, SignatureCache, SignatureCacheKey

SCHEMA_VERSION = 1


class SignatureRegistryError(Exception):
    """Raised for signature-registry governance violations or clear misses."""


@dataclass(frozen=True, slots=True)
class SignatureRegistryEntry:
    topology_hash: str
    hydraulic_regime_hash: str
    cache_key_digest: str
    artifact_hash: str
    source_nodes: tuple[str, ...]
    sensor_nodes: tuple[str, ...]
    fit_split: str
    generator_version: str
    registered_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "topology_hash": self.topology_hash,
            "hydraulic_regime_hash": self.hydraulic_regime_hash,
            "cache_key_digest": self.cache_key_digest,
            "artifact_hash": self.artifact_hash,
            "source_nodes": list(self.source_nodes),
            "sensor_nodes": list(self.sensor_nodes),
            "fit_split": self.fit_split,
            "generator_version": self.generator_version,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SignatureRegistryEntry":
        return cls(
            topology_hash=payload["topology_hash"],
            hydraulic_regime_hash=payload["hydraulic_regime_hash"],
            cache_key_digest=payload["cache_key_digest"],
            artifact_hash=payload["artifact_hash"],
            source_nodes=tuple(payload["source_nodes"]),
            sensor_nodes=tuple(payload["sensor_nodes"]),
            fit_split=payload["fit_split"],
            generator_version=payload["generator_version"],
            registered_at=payload["registered_at"],
        )


class SignatureRegistry:
    """Indexes SignatureCache artifacts by (topology_hash, hydraulic_regime_hash).

    Backed by the same SignatureCache for artifact bytes; this class only
    adds the lightweight, atomically-written pointer index plus the
    train-only-fitting guard.
    """

    def __init__(self, cache: SignatureCache, *, index_path: str | Path | None = None) -> None:
        self.cache = cache
        self.index_path = Path(index_path) if index_path is not None else cache.root / "registry.json"

    def _load_index(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.index_path.exists():
            return {}
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        return payload.get("entries", {})

    def _write_index(self, entries: dict[str, dict[str, dict[str, Any]]]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "entries": entries}
        temporary = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.index_path)

    def register(
        self,
        *,
        topology_hash: str,
        hydraulic_regime_hash: str,
        key: SignatureCacheKey,
        artifact: SignatureArtifact,
        fit_split: str,
        generator_version: str = "hydroswarm-signature-registry-v1",
    ) -> SignatureRegistryEntry:
        """Index an already-built artifact under (topology_hash, regime_hash).

        Raises SignatureRegistryError if fit_split is anything other than
        "train": signature fitting uses training data only, and calibration,
        validation, and test scenarios must never contribute to fitting.
        """

        if fit_split != "train":
            raise SignatureRegistryError(
                f"refusing to register signatures fit from split={fit_split!r}; signature "
                "fitting uses training data only"
            )
        entry = SignatureRegistryEntry(
            topology_hash=topology_hash,
            hydraulic_regime_hash=hydraulic_regime_hash,
            cache_key_digest=key.digest,
            artifact_hash=artifact.artifact_hash,
            source_nodes=tuple(sorted({hypothesis.source_node for hypothesis in artifact.hypotheses})),
            sensor_nodes=artifact.sensor_nodes,
            fit_split=fit_split,
            generator_version=generator_version,
            registered_at=datetime.now(UTC).isoformat(),
        )
        entries = self._load_index()
        entries.setdefault(topology_hash, {})[hydraulic_regime_hash] = entry.to_json()
        self._write_index(entries)
        return entry

    def entry(self, *, topology_hash: str, hydraulic_regime_hash: str) -> SignatureRegistryEntry | None:
        entries = self._load_index()
        raw = entries.get(topology_hash, {}).get(hydraulic_regime_hash)
        return SignatureRegistryEntry.from_json(raw) if raw is not None else None

    def lookup(
        self, *, topology_hash: str, hydraulic_regime_hash: str, key: SignatureCacheKey
    ) -> SignatureArtifact | None:
        """Pure read: returns None (never triggers a simulation) if this
        topology/regime has no registered signatures, or if the caller's
        `key` digest does not match what was registered (a clear signal the
        caller is asking with stale/wrong provenance rather than silently
        returning a mismatched artifact)."""

        entry = self.entry(topology_hash=topology_hash, hydraulic_regime_hash=hydraulic_regime_hash)
        if entry is None or entry.cache_key_digest != key.digest:
            return None
        return self.cache.load(key)

    def require(
        self, *, topology_hash: str, hydraulic_regime_hash: str, key: SignatureCacheKey
    ) -> SignatureArtifact:
        """Like lookup(), but fails closed with a clear, actionable error
        instead of returning None -- for callers (e.g. runtime inference)
        that must not silently proceed without signatures."""

        artifact = self.lookup(topology_hash=topology_hash, hydraulic_regime_hash=hydraulic_regime_hash, key=key)
        if artifact is None:
            raise SignatureRegistryError(
                f"no registered signatures for topology_hash={topology_hash!r} "
                f"hydraulic_regime_hash={hydraulic_regime_hash!r}; refusing to proceed "
                "without an explicit exact-simulation fallback"
            )
        return artifact

    def topology_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._load_index()))
