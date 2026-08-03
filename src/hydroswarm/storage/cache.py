"""Versioned, checksummed JSON cache for deterministic simulation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CACHE_FORMAT_VERSION = "hydroswarm-simulation-v1"


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class SimulationResultCache:
    """Atomic local cache that treats malformed or stale entries as misses."""

    def __init__(self, directory: str | Path, *, version: str = CACHE_FORMAT_VERSION) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.version = version

    def key(self, *, network: Any, state: Any, profile: Any, plan: Any, simulator: Any, config: Any) -> str:
        return canonical_digest(
            {
                "format": self.version,
                "network": network,
                "state": state,
                "profile": profile,
                "plan": plan,
                "simulator": simulator,
                "config": config,
            }
        )

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a SHA-256 hex digest")
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if envelope.get("format") != self.version:
                raise ValueError("cache format mismatch")
            if envelope.get("key") != key:
                raise ValueError("cache key mismatch")
            if envelope.get("payload_sha256") != canonical_digest(payload):
                raise ValueError("cache payload checksum mismatch")
            if not isinstance(payload, dict):
                raise ValueError("cache payload must be an object")
            return payload
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path(key)
        envelope = {
            "format": self.version,
            "key": key,
            "payload_sha256": canonical_digest(payload),
            "payload": payload,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def invalidate(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

