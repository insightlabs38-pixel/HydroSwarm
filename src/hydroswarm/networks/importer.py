"""Fail-closed import of immutable EPANET INP network versions."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydroswarm.simulation import HydraulicSimulator
from hydroswarm.storage import ScenarioStore


MAX_INP_BYTES = 5 * 1024 * 1024
_REQUIRED_SECTIONS = {"JUNCTIONS", "PIPES"}
_SOURCE_SECTIONS = {"RESERVOIRS", "TANKS"}
_UNSAFE_SECTIONS = {"FILES"}


class NetworkImportError(ValueError):
    """A safe, operator-facing import rejection without local path details."""


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:48] or "network"


class NetworkImporter:
    def __init__(
        self,
        store: ScenarioStore,
        storage_directory: str | Path,
        *,
        max_bytes: int = MAX_INP_BYTES,
        run_hydraulic_validation: bool = True,
    ) -> None:
        self.store = store
        self.storage_directory = Path(storage_directory).expanduser().resolve()
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.run_hydraulic_validation = run_hydraulic_validation

    def import_bytes(self, filename: str, content: bytes) -> Any:
        safe_name = self._validate_filename(filename)
        self._validate_content(content)
        sha256 = hashlib.sha256(content).hexdigest()
        existing = self.store.network_by_hash(sha256)
        if existing is not None:
            return existing

        name = _slug(Path(safe_name).stem)
        version = self.store.next_network_version(name)
        network_id = f"{name}-v{version}"
        target = (self.storage_directory / f"{sha256}.inp").resolve()
        if target.parent != self.storage_directory:
            raise NetworkImportError("invalid network storage target")
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(target)
            simulator = HydraulicSimulator(target)
            issues = simulator.validate()
            if issues:
                raise NetworkImportError("network failed validation: " + ", ".join(issues))
            if self.run_hydraulic_validation:
                simulator.calculate_state(at_time=0)
            model = simulator.network
            metadata = self._metadata(model)
            geojson = self._geojson(model)
        except NetworkImportError:
            target.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            raise NetworkImportError(
                f"network could not be parsed or simulated ({type(exc).__name__})"
            ) from None

        from hydroswarm.api.state import NetworkRecord

        record = NetworkRecord(
            network_id=network_id,
            name=name,
            version=version,
            sha256=sha256,
            node_count=len(model.node_name_list),
            link_count=len(model.link_name_list),
            valid=True,
            validated_at=datetime.now(UTC),
            metadata=metadata,
            geojson=geojson,
            validation_errors=(),
        )
        self.store.save_network(record, inp_path=str(target))
        return record

    @staticmethod
    def _validate_filename(filename: str) -> str:
        if not filename or len(filename) > 160:
            raise NetworkImportError("filename is missing or too long")
        if filename != Path(filename).name or "/" in filename or "\\" in filename:
            raise NetworkImportError("filename must not contain a path")
        if Path(filename).suffix.lower() != ".inp":
            raise NetworkImportError("only .inp network files are accepted")
        return filename

    def _validate_content(self, content: bytes) -> str:
        if not content:
            raise NetworkImportError("network file is empty")
        if len(content) > self.max_bytes:
            raise NetworkImportError(f"network file exceeds the {self.max_bytes}-byte limit")
        if b"\x00" in content:
            raise NetworkImportError("network file contains binary data")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise NetworkImportError("network file must be UTF-8 text") from None
        sections = {
            match.group(1).upper()
            for match in re.finditer(r"(?m)^\s*\[([A-Za-z]+)\]\s*(?:;.*)?$", text)
        }
        missing = _REQUIRED_SECTIONS - sections
        if missing or not (_SOURCE_SECTIONS & sections):
            raise NetworkImportError("network file is missing required hydraulic sections")
        if sections & _UNSAFE_SECTIONS:
            raise NetworkImportError("external file references are not allowed")
        if len(text.splitlines()) > 200_000:
            raise NetworkImportError("network file contains too many lines")
        return text

    @staticmethod
    def _metadata(model: Any) -> dict[str, Any]:
        nodes = []
        for name in sorted(model.node_name_list):
            node = model.get_node(name)
            coordinates = tuple(float(value) for value in getattr(node, "coordinates", (0.0, 0.0)))
            nodes.append(
                {
                    "node_id": name,
                    "node_type": type(node).__name__.lower(),
                    "elevation_m": float(getattr(node, "elevation", 0.0)),
                    "coordinates": coordinates,
                }
            )
        links = [
            {
                "link_id": name,
                "link_type": type(model.get_link(name)).__name__.lower(),
                "start_node": model.get_link(name).start_node_name,
                "end_node": model.get_link(name).end_node_name,
            }
            for name in sorted(model.link_name_list)
        ]
        return {
            "node_ids": sorted(model.node_name_list),
            "link_ids": sorted(model.link_name_list),
            "reservoir_ids": sorted(model.reservoir_name_list),
            "tank_ids": sorted(model.tank_name_list),
            "junction_ids": sorted(model.junction_name_list),
            "nodes": nodes,
            "links": links,
        }

    @staticmethod
    def _geojson(model: Any) -> dict[str, Any]:
        features: list[dict[str, Any]] = []
        for name in sorted(model.node_name_list):
            node = model.get_node(name)
            coordinates = [float(value) for value in getattr(node, "coordinates", (0.0, 0.0))]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coordinates},
                    "properties": {"id": name, "kind": type(node).__name__.lower()},
                }
            )
        for name in sorted(model.link_name_list):
            link = model.get_link(name)
            start = model.get_node(link.start_node_name)
            end = model.get_node(link.end_node_name)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(value) for value in getattr(start, "coordinates", (0.0, 0.0))],
                            [float(value) for value in getattr(end, "coordinates", (0.0, 0.0))],
                        ],
                    },
                    "properties": {"id": name, "kind": type(link).__name__.lower()},
                }
            )
        return {"type": "FeatureCollection", "features": features}
