"""Serialization boundary for durable networks and incident scenarios."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from hydroswarm.domain import IncidentCreate, IncidentState, OperationalPlan, PlanVerification
from hydroswarm.storage.database import Database
from hydroswarm.storage.migrations import migrate

if TYPE_CHECKING:
    from hydroswarm.api.state import ApprovalReceipt, IncidentRuntime, NetworkRecord


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class ScenarioStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        migrate(database)

    def save_network(self, record: NetworkRecord, *, inp_path: str | None) -> None:
        with self.database.connect(write=True) as connection:
            connection.execute(
                """INSERT INTO networks (
                    network_id, name, version, sha256, inp_path, node_count, link_count,
                    valid, validated_at, metadata_json, geojson_json, validation_errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(network_id) DO UPDATE SET
                    node_count=excluded.node_count, link_count=excluded.link_count,
                    valid=excluded.valid, validated_at=excluded.validated_at,
                    metadata_json=excluded.metadata_json, geojson_json=excluded.geojson_json,
                    validation_errors_json=excluded.validation_errors_json""",
                (
                    record.network_id,
                    record.name,
                    record.version,
                    record.sha256,
                    inp_path,
                    record.node_count,
                    record.link_count,
                    int(record.valid),
                    record.validated_at.isoformat(),
                    _json(record.metadata),
                    _json(record.geojson),
                    _json(record.validation_errors),
                ),
            )

    def network_by_hash(self, sha256: str) -> NetworkRecord | None:
        from hydroswarm.api.state import NetworkRecord

        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM networks WHERE sha256 = ?", (sha256,)).fetchone()
        return self._network_record(row, NetworkRecord) if row else None

    def network_path(self, network_id: str) -> str | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT inp_path FROM networks WHERE network_id = ?", (network_id,)
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def next_network_version(self, name: str) -> int:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM networks WHERE name = ?", (name,)
            ).fetchone()
        return int(row[0])

    def load_networks(self) -> dict[str, NetworkRecord]:
        from hydroswarm.api.state import NetworkRecord

        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM networks ORDER BY name, version").fetchall()
        return {row["network_id"]: self._network_record(row, NetworkRecord) for row in rows}

    @staticmethod
    def _network_record(row: Any, record_type: Any) -> Any:
        return record_type(
            network_id=row["network_id"],
            name=row["name"],
            version=row["version"],
            sha256=row["sha256"],
            node_count=row["node_count"],
            link_count=row["link_count"],
            valid=bool(row["valid"]),
            validated_at=row["validated_at"],
            metadata=json.loads(row["metadata_json"]),
            geojson=json.loads(row["geojson_json"]),
            validation_errors=tuple(json.loads(row["validation_errors_json"])),
        )

    def save_incident(self, runtime: IncidentRuntime) -> None:
        now = datetime.now(UTC).isoformat()
        incident_id = str(runtime.state.incident_id)
        with self.database.connect(write=True) as connection:
            connection.execute(
                """INSERT INTO incidents(incident_id, network_id, create_json, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET state_json=excluded.state_json,
                    network_id=excluded.network_id, updated_at=excluded.updated_at""",
                (incident_id, runtime.state.network_id, _json(runtime.create), _json(runtime.state), now),
            )
            connection.execute("DELETE FROM observations WHERE incident_id = ?", (incident_id,))
            connection.executemany(
                "INSERT INTO observations(incident_id, ordinal, observation_json) VALUES (?, ?, ?)",
                [(incident_id, index, _json(item)) for index, item in enumerate(runtime.state.observations)],
            )
            if runtime.state.candidates is not None:
                connection.execute(
                    """INSERT INTO posteriors(incident_id, candidate_json, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(incident_id) DO UPDATE SET candidate_json=excluded.candidate_json,
                        updated_at=excluded.updated_at""",
                    (incident_id, _json(runtime.state.candidates), now),
                )
            for plan in runtime.plans.values():
                connection.execute(
                    "INSERT OR REPLACE INTO plans(plan_id, incident_id, plan_json, created_at) VALUES (?, ?, ?, ?)",
                    (str(plan.plan_id), incident_id, _json(plan), plan.created_at.isoformat()),
                )
            for verification in runtime.verifications.values():
                connection.execute(
                    """INSERT OR REPLACE INTO verifications(
                        plan_id, incident_id, verification_json, verified_at
                    ) VALUES (?, ?, ?, ?)""",
                    (
                        str(verification.plan_id),
                        incident_id,
                        _json(verification),
                        verification.verified_at.isoformat(),
                    ),
                )

    def save_approval(self, receipt: ApprovalReceipt) -> None:
        with self.database.connect(write=True) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO approvals(
                    incident_id, plan_id, operator_id, approved_at, receipt_json
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    str(receipt.incident_id),
                    str(receipt.plan_id),
                    receipt.operator_id,
                    receipt.approved_at.isoformat(),
                    _json(receipt),
                ),
            )

    def load_incidents(self) -> dict[UUID, IncidentRuntime]:
        from hydroswarm.api.state import IncidentRuntime

        with self.database.connect() as connection:
            incidents = connection.execute("SELECT * FROM incidents ORDER BY updated_at").fetchall()
            plans = connection.execute("SELECT * FROM plans ORDER BY created_at").fetchall()
            verifications = connection.execute("SELECT * FROM verifications ORDER BY verified_at").fetchall()
        plan_map: dict[str, dict[UUID, OperationalPlan]] = {}
        for row in plans:
            plan = OperationalPlan.model_validate_json(row["plan_json"])
            plan_map.setdefault(row["incident_id"], {})[plan.plan_id] = plan
        verification_map: dict[str, dict[UUID, PlanVerification]] = {}
        for row in verifications:
            item = PlanVerification.model_validate_json(row["verification_json"])
            verification_map.setdefault(row["incident_id"], {})[item.plan_id] = item
        result: dict[UUID, IncidentRuntime] = {}
        for row in incidents:
            create = IncidentCreate.model_validate_json(row["create_json"])
            state = IncidentState.model_validate_json(row["state_json"])
            result[state.incident_id] = IncidentRuntime(
                create=create,
                state=state,
                plans=plan_map.get(row["incident_id"], {}),
                verifications=verification_map.get(row["incident_id"], {}),
            )
        return result

    def table_counts(self) -> dict[str, int]:
        tables = ("networks", "incidents", "observations", "posteriors", "plans", "verifications", "approvals")
        with self.database.connect() as connection:
            return {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}

