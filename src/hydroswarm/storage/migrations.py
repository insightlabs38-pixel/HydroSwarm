"""Idempotent schema migrations for scenario persistence."""

from __future__ import annotations

from hydroswarm.storage.database import Database


SCHEMA_VERSION = 1


def migrate(database: Database) -> None:
    with database.connect(write=True) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS networks (
                network_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                sha256 TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64),
                inp_path TEXT,
                node_count INTEGER NOT NULL CHECK (node_count >= 0),
                link_count INTEGER NOT NULL CHECK (link_count >= 0),
                valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
                validated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                geojson_json TEXT NOT NULL,
                validation_errors_json TEXT NOT NULL,
                UNIQUE(name, version)
            );

            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                network_id TEXT NOT NULL REFERENCES networks(network_id),
                create_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                incident_id TEXT NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                observation_json TEXT NOT NULL,
                PRIMARY KEY (incident_id, ordinal)
            );

            CREATE TABLE IF NOT EXISTS posteriors (
                incident_id TEXT PRIMARY KEY REFERENCES incidents(incident_id) ON DELETE CASCADE,
                candidate_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plans (
                plan_id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
                plan_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS verifications (
                plan_id TEXT PRIMARY KEY REFERENCES plans(plan_id) ON DELETE CASCADE,
                incident_id TEXT NOT NULL REFERENCES incidents(incident_id) ON DELETE CASCADE,
                verification_json TEXT NOT NULL,
                verified_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS approvals (
                approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT NOT NULL REFERENCES incidents(incident_id),
                plan_id TEXT NOT NULL REFERENCES plans(plan_id),
                operator_id TEXT NOT NULL,
                approved_at TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                UNIQUE(incident_id, plan_id)
            );

            CREATE INDEX IF NOT EXISTS incidents_network_idx ON incidents(network_id);
            CREATE INDEX IF NOT EXISTS plans_incident_idx ON plans(incident_id);
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,),
        )

