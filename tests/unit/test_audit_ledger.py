import sqlite3
from uuid import uuid4

import pytest

from hydroswarm.storage.ledger import AuditLedger, GENESIS_HASH


def test_ledger_is_sequenced_hash_chained_and_replayable(tmp_path) -> None:
    ledger = AuditLedger(tmp_path / "audit.db")
    incident_id = uuid4()
    first = ledger.append(
        incident_id=incident_id,
        event_type="INCIDENT_CREATED",
        actor="operator",
        input_state_hash="a" * 64,
        payload={"network_id": "demo"},
        model_version="none",
        simulator_version="wntr-1.5",
    )
    second = ledger.append(
        incident_id=incident_id,
        event_type="PLAN_REJECTED",
        actor="hydroverifier",
        input_state_hash=first.event_hash,
        payload={"reason": "MINIMUM_PRESSURE"},
        model_version="hydrocore-0.1",
        simulator_version="wntr-1.5",
    )

    assert first.previous_event_hash == GENESIS_HASH
    assert second.previous_event_hash == first.event_hash
    assert [event.sequence for event in ledger.events(incident_id)] == [1, 2]
    assert ledger.verify_chain(incident_id)


def test_sqlite_triggers_reject_update_and_delete(tmp_path) -> None:
    path = tmp_path / "audit.db"
    ledger = AuditLedger(path)
    event = ledger.append(
        incident_id=uuid4(),
        event_type="OPERATOR_APPROVAL_REQUIRED",
        actor="system",
        input_state_hash="b" * 64,
        payload={"approved": False},
        model_version="hydrocore-0.1",
        simulator_version="wntr-1.5",
    )

    with sqlite3.connect(path) as connection, pytest.raises(
        sqlite3.IntegrityError, match="append-only"
    ):
        connection.execute(
            "UPDATE audit_events SET actor = 'attacker' WHERE event_id = ?",
            (str(event.event_id),),
        )

    with sqlite3.connect(path) as connection, pytest.raises(
        sqlite3.IntegrityError, match="append-only"
    ):
        connection.execute("DELETE FROM audit_events WHERE event_id = ?", (str(event.event_id),))
