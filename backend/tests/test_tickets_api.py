"""
Tests for the Support Ticket API (/tickets).

Uses the isolated temporary SQLite database configured in conftest.py,
so the dev orders.db is never touched.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api import app as fastapi_app
from app.db import init_db
from app.models_db import SupportTicket

client = TestClient(fastapi_app)

NOW = datetime.now(UTC)


def _seed():
    init_db()
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        if db.query(SupportTicket).count() > 0:
            return
        rows = [
            SupportTicket(
                ticket_id="TKT-TEST0001",
                customer_name="Jordan Lee",
                customer_email="jordan@example.com",
                subject="Damaged item arrived",
                description="The box was crushed and item broken.",
                status="open",
                priority="high",
                order_id="ORD-1001",
                created_at=NOW - timedelta(hours=1),
            ),
            SupportTicket(
                ticket_id="TKT-TEST0002",
                customer_name="Priya Sharma",
                customer_email="priya@example.com",
                subject="Refund not received",
                description="Refund approved but money not credited yet.",
                status="in_progress",
                priority="urgent",
                order_id="ORD-1002",
                created_at=NOW - timedelta(hours=2),
            ),
            SupportTicket(
                ticket_id="TKT-TEST0003",
                customer_name="Marcus Webb",
                customer_email="marcus@example.com",
                subject="Warranty question",
                description="Is my product still under warranty?",
                status="resolved",
                priority="low",
                order_id=None,
                created_at=NOW - timedelta(hours=3),
            ),
        ]
        db.add_all(rows)
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def seeded_db():
    _seed()


def test_list_tickets_newest_first():
    res = client.get("/tickets")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 3
    ids = [t["ticket_id"] for t in body["tickets"]]
    assert ids == ["TKT-TEST0001", "TKT-TEST0002", "TKT-TEST0003"]
    assert set(body["tickets"][0].keys()) >= {
        "ticket_id", "customer_name", "subject", "status", "priority", "created_at",
    }


def test_list_filters_by_status_and_priority():
    res = client.get("/tickets?status=open")
    assert res.status_code == 200
    assert [t["ticket_id"] for t in res.json()["tickets"]] == ["TKT-TEST0001"]

    res = client.get("/tickets?priority=urgent")
    assert res.status_code == 200
    assert [t["ticket_id"] for t in res.json()["tickets"]] == ["TKT-TEST0002"]


def test_list_filters_by_email_order_and_search():
    res = client.get("/tickets?email=priya@example.com")
    assert res.status_code == 200
    assert res.json()["total"] == 1

    res = client.get("/tickets?order_id=ORD-1001")
    assert res.status_code == 200
    assert res.json()["tickets"][0]["ticket_id"] == "TKT-TEST0001"

    res = client.get("/tickets?q=warranty")
    assert res.status_code == 200
    assert [t["ticket_id"] for t in res.json()["tickets"]] == ["TKT-TEST0003"]


def test_list_pagination():
    res = client.get("/tickets?limit=2&offset=1")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 3
    assert len(body["tickets"]) == 2
    assert [t["ticket_id"] for t in body["tickets"]] == ["TKT-TEST0002", "TKT-TEST0003"]


def test_list_rejects_invalid_filters():
    assert client.get("/tickets?status=bogus").status_code == 422
    assert client.get("/tickets?priority=bogus").status_code == 422


def test_stats_aggregation():
    res = client.get("/tickets/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 3
    assert body["by_status"] == {"open": 1, "in_progress": 1, "resolved": 1}
    assert body["by_priority"] == {"high": 1, "urgent": 1, "low": 1}


def test_get_single_ticket_and_404():
    res = client.get("/tickets/TKT-TEST0001")
    assert res.status_code == 200
    assert res.json()["customer_name"] == "Jordan Lee"

    assert client.get("/tickets/TKT-MISSING").status_code == 404


def test_patch_updates_status_and_priority():
    res = client.patch(
        "/tickets/TKT-TEST0003",
        json={"status": "closed", "priority": "medium"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "closed"
    assert body["priority"] == "medium"

    # Reflected in list filters
    res = client.get("/tickets?status=closed")
    assert [t["ticket_id"] for t in res.json()["tickets"]] == ["TKT-TEST0003"]


def test_patch_validation_errors():
    assert (
        client.patch("/tickets/TKT-TEST0001", json={"status": "bogus"}).status_code
        == 422
    )
    assert (
        client.patch("/tickets/TKT-TEST0001", json={}).status_code == 422
    )
    assert (
        client.patch("/tickets/TKT-MISSING", json={"status": "open"}).status_code
        == 404
    )
