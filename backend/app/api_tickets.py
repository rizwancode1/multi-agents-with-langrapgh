"""
Support Ticket API endpoints.

Exposes the tickets created by the multi-agent system (TKT-...) so the
frontend can display live, model-generated support tickets instead of
static demo data.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_

from app.auth import require_api_key
from app.db import db_session
from app.models_db import SupportTicket
from app.monitoring import get_logger

logger = get_logger("api_tickets")

router = APIRouter(prefix="/tickets", tags=["tickets"])

ALLOWED_STATUSES = {"open", "in_progress", "resolved", "closed", "cancelled"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "urgent"}


# === Response / request models ===

class TicketResponse(BaseModel):
    """A single support ticket record."""

    ticket_id: str
    customer_name: str
    customer_email: str
    subject: str
    description: str
    status: str
    priority: str
    order_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class TicketListResponse(BaseModel):
    """Paginated list of support tickets."""

    tickets: list[TicketResponse]
    total: int
    limit: int
    offset: int


class TicketStatsResponse(BaseModel):
    """Dashboard summary counts."""

    total: int
    by_status: dict[str, int]
    by_priority: dict[str, int]


class UpdateTicketRequest(BaseModel):
    """Partial ticket update (human agent workflow)."""

    status: str | None = Field(
        default=None,
        description=f"New status. One of: {', '.join(sorted(ALLOWED_STATUSES))}",
    )
    priority: str | None = Field(
        default=None,
        description=f"New priority. One of: {', '.join(sorted(ALLOWED_PRIORITIES))}",
    )


# === Helpers ===

def _ticket_to_response(ticket: SupportTicket) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        customer_name=ticket.customer_name,
        customer_email=ticket.customer_email,
        subject=ticket.subject,
        description=ticket.description,
        status=ticket.status,
        priority=ticket.priority,
        order_id=ticket.order_id,
        created_at=ticket.created_at.isoformat() if ticket.created_at else None,
        updated_at=ticket.updated_at.isoformat() if ticket.updated_at else None,
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _get_ticket_or_404(db, ticket_id: str) -> SupportTicket:
    ticket = (
        db.query(SupportTicket)
        .filter(SupportTicket.ticket_id == ticket_id.strip().upper())
        .first()
    )
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket


# === Endpoints ===
# NOTE: /stats must be declared before /{ticket_id} so it is not captured as an ID.

@router.get("", response_model=TicketListResponse)
def list_tickets(
    status: str | None = Query(default=None, description="Filter by status"),
    priority: str | None = Query(default=None, description="Filter by priority"),
    email: str | None = Query(default=None, description="Filter by customer email"),
    order_id: str | None = Query(default=None, description="Filter by related order ID"),
    q: str | None = Query(default=None, max_length=200, description="Search subject/description"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List support tickets, newest first, with optional filters and pagination."""
    if status and status.lower() not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Allowed: {', '.join(sorted(ALLOWED_STATUSES))}",
        )
    if priority and priority.lower() not in ALLOWED_PRIORITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid priority. Allowed: {', '.join(sorted(ALLOWED_PRIORITIES))}",
        )

    with db_session() as db:
        query = db.query(SupportTicket)

        if status:
            query = query.filter(SupportTicket.status == status.lower())
        if priority:
            query = query.filter(SupportTicket.priority == priority.lower())
        if email:
            query = query.filter(SupportTicket.customer_email == email.strip().lower())
        if order_id:
            query = query.filter(SupportTicket.order_id == order_id.strip().upper())
        if q:
            pattern = f"%{q.strip()}%"
            query = query.filter(
                or_(
                    SupportTicket.subject.ilike(pattern),
                    SupportTicket.description.ilike(pattern),
                )
            )

        total = query.count()
        tickets = (
            query.order_by(SupportTicket.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return TicketListResponse(
            tickets=[_ticket_to_response(t) for t in tickets],
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/stats", response_model=TicketStatsResponse)
def ticket_stats():
    """Aggregate counts by status and priority for dashboard cards."""
    with db_session() as db:
        total = db.query(func.count(SupportTicket.id)).scalar() or 0

        by_status_rows = (
            db.query(SupportTicket.status, func.count(SupportTicket.id))
            .group_by(SupportTicket.status)
            .all()
        )
        by_priority_rows = (
            db.query(SupportTicket.priority, func.count(SupportTicket.id))
            .group_by(SupportTicket.priority)
            .all()
        )

        return TicketStatsResponse(
            total=total,
            by_status={row[0]: row[1] for row in by_status_rows},
            by_priority={row[0]: row[1] for row in by_priority_rows},
        )


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: str):
    """Retrieve a single support ticket by ID."""
    with db_session() as db:
        ticket = _get_ticket_or_404(db, ticket_id)
        return _ticket_to_response(ticket)


@router.patch("/{ticket_id}", response_model=TicketResponse)
def update_ticket(ticket_id: str, body: UpdateTicketRequest, _auth: None = Depends(require_api_key)):
    """Update a ticket's status and/or priority."""
    if body.status and body.status.lower() not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Allowed: {', '.join(sorted(ALLOWED_STATUSES))}",
        )
    if body.priority and body.priority.lower() not in ALLOWED_PRIORITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid priority. Allowed: {', '.join(sorted(ALLOWED_PRIORITIES))}",
        )
    if body.status is None and body.priority is None:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one field to update: status or priority",
        )

    with db_session() as db:
        ticket = _get_ticket_or_404(db, ticket_id)
        if body.status:
            ticket.status = body.status.lower()
        if body.priority:
            ticket.priority = body.priority.lower()
        db.commit()
        db.refresh(ticket)

        logger.info("ticket_updated", extra={"extra_data": {
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "priority": ticket.priority,
        }})
        return _ticket_to_response(ticket)
