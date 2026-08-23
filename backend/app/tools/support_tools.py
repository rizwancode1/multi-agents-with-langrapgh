"""
Support Ticket Tools
LangChain @tool decorated functions for creating and retrieving support tickets.
"""

import uuid

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models_db import SupportTicket


def _get_db() -> Session:
    return SessionLocal()


def _generate_ticket_id() -> str:
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"


def ticket_to_dict(ticket: SupportTicket) -> dict:
    return {
        "ticket_id": ticket.ticket_id,
        "customer_name": ticket.customer_name,
        "customer_email": ticket.customer_email,
        "subject": ticket.subject,
        "description": ticket.description,
        "status": ticket.status,
        "priority": ticket.priority,
        "order_id": ticket.order_id,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
    }


@tool
def create_support_ticket(customer_name: str, customer_email: str, subject: str, description: str, priority: str = "medium", order_id: str | None = None) -> str:
    """
    Create a new support ticket in the database.

    Args:
        customer_name: Full name of the customer
        customer_email: Customer email address
        subject: Short summary of the issue
        description: Detailed description of the complaint or issue
        priority: Ticket priority (low, medium, high, urgent). Defaults to medium.
        order_id: Optional related order ID (e.g., ORD-1001)

    Returns:
        JSON string with ticket_id, status, and creation timestamp
    """
    db = _get_db()
    try:
        ticket = SupportTicket(
            ticket_id=_generate_ticket_id(),
            customer_name=customer_name,
            customer_email=customer_email,
            subject=subject,
            description=description,
            priority=priority,
            order_id=order_id,
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        result = ticket_to_dict(ticket)
        result["message"] = f"Support ticket {ticket.ticket_id} created successfully."
        return str(result)
    except Exception as e:
        db.rollback()
        return str({"error": str(e), "message": "Failed to create support ticket."})
    finally:
        db.close()


@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Retrieve the current status and details of a support ticket by ticket ID.

    Args:
        ticket_id: The ticket ID (e.g., TKT-ABC12345)

    Returns:
        JSON string with ticket details including status, priority, and timestamps
    """
    db = _get_db()
    try:
        ticket = db.query(SupportTicket).filter(SupportTicket.ticket_id == ticket_id.upper()).first()
        if not ticket:
            return str({"error": "Ticket not found", "ticket_id": ticket_id})
        return str(ticket_to_dict(ticket))
    finally:
        db.close()


@tool
def list_tickets_by_email(customer_email: str) -> str:
    """
    List all support tickets for a given customer email.

    Args:
        customer_email: The customer's email address

    Returns:
        JSON string with a list of tickets (ticket_id, subject, status, created_at)
    """
    db = _get_db()
    try:
        tickets = db.query(SupportTicket).filter(SupportTicket.customer_email == customer_email).order_by(SupportTicket.created_at.desc()).all()
        result = [ticket_to_dict(t) for t in tickets]
        return str({"tickets": result, "count": len(result)})
    finally:
        db.close()


@tool
def update_ticket_status(ticket_id: str, new_status: str) -> str:
    """
    Update the status of a support ticket.

    Args:
        ticket_id: The ticket ID (e.g., TKT-ABC12345)
        new_status: New status (open, in_progress, resolved, closed, cancelled)

    Returns:
        JSON string confirming the update
    """
    db = _get_db()
    try:
        ticket = db.query(SupportTicket).filter(SupportTicket.ticket_id == ticket_id.upper()).first()
        if not ticket:
            return str({"error": "Ticket not found", "ticket_id": ticket_id})
        ticket.status = new_status
        db.commit()
        db.refresh(ticket)
        return str({"message": f"Ticket {ticket_id} status updated to {new_status}.", "ticket": ticket_to_dict(ticket)})
    except Exception as e:
        db.rollback()
        return str({"error": str(e), "message": "Failed to update ticket status."})
    finally:
        db.close()
