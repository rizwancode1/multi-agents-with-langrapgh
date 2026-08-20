"""
Refund Tools
LangChain @tool decorated functions for creating and retrieving refund requests.
"""

import uuid
from typing import Optional
from sqlalchemy.orm import Session
from langchain_core.tools import tool

from app.models_db import RefundRequest, Order, OrderItem
from app.db import SessionLocal


def _get_db() -> Session:
    return SessionLocal()


def _generate_refund_id() -> str:
    return f"REF-{uuid.uuid4().hex[:8].upper()}"


def refund_to_dict(refund: RefundRequest) -> dict:
    return {
        "refund_id": refund.refund_id,
        "order_id": refund.order_id,
        "reason": refund.reason,
        "status": refund.status,
        "requested_amount": refund.requested_amount,
        "approved_amount": refund.approved_amount,
        "refund_method": refund.refund_method,
        "requested_at": refund.requested_at.isoformat() if refund.requested_at else None,
        "processed_at": refund.processed_at.isoformat() if refund.processed_at else None,
        "notes": refund.notes,
    }


@tool
def create_refund_request(order_id: str, reason: str, requested_amount: float, refund_method: Optional[str] = None, notes: Optional[str] = None) -> str:
    """
    Create a new refund request in the database.

    Args:
        order_id: The order ID (e.g., ORD-1001)
        reason: Reason for the refund request
        requested_amount: Amount to be refunded
        refund_method: Optional refund method (original_payment, store_credit, etc.)
        notes: Optional additional notes

    Returns:
        JSON string with refund_id, status, and creation timestamp
    """
    db = _get_db()
    try:
        order = db.query(Order).filter(Order.order_id == order_id.upper()).first()
        if not order:
            return str({"error": "Order not found", "order_id": order_id})

        refund = RefundRequest(
            refund_id=_generate_refund_id(),
            order_id=order.order_id,
            reason=reason,
            requested_amount=requested_amount,
            refund_method=refund_method,
            notes=notes,
        )
        db.add(refund)
        db.commit()
        db.refresh(refund)
        result = refund_to_dict(refund)
        result["message"] = f"Refund request {refund.refund_id} created successfully for order {order_id}."
        return str(result)
    except Exception as e:
        db.rollback()
        return str({"error": str(e), "message": "Failed to create refund request."})
    finally:
        db.close()


@tool
def get_refund_status(refund_id: str) -> str:
    """
    Retrieve the current status and details of a refund request by refund ID.

    Args:
        refund_id: The refund request ID (e.g., REF-ABC12345)

    Returns:
        JSON string with refund details including status, amounts, and timestamps
    """
    db = _get_db()
    try:
        refund = db.query(RefundRequest).filter(RefundRequest.refund_id == refund_id.upper()).first()
        if not refund:
            return str({"error": "Refund request not found", "refund_id": refund_id})
        return str(refund_to_dict(refund))
    finally:
        db.close()


@tool
def list_refunds_by_order(order_id: str) -> str:
    """
    List all refund requests for a given order ID.

    Args:
        order_id: The order ID (e.g., ORD-1001)

    Returns:
        JSON string with a list of refund requests (refund_id, status, requested_amount, requested_at)
    """
    db = _get_db()
    try:
        refunds = db.query(RefundRequest).filter(RefundRequest.order_id == order_id.upper()).order_by(RefundRequest.requested_at.desc()).all()
        result = [refund_to_dict(r) for r in refunds]
        return str({"refunds": result, "count": len(result)})
    finally:
        db.close()


@tool
def calculate_eligible_refund(order_id: str) -> str:
    """
    Calculate the eligible refund amount for an order based on order total and items.
    This is a utility tool to help the agent determine refund amounts before creating a request.

    Args:
        order_id: The order ID (e.g., ORD-1001)

    Returns:
        JSON string with order total, eligible refund amount, and item breakdown
    """
    db = _get_db()
    try:
        order = db.query(Order).filter(Order.order_id == order_id.upper()).first()
        if not order:
            return str({"error": "Order not found", "order_id": order_id})

        items = db.query(OrderItem).filter(OrderItem.order_id == order.order_id).all()
        item_details = [
            {
                "product_id": item.product_id,
                "name": item.name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "line_total": round(item.quantity * item.unit_price, 2),
            }
            for item in items
        ]

        eligible_amount = round(order.total, 2)
        return str({
            "order_id": order.order_id,
            "order_total": round(order.total, 2),
            "eligible_refund_amount": eligible_amount,
            "items": item_details,
            "message": f"Order {order_id} total is ${eligible_amount}. This is the maximum eligible refund amount if the return is approved.",
        })
    finally:
        db.close()
