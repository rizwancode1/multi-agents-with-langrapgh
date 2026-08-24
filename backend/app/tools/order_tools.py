"""
Order Tools
LangChain @tool decorated functions for retrieving order information from the database.
"""

import re

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models_db import Order, OrderItem


def _get_db() -> Session:
    return SessionLocal()


def order_to_dict(order: Order) -> dict:
    return {
        "order_id": order.order_id,
        "customer_name": order.customer_name,
        "customer_email": order.customer_email,
        "customer_phone": order.customer_phone,
        "order_date": order.order_date.isoformat() if order.order_date else None,
        "status": order.status,
        "items": [
            {
                "product_id": item.product_id,
                "name": item.name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
            }
            for item in order.items
        ],
        "shipping_address": order.shipping_address,
        "payment": {
            "method": order.payment_method,
            "transaction_id": order.transaction_id,
            "subtotal": order.subtotal,
            "tax": order.tax,
            "shipping_fee": order.shipping_fee,
            "total": order.total,
        },
    }


@tool
def get_order_by_id(order_id: str) -> str:
    """
    Retrieve order details by exact order ID.

    Args:
        order_id: The order ID (e.g., ORD-1001)

    Returns:
        JSON string with order details including items, status, and payment info
    """
    db = _get_db()
    try:
        order = db.query(Order).filter(Order.order_id == order_id.upper()).first()
        if not order:
            return str({"error": "Order not found", "order_id": order_id})
        return str(order_to_dict(order))
    finally:
        db.close()


@tool
def get_order_by_customer_name(customer_name: str) -> str:
    """
    Retrieve the most recent order for a customer by name (case-insensitive partial match).

    Args:
        customer_name: Full or partial customer name (e.g., "Sarah Jenkins")

    Returns:
        JSON string with the most recent order details
    """
    db = _get_db()
    try:
        order = (
            db.query(Order)
            .filter(Order.customer_name.ilike(f"%{customer_name}%"))
            .order_by(Order.order_date.desc())
            .first()
        )
        if not order:
            return str({"error": "No orders found for customer", "customer_name": customer_name})
        return str(order_to_dict(order))
    finally:
        db.close()


@tool
def get_orders_by_email(customer_email: str) -> str:
    """
    Retrieve all orders for a customer by email address (case-insensitive exact match).

    Args:
        customer_email: The customer's email address (e.g., jtaylor@example.com)

    Returns:
        JSON string with a list of matching orders
    """
    db = _get_db()
    try:
        orders = (
            db.query(Order)
            .filter(Order.customer_email.ilike(customer_email.lower()))
            .order_by(Order.order_date.desc())
            .all()
        )
        if not orders:
            return str({"error": "No orders found for email", "customer_email": customer_email})
        return str([order_to_dict(o) for o in orders])
    finally:
        db.close()


@tool
def search_orders(query: str) -> str:
    """
    Search orders by order ID or customer email. This is the main lookup tool.

    Args:
        query: Order ID like ORD-1001, or the customer's email address

    Returns:
        JSON string with matching order details
    """
    db = _get_db()
    try:
        order_id_match = re.search(r"ORD-\d+", query, re.IGNORECASE)
        if order_id_match:
            order_id = order_id_match.group(0).upper()
            order = db.query(Order).filter(Order.order_id == order_id).first()
            if order:
                return str(order_to_dict(order))
            return str({"error": "Order not found", "order_id": order_id})

        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
        if email_match:
            email = email_match.group(0).lower()
            orders = (
                db.query(Order)
                .filter(Order.customer_email.ilike(email))
                .order_by(Order.order_date.desc())
                .all()
            )
            if orders:
                return str([order_to_dict(o) for o in orders])
            return str({"error": "No orders found for email", "email": email})

        name_candidates = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", query)
        if name_candidates:
            # Privacy: name-only lookups are disabled. Names are not unique
            # identifiers, so they must never be used to pull customer data.
            return str({
                "error": "Identifier required",
                "message": "For privacy, lookups require the customer's email address or an order ID (ORD-...).",
            })

        # Privacy: never fall back to returning arbitrary/recent orders —
        # that would leak another customer's data.
        return str({
            "error": "Identifier required",
            "message": "Please provide a valid order ID (ORD-...) or the customer's email address.",
        })
    finally:
        db.close()


@tool
def get_order_items(order_id: str) -> str:
    """
    Retrieve line items for a specific order.

    Args:
        order_id: The order ID (e.g., ORD-1001)

    Returns:
        JSON string with list of items (product_id, name, quantity, unit_price)
    """
    db = _get_db()
    try:
        order = db.query(Order).filter(Order.order_id == order_id.upper()).first()
        if not order:
            return str({"error": "Order not found", "order_id": order_id})

        items = db.query(OrderItem).filter(OrderItem.order_id == order.order_id).all()
        result = [
            {
                "product_id": item.product_id,
                "name": item.name,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "line_total": round(item.quantity * item.unit_price, 2),
            }
            for item in items
        ]
        return str({"order_id": order.order_id, "items": result})
    finally:
        db.close()
