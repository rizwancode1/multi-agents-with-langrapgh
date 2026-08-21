"""
Database Initialization Script
Creates tables and seeds initial data from orders.json.
Run this script once to bootstrap the database.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from app.db import init_db, get_db
from app.models_db import Order, OrderItem, SupportTicket, RefundRequest
from app.config import get_settings

settings = get_settings()
ORDERS_JSON_PATH = Path(__file__).resolve().parent / "data" / "orders.json"


def seed_orders():
    """Seed the database with orders from orders.json."""
    if not ORDERS_JSON_PATH.exists():
        print(f"Orders JSON not found at {ORDERS_JSON_PATH}")
        return

    with open(ORDERS_JSON_PATH, "r", encoding="utf-8") as f:
        orders_data = json.load(f)

    db = get_db()
    try:
        existing_count = db.query(Order).count()
        if existing_count > 0:
            print(f"Database already contains {existing_count} orders. Skipping seed.")
            return

        for order_data in orders_data:
            order_date = datetime.fromisoformat(order_data["order_date"].replace("Z", "+00:00"))
            customer = order_data["customer"]
            payment = order_data["payment"]
            shipping = order_data["shipping_address"]

            order = Order(
                order_id=order_data["order_id"],
                customer_name=customer["name"],
                customer_email=customer["email"],
                customer_phone=customer.get("phone"),
                order_date=order_date,
                status=order_data["status"],
                shipping_address=shipping,
                payment_method=payment["method"],
                transaction_id=payment.get("transaction_id"),
                subtotal=payment["subtotal"],
                tax=payment["tax"],
                shipping_fee=payment.get("shipping_fee", 0.0),
                total=payment["total"],
            )
            db.add(order)

            for item_data in order_data.get("items", []):
                item = OrderItem(
                    order_id=order_data["order_id"],
                    product_id=item_data["product_id"],
                    name=item_data["name"],
                    quantity=item_data["quantity"],
                    unit_price=item_data["unit_price"],
                )
                db.add(item)

        db.commit()
        print(f"Successfully seeded {len(orders_data)} orders.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()


def main():
    print("Initializing database...")
    init_db()
    print("Tables created.")
    seed_orders()
    print("Database initialization complete.")


if __name__ == "__main__":
    main()
