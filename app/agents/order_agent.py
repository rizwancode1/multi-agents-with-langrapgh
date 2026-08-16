import json
import re
from pathlib import Path

from app.agents.state import AgentState
from app.config import get_settings

settings = get_settings()

ORDERS_PATH = Path(__file__).resolve().parents[2] / "app" / "data" / "orders.json"


def load_orders() -> list[dict]:
    with open(ORDERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_order(query: str) -> dict | None:
    orders = load_orders()

    order_id_match = re.search(r"ORD-\d+", query, re.IGNORECASE)
    if order_id_match:
        order_id = order_id_match.group(0).upper()
        for order in orders:
            if order["order_id"].upper() == order_id:
                return order

    name_candidates = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", query)
    for candidate in sorted(name_candidates, key=len, reverse=True):
        name_lower = candidate.lower()
        for order in orders:
            if name_lower in order["customer"]["name"].lower():
                return order

    return orders[0] if orders else None


def order_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    order = find_order(query)

    response_text = ""
    if order:
        response_text = (
            f"Order {order['order_id']} for {order['customer']['name']} - "
            f"Status: {order['status']}, "
            f"Items: {len(order['items'])}, "
            f"Total: ${order['payment']['total']}"
        )
    else:
        response_text = "No matching order found."

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    summary = llm.invoke(
        f"""
You are an order assistant. Summarize the following order information for the user.

User question:
{query}

Order data:
{response_text}

Provide a concise, friendly response.
"""
    )

    return {
        "current_agent": "order",
        "response": summary.content,
        "order_data": order or {},
        "visited_agents": state.get("visited_agents", []) + ["order"],
        "handoff_count": state.get("handoff_count", 0),
    }
