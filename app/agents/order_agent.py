import json
import re
from pathlib import Path

from app.agents.state import AgentState, AgentName
from app.config import get_settings
from langchain_core.prompts import ChatPromptTemplate

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


def _needs_policy_info(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in [
        "policy", "policies", "shipping", "return", "refund",
        "compare", "comparison", "fee", "cost", "standard",
        "free shipping", "delivery", "time", "days",
    ])


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "order",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


ORDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an order assistant. Return ONLY the order facts provided."
        " Do not infer item categories, eligibility, or policy conclusions from order data alone."
    ),
    (
        "user",
        "Question:\n{question}\n\nOrder data:\n{order_data}"
    ),
])


def order_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    order = find_order(query)

    if not order:
        return {
            "current_agent": "order",
            "response": "No matching order found.",
            "order_data": {},
            "order_id": None,
            "visited_agents": state.get("visited_agents", []) + ["order"],
            "handoff_count": state.get("handoff_count", 0),
            "route": _add_route(state, "no_order_found"),
        }

    order_summary = {
        "order_id": order["order_id"],
        "customer_name": order["customer"]["name"],
        "status": order["status"],
        "order_date": order["order_date"],
        "items": [
            {
                "name": item["name"],
                "quantity": item["quantity"],
                "unit_price": item["unit_price"],
            }
            for item in order.get("items", [])
        ],
        "shipping_address": order.get("shipping_address", {}),
        "payment": {
            "subtotal": order["payment"]["subtotal"],
            "tax": order["payment"]["tax"],
            "shipping_fee": order["payment"]["shipping_fee"],
            "total": order["payment"]["total"],
            "method": order["payment"]["method"],
        },
    }

    next_agent: AgentName | None = "evaluator"
    handoff_reason = "Order data retrieved, ready for evaluation."

    if _needs_policy_info(query):
        next_agent = "rag"
        handoff_reason = "Query requires policy/shipping information beyond order data."

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    try:
        chain = ORDER_PROMPT | llm
        answer = chain.invoke({
            "question": query,
            "order_data": str(order_summary),
        })
        response_text = answer.content
    except Exception:
        response_text = f"Order {order['order_id']} found. Status: {order['status']}."

    return {
        "current_agent": "order",
        "order_data": order_summary,
        "order_id": order["order_id"],
        "response": response_text,
        "next_agent": next_agent,
        "handoff_reason": handoff_reason,
        "visited_agents": state.get("visited_agents", []) + ["order"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, f"lookup_order_handoff_to_{next_agent or 'evaluator'}"),
    }
