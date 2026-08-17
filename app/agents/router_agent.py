from typing import Literal
import re

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.agents.state import AgentState, AgentName
from app.config import get_settings

settings = get_settings()


class RouteDecision(BaseModel):
    agent: Literal["rag", "order", "coding"]
    intents: list[str] = Field(description="List of specific user intents. Use exactly: order_status, shipping_policy, return_policy, refund_policy, account_security, coding, code_review, documentation, faq, general")
    order_id: str | None = Field(description="Extracted order ID like ORD-1001, or null if none found")
    reason: str = Field(description="Why this agent should handle the request")


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": state.get("current_agent", "router"),
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


def _fallback_route(query: str) -> dict:
    q = query.lower()
    if any(k in q for k in ["order", "purchase", "shipped", "delivered", "status", "ord-"]):
        return {"agent": "order", "reason": "Query appears to be about an order."}
    if any(k in q for k in ["code", "python", "function", "bug", "implement", "program"]):
        return {"agent": "coding", "reason": "Query appears to be about coding."}
    return {"agent": "rag", "reason": "Defaulting to knowledge base lookup."}


def _extract_intents_fallback(query: str) -> list[str]:
    q = query.lower()
    intents = []
    if any(k in q for k in ["order", "purchase", "shipped", "delivered", "status", "ord-"]):
        intents.append("order_status")
    if any(k in q for k in ["shipping", "delivery", "days", "standard shipping"]):
        intents.append("shipping_policy")
    if any(k in q for k in ["return", "refund", "money back"]):
        intents.append("return_policy")
        intents.append("refund_policy")
    if any(k in q for k in ["account", "security", "password", "login", "personal"]):
        intents.append("account_security")
    if any(k in q for k in ["code", "python", "function", "bug", "implement", "program"]):
        intents.append("coding")
    if any(k in q for k in ["review", "refactor", "improve", "fix"]):
        intents.append("code_review")
    if not intents:
        intents.append("general")
    return intents


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing agent. Choose the ONE specialized agent that should handle the user's request."
        " Also extract ALL specific user intents and any order ID."
        "\n\nAvailable agents:"
        "\n1. rag - documentation, policies, FAQs, knowledge base"
        "\n2. order - customer orders, order status, order history"
        "\n3. coding - programming, debugging, implementation"
        "\n\nAllowed intents (use exactly these labels when applicable):"
        "\n- order_status"
        "\n- shipping_policy"
        "\n- return_policy"
        "\n- refund_policy"
        "\n- account_security"
        "\n- coding"
        "\n- code_review"
        "\n- documentation"
        "\n- faq"
        "\n- general"
    ),
    (
        "user",
        "{query}"
    ),
])


def router_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    router_llm = llm.with_structured_output(RouteDecision)

    try:
        chain = ROUTER_PROMPT | router_llm
        decision = chain.invoke({"query": query})
        agent = decision.agent
        reason = decision.reason
        intents = decision.intents
        order_id = decision.order_id
    except Exception:
        fallback = _fallback_route(query)
        agent = fallback["agent"]
        reason = fallback["reason"] + " (fallback routing)"
        intents = _extract_intents_fallback(query)
        order_id = None
        order_match = re.search(r"ORD-\d+", query, re.IGNORECASE)
        if order_match:
            order_id = order_match.group(0).upper()

    return {
        "current_agent": "router",
        "next_agent": agent,
        "handoff_reason": reason,
        "intents": intents,
        "order_id": order_id,
        "route": _add_route(state, f"route_to_{agent}"),
    }
