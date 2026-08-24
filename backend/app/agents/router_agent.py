import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agents.route_utils import add_route
from app.agents.state import AgentState, format_history
from app.config import get_settings
from app.monitoring import get_logger

settings = get_settings()
logger = get_logger("router_agent")


class RouteDecision(BaseModel):
    agent: Literal["order", "policy_rag", "support_ticket", "out_of_scope"]
    intents: list[str] = Field(description="List of specific user intents. Use exactly: order_status, shipping_policy, return_policy, refund_policy, account_security, documentation, faq, support_ticket, complaint, general")
    order_id: str | None = Field(description="Extracted order ID like ORD-1001, or null if none found")
    reason: str = Field(description="Why this agent should handle the request")


def _fallback_route(query: str) -> dict:
    q = query.lower()
    if any(k in q for k in ["ord-", "order status", "track", "where is my order", "my order"]):
        return {"agent": "order", "reason": "Query appears to be about an order."}
    # Action requests (tickets/complaints) must beat policy lookups, otherwise
    # phrases like "i want a refund" drag actionable requests into policy_rag.
    if any(k in q for k in ["ticket", "complaint", "complain", "damaged", "broken",
                            "defective", "not working", "issue", "problem",
                            "unhappy", "bad", "never arrived", "missing", "faulty"]):
        return {"agent": "support_ticket", "reason": "Query appears to be a support request."}
    if any(k in q for k in ["order", "purchase", "shipped", "delivered", "status"]):
        return {"agent": "order", "reason": "Query appears to be about an order."}
    if any(k in q for k in ["policy", "faq", "documentation", "rules", "guidelines", "shipping", "return", "refund"]):
        return {"agent": "policy_rag", "reason": "Query appears to be about policies or documentation."}
    return {"agent": "policy_rag", "reason": "Defaulting to knowledge base lookup."}


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
    if any(k in q for k in ["policy", "faq", "documentation", "rules", "guidelines"]):
        intents.append("documentation")
        intents.append("faq")
    if any(k in q for k in ["complaint", "issue", "problem", "support", "ticket", "help", "unhappy", "bad"]):
        intents.append("support_ticket")
        intents.append("complaint")
    if not intents:
        intents.append("general")
    return intents


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the supervisor of a RETAIL CUSTOMER SUPPORT assistant for an online store."
        " You know exactly what this assistant IS and what it is ALLOWED to do:"
        " it answers questions about customer orders, shipping, returns and refunds,"
        " store policies, and it manages support tickets. Nothing else."
        "\n\nChoose the ONE specialized agent that should handle the user's request."
        " Also extract ALL specific user intents and any order ID."
        "\n\nConversation history is provided. Use it to understand follow-up requests"
        " (e.g. 'what is the price?' after an order discussion means the ORDER price,"
        " not a general pricing FAQ). When in doubt, route to the agent from the previous relevant turn."
        "\n\nAvailable agents:"
        "\n1. order - customer orders, order status, order history, tracking, order prices"
        "\n2. policy_rag - policies, FAQs, documentation, rules, guidelines"
        "\n3. support_ticket - complaints, issues, support tickets, general help"
        "\n4. return_refund - returns, eligibility, refunds (usually reached via policy_rag)"
        "\n5. out_of_scope - ANY request outside this store's support domain. Examples:"
        " math or homework ('what is 2/2'), writing or debugging code, writing essays or"
        " stories, general trivia or news, translations, image generation, medical/legal"
        " advice, or asking the assistant to modify/delete systems."
        " If the request is not about THIS customer's orders, tickets, or store policies,"
        " choose out_of_scope. Do NOT try to solve out-of-domain tasks yourself."
        "\n\nAllowed intents (use exactly these labels when applicable):"
        "\n- order_status"
        "\n- shipping_policy"
        "\n- return_policy"
        "\n- refund_policy"
        "\n- account_security"
        "\n- documentation"
        "\n- faq"
        "\n- support_ticket"
        "\n- complaint"
        "\n- general"
    ),
    (
        "user",
        "Conversation history:\n{history}\n\nCurrent user request:\n{query}"
    ),
])


# Slot name -> specialist that should resume when the slot is filled.
_SLOT_TARGETS = {
    "order_identifier": "order",
    "support_details": "support_ticket",
}

_IDENTIFIER_RE = re.compile(
    r"(ORD-\d+|TKT-[A-Z0-9]+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
    re.IGNORECASE,
)


def out_of_scope_node(state: AgentState):
    """Terminal refusal for requests outside the support domain.

    Deterministic on purpose: no LLM call, no tools. The router already made
    the semantic judgement; this node only states the assistant's boundaries
    and redirects to what it CAN do.
    """
    query = state.get("query", "")
    logger.info("out_of_scope_refusal", extra={"extra_data": {"query": query[:200]}})
    return {
        "current_agent": "out_of_scope",
        "response": (
            "I'm Relay, a customer support assistant — I can help with your orders, "
            "shipping, returns and refunds, store policies, and support tickets. "
            "I'm not able to help with anything outside of that, like calculations, "
            "writing code or essays, or general topics. Is there something with your "
            "orders or a support issue I can help you with?"
        ),
        "next_agent": None,
        "handoff_reason": "Request is outside the support domain.",
        "awaiting_slot": None,
        "visited_agents": [*state.get("visited_agents", []), "out_of_scope"],
        "route": add_route(state, "router", "refuse_out_of_scope"),
    }


def router_node(state: AgentState):
    from app.utils import get_chat_llm

    query = state["query"]
    history = format_history(state.get("messages", []))

    # ---- Slot-filling fast path ----
    # A specialist previously asked for identifying info. If this turn
    # provides it, resume that specialist directly — no LLM routing call,
    # no re-classification mistakes, seconds saved per follow-up.
    awaiting_slot = state.get("awaiting_slot")
    if awaiting_slot in _SLOT_TARGETS and _IDENTIFIER_RE.search(query):
        target = _SLOT_TARGETS[awaiting_slot]
        logger.info("router_slot_fill_resume", extra={"extra_data": {
            "slot": awaiting_slot,
            "agent": target,
        }})
        order_match = re.search(r"ORD-\d+", query, re.IGNORECASE)
        return {
            "current_agent": "router",
            "next_agent": target,
            "handoff_reason": f"Follow-up: user supplied the awaited {awaiting_slot.replace('_', ' ')}.",
            "intents": ["order_status"] if target == "order" else ["support_ticket"],
            "order_id": order_match.group(0).upper() if order_match else None,
            "visited_agents": [*state.get("visited_agents", []), "router"],
            "route": add_route(state, "router", f"slot_fill_to_{target}"),
        }

    logger.info("router_node_started", extra={"extra_data": {"query": query[:200]}})

    router_llm = get_chat_llm(schema=RouteDecision)

    try:
        chain = ROUTER_PROMPT | router_llm
        decision = chain.invoke({"query": query, "history": history})
        agent = decision.agent
        reason = decision.reason
        intents = decision.intents
        order_id = decision.order_id
        logger.info("router_llm_decision", extra={"extra_data": {
            "agent": agent,
            "intents": intents,
            "order_id": order_id,
            "reason": reason,
        }})
    except Exception:
        logger.exception("router_llm_failed", extra={"extra_data": {
            "query": query[:200],
        }})
        fallback = _fallback_route(query)
        agent = fallback["agent"]
        reason = fallback["reason"] + " (fallback routing)"
        intents = _extract_intents_fallback(query)
        order_id = None
        order_match = re.search(r"ORD-\d+", query, re.IGNORECASE)
        if order_match:
            order_id = order_match.group(0).upper()
        logger.warning("router_fallback_used", extra={"extra_data": {
            "agent": agent,
            "intents": intents,
            "order_id": order_id,
            "reason": reason,
            "query": query[:200],
        }})

    # Any normal (LLM-routed) turn invalidates a stale pending slot.
    return {
        "current_agent": "router",
        "awaiting_slot": None,
        "next_agent": agent,
        "handoff_reason": reason,
        "intents": intents,
        "order_id": order_id,
        "visited_agents": [*state.get("visited_agents", []), "router"],
        "route": add_route(state, "router", f"route_to_{agent}"),
    }
