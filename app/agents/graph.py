from __future__ import annotations

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.state import AgentState, AgentName
from app.agents.router_agent import router_node
from app.agents.rag_agent import rag_node as policy_rag_node
from app.agents.order_agent import order_node
from app.agents.support_ticket_agent import support_ticket_node
from app.agents.return_refund_agent import return_refund_node
from app.agents.formatter_agent import formatter_node
from app.agents.evaluator_agent import evaluator_node, evaluator_router


AGENT_CAPABILITIES = {
    "policy_rag": {
        "description": "Retrieves information from company documents, policies, FAQs",
        "can_handle": ["documentation", "policies", "FAQs", "knowledge base", "shipping_policy", "return_policy", "refund_policy", "account_security"],
        "can_handoff_to": ["return_refund", "evaluator"],
    },
    "order": {
        "description": "Retrieves customer order information",
        "can_handle": ["order status", "order history", "customer order", "order_tracking"],
        "can_handoff_to": ["evaluator"],
    },
    "support_ticket": {
        "description": "Handles complaints, issues, and support tickets",
        "can_handle": ["complaint", "issue", "support_ticket", "general_help"],
        "can_handoff_to": ["evaluator"],
    },
    "return_refund": {
        "description": "Handles returns, eligibility checks, and refunds",
        "can_handle": ["return_policy", "refund_policy", "return_refund"],
        "can_handoff_to": ["evaluator"],
    },
}


def request_handoff(state: AgentState, target: AgentName, reason: str):
    current_agent = state["current_agent"]
    allowed = AGENT_CAPABILITIES.get(current_agent, {}).get("can_handoff_to", [])

    if target not in allowed:
        raise ValueError(f"{current_agent} cannot handoff to {target}")

    count = state.get("handoff_count", 0)
    if count >= 5:
        raise RuntimeError("Maximum handoff limit reached")

    return {
        "next_agent": target,
        "handoff_reason": reason,
        "handoff_count": count + 1,
    }


def agent_router(state: AgentState):
    next_agent = state.get("next_agent")
    if next_agent:
        visited = state.get("visited_agents", [])
        agent_visits = sum(1 for a in visited if a == next_agent)
        if agent_visits >= 3:
            return "formatter"
        return next_agent
    return "evaluator"


def initial_router(state: AgentState):
    return state["next_agent"]


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("policy_rag", policy_rag_node)
    builder.add_node("order", order_node)
    builder.add_node("support_ticket", support_ticket_node)
    builder.add_node("return_refund", return_refund_node)
    builder.add_node("formatter", formatter_node)
    builder.add_node("evaluator", evaluator_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        initial_router,
        {
            "order": "order",
            "policy_rag": "policy_rag",
            "support_ticket": "support_ticket",
        },
    )

    builder.add_conditional_edges(
        "order",
        agent_router,
        {
            "evaluator": "evaluator",
        },
    )

    builder.add_conditional_edges(
        "policy_rag",
        agent_router,
        {
            "return_refund": "return_refund",
            "evaluator": "evaluator",
        },
    )

    builder.add_conditional_edges(
        "support_ticket",
        agent_router,
        {
            "evaluator": "evaluator",
        },
    )

    builder.add_conditional_edges(
        "return_refund",
        agent_router,
        {
            "evaluator": "evaluator",
        },
    )

    builder.add_edge("formatter", END)

    builder.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            "formatter": "formatter",
            "order": "order",
            "policy_rag": "policy_rag",
            "support_ticket": "support_ticket",
            "return_refund": "return_refund",
            "end": END,
        },
    )

    return builder.compile(checkpointer=checkpointer)


# Module-level graph (no checkpointer by default for backward compatibility)
graph = build_graph()

