from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.evaluator_agent import evaluator_node, evaluator_router
from app.agents.formatter_agent import formatter_node
from app.agents.order_agent import order_node
from app.agents.rag_agent import rag_node as policy_rag_node
from app.agents.return_refund_agent import return_refund_node
from app.agents.router_agent import router_node
from app.agents.state import AgentState
from app.agents.support_ticket_agent import support_ticket_node

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

# Safety caps advertised in the README. Enforced in agent_router so they apply
# no matter which node set next_agent.
MAX_HANDOFFS_PER_REQUEST = 5
MAX_VISITS_PER_AGENT = 3

VALID_AGENTS = set(AGENT_CAPABILITIES.keys())


def resolve_next_agent(state: AgentState) -> str:
    """Validate and sanitize next_agent before routing.

    Enforces:
      - the agent actually exists,
      - the handoff is allowed by AGENT_CAPABILITIES (or comes from the
        router/evaluator, which may target any specialist),
      - the per-request handoff cap and per-agent visit cap.

    Violations degrade gracefully to evaluator/formatter instead of raising,
    so a misbehaving LLM can never crash a request or loop forever.
    """
    next_agent = state.get("next_agent")
    if not next_agent:
        return "evaluator"

    if next_agent == "formatter":
        return "formatter"
    if next_agent == "evaluator":
        return "evaluator"

    if next_agent not in VALID_AGENTS:
        return "evaluator"

    visited = state.get("visited_agents", [])
    agent_visits = sum(1 for a in visited if a == next_agent)
    if agent_visits >= MAX_VISITS_PER_AGENT:
        return "formatter"

    if state.get("handoff_count", 0) >= MAX_HANDOFFS_PER_REQUEST:
        return "formatter"

    current_agent = state.get("current_agent")
    allowed = AGENT_CAPABILITIES.get(current_agent, {}).get("can_handoff_to", [])
    # Router and evaluator may route to any specialist; specialists are
    # restricted to their declared can_handoff_to list.
    privileged_sources = {None, "router", "evaluator"}
    if current_agent not in privileged_sources and next_agent not in allowed:
        return "evaluator"

    return next_agent


def agent_router(state: AgentState):
    return resolve_next_agent(state)


def initial_router(state: AgentState):
    return state["next_agent"]


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
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
