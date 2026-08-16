from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from app.agents.state import AgentState, AgentName
from app.agents.router_agent import router_node
from app.agents.rag_agent import rag_node
from app.agents.order_agent import order_node
from app.agents.coding_agent import coding_node
from app.agents.review_agent import review_node
from app.agents.evaluator_agent import evaluator_node, evaluator_router


AGENT_CAPABILITIES = {
    "rag": {
        "description": "Retrieves information from company documents",
        "can_handle": ["documentation", "policies", "FAQs", "knowledge base"],
        "can_handoff_to": ["evaluator"],
    },
    "order": {
        "description": "Retrieves customer order information",
        "can_handle": ["order status", "order history", "customer order"],
        "can_handoff_to": ["rag", "evaluator"],
    },
    "coding": {
        "description": "Writes and analyzes code",
        "can_handle": ["coding", "bugs", "implementation"],
        "can_handoff_to": ["review", "evaluator"],
    },
    "review": {
        "description": "Reviews generated code",
        "can_handle": ["code review", "security review", "quality review"],
        "can_handoff_to": ["coding", "evaluator"],
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


def initial_router(state: AgentState):
    return state["next_agent"]


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("order", order_node)
    builder.add_node("coding", coding_node)
    builder.add_node("review", review_node)
    builder.add_node("evaluator", evaluator_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        initial_router,
        {
            "rag": "rag",
            "order": "order",
            "coding": "coding",
        },
    )

    builder.add_conditional_edges(
        "rag",
        lambda state: "evaluator",
        {
            "evaluator": "evaluator",
        },
    )

    builder.add_conditional_edges(
        "order",
        lambda state: "evaluator",
        {
            "evaluator": "evaluator",
        },
    )

    builder.add_conditional_edges(
        "coding",
        lambda state: "review",
        {
            "review": "review",
        },
    )

    builder.add_edge("review", "evaluator")

    builder.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            "end": END,
            "rag": "rag",
            "order": "order",
            "coding": "coding",
            "review": "review",
        },
    )

    return builder.compile()


graph = build_graph()
