from typing import Literal, TypedDict

AgentName = Literal[
    "router",
    "policy_rag",
    "order",
    "support_ticket",
    "return_refund",
    "evaluator",
    "formatter",
    "out_of_scope",
]


class AgentState(TypedDict, total=False):
    query: str
    current_agent: AgentName
    response: str
    context: list[str]
    order_data: dict
    order_id: str | None
    citations: list[str]
    intents: list[str]
    retrieved_documents: list[dict]
    next_agent: AgentName | None
    handoff_reason: str | None
    evaluation: dict
    handoff_count: int
    visited_agents: list[str]
    error: str | None
    route: list[dict]
    return_refund_data: dict | None
    messages: list[dict]
    context_state: str | None
    missing_aspects: list[str]
    relevant_sources: list[str]
    is_grounded: bool | None
    rag_trace: list[str]
    # Slot-filling: when a specialist asked the user for missing info (e.g.
    # their email/order ID), the pending slot name persists in the checkpoint
    # so the next turn can skip routing and resume the right agent directly.
    awaiting_slot: str | None


def format_history(messages: list[dict] | None) -> str:
    """Render conversation history (list of {role, text}) for agent prompts."""
    if not messages:
        return "No prior conversation."
    lines = []
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        text = m.get("text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)
