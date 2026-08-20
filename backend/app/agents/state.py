from typing import TypedDict, Literal, Optional

AgentName = Literal[
    "router",
    "policy_rag",
    "order",
    "support_ticket",
    "return_refund",
    "evaluator",
    "formatter",
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
