from typing import TypedDict, Literal, Optional

AgentName = Literal[
    "router",
    "rag",
    "order",
    "coding",
    "review",
    "evaluator",
]


class AgentState(TypedDict, total=False):
    query: str
    current_agent: AgentName
    response: str
    context: list[str]
    order_data: dict
    code_result: str
    citations: list[str]
    next_agent: AgentName | None
    handoff_reason: str | None
    evaluation: dict
    handoff_count: int
    visited_agents: list[str]
    error: str | None
