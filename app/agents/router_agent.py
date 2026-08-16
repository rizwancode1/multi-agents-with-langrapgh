from typing import Literal

from pydantic import BaseModel, Field

from app.agents.state import AgentState, AgentName
from app.config import get_settings

settings = get_settings()


class RouteDecision(BaseModel):
    agent: Literal["rag", "order", "coding"]
    reason: str = Field(description="Why this agent should handle the request")


def _fallback_route(query: str) -> dict:
    q = query.lower()
    if any(k in q for k in ["order", "purchase", "shipped", "delivered", "status", "ord-"]):
        return {"agent": "order", "reason": "Query appears to be about an order."}
    if any(k in q for k in ["code", "python", "function", "bug", "implement", "program"]):
        return {"agent": "coding", "reason": "Query appears to be about coding."}
    return {"agent": "rag", "reason": "Defaulting to knowledge base lookup."}


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
        decision = router_llm.invoke(
            f"""
You are a routing agent.

Choose the ONE specialized agent that should handle the user's request.

Available agents:
1. rag - documentation, policies, FAQs, knowledge base
2. order - customer orders, order status, order history
3. coding - programming, debugging, implementation

User request:
{query}
"""
        )
        agent = decision.agent
        reason = decision.reason
    except Exception:
        fallback = _fallback_route(query)
        agent = fallback["agent"]
        reason = fallback["reason"] + " (fallback routing)"

    return {
        "current_agent": "router",
        "next_agent": agent,
        "handoff_reason": reason,
    }
