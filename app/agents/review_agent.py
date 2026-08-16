from app.agents.state import AgentState, AgentName
from app.config import get_settings

settings = get_settings()


def review_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    code = state.get("code_result", "")

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    review = llm.invoke(
        f"""
Review the following code for:
- correctness
- security
- performance
- maintainability

Code:
{code}

Provide actionable feedback.
"""
    )

    return {
        "current_agent": "review",
        "response": review.content,
        "visited_agents": state.get("visited_agents", []) + ["review"],
        "handoff_count": state.get("handoff_count", 0),
    }
