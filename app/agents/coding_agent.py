from app.agents.state import AgentState
from app.config import get_settings

settings = get_settings()


def coding_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    generated_code = llm.invoke(
        f"""
Implement the following requirement in Python:

{query}

Provide clean, production-ready code with comments.
"""
    )

    return {
        "current_agent": "coding",
        "code_result": generated_code.content,
        "response": generated_code.content,
        "visited_agents": state.get("visited_agents", []) + ["coding"],
        "handoff_count": state.get("handoff_count", 0),
    }
