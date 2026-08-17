from app.agents.state import AgentState, AgentName
from app.config import get_settings
from langchain_core.prompts import ChatPromptTemplate

settings = get_settings()


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "review",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a code reviewer. Review the provided code for correctness, security, performance, and maintainability."
        " Provide actionable feedback only."
    ),
    (
        "user",
        "Code:\n{code}"
    ),
])


def review_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    code = state.get("code_result", "")

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    try:
        chain = REVIEW_PROMPT | llm
        answer = chain.invoke({"code": code})
        review_text = answer.content
    except Exception:
        review_text = "Could not complete code review at this time."

    return {
        "current_agent": "review",
        "review_feedback": review_text,
        "response": review_text,
        "next_agent": "evaluator",
        "visited_agents": state.get("visited_agents", []) + ["review"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, "review_code"),
    }
