from app.agents.state import AgentState
from app.config import get_settings
from langchain_core.prompts import ChatPromptTemplate

settings = get_settings()


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "coding",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


CODING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a coding assistant. Provide clean, production-ready Python code with comments."
        " Only output code and brief explanations; do not add unrelated content."
    ),
    (
        "user",
        "Implement the following requirement in Python:\n\n{question}"
    ),
])


def coding_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    try:
        chain = CODING_PROMPT | llm
        answer = chain.invoke({"question": query})
        code_text = answer.content
    except Exception:
        code_text = "# Could not generate code at this time."

    return {
        "current_agent": "coding",
        "code_result": code_text,
        "response": code_text,
        "next_agent": "evaluator",
        "visited_agents": state.get("visited_agents", []) + ["coding"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, "generate_code"),
    }
