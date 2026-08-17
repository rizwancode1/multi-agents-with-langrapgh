from app.agents.state import AgentState, AgentName
from app.config import get_settings
from langchain_core.prompts import ChatPromptTemplate

settings = get_settings()


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "formatter",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a response formatter. Produce a final answer using ONLY the data provided below."
        " Do NOT add information from general knowledge."
        " IMPORTANT: Absence of mention does NOT imply the opposite is true."
        " If data does not state whether something is true or false, do not make any claim about it."
    ),
    (
        "user",
        "User question:\n{question}\n\nOrder data:\n{order_data}\n\nGenerated code:\n{code_result}\n\nReview feedback:\n{review_feedback}\n\nRetrieved context:\n{context}"
    ),
])


def formatter_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    order_data = state.get("order_data", {})
    code_result = state.get("code_result", "")
    review_feedback = state.get("review_feedback", "")
    context = state.get("context", [])

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    try:
        chain = FORMATTER_PROMPT | llm
        formatted = chain.invoke({
            "question": query,
            "order_data": str(order_data) if order_data else "",
            "code_result": code_result,
            "review_feedback": review_feedback,
            "context": "\n".join(f"[{i}] {c}" for i, c in enumerate(context, 1)) if context else "",
        })
        response_text = formatted.content
    except Exception:
        response_text = "I couldn't format the response at this time."

    return {
        "current_agent": "formatter",
        "response": response_text,
        "visited_agents": state.get("visited_agents", []) + ["formatter"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, "format_response"),
    }
