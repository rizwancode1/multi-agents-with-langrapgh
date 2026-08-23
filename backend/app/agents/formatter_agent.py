from langchain_core.prompts import ChatPromptTemplate

from app.agents.route_utils import add_route
from app.agents.state import AgentState, format_history
from app.config import get_settings
from app.monitoring import get_logger

settings = get_settings()
logger = get_logger("formatter_agent")


FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a response formatter. Produce a final answer using ONLY the data provided below."
        " Do NOT add information from general knowledge."
        " IMPORTANT: Absence of mention does NOT imply the opposite is true."
        " If data does not state whether something is true or false, do not make any claim about it."
        " If no data is provided, produce a helpful response based on the available context."
    ),
    (
        "user",
        "Conversation history:\n{history}\n\nUser question:\n{question}\n\nOrder data:\n{order_data}\n\nReturn/Refund data:\n{return_refund_data}\n\nRetrieved context:\n{context}"
    ),
])


def formatter_node(state: AgentState):
    from app.utils import get_chat_llm

    query = state["query"]
    history = format_history(state.get("messages", []))
    order_data = state.get("order_data", {})
    return_refund_data = state.get("return_refund_data", {})
    context = state.get("context", [])

    logger.info("formatter_node_started", extra={"extra_data": {
        "query": query[:200],
        "order_data_type": type(order_data).__name__,
        "order_data_empty": not bool(order_data),
        "return_refund_data_type": type(return_refund_data).__name__,
        "return_refund_data_empty": not bool(return_refund_data),
        "context_length": len(context) if context else 0,
        "context_empty": not bool(context),
        "visited_agents": state.get("visited_agents", []),
    }})

    llm = get_chat_llm()

    try:
        chain = FORMATTER_PROMPT | llm
        formatted = chain.invoke({
            "question": query,
            "history": history,
            "order_data": str(order_data) if order_data else "No order data available.",
            "return_refund_data": str(return_refund_data) if return_refund_data else "No return/refund data available.",
            "context": "\n".join(f"[{i}] {c}" for i, c in enumerate(context, 1)) if context else "No retrieved context available.",
        })
        response_text = formatted.content
        logger.info("formatter_llm_invoked", extra={"extra_data": {
            "response_length": len(response_text) if response_text else 0,
        }})
    except Exception as e:
        logger.error("formatter_llm_error", extra={"extra_data": {
            "error": str(e),
        }})
        response_text = "I couldn't format the response at this time."

    return {
        "current_agent": "formatter",
        "response": response_text,
        "visited_agents": state.get("visited_agents", []) + ["formatter"],
        "handoff_count": state.get("handoff_count", 0),
        "route": add_route(state, "formatter", "format_response"),
    }