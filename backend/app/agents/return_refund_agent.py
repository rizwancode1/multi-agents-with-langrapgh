from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from app.agents.route_utils import add_route
from app.agents.state import AgentState, format_history
from app.agents.tool_loop import run_tool_loop, successful_results
from app.config import get_settings
from app.monitoring import get_logger
from app.tools.refund_tools import (
    calculate_eligible_refund,
    create_refund_request,
    get_refund_status,
    list_refunds_by_order,
)

settings = get_settings()
logger = get_logger("return_refund_agent")


RETURN_REFUND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a return and refund agent. Your job is to process return and refund requests by creating actual refund requests in the database."
        "\n\nYou have access to the following tools:"
        "\n- calculate_eligible_refund: Calculates the eligible refund amount for an order"
        "\n- create_refund_request: Creates a new refund request and returns the refund ID"
        "\n- get_refund_status: Retrieves the status of an existing refund request by refund ID"
        "\n- list_refunds_by_order: Lists all refund requests for a given order"
        "\n\nWhen a customer wants to return an item or request a refund, use calculate_eligible_refund first to determine the amount, then use create_refund_request to create the actual refund request."
        " When a customer asks about their refund status, use get_refund_status or list_refunds_by_order."
        " Use ONLY the provided policy context and order data. Do not infer eligibility from absence of information."
        " Return the actual refund ID so the customer can track their request."
        "\nYou may call multiple tools in sequence. Tool results are returned to you so you can decide the next call."
        "\nWhen you have all the information you need, reply to the customer directly WITHOUT calling any more tools."
    ),
    (
        "user",
        "Conversation history:\n{history}\n\nUser request:\n{question}\n\nOrder data:\n{order_data}\n\nPolicy context:\n{context}\n\nAvailable tools: {tools}"
    ),
])


REFUND_TOOLS: list[BaseTool] = [
    calculate_eligible_refund,
    create_refund_request,
    get_refund_status,
    list_refunds_by_order,
]


def _build_response(executed_calls: list[dict], fallback_text: str | None) -> str:
    """Prefer grounded tool output; fall back to the model's final message."""
    for tc in reversed(successful_results(executed_calls)):
        result = tc["result"]
        if tc["tool"] == "create_refund_request" and "refund_id" in result:
            return f"I've created a refund request for you. {result}"
        if tc["tool"] in ("get_refund_status", "list_refunds_by_order") and "refund_id" in result:
            return f"Here are your refund details. {result}"
        if tc["tool"] == "calculate_eligible_refund":
            return f"Based on your order, here's the refund calculation. {result}"
    return fallback_text or "I've processed your return/refund request."


def return_refund_node(state: AgentState):
    from app.utils import get_chat_llm

    query = state["query"]
    history = format_history(state.get("messages", []))
    order_data = state.get("order_data", {})
    context = state.get("context", [])

    llm = get_chat_llm(tools=REFUND_TOOLS)
    executed_calls: list[dict] = []

    try:
        answer, executed_calls = run_tool_loop(
            llm,
            RETURN_REFUND_PROMPT,
            {
                "question": query,
                "history": history,
                "order_data": str(order_data) if order_data else "No order data available.",
                "context": "\n".join(f"[{i}] {c}" for i, c in enumerate(context, 1)) if context else "No policy context available.",
                "tools": ", ".join([t.name for t in REFUND_TOOLS]),
            },
            REFUND_TOOLS,
        )
        response_text = _build_response(
            executed_calls,
            answer.content if (answer and answer.content) else None,
        )
    except Exception:
        response_text = "I couldn't process the return/refund request at this time. Please try again later."
        logger.error("refund_agent_exception", extra={"extra_data": {
            "query": query[:200],
        }})

    return_refund_summary = {
        "order_id": state.get("order_id"),
        "eligibility": "pending_review",
        "refund_amount": None,
        "reasoning": response_text,
    }

    logger.info("refund_agent_result", extra={"extra_data": {
        "response_preview": response_text[:200],
        "tool_calls_count": len(executed_calls),
        "order_id": state.get("order_id"),
    }})

    return {
        "current_agent": "return_refund",
        "response": response_text,
        "return_refund_data": return_refund_summary,
        "next_agent": "evaluator",
        "visited_agents": state.get("visited_agents", []) + ["return_refund"],
        "handoff_count": state.get("handoff_count", 0),
        "route": add_route(state, "return_refund", "process_return_refund"),
    }
