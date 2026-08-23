from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from app.agents.route_utils import add_route
from app.agents.state import AgentState, format_history
from app.agents.tool_loop import run_tool_loop, successful_results
from app.config import get_settings
from app.monitoring import get_logger
from app.tools.support_tools import (
    create_support_ticket,
    get_ticket_status,
    list_tickets_by_email,
    update_ticket_status,
)

settings = get_settings()
logger = get_logger("support_ticket_agent")


SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a support ticket agent. Your job is to handle customer complaints, issues, and support requests by creating actual support tickets in the database."
        "\n\nYou have access to the following tools:"
        "\n- create_support_ticket: Creates a new support ticket and returns the ticket ID"
        "\n- get_ticket_status: Retrieves the status of an existing ticket by ticket ID"
        "\n- list_tickets_by_email: Lists all tickets for a customer by email"
        "\n- update_ticket_status: Updates the status of an existing ticket"
        "\n\nWhen a customer reports an issue, use create_support_ticket to create a real ticket and return the ticket ID."
        " When a customer asks about their ticket status, use get_ticket_status or list_tickets_by_email."
        " Acknowledge the problem, show empathy, and provide the actual ticket ID and next steps."
        "\nYou may call multiple tools in sequence. Tool results are returned to you so you can decide the next call."
        "\nWhen you have all the information you need, reply to the customer directly WITHOUT calling any more tools."
        "\nConversation history:\n{history}"
    ),
    (
        "user",
        "Customer issue:\n{question}\n\nOrder context:\n{order_data}\n\nAvailable tools: {tools}"
    ),
])


SUPPORT_TOOLS: list[BaseTool] = [
    create_support_ticket,
    get_ticket_status,
    list_tickets_by_email,
    update_ticket_status,
]


def _build_response(executed_calls: list[dict], fallback_text: str | None) -> str:
    """Prefer grounded tool output; fall back to the model's final message."""
    for tc in reversed(successful_results(executed_calls)):
        result = tc["result"]
        if tc["tool"] == "create_support_ticket" and "ticket_id" in result:
            return f"I've created a support ticket for you. {result}"
        if tc["tool"] in ("get_ticket_status", "list_tickets_by_email") and "ticket_id" in result:
            return f"Here are your ticket details. {result}"
        if tc["tool"] == "update_ticket_status" and "error" not in result:
            return f"I've updated your ticket. {result}"
    return fallback_text or "I've processed your support request."


def support_ticket_node(state: AgentState):
    from app.utils import get_chat_llm

    query = state["query"]
    history = format_history(state.get("messages", []))
    order_data = state.get("order_data", {})

    llm = get_chat_llm(tools=SUPPORT_TOOLS)
    executed_calls: list[dict] = []

    try:
        answer, executed_calls = run_tool_loop(
            llm,
            SUPPORT_PROMPT,
            {
                "question": query,
                "history": history,
                "order_data": str(order_data) if order_data else "No order data available.",
                "tools": ", ".join([t.name for t in SUPPORT_TOOLS]),
            },
            SUPPORT_TOOLS,
        )
        response_text = _build_response(
            executed_calls,
            answer.content if (answer and answer.content) else None,
        )
    except Exception as e:
        response_text = "I couldn't process your support request at this time. Please try again later."
        logger.error("support_agent_exception", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
        }})

    logger.info("support_agent_result", extra={"extra_data": {
        "response_preview": response_text[:200],
        "tool_calls_count": len(executed_calls),
    }})

    return {
        "current_agent": "support_ticket",
        "response": response_text,
        "next_agent": "evaluator",
        "visited_agents": [*state.get("visited_agents", []), "support_ticket"],
        "handoff_count": state.get("handoff_count", 0),
        "route": add_route(state, "support_ticket", "handle_support_ticket"),
    }
