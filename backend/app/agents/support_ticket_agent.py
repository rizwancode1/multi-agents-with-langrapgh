from app.agents.state import AgentState, AgentName
from app.config import get_settings
from app.monitoring import get_logger
from app.tools.support_tools import create_support_ticket, get_ticket_status, list_tickets_by_email, update_ticket_status
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

settings = get_settings()
logger = get_logger("support_ticket_agent")


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "support_ticket",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


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


def support_ticket_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    order_data = state.get("order_data", {})

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    ).bind_tools(SUPPORT_TOOLS)

    try:
        chain = SUPPORT_PROMPT | llm
        answer = chain.invoke({
            "question": query,
            "order_data": str(order_data) if order_data else "No order data available.",
            "tools": ", ".join([t.name for t in SUPPORT_TOOLS]),
        })

        response_text = answer.content if answer.content else "I've processed your support request."

        tool_calls = []
        if hasattr(answer, "tool_calls") and answer.tool_calls:
            for tool_call in answer.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                selected_tool = next((t for t in SUPPORT_TOOLS if t.name == tool_name), None)
                if selected_tool:
                    try:
                        tool_result = selected_tool.invoke(tool_args)
                        tool_calls.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                        })
                        logger.info("support_tool_call", extra={"extra_data": {
                            "tool": tool_name,
                            "args": tool_args,
                            "result_preview": str(tool_result)[:200],
                        }})
                        if tool_name == "create_support_ticket" and "ticket_id" in str(tool_result):
                            response_text = f"I've created a support ticket for you. {tool_result}"
                        elif tool_name in ["get_ticket_status", "list_tickets_by_email"] and "ticket_id" in str(tool_result):
                            response_text = f"Here are your ticket details. {tool_result}"
                    except Exception as e:
                        tool_calls.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "error": str(e),
                        })
                        logger.error("support_tool_error", extra={"extra_data": {
                            "tool": tool_name,
                            "error": str(e),
                        }})

        if not tool_calls:
            response_text = answer.content or "I've processed your support request."

    except Exception as e:
        response_text = "I couldn't process your support request at this time. Please try again later."
        logger.error("support_agent_exception", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
        }})

    logger.info("support_agent_result", extra={"extra_data": {
        "response_preview": response_text[:200],
        "tool_calls_count": len(tool_calls),
    }})

    return {
        "current_agent": "support_ticket",
        "response": response_text,
        "next_agent": "evaluator",
        "visited_agents": state.get("visited_agents", []) + ["support_ticket"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, "handle_support_ticket"),
    }
