from app.agents.state import AgentState, AgentName
from app.config import get_settings
from app.monitoring import get_logger
from app.tools.refund_tools import create_refund_request, get_refund_status, list_refunds_by_order, calculate_eligible_refund
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

settings = get_settings()
logger = get_logger("return_refund_agent")


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "return_refund",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


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
    ),
    (
        "user",
        "User request:\n{question}\n\nOrder data:\n{order_data}\n\nPolicy context:\n{context}\n\nAvailable tools: {tools}"
    ),
])


REFUND_TOOLS: list[BaseTool] = [
    calculate_eligible_refund,
    create_refund_request,
    get_refund_status,
    list_refunds_by_order,
]


def return_refund_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    order_data = state.get("order_data", {})
    context = state.get("context", [])

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    ).bind_tools(REFUND_TOOLS)

    try:
        chain = RETURN_REFUND_PROMPT | llm
        answer = chain.invoke({
            "question": query,
            "order_data": str(order_data) if order_data else "No order data available.",
            "context": "\n".join(f"[{i}] {c}" for i, c in enumerate(context, 1)) if context else "No policy context available.",
            "tools": ", ".join([t.name for t in REFUND_TOOLS]),
        })

        response_text = answer.content if answer.content else "I've processed your return/refund request."

        tool_calls = []
        if hasattr(answer, "tool_calls") and answer.tool_calls:
            for tool_call in answer.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                selected_tool = next((t for t in REFUND_TOOLS if t.name == tool_name), None)
                if selected_tool:
                    try:
                        tool_result = selected_tool.invoke(tool_args)
                        tool_calls.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                        })
                        logger.info("refund_tool_call", extra={"extra_data": {
                            "tool": tool_name,
                            "args": tool_args,
                            "result_preview": str(tool_result)[:200],
                        }})
                        if tool_name == "create_refund_request" and "refund_id" in str(tool_result):
                            response_text = f"I've created a refund request for you. {tool_result}"
                        elif tool_name in ["get_refund_status", "list_refunds_by_order"] and "refund_id" in str(tool_result):
                            response_text = f"Here are your refund details. {tool_result}"
                        elif tool_name == "calculate_eligible_refund":
                            response_text = f"Based on your order, here's the refund calculation. {tool_result}"
                    except Exception as e:
                        tool_calls.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "error": str(e),
                        })
                        logger.error("refund_tool_error", extra={"extra_data": {
                            "tool": tool_name,
                            "error": str(e),
                        }})

        if not tool_calls:
            response_text = answer.content or "I've processed your return/refund request."

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
        "tool_calls_count": len(tool_calls),
        "order_id": state.get("order_id"),
    }})

    return {
        "current_agent": "return_refund",
        "response": response_text,
        "return_refund_data": return_refund_summary,
        "next_agent": "evaluator",
        "visited_agents": state.get("visited_agents", []) + ["return_refund"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, "process_return_refund"),
    }
