import ast
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from app.agents.route_utils import add_route
from app.agents.state import AgentName, AgentState, format_history
from app.agents.tool_loop import run_tool_loop, successful_results
from app.config import get_settings
from app.monitoring import get_logger
from app.tools.order_tools import (
    get_order_by_customer_name,
    get_order_by_id,
    get_order_items,
    get_orders_by_email,
    search_orders,
)

settings = get_settings()
logger = get_logger("order_agent")


def _parse_tool_result(tool_result) -> tuple[list[dict], str | None]:
    """Parse an order-tool result into (list of order dicts, order_id).

    Tools return a stringified dict or list of dicts (e.g. "{'order_id': ...}"
    or "[{'order_id': ...}, ...]"). This normalizes every order tool's output
    so order_data/order_id are populated regardless of which tool was called.
    """
    data = tool_result
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except Exception:
            return [], None

    if isinstance(data, dict):
        # get_order_items returns {"order_id": ..., "items": [...]} — keep it,
        # but plain error dicts are dropped.
        if "error" in data:
            return [], None
        if "order_id" not in data:
            return [], None
        return [data], data.get("order_id")
    if isinstance(data, list):
        orders = [d for d in data if isinstance(d, dict) and d.get("order_id")]
        return orders, orders[0].get("order_id") if orders else None
    return [], None


def _format_order_summary(orders: list[dict]) -> str | None:
    """Build a concise, grounded summary from parsed order data."""
    if not orders:
        return None
    if len(orders) == 1:
        o = orders[0]
        total = (o.get("payment") or {}).get("total")
        summary = (
            f"I found order {o.get('order_id')} for {o.get('customer_name')} "
            f"(placed on {o.get('order_date')}). Status: {o.get('status')}."
        )
        if total is not None:
            summary += f" Total: {total}."
        return summary
    details = ", ".join(f"{o.get('order_id')} ({o.get('status')})" for o in orders)
    total_spend = sum(
        ((o.get("payment") or {}).get("total") or 0)
        for o in orders
        if isinstance(o, dict)
    )
    summary = f"I found {len(orders)} orders for you: {details}."
    if total_spend:
        summary += f" Combined total spend: {round(total_spend, 2)}."
    return summary


ORDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an order assistant. Your job is to retrieve customer order information from the database using the available tools."
        "\n\nYou have access to the following tools:"
        "\n- search_orders: Search orders by order ID, customer name, or email (main lookup tool)"
        "\n- get_order_by_id: Retrieve order details by exact order ID"
        "\n- get_order_by_customer_name: Retrieve the most recent order for a customer by name"
        "\n- get_orders_by_email: Retrieve ALL orders for a customer by email address"
        "\n- get_order_items: Retrieve line items for a specific order"
        "\n\nCRITICAL RULES:"
        "\n1. When the user provides an email address, ALWAYS use get_orders_by_email. It returns a LIST of ALL orders for that email. Do NOT limit to one order."
        "\n2. When the user asks for 'all orders', 'order history', or 'total spend', you MUST:"
        "\n   - Retrieve ALL orders for the customer"
        "\n   - Calculate the total spend by summing the 'total' field from each order's payment info"
        "\n   - Present ALL orders and the calculated total"
        "\n3. Do not infer or make up order information. Use ONLY the data returned by the tools."
        "\n4. If the tool returns an empty list or error, clearly state that no orders were found."
        "\n5. You may call multiple tools in sequence (e.g. look up the order, then fetch its items). Tool results are returned to you so you can decide the next call."
        "\n6. {handoff_feedback}"
        ,
    ),
    (
        "user",
        "Conversation history:\n{history}\n\nQuestion:\n{question}\n\nAvailable tools: {tools}"
    ),
])


ORDER_TOOLS: list[BaseTool] = [
    search_orders,
    get_order_by_id,
    get_order_by_customer_name,
    get_orders_by_email,
    get_order_items,
]


def _fallback_email_lookup(query: str) -> tuple[list[dict], str | None, str | None]:
    """Direct email lookup when the LLM produced no tool calls.

    Returns (orders, order_id, response_text_or_None).
    """
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
    if not email_match:
        return [], None, None
    fallback_tool = next((t for t in ORDER_TOOLS if t.name == "get_orders_by_email"), None)
    if fallback_tool is None:
        return [], None, None
    try:
        tool_result = fallback_tool.invoke({"customer_email": email_match.group(0).lower()})
        parsed_orders, parsed_order_id = _parse_tool_result(tool_result)
        if "error" in str(tool_result).lower():
            return [], None, "I could not find any orders for that email address."
        if parsed_orders:
            summary = _format_order_summary(parsed_orders)
            return parsed_orders, parsed_order_id, summary
        return [], None, f"I found your orders using your email address. {tool_result}"
    except Exception as e:
        logger.error("order_email_fallback_error", extra={"extra_data": {"error": str(e)}})
        return [], None, None


def order_node(state: AgentState):
    from app.utils import get_chat_llm

    query = state["query"]
    history = format_history(state.get("messages", []))
    handoff_reason = state.get("handoff_reason", "")
    handoff_feedback = f"PREVIOUS ATTEMPT FEEDBACK: {handoff_reason}. Address this issue in your response." if handoff_reason else "This is your first attempt. Retrieve the order information completely."

    llm = get_chat_llm(tools=ORDER_TOOLS)

    try:
        answer, executed_calls = run_tool_loop(
            llm,
            ORDER_PROMPT,
            {
                "question": query,
                "history": history,
                "tools": ", ".join([t.name for t in ORDER_TOOLS]),
                "handoff_feedback": handoff_feedback,
            },
            ORDER_TOOLS,
        )

        # Aggregate results across ALL successful tool calls (previously only
        # the first call was inspected).
        order_data: list[dict] = []
        order_id = None
        any_tool_failed = False
        saw_not_found = False

        for tc in executed_calls:
            if "error" in tc:
                any_tool_failed = True
                continue
            parsed_orders, parsed_order_id = _parse_tool_result(tc["result"])
            if not parsed_orders and "not found" in str(tc["result"]).lower():
                saw_not_found = True
                continue
            if parsed_orders:
                known_ids = {o.get("order_id") for o in order_data}
                for o in parsed_orders:
                    if o.get("order_id") not in known_ids:
                        order_data.append(o)
                if order_id is None:
                    order_id = parsed_order_id

        if executed_calls:
            ok_results = successful_results(executed_calls)
            if order_data:
                response_text = _format_order_summary(order_data) or (
                    f"I found your order information. {ok_results[-1]['result']}"
                )
            elif saw_not_found or not ok_results:
                response_text = "I could not find any orders matching your information. Please double-check your name or email."
            else:
                response_text = f"I found your order information. {ok_results[-1]['result']}"
            if any_tool_failed and order_data:
                logger.warning("order_partial_tool_failure", extra={"extra_data": {
                    "failed": [tc["tool"] for tc in executed_calls if "error" in tc],
                }})
        else:
            # No tool calls at all — try direct email lookup before giving up.
            fb_orders, fb_order_id, fb_response = _fallback_email_lookup(query)
            if fb_response is not None:
                order_data, order_id, response_text = fb_orders, fb_order_id, fb_response
            else:
                response_text = answer.content if (answer and answer.content) else \
                    "I couldn't find matching order information. Please provide your order ID, name, or email."

    except Exception as e:
        order_data = []
        order_id = None
        response_text = "I couldn't retrieve the order information at this time. Please try again later."
        logger.error("order_agent_exception", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
        }})

    next_agent: AgentName | None = "evaluator"

    logger.info("order_agent_result", extra={"extra_data": {
        "order_id": order_id,
        "order_data_count": len(order_data) if isinstance(order_data, list) else 0,
        "order_data_preview": str(order_data)[:200],
        "response_preview": response_text[:200],
        "next_agent": next_agent,
    }})

    return {
        "current_agent": "order",
        "order_data": order_data,
        "order_id": order_id,
        "response": response_text,
        "next_agent": next_agent,
        "handoff_reason": "Order data retrieved, ready for evaluation.",
        "visited_agents": state.get("visited_agents", []) + ["order"],
        "handoff_count": state.get("handoff_count", 0),
        "route": add_route(state, "order", "lookup_order_handoff_to_evaluator"),
    }
