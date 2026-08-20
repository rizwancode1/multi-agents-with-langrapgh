import ast
import re

from app.agents.state import AgentState, AgentName, format_history
from app.config import get_settings
from app.monitoring import get_logger
from app.tools.order_tools import search_orders, get_order_by_id, get_order_by_customer_name, get_orders_by_email, get_order_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

settings = get_settings()
logger = get_logger("order_agent")


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "order",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


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
        if "error" in data or "order_id" not in data:
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
    return f"I found {len(orders)} orders for you: {details}."


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
        "\n5. {handoff_feedback}"
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


def order_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    history = format_history(state.get("messages", []))
    handoff_reason = state.get("handoff_reason", "")
    handoff_feedback = f"PREVIOUS ATTEMPT FEEDBACK: {handoff_reason}. Address this issue in your response." if handoff_reason else "This is your first attempt. Retrieve the order information completely."

    tool_calls = []
    order_data = []
    order_id = None

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    ).bind_tools(ORDER_TOOLS)

    try:
        chain = ORDER_PROMPT | llm
        answer = chain.invoke({
            "question": query,
            "history": history,
            "tools": ", ".join([t.name for t in ORDER_TOOLS]),
            "handoff_feedback": handoff_feedback,
        })

        response_text = answer.content if answer.content else "I've looked up the order information."

        if hasattr(answer, "tool_calls") and answer.tool_calls:
            for tool_call in answer.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                selected_tool = next((t for t in ORDER_TOOLS if t.name == tool_name), None)
                if selected_tool:
                    try:
                        tool_result = selected_tool.invoke(tool_args)
                        tool_calls.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result": tool_result,
                        })
                        logger.info("order_tool_call", extra={"extra_data": {
                            "tool": tool_name,
                            "args": tool_args,
                            "result_preview": str(tool_result)[:200],
                        }})
                        parsed_orders, parsed_order_id = _parse_tool_result(tool_result)
                        if parsed_orders:
                            order_data = parsed_orders
                            if order_id is None:
                                order_id = parsed_order_id
                    except Exception as e:
                        tool_calls.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "error": str(e),
                        })
                        logger.error("order_tool_error", extra={"extra_data": {
                            "tool": tool_name,
                            "error": str(e),
                        }})

        if tool_calls:
            for tc in tool_calls:
                if "error" in tc:
                    continue
                if "error" in str(tc.get("result", "")).lower():
                    response_text = "I could not find any orders matching your information. Please double-check your name or email."
                elif order_data:
                    summary = _format_order_summary(order_data)
                    response_text = summary or f"I found your order information. {tc.get('result', '')}"
                else:
                    response_text = f"I found your order information. {tc.get('result', '')}"
                break

        if not tool_calls:
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
            if email_match:
                try:
                    fallback_tool = next((t for t in ORDER_TOOLS if t.name == "get_orders_by_email"), None)
                    if fallback_tool:
                        tool_result = fallback_tool.invoke({"customer_email": email_match.group(0).lower()})
                        parsed_orders, parsed_order_id = _parse_tool_result(tool_result)
                        if parsed_orders:
                            order_data = parsed_orders
                            order_id = parsed_order_id
                        if "error" in str(tool_result).lower():
                            response_text = "I could not find any orders for that email address."
                        elif order_data:
                            summary = _format_order_summary(order_data)
                            response_text = summary or f"I found your orders using your email address. {tool_result}"
                        else:
                            response_text = f"I found your orders using your email address. {tool_result}"
                        logger.info("order_email_fallback", extra={"extra_data": {
                            "email": email_match.group(0).lower(),
                            "result": str(tool_result)[:200],
                        }})
                except Exception:
                    pass

    except Exception as e:
        response_text = "I couldn't retrieve the order information at this time. Please try again later."
        logger.error("order_agent_exception", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
        }})

    next_agent: AgentName | None = "evaluator"
    handoff_reason = "Order data retrieved, ready for evaluation."

    logger.info("order_agent_result", extra={"extra_data": {
        "order_id": order_id,
        "order_data_count": len(order_data) if isinstance(order_data, list) else (1 if order_data else 0),
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
        "handoff_reason": handoff_reason,
        "visited_agents": state.get("visited_agents", []) + ["order"],
        "handoff_count": state.get("handoff_count", 0) + 1,
        "route": _add_route(state, f"lookup_order_handoff_to_{next_agent or 'evaluator'}"),
    }
