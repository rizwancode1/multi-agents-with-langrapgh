from app.agents.state import AgentState, AgentName
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
        "Question:\n{question}\n\nAvailable tools: {tools}"
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
                        if tool_name == "search_orders" and "order_id" in str(tool_result):
                            try:
                                import ast
                                result_dict = ast.literal_eval(tool_result)
                                if isinstance(result_dict, dict) and "order_id" in result_dict:
                                    order_id = result_dict.get("order_id")
                                    order_data = [result_dict]
                                elif isinstance(result_dict, list) and result_dict:
                                    order_data = result_dict
                                    order_id = order_data[0].get("order_id") if order_data else None
                            except Exception:
                                pass
                        if tool_name == "get_orders_by_email" and "order_id" in str(tool_result):
                            try:
                                import ast
                                result_dict = ast.literal_eval(tool_result)
                                if isinstance(result_dict, list) and result_dict:
                                    order_data = result_dict
                                    order_id = order_data[0].get("order_id") if order_data else None
                            except Exception:
                                pass
                        if tool_name == "get_orders_by_email" and isinstance(tool_result, str) and tool_result.strip().startswith("[") :
                            try:
                                import ast
                                result_dict = ast.literal_eval(tool_result)
                                if isinstance(result_dict, list) and result_dict:
                                    order_data = result_dict
                                    order_id = order_data[0].get("order_id") if order_data else None
                            except Exception:
                                pass
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
                if "error" not in tc:
                    response_text = f"I found your order information. {tc.get('result', '')}"
                    break

        if not tool_calls:
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query)
            if email_match:
                try:
                    fallback_tool = next((t for t in ORDER_TOOLS if t.name == "get_orders_by_email"), None)
                    if fallback_tool:
                        tool_result = fallback_tool.invoke({"customer_email": email_match.group(0).lower()})
                        response_text = f"I found your orders using your email address. {tool_result}"
                        logger.info("order_email_fallback", extra={"extra_data": {
                            "email": email_match.group(0).lower(),
                            "result": str(tool_result)[:200],
                        }})
                        if "order_id" in str(tool_result):
                            try:
                                import ast
                                result_dict = ast.literal_eval(tool_result)
                                if isinstance(result_dict, list) and result_dict:
                                    order_data = result_dict
                                    order_id = order_data[0].get("order_id") if order_data else None
                            except Exception:
                                pass
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
