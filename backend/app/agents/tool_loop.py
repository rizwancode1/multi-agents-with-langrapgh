"""
Multi-round tool execution loop for agent nodes.

The previous agents executed only the FIRST batch of tool calls emitted by
the LLM, so the model could never chain tools (e.g. get_order_by_id ->
get_order_items). This helper feeds tool results back into the conversation
and re-invokes the model until it produces a final answer or the round cap
is hit.
"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from app.monitoring import get_logger
from app.security import security

logger = get_logger("tool_loop")

MAX_TOOL_ROUNDS = 4


def _scan_tool_result(result_text: str) -> str:
    """Guard against indirect prompt injection arriving via tool output.

    Tool results are untrusted content (DB fields could contain attacker-
    controlled text). If they carry instruction-style payloads, replace them
    before the text is fed back into the LLM conversation.
    """
    is_safe, reason = security.sanitizer.check(result_text)
    if not is_safe:
        logger.warning("tool_result_injection_blocked", extra={"extra_data": {
            "reason": reason,
            "preview": result_text[:200],
        }})
        return ('{"error": "Tool output blocked by security scan '
                '(potential embedded instructions)."}')
    return result_text


def run_tool_loop(
    llm,
    prompt: ChatPromptTemplate | None,
    prompt_inputs: dict,
    tools: Sequence[BaseTool],
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[AIMessage | None, list[dict]]:
    """Invoke the LLM with tools bound and execute tool calls until done.

    Args:
        llm: chat model (already bound to ``tools``).
        prompt: rendered into the initial messages; may be None when the caller
            supplies a ready conversation via ``prompt_inputs`` (tests).
        prompt_inputs: variables for ``prompt``.
        tools: executable tool set matching what was bound to ``llm``.
        max_rounds: safety cap on LLM round-trips.

    Returns:
        (final_ai_message, executed_tool_calls) where executed_tool_calls is a
        list of {"tool", "args", "result"|"error"} dicts in call order.
    """
    messages: list[BaseMessage] = (
        list(prompt.invoke(prompt_inputs).to_messages()) if prompt is not None else []
    )
    executed: list[dict] = []
    ai_message: AIMessage | None = None

    for round_num in range(max_rounds):
        ai_message = llm.invoke(messages)
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            return ai_message, executed

        messages.append(ai_message)
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if selected_tool is None:
                result_text = f'{{"error": "Unknown tool: {tool_name}"}}'
                executed.append({"tool": tool_name, "args": tool_args, "error": result_text})
            else:
                try:
                    raw_result = selected_tool.invoke(tool_args)
                    result_text = raw_result if isinstance(raw_result, str) else str(raw_result)
                    result_text = _scan_tool_result(result_text)
                    executed.append({"tool": tool_name, "args": tool_args, "result": result_text})
                    logger.info("tool_call_executed", extra={"extra_data": {
                        "tool": tool_name,
                        "round": round_num + 1,
                        "result_preview": result_text[:200],
                    }})
                except Exception as e:
                    result_text = f'{{"error": {e!r}}}'
                    executed.append({"tool": tool_name, "args": tool_args, "error": str(e)})
                    logger.error("tool_call_failed", extra={"extra_data": {
                        "tool": tool_name,
                        "round": round_num + 1,
                        "error": str(e),
                    }})
            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call.get("id") or tool_name)
            )

    # Round cap reached — force a final answer without tools.
    logger.warning("tool_loop_round_cap", extra={"extra_data": {"rounds": max_rounds}})
    ai_message = llm.invoke(messages)
    return ai_message, executed


def successful_results(executed: list[dict]) -> list[dict]:
    """Return only the tool-call records that did not error."""
    return [tc for tc in executed if "error" not in tc]
