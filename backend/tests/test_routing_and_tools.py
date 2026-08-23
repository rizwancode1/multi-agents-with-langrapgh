"""
Unit tests for multi-agent orchestration logic (no network, no LLM).

Covers the previously untested core: fallback routing, handoff caps and
validation in the graph router, evaluator retry caps, order-tool result
parsing, the RAG sub-graph loop bounds, and the multi-round tool executor.
"""

import pytest

from app.agents.graph import (
    AGENT_CAPABILITIES,
    MAX_HANDOFFS_PER_REQUEST,
    MAX_VISITS_PER_AGENT,
    resolve_next_agent,
)
from app.agents.evaluator_agent import _safe_default_evaluation, evaluator_router
from app.agents.order_agent import _format_order_summary, _parse_tool_result
from app.agents.rag_agent import route_after_answer_eval, route_after_grade_context
from app.agents.router_agent import _extract_intents_fallback, _fallback_route


# === Router fallbacks ===

def test_fallback_route_detects_order_id_query():
    assert _fallback_route("Where is my ORD-1002?")["agent"] == "order"


def test_fallback_route_action_beats_policy():
    # "I want a refund" is actionable — must NOT go to policy_rag.
    assert _fallback_route("this product arrived broken, i want a refund")["agent"] == "support_ticket"


def test_fallback_route_policy_default():
    assert _fallback_route("what is your return policy?")["agent"] == "policy_rag"
    assert _fallback_route("hello there")["agent"] == "policy_rag"  # safe default


def test_extract_intents_fallback_labels():
    intents = _extract_intents_fallback("my order never shipped")
    assert "order_status" in intents


# === Graph handoff enforcement ===

def base_state(**overrides):
    state = {
        "next_agent": None,
        "current_agent": None,
        "visited_agents": [],
        "handoff_count": 0,
    }
    state.update(overrides)
    return state


def test_resolve_routes_to_requested_specialist():
    state = base_state(next_agent="evaluator")
    assert resolve_next_agent(state) == "evaluator"


def test_resolve_rejects_unknown_agent():
    assert resolve_next_agent(base_state(next_agent="nonexistent")) == "evaluator"


def test_resolve_blocks_disallowed_specialist_handoff():
    # order -> return_refund is not in order's can_handoff_to.
    state = base_state(current_agent="order", next_agent="return_refund")
    assert resolve_next_agent(state) == "evaluator"


def test_resolve_allows_privileged_sources_any_target():
    state = base_state(current_agent="router", next_agent="order")
    assert resolve_next_agent(state) == "order"


def test_resolve_caps_visits_per_agent():
    visited = ["policy_rag"] * MAX_VISITS_PER_AGENT
    state = base_state(current_agent="router", next_agent="policy_rag", visited_agents=visited)
    assert resolve_next_agent(state) == "formatter"


def test_resolve_caps_total_handoffs():
    state = base_state(
        current_agent="router",
        next_agent="order",
        handoff_count=MAX_HANDOFFS_PER_REQUEST,
    )
    assert resolve_next_agent(state) == "formatter"


def test_capabilities_declare_valid_targets_only():
    for name, caps in AGENT_CAPABILITIES.items():
        for target in caps["can_handoff_to"]:
            assert target == "evaluator" or target in AGENT_CAPABILITIES, (
                f"{name} declares invalid handoff target {target}"
            )


# === Evaluator router ===

def test_evaluator_router_pass_goes_to_formatter():
    assert evaluator_router({"evaluation": {"passed": True}}) == "formatter"


def test_evaluator_router_no_recommendation_ends():
    assert evaluator_router({"evaluation": {"passed": False, "recommended_agent": None}}) == "end"


def test_evaluator_router_respects_visit_cap():
    visited = ["order"] * MAX_VISITS_PER_AGENT
    state = {"evaluation": {"passed": False, "recommended_agent": "order"}, "visited_agents": visited}
    assert evaluator_router(state) == "formatter"


def test_safe_default_is_fail_open_but_flagged():
    evaluation = _safe_default_evaluation()
    assert evaluation["passed"] is True
    assert evaluation["evaluation_available"] is False


# === Order tool result parsing ===

def test_parse_tool_result_single_order_dict():
    orders, order_id = _parse_tool_result(str({
        "order_id": "ORD-1001", "customer_name": "A", "status": "shipped",
    }))
    assert len(orders) == 1
    assert order_id == "ORD-1001"


def test_parse_tool_result_list_of_orders():
    data = str([
        {"order_id": "ORD-1001", "status": "shipped"},
        {"order_id": "ORD-1002", "status": "delivered"},
    ])
    orders, order_id = _parse_tool_result(data)
    assert [o["order_id"] for o in orders] == ["ORD-1001", "ORD-1002"]
    assert order_id == "ORD-1001"


def test_parse_tool_result_error_dict_yields_empty():
    orders, order_id = _parse_tool_result(str({"error": "Order not found"}))
    assert orders == [] and order_id is None


def test_parse_tool_result_items_payload_kept():
    payload = str({"order_id": "ORD-1", "items": [{"name": "Keyboard"}]})
    orders, order_id = _parse_tool_result(payload)
    assert orders and order_id == "ORD-1"


def test_format_order_summary_multi_order_totals():
    summary = _format_order_summary([
        {"order_id": "ORD-1", "status": "shipped", "payment": {"total": 50.0}},
        {"order_id": "ORD-2", "status": "delivered", "payment": {"total": 25.0}},
    ])
    assert "ORD-1" in summary and "ORD-2" in summary
    assert "75.0" in summary


# === RAG sub-graph route bounds ===

def rag_state(**overrides):
    state = {
        "context_state": "NONE",
        "retry_count": 0,
        "complementary_count": 0,
    }
    state.update(overrides)
    return state


def test_partial_first_round_retrieves_complementary():
    assert route_after_grade_context(rag_state(context_state="PARTIAL")) == "retrieve_complementary"


def test_partial_second_round_generates_instead_of_looping():
    state = rag_state(context_state="PARTIAL", complementary_count=1)
    assert route_after_grade_context(state) == "generate"


def test_none_rewrites_until_budget_spent():
    assert route_after_grade_context(rag_state()) == "rewrite"
    assert route_after_grade_context(rag_state(retry_count=1)) == "no_answer"


def test_ungrounded_answer_loops_bounded():
    assert route_after_answer_eval(rag_state(is_grounded=False)) == "rewrite"
    assert route_after_answer_eval(rag_state(is_grounded=False, retry_count=5)) == "no_answer"


# === Tool loop (multi-round chaining) ===

class ScriptedLLM:
    """Returns queued AIMessages; records the message history it receives."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


class FakeTool:
    def __init__(self, name):
        self.name = name

    def invoke(self, args):
        return f"{self.name}({args}) ok"


def fake_ai(tool_calls=None, content=""):
    msg = type("AIMessage", (), {})()
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def test_tool_loop_executes_and_feeds_results_back():
    llm = ScriptedLLM([
        fake_ai([{"name": "tool_a", "args": {"x": 1}, "id": "c1"}]),
        fake_ai(content="final answer"),
    ])
    from app.agents.tool_loop import run_tool_loop
    answer, executed = run_tool_loop(llm, None, {}, [FakeTool("tool_a")])

    assert executed == [{"tool": "tool_a", "args": {"x": 1}, "result": "tool_a({'x': 1}) ok"}]
    assert len(llm.calls) == 2
    # Second call must include the ToolMessage result of round one.
    second_call_contents = [
        getattr(m, "content", "") if not isinstance(m, dict) else ""
        for m in llm.calls[1]
    ]
    assert any("tool_a" in c and "ok" in c for c in map(str, second_call_contents))


def test_tool_loop_stops_when_no_tool_calls():
    llm = ScriptedLLM([fake_ai(content="done immediately")])
    from app.agents.tool_loop import run_tool_loop
    answer, executed = run_tool_loop(llm, None, {}, [FakeTool("tool_a")])
    assert answer.content == "done immediately"
    assert executed == []
    assert len(llm.calls) == 1


def test_tool_loop_survives_unknown_and_failing_tools():
    llm = ScriptedLLM([
        fake_ai([
            {"name": "ghost_tool", "args": {}, "id": "g1"},
            {"name": "boom", "args": {}, "id": "b1"},
        ]),
        fake_ai(content="recovered"),
    ])

    class Boom:
        name = "boom"

        def invoke(self, args):
            raise RuntimeError("kaput")

    from app.agents.tool_loop import run_tool_loop, successful_results
    _, executed = run_tool_loop(llm, None, {}, [Boom()])
    assert len(executed) == 2
    assert successful_results(executed) == []


@pytest.mark.parametrize("text,expected", [
    ("What is the status of ORD-1004?", "order"),
    ("How long does standard shipping take?", "policy_rag"),
])
def test_fallback_routing_parametrized(text, expected):
    assert _fallback_route(text)["agent"] == expected
