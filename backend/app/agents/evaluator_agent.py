from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.agents.route_utils import add_route
from app.agents.state import AgentName, AgentState, format_history
from app.config import get_settings
from app.monitoring import get_logger

settings = get_settings()
logger = get_logger("evaluator_agent")


class EvaluationResult(BaseModel):
    passed: bool
    grounded: bool
    complete: bool
    hallucination: bool
    user_action_required: bool = Field(
        default=False,
        description=(
            "True when the correct next step is for the USER to provide "
            "information (e.g. their email or order ID), so the turn should END."
        ),
    )
    unsupported_claims: list[str]
    invalid_citations: list[str]
    missing_information: list[str]
    recommended_agent: AgentName | None
    reason: str


def _safe_default_evaluation() -> dict:
    # Fail-OPEN so an evaluator outage never blocks users, but the result is
    # explicitly flagged (evaluation_available=False + metric) instead of the
    # previous silent pass.
    return {
        "passed": True,
        "grounded": True,
        "complete": True,
        "hallucination": False,
        "user_action_required": False,
        "unsupported_claims": [],
        "invalid_citations": [],
        "missing_information": [],
        "recommended_agent": None,
        "reason": "Evaluation unavailable; passing by default.",
        "evaluation_available": False,
    }


EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict evaluator. Assess whether the current agent outputs are safe, grounded, and complete."
        "\n\nCRITICAL RULES:"
        "\n1. ABSENCE OF MENTION IS NOT EVIDENCE. If the context does not state whether something is true or false, you must mark any claim asserting the opposite as UNSUPPORTED."
        "\n   - Example: If the context only says \"Clearance items are non-refundable\" but does NOT say whether ORD-1007 items are clearance, the claim \"ORD-1007 items are not clearance\" is UNSUPPORTED."
        "\n   - Example: If order_data does not include item categories, any claim about item types is UNSUPPORTED."
        "\n2. GROUNDING: Every factual claim must be directly supported by the provided data. No inference from absence is allowed."
        "\n3. CITATIONS: Mark any citation document_id that is irrelevant to the user's query as invalid. For example, a security document should not be cited for an order-status answer."
        "\n4. COMPLETENESS: Check whether all detected intents are addressed."
        "\n   - order_status: requires order_data with matching order_id"
        "\n   - shipping_policy: requires retrieved_documents covering shipping"
        "\n   - return_policy / refund_policy: requires retrieved_documents covering returns/refunds"
        "\n   - account_security: requires retrieved_documents covering security"
        "\n   - support_ticket / complaint: requires a response acknowledging the issue"
        "\n   - return_refund: requires return_refund_data with eligibility and reasoning"
        "\n4b. AWAITING USER INPUT IS A VALID COMPLETED TURN. If the agent's response is a legitimate request for information ONLY the user can provide"
        "\n   (e.g. their email address or order ID, required by privacy policy before revealing account data), then set user_action_required=true,"
        "\n   complete=true, passed=true, recommended_agent=null. Do NOT retry the agent for asking."
        "\n   Retrying is only for when the agent COULD have answered from the available data but failed to."
        "\n5. HALLUCINATION: Set hallucination=true if ANY claim is made that is not directly supported by the provided data."
        "\n6. FEEDBACK: If the response is incomplete, provide specific, actionable feedback in 'reason' so the next agent attempt can address the gap."
        "\n7. If the agent has already been retried 2+ times for the same issue, consider passing rather than retrying indefinitely."
        "\n   - The 'retry_count' reflects how many attempts have been made. If it is 2 or more, prefer passing."
        "\n\nReturn structured evaluation."
    ),
    (
        "user",
        "Conversation history:\n{history}\n\nUser query:\n{query}\n\nDetected intents:\n{intents}\n\nAgent response:\n{response}\n\nHandoff feedback:\n{handoff_reason}\n\nOrder data:\n{order_data}\n\nOrder ID:\n{order_id}\n\nRetrieved documents:\n{retrieved_documents}\n\nReturn/Refund data:\n{return_refund_data}\n\nContext:\n{context}\n\nRetry count:\n{retry_count}"
    ),
])


def _looks_like_user_input_request(response: str, state: AgentState) -> bool:
    """Heuristic fallback for 'the agent asked the user for their identifier'.

    Complements the LLM's user_action_required flag so that even a weak
    evaluator model can never trap the request in an endless retry loop when
    the specialist correctly asked for missing info (privacy verification).
    """
    if not response or "?" not in response:
        return False
    r = response.lower()
    asks_identifier = "email" in r or "order id" in r or "order-id" in r
    has_data = bool(
        state.get("order_data")
        or state.get("return_refund_data")
        or state.get("retrieved_documents")
    )
    return asks_identifier and not has_data


def evaluator_node(state: AgentState):
    from app.utils import get_chat_llm

    query = state.get("query", "")
    intents = state.get("intents") or []
    history = format_history(state.get("messages", []))
    order_data = state.get("order_data", {})
    order_id = state.get("order_id")
    retrieved_documents = state.get("retrieved_documents", [])
    return_refund_data = state.get("return_refund_data", {})
    context = state.get("context", [])
    response = state.get("response", "")
    handoff_reason = state.get("handoff_reason", "")
    retry_count = state.get("handoff_count", 0)

    evaluator_llm = get_chat_llm(schema=EvaluationResult)

    try:
        chain = EVALUATOR_PROMPT | evaluator_llm
        result = chain.invoke({
            "query": query,
            "history": history,
            "intents": intents,
            "response": response,
            "handoff_reason": handoff_reason,
            "order_data": order_data,
            "order_id": order_id,
            "retrieved_documents": retrieved_documents,
            "return_refund_data": return_refund_data,
            "context": context,
            "retry_count": retry_count,
        })
        evaluation = result.model_dump()
        logger.info("evaluator_result", extra={"extra_data": {
            "passed": evaluation.get("passed"),
            "grounded": evaluation.get("grounded"),
            "complete": evaluation.get("complete"),
            "hallucination": evaluation.get("hallucination"),
            "user_action_required": evaluation.get("user_action_required"),
            "recommended_agent": evaluation.get("recommended_agent"),
            "reason": evaluation.get("reason"),
        }})
    except Exception as e:
        evaluation = _safe_default_evaluation()
        from app.monitoring import get_metrics
        get_metrics().record_request(latency_ms=0, error=True)
        logger.error("evaluator_exception_fail_open", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
            "evaluation_available": False,
        }})

    # Deterministic override: a clarification question that asks the user for
    # their own identifier is the CORRECT end of turn — never retry it.
    awaiting_user_input = evaluation.get("user_action_required", False) or (
        not evaluation.get("hallucination", False)
        and _looks_like_user_input_request(state.get("response", ""), state)
    )
    if awaiting_user_input and not evaluation.get("passed", True):
        logger.info("evaluator_awaiting_user_input_override", extra={"extra_data": {
            "query": query[:200],
            "original_reason": evaluation.get("reason", ""),
        }})
        evaluation.update({
            "passed": True,
            "complete": True,
            "recommended_agent": None,
            "user_action_required": True,
            "reason": (evaluation.get("reason") or "") + " [Awaiting user-provided identifier — ending turn.]",
        })

    if not evaluation.get("passed", True):
        evaluation["recommended_agent"] = evaluation.get("recommended_agent") or "policy_rag"

    action = "pass" if evaluation.get("passed", True) else f"retry_{evaluation.get('recommended_agent', 'end')}"

    logger.info("evaluator_routing", extra={"extra_data": {
        "passed": evaluation.get("passed"),
        "next_agent": evaluation.get("recommended_agent") if not evaluation.get("passed", True) else None,
        "action": action,
    }})

    return {
        "current_agent": "evaluator",
        "evaluation": evaluation,
        "next_agent": evaluation.get("recommended_agent") if not evaluation.get("passed", True) else None,
        "handoff_reason": evaluation.get("reason", ""),
        "route": add_route(state, "evaluator", action),
    }


def evaluator_router(state: AgentState):
    from app.agents.graph import MAX_HANDOFFS_PER_REQUEST, MAX_VISITS_PER_AGENT

    evaluation = state["evaluation"]

    if evaluation.get("passed", True):
        return "formatter"

    retry_agent = evaluation.get("recommended_agent")
    if not retry_agent:
        return "end"

    # Same safety caps as agent_router: never bounce to an agent that has
    # already been tried too many times, or once the handoff budget is spent.
    visited = state.get("visited_agents", [])
    if sum(1 for a in visited if a == retry_agent) >= MAX_VISITS_PER_AGENT:
        return "formatter"
    if state.get("handoff_count", 0) >= MAX_HANDOFFS_PER_REQUEST:
        return "formatter"

    return retry_agent
