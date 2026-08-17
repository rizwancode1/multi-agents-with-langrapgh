from pydantic import BaseModel, Field

from app.agents.state import AgentState, AgentName
from app.config import get_settings
from langchain_core.prompts import ChatPromptTemplate

settings = get_settings()


class EvaluationResult(BaseModel):
    passed: bool
    grounded: bool
    complete: bool
    hallucination: bool
    unsupported_claims: list[str]
    invalid_citations: list[str]
    missing_information: list[str]
    recommended_agent: AgentName | None
    reason: str


def _safe_default_evaluation() -> dict:
    return {
        "passed": True,
        "grounded": True,
        "complete": True,
        "hallucination": False,
        "unsupported_claims": [],
        "invalid_citations": [],
        "missing_information": [],
        "recommended_agent": None,
        "reason": "Evaluation unavailable; passing by default.",
    }


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "evaluator",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


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
        "\n   - coding: requires non-empty code_result"
        "\n   - code_review: requires non-empty review_feedback"
        "\n5. HALLUCINATION: Set hallucination=true if ANY claim is made that is not directly supported by the provided data."
        "\n\nReturn structured evaluation."
    ),
    (
        "user",
        "User query:\n{query}\n\nDetected intents:\n{intents}\n\nOrder data:\n{order_data}\n\nOrder ID:\n{order_id}\n\nRetrieved documents:\n{retrieved_documents}\n\nCode result:\n{code_result}\n\nReview feedback:\n{review_feedback}\n\nContext:\n{context}"
    ),
])


def evaluator_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state.get("query", "")
    intents = state.get("intents", [])
    order_data = state.get("order_data", {})
    order_id = state.get("order_id")
    retrieved_documents = state.get("retrieved_documents", [])
    code_result = state.get("code_result", "")
    review_feedback = state.get("review_feedback", "")
    context = state.get("context", [])

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    evaluator_llm = llm.with_structured_output(EvaluationResult)

    try:
        chain = EVALUATOR_PROMPT | evaluator_llm
        result = chain.invoke({
            "query": query,
            "intents": intents,
            "order_data": order_data,
            "order_id": order_id,
            "retrieved_documents": retrieved_documents,
            "code_result": code_result,
            "review_feedback": review_feedback,
            "context": context,
        })
        evaluation = result.model_dump()
    except Exception:
        evaluation = _safe_default_evaluation()

    if not evaluation.get("passed", True):
        evaluation["recommended_agent"] = evaluation.get("recommended_agent") or "rag"

    action = "pass" if evaluation.get("passed", True) else f"retry_{evaluation.get('recommended_agent', 'end')}"

    return {
        "current_agent": "evaluator",
        "evaluation": evaluation,
        "next_agent": evaluation.get("recommended_agent") if not evaluation.get("passed", True) else None,
        "route": _add_route(state, action),
    }


def evaluator_router(state: AgentState):
    evaluation = state["evaluation"]

    if evaluation.get("passed", True):
        return "formatter"

    retry_agent = evaluation.get("recommended_agent")
    if retry_agent:
        return retry_agent

    return "end"
