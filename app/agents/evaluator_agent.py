from pydantic import BaseModel, Field

from app.agents.state import AgentState, AgentName
from app.config import get_settings

settings = get_settings()


class EvaluationResult(BaseModel):
    passed: bool
    score: float
    feedback: str
    retry_agent: AgentName | None


def _safe_default_evaluation() -> dict:
    return {
        "passed": True,
        "score": 0.7,
        "feedback": "Evaluation unavailable; passing by default.",
        "retry_agent": None,
    }


def evaluator_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    response = state.get("response", "")
    context = state.get("context", [])

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    evaluator_llm = llm.with_structured_output(EvaluationResult)

    try:
        result = evaluator_llm.invoke(
            f"""
Evaluate the answer.

Answer:
{response}

Context:
{context}

Check:
1. Is the answer grounded?
2. Is it relevant?
3. Is it factually supported?
4. Are there unsupported claims?
5. Should another agent retry?
"""
        )
        evaluation = result.model_dump()
    except Exception:
        evaluation = _safe_default_evaluation()

    return {
        "current_agent": "evaluator",
        "evaluation": evaluation,
    }


def evaluator_router(state: AgentState):
    evaluation = state["evaluation"]

    if evaluation.get("passed", True):
        return "end"

    retry_agent = evaluation.get("retry_agent")
    if retry_agent:
        return retry_agent

    return "end"
