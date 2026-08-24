"""
Policy RAG Agent — Agentic RAG sub-graph.

Flow (ported from production-rag-api):
    contextualize -> hybrid_retrieve -> rerank -> grade_context
        SUFFICIENT   -> generate_answer -> groundedness_check -> END (or rewrite)
        PARTIAL      -> retrieve_complementary -> rerank -> grade_context
        NONE         -> rewrite_query -> hybrid_retrieve (bounded by max_retrieval_retries)
        UNANSWERABLE -> "I don't have sufficient information..."

The compiled sub-graph is invoked by ``rag_node`` so the outer multi-agent
topology (router -> agents -> evaluator -> formatter) stays unchanged.
"""

import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.agents.route_utils import add_route
from app.agents.state import AgentState
from app.config import get_settings
from app.monitoring import get_logger
from app.utils import get_chat_llm

settings = get_settings()
logger = get_logger("rag_agent")


# === Sub-graph State ===

class RagState(TypedDict):
    question: str
    chat_history: str
    rewritten_question: str
    context: list[Document]
    answer: str
    context_state: str
    relevant_sources: list[str]
    missing_aspects: list[str]
    is_grounded: bool
    retry_count: int
    # Number of complementary-retrieval rounds already run. The PARTIAL branch
    # of grade_context used to loop forever on repeated PARTIAL grades because
    # nothing incremented or checked a bound.
    complementary_count: int
    max_retries: int
    model_used: str
    node_latencies: dict[str, float]
    trace: list[str]


# === Prompts (verbatim from production-rag-api) ===

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query contextualization agent for a closed-domain retrieval system. "
            "Given the conversation history and a follow-up question, rephrase it to be a standalone question. "
            "If the question is already standalone, return it unchanged. "
            "STRICTLY preserve the user's original intent and scope. "
            "Do NOT introduce new architectures, datasets, methods, or technical concepts from general knowledge. "
            "Do NOT answer the question.",
        ),
        ("human", "Chat history:\n{chat_history}\n\nFollow-up question: {input}"),
    ]
)

GRADE_CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """You are an evidence grader for a closed-domain retrieval system.
Assess whether the retrieved context is sufficient to answer the question.

Question: {question}

Retrieved context:
{context}

Classify the evidence as one of:
- SUFFICIENT: The context contains enough evidence to fully answer the question.
- PARTIAL: The context supports part of the question, but additional evidence from other sources is needed.
- NONE: The context does not meaningfully support the question.
- UNANSWERABLE: The corpus does not contain information relevant to the question.

Respond with ONLY the state name and a brief reason.
Format: STATE: <state_name>\nREASON: <brief reason>\nSOURCES: <comma-separated source names found>\nMISSING: <comma-separated missing aspects, or none>"""
)

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use ONLY the provided context to answer the question.
If the context does not contain enough information, say "I don't have sufficient information in the provided documents to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_template(
    """You are a grader assessing whether an answer is grounded in and supported by the given facts.

Facts:
{context}

Answer: {answer}

Return 'yes' if the answer is grounded in the facts and addresses the question.
Return 'no' otherwise. Respond with only 'yes' or 'no'."""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a conservative query rewriter for a closed-domain retrieval system.
The original question did not yield relevant documents. Rewrite it to be clearer and more specific,
but STRICTLY preserve the original user intent and scope.

Rules:
1. Do NOT introduce new architectures, datasets, methods, or technical concepts not present in the original question.
2. Do NOT expand a closed-domain comparison into concepts the user did not mention.
3. Keep the rewritten query concise and focused.
4. Return ONLY the rewritten query, nothing else.

Original question: {question}

Rewritten question:"""
)


# === Lazy singletons ===

@lru_cache(maxsize=1)
def _get_llm():
    return get_chat_llm()


@lru_cache(maxsize=1)
def _get_retriever():
    from app.retrieval import HybridRetriever
    return HybridRetriever()


@lru_cache(maxsize=1)
def _get_reranker():
    from app.reranker import LLMReranker
    return LLMReranker()


def _context_text(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def sanitize_documents(docs: list[Document], source: str) -> list[Document]:
    """Guard against indirect prompt injection hidden in retrieved documents.

    Retrieved content is untrusted (a poisoned corpus or indexed page can carry
    instruction-style payloads). Flagged chunks are replaced with a neutral
    marker before they reach any LLM prompt.
    """
    from app.security import security

    sanitized: list[Document] = []
    for doc in docs:
        is_safe, reason = security.sanitizer.check(doc.page_content)
        if not is_safe:
            logger.warning("rag_doc_injection_blocked", extra={"extra_data": {
                "source": doc.metadata.get("source", "unknown"),
                "reason": reason,
                "stage": source,
                "preview": doc.page_content[:200],
            }})
            doc = Document(
                page_content="[Content blocked by security scan (potential embedded instructions).]",
                metadata=doc.metadata,
            )
        sanitized.append(doc)
    return sanitized


# === Sub-graph builder ===

# === Routing (module-level & pure for testability) ===

MAX_COMPLEMENTARY_ROUNDS = 1


def route_after_grade_context(state: RagState) -> str:
    ctx_state = state.get("context_state", "NONE")
    if ctx_state == "SUFFICIENT":
        return "generate"
    if ctx_state == "PARTIAL":
        # Bounded: after one complementary round, generate from the best
        # available evidence instead of looping PARTIAL forever.
        if state.get("complementary_count", 0) < MAX_COMPLEMENTARY_ROUNDS:
            return "retrieve_complementary"
        return "generate"
    if ctx_state == "NONE":
        if state.get("retry_count", 0) < settings.max_retrieval_retries:
            return "rewrite"
        return "no_answer"
    return "no_answer"


def route_after_rewrite(state: RagState) -> str:
    return "hybrid_retrieve"


def route_after_answer_eval(state: RagState) -> str:
    if state.get("is_grounded"):
        return "end"
    if state.get("retry_count", 0) < settings.max_retrieval_retries:
        return "rewrite"
    return "no_answer"


def build_rag_subgraph():
    llm = _get_llm()

    def _record_latency(state: RagState, node_name: str, fn):
        start = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies = dict(state.get("node_latencies", {}))
        latencies[node_name] = round(elapsed_ms, 2)
        trace = list(state.get("trace", []))
        trace.append(f"{node_name}:{round(elapsed_ms, 2)}ms")
        return result, latencies, trace

    def contextualize(state: RagState) -> dict[str, Any]:
        def _run():
            history = state.get("chat_history", "")
            question = state["question"]
            if history and history != "No prior conversation.":
                chain = CONTEXTUALIZE_PROMPT | llm
                rewritten = chain.invoke({
                    "chat_history": history,
                    "input": question,
                }).content.strip()
            else:
                rewritten = question
            return {"rewritten_question": rewritten}

        result, latencies, trace = _record_latency(state, "contextualize", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def hybrid_retrieve(state: RagState) -> dict[str, Any]:
        def _run():
            question = state.get("rewritten_question") or state["question"]
            preserve_sources = state.get("relevant_sources", [])
            docs = _get_retriever().retrieve(
                question,
                preserve_sources=preserve_sources if preserve_sources else None,
            )
            return {"context": docs}

        result, latencies, trace = _record_latency(state, "hybrid_retrieve", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def rerank(state: RagState) -> dict[str, Any]:
        def _run():
            question = state.get("rewritten_question") or state["question"]
            docs = state.get("context", [])
            docs = sanitize_documents(docs, source="rerank")
            reranked = _get_reranker().rerank(question, docs)
            return {"context": reranked}

        result, latencies, trace = _record_latency(state, "rerank", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def grade_context(state: RagState) -> dict[str, Any]:
        def _run():
            question = state["question"]
            docs = state.get("context", [])
            if not docs:
                return {
                    "context_state": "NONE",
                    "relevant_sources": [],
                    "missing_aspects": ["any relevant evidence"],
                }

            context_block = "\n\n".join(
                f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )
            chain = GRADE_CONTEXT_PROMPT | llm
            resp = chain.invoke({
                "question": question,
                "context": context_block,
            }).content.strip()

            state_name = "NONE"
            sources: list[str] = []
            missing: list[str] = []

            for line in resp.splitlines():
                line = line.strip()
                if line.startswith("STATE:"):
                    state_name = line.split(":", 1)[1].strip().upper()
                elif line.startswith("SOURCES:"):
                    sources = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
                elif line.startswith("MISSING:"):
                    missing = [
                        s.strip()
                        for s in line.split(":", 1)[1].split(",")
                        if s.strip() and s.strip().lower() != "none"
                    ]

            if state_name not in {"SUFFICIENT", "PARTIAL", "NONE", "UNANSWERABLE"}:
                state_name = "NONE"

            return {
                "context_state": state_name,
                "relevant_sources": sources,
                "missing_aspects": missing,
            }

        result, latencies, trace = _record_latency(state, "grade_context", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def retrieve_complementary(state: RagState) -> dict[str, Any]:
        def _run():
            question = state.get("rewritten_question") or state["question"]
            missing = state.get("missing_aspects", [])
            existing = state.get("relevant_sources", [])
            docs = _get_retriever().retrieve_complementary(
                original_query=question,
                missing_aspects=missing,
                existing_sources=existing,
                top_k=settings.retrieval_top_k,
            )
            existing_context = list(state.get("context", []))
            seen_ids = {
                d.metadata.get("chunk_id") or id(d) for d in existing_context
            }
            combined = existing_context + [
                d for d in docs
                if (d.metadata.get("chunk_id") or id(d)) not in seen_ids
            ]
            return {
                "context": combined,
                "complementary_count": state.get("complementary_count", 0) + 1,
            }

        result, latencies, trace = _record_latency(state, "retrieve_complementary", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def rewrite_query(state: RagState) -> dict[str, Any]:
        def _run():
            question = state["question"]
            chain = REWRITE_PROMPT | llm
            rewritten = chain.invoke({"question": question}).content.strip()
            return {
                "rewritten_question": rewritten,
                "retry_count": state["retry_count"] + 1,
            }

        result, latencies, trace = _record_latency(state, "rewrite_query", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def generate_answer(state: RagState) -> dict[str, Any]:
        def _run():
            question = state["question"]
            docs = state.get("context", [])
            chain = GENERATE_PROMPT | llm
            answer = chain.invoke({
                "context": _context_text(docs),
                "question": question,
            }).content.strip()
            return {"answer": answer, "model_used": "primary"}

        result, latencies, trace = _record_latency(state, "generate", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def groundedness_check(state: RagState) -> dict[str, Any]:
        def _run():
            docs = state.get("context", [])
            chain = GROUNDEDNESS_PROMPT | llm
            resp = chain.invoke({
                "context": _context_text(docs),
                "answer": state["answer"],
            }).content.strip().lower()
            return {"is_grounded": resp.startswith("y")}

        result, latencies, trace = _record_latency(state, "groundedness_check", _run)
        return {**result, "node_latencies": latencies, "trace": trace}

    def handle_no_answer(state: RagState) -> dict[str, Any]:
        return {
            "answer": "I don't have sufficient information in the provided documents to answer that.",
            "model_used": "no_answer",
            "context_state": "UNANSWERABLE",
        }

    # --- Build Graph ---

    graph = StateGraph(RagState)

    graph.add_node("contextualize", contextualize)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade_context", grade_context)
    graph.add_node("retrieve_complementary", retrieve_complementary)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("groundedness_check", groundedness_check)
    graph.add_node("handle_no_answer", handle_no_answer)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "rerank")
    graph.add_edge("rerank", "grade_context")

    graph.add_conditional_edges(
        "grade_context",
        route_after_grade_context,
        {
            "generate": "generate_answer",
            "retrieve_complementary": "retrieve_complementary",
            "rewrite": "rewrite_query",
            "no_answer": "handle_no_answer",
        },
    )

    graph.add_edge("retrieve_complementary", "rerank")

    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"hybrid_retrieve": "hybrid_retrieve"},
    )

    graph.add_edge("generate_answer", "groundedness_check")

    graph.add_conditional_edges(
        "groundedness_check",
        route_after_answer_eval,
        {
            "end": END,
            "rewrite": "rewrite_query",
            "no_answer": "handle_no_answer",
        },
    )

    graph.add_edge("handle_no_answer", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_rag_subgraph():
    return build_rag_subgraph()


# === Outer multi-agent node ===

def rag_node(state: AgentState):
    query = state["query"]

    from app.agents.state import format_history
    chat_history = format_history(state.get("messages"))

    initial_state: RagState = {
        "question": query,
        "chat_history": chat_history,
        "rewritten_question": "",
        "context": [],
        "answer": "",
        "context_state": "NONE",
        "relevant_sources": [],
        "missing_aspects": [],
        "is_grounded": False,
        "retry_count": 0,
        "complementary_count": 0,
        "max_retries": settings.max_retrieval_retries,
        "model_used": "",
        "node_latencies": {},
        "trace": [],
    }

    try:
        result = get_rag_subgraph().invoke(initial_state)
    except Exception as e:
        logger.error("rag_pipeline_error", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
        }})
        result = {
            **initial_state,
            "answer": "I couldn't retrieve an answer at this time.",
            "model_used": "error",
            "context": [],
            "trace": [],
        }

    documents: list[Document] = result.get("context", [])

    retrieved_documents = [
        {
            "document_id": doc.metadata.get("document_id", ""),
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content,
            "score": float(doc.metadata.get("rerank_score", 0.0)),
        }
        for doc in documents
    ]

    logger.info("rag_docs_retrieved", extra={"extra_data": {
        "query": query[:200],
        "doc_count": len(retrieved_documents),
        "doc_ids": [d["document_id"] for d in retrieved_documents],
        "scores": [d["score"] for d in retrieved_documents],
        "context_state": result.get("context_state", "NONE"),
        "trace": result.get("trace", []),
    }})

    return {
        "current_agent": "rag",
        "response": result.get("answer", ""),
        "context": [doc.page_content for doc in documents],
        "citations": list({doc.metadata.get("document_id", "") for doc in documents} - {""}),
        "retrieved_documents": retrieved_documents,
        "context_state": result.get("context_state", "NONE"),
        "missing_aspects": result.get("missing_aspects", []),
        "relevant_sources": result.get("relevant_sources", []),
        "is_grounded": result.get("is_grounded", False),
        "rag_trace": result.get("trace", []),
        "next_agent": "evaluator",
        "visited_agents": [*state.get("visited_agents", []), "rag"],
        "handoff_count": state.get("handoff_count", 0),
        "route": add_route(state, "rag", "agentic_rag"),
    }
