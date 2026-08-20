import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.agents.state import AgentState
from app.config import get_settings
from app.monitoring import get_logger

settings = get_settings()
logger = get_logger("rag_agent")

KB_PATH = Path(__file__).resolve().parents[2] / "app" / "data" / "rag_knowledge_base.json"


def load_kb() -> list[dict]:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_documents(query: str, top_k: int = 3) -> list[Document]:
    kb = load_kb()
    query_terms = set(query.lower().split())

    scored = []
    for doc in kb:
        content = doc.get("content", "")
        title = doc.get("title", "")
        tags = " ".join(doc.get("tags", []))
        text = f"{title} {tags} {content}".lower()
        score = sum(1 for term in query_terms if term in text)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [
        Document(
            page_content=doc["content"],
            metadata={
                "title": doc.get("title", ""),
                "document_id": doc.get("document_id", ""),
                "score": float(score),
            },
        )
        for score, doc in top if score > 0
    ]


def _add_route(state: AgentState, action: str) -> list[dict]:
    route = list(state.get("route", []))
    route.append({
        "agent": "rag",
        "action": action,
        "timestamp": __import__("time").time(),
    })
    return route


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledge-base assistant. Answer using ONLY the provided context."
        " Do not add information from general knowledge."
        " IMPORTANT: Absence of mention does NOT imply the opposite is true."
        " If the context does not state whether something is true or false, do not make any claim about it."
    ),
    (
        "user",
        "Question:\n{question}\n\nContext:\n{context}"
    ),
])


def rag_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    documents = retrieve_documents(query)
    context = [doc.page_content for doc in documents]

    retrieved_documents = [
        {
            "document_id": doc.metadata.get("document_id", ""),
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content,
            "score": doc.metadata.get("score", 0.0),
        }
        for doc in documents
    ]

    logger.info("rag_docs_retrieved", extra={"extra_data": {
        "query": query[:200],
        "doc_count": len(retrieved_documents),
        "doc_ids": [d["document_id"] for d in retrieved_documents],
        "scores": [d["score"] for d in retrieved_documents],
    }})

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    try:
        chain = RAG_PROMPT | llm
        answer = chain.invoke({"question": query, "context": context})
        response_text = answer.content
        logger.info("rag_llm_response", extra={"extra_data": {
            "response_length": len(response_text) if response_text else 0,
        }})
    except Exception as e:
        response_text = "I couldn't retrieve an answer at this time."
        logger.error("rag_llm_error", extra={"extra_data": {
            "query": query[:200],
            "error": str(e),
        }})

    citations = [doc["document_id"] for doc in retrieved_documents]

    return {
        "current_agent": "rag",
        "response": response_text,
        "context": context,
        "citations": citations,
        "retrieved_documents": retrieved_documents,
        "next_agent": "evaluator",
        "visited_agents": state.get("visited_agents", []) + ["rag"],
        "handoff_count": state.get("handoff_count", 0),
        "route": _add_route(state, "retrieve_and_answer"),
    }
