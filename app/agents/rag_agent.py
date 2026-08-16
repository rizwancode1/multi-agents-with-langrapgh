import json
import re
from pathlib import Path

from langchain_core.documents import Document

from app.agents.state import AgentState
from app.config import get_settings

settings = get_settings()

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
            metadata={"title": doc.get("title", ""), "document_id": doc.get("document_id", "")},
        )
        for score, doc in top if score > 0
    ]


def rag_node(state: AgentState):
    from langchain_openai import ChatOpenAI

    query = state["query"]
    documents = retrieve_documents(query)
    context = [doc.page_content for doc in documents]

    llm = ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=settings.primary_model,
        temperature=0,
    )

    answer = llm.invoke(
        f"""
Answer the user using ONLY the provided context.

Question:
{query}

Context:
{context}
"""
    )

    citations = [doc.metadata.get("document_id", "") for doc in documents]

    return {
        "current_agent": "rag",
        "response": answer.content,
        "context": context,
        "citations": citations,
        "visited_agents": state.get("visited_agents", []) + ["rag"],
        "handoff_count": state.get("handoff_count", 0),
    }
