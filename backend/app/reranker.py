"""
Reranker
Uses the primary LLM to score retrieved chunks for relevance,
then keeps only the top-k. Includes a simple in-memory cache to avoid
re-scoring identical queries.
"""

import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings


class LLMReranker:
    """Score and filter retrieved documents using an LLM grader."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.primary_model,
            temperature=0,
            api_key=self.settings.openrouter_api_key or "sk-placeholder",
            base_url=self.settings.openrouter_base_url,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """You are a relevance grader for a retrieval-augmented QA system.

Question: {question}

Context chunk:
{context}

Rate the relevance of this chunk to the question on a scale of 0 to 10.
Respond with ONLY a single integer between 0 and 10.
Do not include any other text."""
        )
        self._cache: dict[str, List[tuple[Document, float]]] = {}

    def _cache_key(self, question: str, docs: List[Document]) -> str:
        ids = "|".join(
            d.metadata.get("chunk_id", d.metadata.get("document_id", "")) for d in docs
        )
        return hashlib.md5(f"{question}:{ids}".encode()).hexdigest()

    def rerank(self, question: str, documents: List[Document], top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k or self.settings.rerank_top_k
        if not documents:
            return []

        key = self._cache_key(question, documents)
        if key in self._cache:
            cached = self._cache[key]
            return [doc for doc, _ in cached[:top_k]]

        scored: List[tuple[Document, float]] = []
        for doc in documents:
            try:
                response = self.llm.invoke(
                    self.prompt.format_messages(question=question, context=doc.page_content)
                )
                score = float(response.content.strip().split()[0])
                score = max(0.0, min(10.0, score))
            except Exception:
                score = 0.0
            doc.metadata["rerank_score"] = score
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        self._cache[key] = scored
        return [doc for doc, _ in scored[:top_k]]
