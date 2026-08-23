"""
Reranker
Uses the primary LLM to score retrieved chunks for relevance,
then keeps only the top-k.

The whole candidate set is scored in ONE batched prompt (previously one
sequential LLM call per chunk, which dominated query latency). Includes a
bounded in-memory cache to avoid re-scoring identical queries.
"""

import hashlib

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.monitoring import get_logger
from app.utils import get_chat_llm

logger = get_logger("reranker")

MAX_CACHE_ENTRIES = 256


class LLMReranker:
    """Score and filter retrieved documents using an LLM grader."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = get_chat_llm()
        self.prompt = ChatPromptTemplate.from_template(
            """You are a relevance grader for a retrieval-augmented QA system.

Question: {question}

Context chunks:
{chunks}

Rate the relevance of EACH numbered chunk to the question on a scale of 0 to 10.
Respond with EXACTLY one line per chunk in this format and nothing else:
1: <integer 0-10>
2: <integer 0-10>
..."""
        )
        self._cache: dict[str, list[tuple[Document, float]]] = {}

    def _cache_key(self, question: str, docs: list[Document]) -> str:
        ids = "|".join(
            d.metadata.get("chunk_id", d.metadata.get("document_id", "")) for d in docs
        )
        return hashlib.md5(f"{question}:{ids}".encode()).hexdigest()

    def _parse_scores(self, raw: str, expected: int) -> list[float]:
        """Parse 'N: score' lines into a fixed-length score list (0.0 fallback)."""
        scores = [0.0] * expected
        for line in raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            idx_part, _, score_part = line.partition(":")
            try:
                idx = int(idx_part.strip())
            except ValueError:
                continue
            if not 1 <= idx <= expected:
                continue
            token = score_part.strip().split()[0] if score_part.strip() else ""
            try:
                scores[idx - 1] = max(0.0, min(10.0, float(token)))
            except ValueError:
                continue
        return scores

    def _score_batch(self, question: str, documents: list[Document]) -> list[float]:
        chunks_block = "\n\n".join(
            f"[{i}] {doc.page_content[:800]}"
            for i, doc in enumerate(documents, start=1)
        )
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(question=question, chunks=chunks_block)
            )
            return self._parse_scores(response.content, len(documents))
        except Exception as e:
            logger.warning("rerank_batch_failed", extra={"extra_data": {
                "error": str(e),
                "docs": len(documents),
            }})
            return [0.0] * len(documents)

    def rerank(self, question: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        top_k = top_k or self.settings.rerank_top_k
        if not documents:
            return []

        key = self._cache_key(question, documents)
        if key in self._cache:
            cached = self._cache[key]
            return [doc for doc, _ in cached[:top_k]]

        scores = self._score_batch(question, documents)
        scored = []
        for doc, score in zip(documents, scores, strict=False):
            doc.metadata["rerank_score"] = score
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Bound memory: drop oldest entries beyond the cap.
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = scored

        return [doc for doc, _ in scored[:top_k]]
