"""
Hybrid Retrieval
Combines dense (ChromaDB / PGVector) and sparse (BM25) retrieval
with Reciprocal Rank Fusion. Supports complementary retrieval for
partial evidence gathering.
"""

import hashlib
from typing import List, Optional, Dict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import get_settings
from app.ingestion import get_ingestion_pipeline
from app.utils import bm25_tokenize


class HybridRetriever:
    """Dense + sparse hybrid retriever with RRF fusion."""

    def __init__(self):
        self.settings = get_settings()
        self.ingestion = get_ingestion_pipeline()
        self.vector_store = self.ingestion.load_vector_store()
        self.bm25_payload = self.ingestion.load_bm25_index()

    def _dense_search(self, query: str, k: int, filter_dict: Optional[Dict] = None) -> List[tuple[Document, float]]:
        kwargs = {"k": k * 2}
        if filter_dict:
            kwargs["filter"] = filter_dict
        docs = self.vector_store.similarity_search_with_score(query, **kwargs)
        return [(doc, score) for doc, score in docs]

    def _sparse_search(self, query: str, k: int) -> List[tuple[Document, float]]:
        if not self.bm25_payload:
            return []
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return []

        bm25: BM25Okapi = self.bm25_payload["bm25"]
        docs: List[Document] = self.bm25_payload["docs"]

        tokenized_query = bm25_tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        ranked = sorted(
            [(docs[i], float(scores[i])) for i in range(len(docs))],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[: k * 2]

    def _rrf_fuse(
        self,
        dense_results: List[tuple[Document, float]],
        sparse_results: List[tuple[Document, float]],
        top_k: int = 5,
        k: int = 60,
        preserve_sources: Optional[List[str]] = None,
    ) -> List[Document]:
        """Reciprocal Rank Fusion with optional source diversity preservation."""
        scores: dict[str, tuple[Document, float]] = {}

        def _add(results: List[tuple[Document, float]], weight: float = 1.0):
            for rank, (doc, _) in enumerate(results, start=1):
                key = (
                    doc.metadata.get("chunk_id")
                    or doc.metadata.get("doc_id")
                    or doc.metadata.get("document_id")
                    or hashlib.md5(doc.page_content.encode()).hexdigest()
                )
                if key not in scores:
                    scores[key] = (doc, 0.0)
                scores[key] = (
                    scores[key][0],
                    scores[key][1] + weight / (k + rank),
                )

        _add(dense_results, weight=1.0)
        _add(sparse_results, weight=1.0)

        fused = sorted(scores.values(), key=lambda x: x[1], reverse=True)

        if preserve_sources:
            selected: List[Document] = []
            seen_sources = set()
            for doc, score in fused:
                source = doc.metadata.get("source", "")
                if source in preserve_sources and source not in seen_sources:
                    selected.append(doc)
                    seen_sources.add(source)
                elif source not in preserve_sources:
                    selected.append(doc)
            return selected[:top_k]

        return [doc for doc, score in fused[:top_k]]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        preserve_sources: Optional[List[str]] = None,
    ) -> List[Document]:
        """Retrieve top-k documents using hybrid dense + BM25 search."""
        top_k = top_k or self.settings.retrieval_top_k
        dense = self._dense_search(query, top_k, filter_dict=filter_dict)
        sparse = self._sparse_search(query, top_k)
        return self._rrf_fuse(dense, sparse, top_k=top_k, preserve_sources=preserve_sources)

    def retrieve_complementary(
        self,
        original_query: str,
        missing_aspects: List[str],
        existing_sources: List[str],
        top_k: int = 5,
    ) -> List[Document]:
        """Retrieve complementary evidence for missing aspects while avoiding already-seen sources."""
        all_docs: List[Document] = []
        seen_ids = set()

        for aspect in missing_aspects:
            aspect_query = f"{original_query} {aspect}".strip()
            dense = self._dense_search(aspect_query, top_k)
            sparse = self._sparse_search(aspect_query, top_k)
            fused = self._rrf_fuse(dense, sparse, top_k=top_k, preserve_sources=existing_sources)

            for doc in fused:
                doc_id = (
                    doc.metadata.get("chunk_id")
                    or doc.metadata.get("doc_id")
                    or doc.page_content[:64]
                )
                if doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)

        return all_docs[:top_k]


# LangChain-compatible wrapper so we can pass it to chains if needed
class HybridRetrieverWrapper(BaseRetriever):
    """LangChain BaseRetriever wrapper around HybridRetriever."""

    def _get_relevant_documents(self, query: str) -> List[Document]:
        retriever = HybridRetriever()
        return retriever.retrieve(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
