"""Unit tests for hybrid retrieval RRF fusion (no network / no index required)."""

from langchain_core.documents import Document

from app.retrieval import HybridRetriever


def make_retriever() -> HybridRetriever:
    """Bypass __init__ so no vector store / BM25 index is loaded."""
    return object.__new__(HybridRetriever)


def test_rrf_fuse_merges_both_lists():
    r = make_retriever()
    d1 = Document(page_content="a", metadata={"chunk_id": "c1"})
    d2 = Document(page_content="b", metadata={"chunk_id": "c2"})
    d3 = Document(page_content="c", metadata={"chunk_id": "c3"})

    dense = [(d1, 0.9), (d2, 0.8)]
    sparse = [(d2, 5.0), (d3, 4.0)]

    fused = r._rrf_fuse(dense, sparse, top_k=3)

    ids = [doc.metadata["chunk_id"] for doc in fused]
    # c2 appears in both lists -> highest fused score; c1 and c3 tie at rank 1
    assert ids[0] == "c2"
    assert set(ids) == {"c1", "c2", "c3"}


def test_rrf_fuse_respects_top_k():
    r = make_retriever()
    docs = [Document(page_content=str(i), metadata={"chunk_id": f"c{i}"}) for i in range(6)]
    dense = [(doc, 1.0) for doc in docs]
    sparse = [(doc, 1.0) for doc in docs]

    fused = r._rrf_fuse(dense, sparse, top_k=2)
    assert len(fused) == 2


def test_rrf_fuse_empty_inputs():
    r = make_retriever()
    assert r._rrf_fuse([], [], top_k=3) == []


def test_rrf_fuse_preserve_sources_prefers_new_sources():
    r = make_retriever()
    seen = Document(page_content="seen", metadata={"chunk_id": "s1", "source": "DOC-1"})
    new1 = Document(page_content="new1", metadata={"chunk_id": "n1", "source": "DOC-2"})
    new2 = Document(page_content="new2", metadata={"chunk_id": "n2", "source": "DOC-3"})
    another_seen = Document(page_content="seen2", metadata={"chunk_id": "s2", "source": "DOC-1"})

    dense = [(another_seen, 0.9), (new1, 0.8), (seen, 0.7), (new2, 0.6)]

    fused = r._rrf_fuse(dense, [], top_k=4, preserve_sources=["DOC-1"])

    ids = [doc.metadata["chunk_id"] for doc in fused]
    # Only one DOC-1 doc kept, new sources kept
    assert ids == ["s2", "n1", "n2"]
