"""Unit tests for the ingestion pipeline (KB loading + chunking, no embedding calls)."""

from app.config import get_settings
from app.ingestion import IngestionPipeline


def make_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


def test_load_kb_documents_enriches_metadata():
    p = make_pipeline()
    docs = p.load_kb_documents()

    assert len(docs) >= 3

    first = docs[0]
    assert first.metadata["document_id"].startswith("DOC-")
    assert first.metadata["source"] == first.metadata["document_id"]
    assert "Title:" in first.page_content
    assert first.metadata["title"] in first.page_content


def test_chunk_documents_assigns_stable_chunk_ids():
    p = make_pipeline()
    docs = p.load_kb_documents()
    chunks = p.chunk_documents(docs)

    assert len(chunks) >= len(docs)

    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))  # unique

    for chunk in chunks:
        assert chunk.metadata["document_id"]
        assert "chunk_index" in chunk.metadata


def test_backend_selection_defaults_to_chroma():
    p = make_pipeline()
    settings = get_settings()

    if not settings.use_pgvector:
        assert p.use_pgvector is False
        assert p.collection_name == settings.chroma_collection
