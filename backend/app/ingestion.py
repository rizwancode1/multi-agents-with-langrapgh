"""
Ingestion Pipeline
Loads the policy knowledge base (JSON), enriches chunks with contextual
prefixes, embeds them via CustomOpenRouterEmbeddings, and persists to
ChromaDB or PGVector + a pickled BM25 index.

Vector backend is selected by USE_PGVECTOR:
- false (default): local ChromaDB at chroma_persist_dir
- true:            PGVector collection in the Postgres database (DATABASE_URL)
BM25 always persists locally as a pickle alongside the vector index.
"""

import contextlib
import hashlib
import json
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.monitoring import get_logger
from app.utils import bm25_tokenize, embeddings, generate_stable_id

logger = get_logger("ingestion")

KB_PATH = Path(__file__).resolve().parent / "data" / "rag_knowledge_base.json"


class IngestionPipeline:
    """Load, chunk, embed, and index the policy knowledge base."""

    def __init__(self):
        self.settings = get_settings()
        self.kb_path = KB_PATH
        self.persist_dir = Path(self.settings.chroma_persist_dir)
        self.bm25_index_path = self.persist_dir / "bm25_index.pkl"
        self.manifest_path = self.persist_dir / "index_manifest.json"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # === Backend selection ===

    @property
    def use_pgvector(self) -> bool:
        return self.settings.use_pgvector

    @property
    def collection_name(self) -> str:
        if self.use_pgvector:
            return self.settings.pgvector_collection
        return self.settings.chroma_collection

    # === Loading ===

    def load_kb_documents(self) -> list[Document]:
        """Load the JSON knowledge base into enriched Documents."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_path}")

        with open(self.kb_path, encoding="utf-8") as f:
            entries = json.load(f)

        documents: list[Document] = []
        for entry in entries:
            content = entry.get("content", "")
            title = entry.get("title", "")
            category = entry.get("category", "")
            tags = entry.get("tags", [])
            document_id = entry.get("document_id") or generate_stable_id(title)

            prefix_parts = [
                f"Document: {document_id}",
                f"Title: {title}",
                f"Category: {category}",
            ]
            if tags:
                prefix_parts.append(f"Tags: {', '.join(tags)}")
            enriched_content = f"{' | '.join(prefix_parts)}\n\n{content}"

            documents.append(
                Document(
                    page_content=enriched_content,
                    metadata={
                        "source": document_id,
                        "document_id": document_id,
                        "title": title,
                        "category": category,
                        "tags": tags,
                        "section": category,
                        "document_type": "policy_doc",
                    },
                )
            )
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into overlapping chunks with stable chunk IDs."""
        chunks: list[Document] = []
        for doc in documents:
            split_docs = self.text_splitter.create_documents(
                [doc.page_content],
                metadatas=[dict(doc.metadata)],
            )
            for idx, split_doc in enumerate(split_docs):
                split_doc.metadata["chunk_id"] = generate_stable_id(
                    doc.metadata["document_id"], str(idx)
                )
                split_doc.metadata["chunk_index"] = idx
                chunks.append(split_doc)
        return chunks

    # === Vector store ===

    def build_vector_store(self, chunks: list[Document]):
        """Create / upsert chunks into the configured vector store."""
        if self.use_pgvector:
            from langchain_postgres import PGVector

            self.vector_store = PGVector.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.collection_name,
                connection=self.settings.database_url,
                use_jsonb=True,
            )
        else:
            from langchain_chroma import Chroma

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            # Drop any previous version of this collection so a rebuild never
            # accumulates duplicates next to stale chunks.
            with contextlib.suppress(Exception):
                # Raises on first run / nothing to clean.
                Chroma(
                    embedding_function=embeddings,
                    persist_directory=str(self.persist_dir),
                    collection_name=self.collection_name,
                ).delete_collection()
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(self.persist_dir),
                collection_name=self.collection_name,
            )
        return self.vector_store

    def load_vector_store(self):
        """Load an existing persistent vector store."""
        if self.use_pgvector:
            from langchain_postgres import PGVector

            self.vector_store = PGVector(
                embeddings=embeddings,
                collection_name=self.collection_name,
                connection=self.settings.database_url,
                use_jsonb=True,
            )
        else:
            from langchain_chroma import Chroma

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=str(self.persist_dir),
                collection_name=self.collection_name,
            )
        return self.vector_store

    # === BM25 ===

    def build_bm25_index(self, chunks: list[Document]) -> dict:
        """Build a BM25 index and persist it as pickle."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 is required. Install with: uv add rank-bm25") from e

        tokenized_corpus = [bm25_tokenize(doc.page_content) for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        payload = {
            "bm25": bm25,
            "docs": chunks,
        }
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(payload, f)
        return payload

    def load_bm25_index(self) -> dict:
        """Load persisted BM25 index."""
        if not self.bm25_index_path.exists():
            return {}
        with open(self.bm25_index_path, "rb") as f:
            return pickle.load(f)

    # === Pipeline ===

    def _kb_hash(self) -> str:
        """SHA-256 of the knowledge-base file so edits invalidate the index."""
        return hashlib.sha256(self.kb_path.read_bytes()).hexdigest()

    def _write_manifest(self, document_count: int) -> None:
        self.manifest_path.write_text(
            json.dumps({
                "status": "ingested",
                "backend": "pgvector" if self.use_pgvector else "chroma",
                "collection": self.collection_name,
                "embedding_model": self.settings.embedding_model,
                "documents": document_count,
                "kb_hash": self._kb_hash(),
            }),
            encoding="utf-8",
        )

    def run(self, force_rebuild: bool = False) -> dict:
        """Execute the full ingestion pipeline."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        manifest_exists = self.manifest_path.exists()
        bm25_exists = self.bm25_index_path.exists()

        if not force_rebuild and manifest_exists and bm25_exists:
            # Rebuild automatically when the KB content changed since the last
            # ingestion (previously stale indexes were served forever).
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                kb_unchanged = manifest.get("kb_hash") == self._kb_hash()
            except Exception:
                kb_unchanged = False

            if kb_unchanged:
                self.load_vector_store()
                bm25_payload = self.load_bm25_index()
                return {
                    "status": "loaded",
                    "vector_store": "existing",
                    "bm25_index": "existing",
                    "documents": len(bm25_payload.get("docs", [])),
                }
            logger.info("rag_kb_changed_rebuilding", extra={"extra_data": {}})

        documents = self.load_kb_documents()
        chunks = self.chunk_documents(documents)

        self.build_vector_store(chunks)
        self.build_bm25_index(chunks)
        self._write_manifest(len(chunks))

        logger.info("rag_corpus_ingested", extra={"extra_data": {
            "backend": "pgvector" if self.use_pgvector else "chroma",
            "chunks": len(chunks),
        }})

        return {
            "status": "ingested",
            "vector_store": "built",
            "bm25_index": "built",
            "documents": len(chunks),
        }


# Module-level singleton (lazy-initialized on first use)
_pipeline: IngestionPipeline | None = None


def get_ingestion_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
