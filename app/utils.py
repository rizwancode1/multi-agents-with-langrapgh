from typing import List
import hashlib
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

import requests
import json
from app.config import get_settings




settings = get_settings()



# 1. Define a custom Embeddings wrapper for your OpenRouter call
class CustomOpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if model_name.startswith("openrouter/"):
            model_name = model_name[len("openrouter/"):]
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (batches supported by API)."""
        batch_size = 20
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = requests.post(
                url=f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend(item["embedding"] for item in data["data"])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single search query."""
        return self.embed_documents([text])[0]



llm = ChatOpenAI(
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key,
    model="openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0,
)


# 2. Instantiate your custom embedding class
embeddings = CustomOpenRouterEmbeddings(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    model_name=settings.embedding_model
)


# 3. Stable ID generation
def generate_stable_id(*parts: str) -> str:
    """Generate a stable hash ID from multiple string parts."""
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


# 4. BM25 tokenizer for technical research corpora
def bm25_tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing, preserving technical identifiers
    and normalizing punctuation carefully.
    """
    text = text.lower()
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\.{2,}", " ", text)
    tokens = re.findall(r"[a-z0-9_]+", text)
    return [t for t in tokens if len(t) > 1]


# 5. Contextual enrichment prefix
def build_context_prefix(
    source: str,
    section: str,
    domain: str,
    document_type: str = "research_paper",
) -> str:
    """Build a contextual prefix to prepend to chunks for better standalone meaning."""
    parts = [f"Document: {source}", f"Section: {section}", f"Domain: {domain}"]
    return " | ".join(parts)