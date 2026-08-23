import hashlib
import re
import time
from collections.abc import Sequence

import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.monitoring import TokenUsageHandler

settings = get_settings()



# 1. Define a custom Embeddings wrapper for your OpenRouter call
class CustomOpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model_name: str, timeout: int = 30, max_retries: int = 2):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        model_name = model_name.removeprefix("openrouter/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def _post_with_retry(self, batch: list[str]) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
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
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (batches supported by API)."""
        batch_size = 20
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            data = self._post_with_retry(batch)
            all_embeddings.extend(item["embedding"] for item in data["data"])
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_documents([text])[0]



def _build_chat_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key or "sk-placeholder",
        model=model,
        temperature=temperature,
        timeout=60,
        max_retries=0,
        callbacks=[TokenUsageHandler()],
    )


def get_chat_llm(
    temperature: float = 0.0,
    tools: Sequence | None = None,
    schema: type | None = None,
):
    """Chat model on PRIMARY_MODEL that automatically fails over to FALLBACK_MODEL.

    Use this instead of constructing ChatOpenAI directly so free-tier rate
    limits (429) degrade gracefully to the fallback model.
    """
    primary = _build_chat_llm(settings.primary_model, temperature)
    fallback = _build_chat_llm(settings.fallback_model, temperature)

    def _bind(model: ChatOpenAI):
        if schema is not None:
            return model.with_structured_output(schema)
        if tools:
            return model.bind_tools(list(tools))
        return model

    return _bind(primary).with_fallbacks([_bind(fallback)])


# 2. Instantiate your custom embedding class
embeddings = CustomOpenRouterEmbeddings(
    api_key=settings.openrouter_api_key or "sk-placeholder",
    base_url=settings.openrouter_base_url,
    model_name=settings.embedding_model
)


# 3. Stable ID generation
def generate_stable_id(*parts: str) -> str:
    """Generate a stable hash ID from multiple string parts."""
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


# 4. BM25 tokenizer for technical research corpora
def bm25_tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing, preserving technical identifiers
    and normalizing punctuation carefully.
    """
    text = text.lower()
    text = re.sub(r"[-\u2013\u2014]", " ", text)
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
