"""Unit tests for the LLM reranker (LLM mocked, no network)."""

from types import SimpleNamespace

from langchain_core.documents import Document

from app.reranker import LLMReranker


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.responses.pop(0))


def make_reranker(responses) -> LLMReranker:
    r = object.__new__(LLMReranker)
    r.settings = None
    r.llm = FakeLLM(responses)
    r.prompt = LLMReranker.__dict__ is None  # placeholder, replaced below
    from langchain_core.prompts import ChatPromptTemplate
    r.prompt = ChatPromptTemplate.from_template("{question} {context}")
    r._cache = {}
    return r


def test_rerank_sorts_by_score_and_clamps():
    docs = [
        Document(page_content="low", metadata={"chunk_id": "d1"}),
        Document(page_content="high", metadata={"chunk_id": "d2"}),
        Document(page_content="mid", metadata={"chunk_id": "d3"}),
    ]
    r = make_reranker(["2", "9", "5"])

    result = r.rerank("q", docs, top_k=3)

    ids = [doc.metadata["chunk_id"] for doc in result]
    assert ids == ["d2", "d3", "d1"]
    assert [doc.metadata["rerank_score"] for doc in result] == [9.0, 5.0, 2.0]


def test_rerank_failure_scores_zero():
    docs = [
        Document(page_content="bad", metadata={"chunk_id": "d1"}),
        Document(page_content="good", metadata={"chunk_id": "d2"}),
    ]
    r = make_reranker(["not-a-number", "7"])

    result = r.rerank("q", docs, top_k=2)

    assert result[0].metadata["chunk_id"] == "d2"
    assert result[1].metadata["rerank_score"] == 0.0


def test_rerank_uses_cache_for_identical_query():
    docs = [Document(page_content="x", metadata={"chunk_id": "d1"})]
    r = make_reranker(["8"])

    first = r.rerank("same question", docs, top_k=1)
    second = r.rerank("same question", docs, top_k=1)

    assert r.llm.calls == 1
    assert first[0].metadata["chunk_id"] == second[0].metadata["chunk_id"]


def test_rerank_empty_documents():
    r = make_reranker([])
    assert r.rerank("q", [], top_k=3) == []
