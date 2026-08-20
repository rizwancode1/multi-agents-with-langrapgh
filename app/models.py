"""
API Request and Response Models
Pydantic models for input validation and response structure.
"""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone


class QueryRequest(BaseModel):
    """Incoming multi-agent query request."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's query to the multi-agent system",
    )
    thread_id: str | None = Field(
        default=None,
        description="Optional thread ID for checkpointed multi-turn conversations",
    )


class QueryResponse(BaseModel):
    """Response from the multi-agent system."""

    query: str
    response: str
    entry_agent: str
    final_agent: str
    visited_agents: list[str]
    route: list[dict]
    intents: list[str]
    order_id: str | None
    thread_id: str
    retrieved_documents: list[dict]
    citations: list[str] | None = None
    cached: bool = False
    processing_time_ms: float = 0.0
    security_notes: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    environment: str
    version: str = "0.1.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Metrics endpoint response."""

    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


class StreamEvent(BaseModel):
    """SSE event payload for streaming agent status."""

    type: Literal["status", "step", "error", "done"]
    agent: str | None = None
    message: str | None = None
    data: dict | None = None
    response: str | None = None
    cached: bool = False


STATUS_MESSAGES = {
    "router": "Analyzing request...",
    "order": "Looking up your order...",
    "policy_rag": "Searching knowledge base...",
    "support_ticket": "Drafting support ticket...",
    "return_refund": "Calculating refund eligibility...",
    "evaluator": "Validating response...",
    "formatter": "Finalizing answer...",
}
