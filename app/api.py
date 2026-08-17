"""
Production-Ready Multi-Agent FastAPI Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate limiting (slowapi)
- LangGraph multi-agent workflow
- Structured logging + metrics
- Health checks
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    QueryRequest, QueryResponse,
    HealthResponse, MetricsResponse,
    ErrorResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agents.graph import graph
from app.agents.state import AgentState
from pathlib import Path

load_dotenv()


# === Global instances (initialized in lifespan) ===
security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
multi_agent_graph = None
logger = get_logger()


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event).
    """
    global security, cache, metrics, multi_agent_graph

    settings = get_settings()

    logger.info("Starting multi-agent API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing_v2,
    }})

    # Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    multi_agent_graph = graph

    output_path = Path.cwd() / "multi_agent_graph.png"

    try:
        output_path.write_bytes(
            multi_agent_graph.get_graph().draw_mermaid_png()
        )
        print(f"Graph Image saved to: {output_path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")

    logger.info("All components initialized. Ready to serve requests.")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})


# === Rate Limiter Setup ===
limiter = Limiter(key_func=get_remote_address)


# === FastAPI App ===
app = FastAPI(
    title="Multi-Agent API",
    description="A production-ready multi-agent API with security, caching, observability, and peer-to-peer agent handoffs.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


# === Exception Handlers ===

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning("Rate limit exceeded", extra={"extra_data": {
        "client_ip": get_remote_address(request),
    }})
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", extra={"extra_data": {
        "path": request.url.path,
        "error": str(exc),
    }})
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if get_settings().app_env != "production" else None,
        ).model_dump(),
    )


# =============================================
# ENDPOINTS
# =============================================

@app.post("/query", response_model=QueryResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="query_endpoint")
async def query_agents(request: Request, body: QueryRequest):
    """
    Main multi-agent query endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph multi-agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        is_allowed, cleaned_message, notes = security.check_input(body.query)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters.",
            )

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "query": body.query[:100],
            }})
            return QueryResponse(
                query=body.query,
                response=cached_response,
                entry_agent="cache",
                final_agent="cache",
                visited_agents=[],
                route=[],
                intents=[],
                order_id=None,
                retrieved_documents=[],
                citations=None,
                cached=True,
                processing_time_ms=0,
                security_notes=security_notes,
            )

        # ---- Step 3: Invoke Multi-Agent Graph ----
        try:
            initial_state: AgentState = {
                "query": cleaned_message,
                "next_agent": None,
                "visited_agents": [],
                "handoff_count": 0,
                "route": [],
            }
            result = multi_agent_graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                "query": body.query[:100],
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        response_text = result.get("response", "")
        visited = result.get("visited_agents", [])
        entry_agent = visited[0] if visited else "unknown"
        final_agent = result.get("current_agent", "unknown")
        route_trace = result.get("route", [])
        intents = result.get("intents", [])
        order_id = result.get("order_id")
        retrieved_documents = result.get("retrieved_documents", [])

        evaluation = result.get("evaluation", {})
        invalid_citations = set(evaluation.get("invalid_citations", []))
        raw_citations = result.get("citations", []) or []
        citations = [c for c in raw_citations if c not in invalid_citations]

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

    # ---- Step 6: Log & Record Metrics ----
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "query": body.query[:100],
        }})

    logger.info("Request completed", extra={"extra_data": {
        "query": body.query[:100],
        "entry_agent": entry_agent,
        "final_agent": final_agent,
        "latency_ms": round(timer.elapsed_ms, 2),
        "visited_agents": visited,
        "citations": result.get("citations", []),
        "route_length": len(route_trace),
    }})

    return QueryResponse(
        query=body.query,
        response=validated_response,
        entry_agent=entry_agent,
        final_agent=final_agent,
        visited_agents=visited,
        route=route_trace,
        intents=intents,
        order_id=order_id,
        retrieved_documents=retrieved_documents,
        citations=citations,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": multi_agent_graph is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards."""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats
