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

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from langsmith import traceable
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.api_conversations import (
    get_conversation_messages,
    get_conversation_thread_id,
    save_conversation_message,
)
from app.api_conversations import router as conversations_router
from app.api_tickets import router as tickets_router
from app.auth import (  # single source for auth + limiter keying
    _rate_limit_key,
    require_api_key,
)
from app.cache import create_cache
from app.checkpoints import get_checkpointer
from app.config import get_settings
from app.models import (
    STATUS_MESSAGES,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    StreamEvent,
)
from app.monitoring import MetricsCollector, RequestTimer, get_logger, get_metrics
from app.security import SecurityPipeline

load_dotenv()

settings = get_settings()


def _build_limiter() -> Limiter:
    """Limiter backed by Redis when reachable in production so the limit is
    shared across workers; in-memory otherwise (per-worker)."""
    storage_uri = "memory"
    if settings.cache_backend == "redis" or (
        settings.cache_backend == "auto" and settings.is_production
    ):
        try:
            import redis as redis_lib
            client = redis_lib.Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            storage_uri = settings.redis_url
        except Exception:
            logger.warning("rate_limiter_redis_unreachable", extra={"extra_data": {
                "redis_url": settings.redis_url,
            }})
    # slowapi/limits requires an explicit scheme; in-memory storage is
    # per-worker, redis storage shares the budget across workers.
    return Limiter(key_func=_rate_limit_key, storage_uri="memory://" if storage_uri == "memory" else storage_uri)


# Graph agent nodes that produce user-facing status updates
AGENT_NODES = {
    "router",
    "policy_rag",
    "order",
    "support_ticket",
    "return_refund",
    "evaluator",
    "formatter",
}


def _build_initial_state(body: "QueryRequest", cleaned_message: str, thread_id: str) -> AgentState:
    """Build the starting graph state for a request.

    Loads persisted conversation history (when a conversation_id is provided) so
    agents can understand follow-up requests, and resets all transient per-turn
    data so checkpointed values from a previous turn cannot leak into this one.

    History is PII-masked on the way in — raw messages are persisted for audit
    but agents never see unmasked PII (previously masking applied only to the
    current message, so older turns re-introduced it).
    """
    history: list[dict] = []
    if body.conversation_id:
        history = get_conversation_messages(body.conversation_id)
        # The current user message is persisted before the graph runs; drop it so
        # the history shows only PRIOR turns (the current query is passed separately).
        if history and history[-1]["role"] == "user" and history[-1]["text"] == body.query:
            history = history[:-1]
        history = [
            {**m, "text": security.pii_detector.mask(m.get("text", ""))}
            for m in history
        ]

    return {
        "query": cleaned_message,
        "messages": history,
        "next_agent": None,
        "current_agent": None,
        "visited_agents": [],
        "handoff_count": 0,
        "route": [],
        "response": "",
        "context": [],
        "intents": [],
        "order_data": [],
        "order_id": None,
        "citations": [],
        "retrieved_documents": [],
        "return_refund_data": None,
        "evaluation": None,
        "handoff_reason": None,
    }


# === Global instances (initialized in lifespan) ===
security: SecurityPipeline = None
cache = None
metrics: MetricsCollector = get_metrics()  # process-wide singleton (LLM callbacks write here)
multi_agent_graph = None
logger = get_logger()


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event).
    """
    global security, cache, multi_agent_graph

    logger.info("Starting multi-agent API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing_v2,
    }})

    # Initialize components
    security = SecurityPipeline()
    cache = create_cache(ttl_seconds=settings.cache_ttl_seconds)

    # Ingest RAG corpus (vector store + BM25) if enabled
    if settings.ingest_on_startup:
        try:
            from app.ingestion import get_ingestion_pipeline
            ingest_result = get_ingestion_pipeline().run()
            logger.info("rag_ingestion_ready", extra={"extra_data": ingest_result})
        except Exception as e:
            logger.error("rag_ingestion_failed", extra={"extra_data": {"error": str(e)}})

    checkpointer = await get_checkpointer()
    multi_agent_graph = build_graph(checkpointer=checkpointer)

    # Initialize database and seed data
    from app.db import init_db
    from app.db_init import seed_orders
    init_db()
    seed_orders()

    if settings.debug_draw_graph:
        output_path = Path.cwd() / "multi_agent_graph.png"
        try:
            output_path.write_bytes(
                multi_agent_graph.get_graph().draw_mermaid_png()
            )
            logger.info("graph_image_saved", extra={"extra_data": {"path": str(output_path)}})
        except Exception as e:
            logger.warning("graph_image_skipped", extra={"extra_data": {"error": str(e)}})

    logger.info("All components initialized. Ready to serve requests.", extra={"extra_data": {
        "checkpoint_storage": settings.checkpoint_storage,
        "checkpoint_path": settings.checkpoint_path,
    }})

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})

    # Release the checkpoint DB connection (previously leaked until GC).
    conn = getattr(checkpointer, "conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception as e:
            logger.warning("checkpointer_close_failed", extra={"extra_data": {"error": str(e)}})


# === Rate Limiter Setup ===
limiter = _build_limiter()


# === FastAPI App ===
app = FastAPI(
    title="Multi-Agent API",
    description="A production-ready multi-agent API with security, caching, observability, and peer-to-peer agent handoffs.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.include_router(conversations_router)
app.include_router(tickets_router)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the test chat UI."""
    return FileResponse("app/static/index.html")


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
async def query_agents(request: Request, body: QueryRequest, _auth: None = Depends(require_api_key)):
    """
    Main multi-agent query endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup (keyed on query + conversation scope)
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

        # ---- Step 1b: Persist the user message ----
        if body.conversation_id:
            save_conversation_message(body.conversation_id, "user", body.query)

        # ---- Step 1c: Resolve thread scope BEFORE the cache so responses that
        # depend on conversation history are never served across conversations
        # (previously identical query text returned whichever answer was cached first).
        if body.conversation_id:
            thread_id = body.thread_id or get_conversation_thread_id(body.conversation_id) or str(uuid.uuid4())
        else:
            thread_id = body.thread_id or str(uuid.uuid4())
        cache_scope = str(body.conversation_id) if body.conversation_id else body.thread_id or ""
        cache_key = f"{cache_scope}::{cleaned_message}" if cache_scope else cleaned_message

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "query": body.query[:100],
            }})
            if body.conversation_id:
                save_conversation_message(body.conversation_id, "assistant", cached_response)
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
                thread_id=thread_id,
                conversation_id=body.conversation_id,
            )

        # ---- Step 3: Invoke Multi-Agent Graph (non-blocking) ----
        graph_config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        try:
            initial_state = _build_initial_state(body, cleaned_message, thread_id)
            result = await multi_agent_graph.ainvoke(initial_state, config=graph_config)
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

        logger.info("graph_invoke_result", extra={"extra_data": {
            "query": body.query[:200],
            "response_preview": response_text[:200],
            "visited_agents": visited,
            "final_agent": final_agent,
            "route_length": len(route_trace),
            "intents": intents,
            "order_id": order_id,
            "retrieved_doc_count": len(retrieved_documents),
            "evaluation_passed": evaluation.get("passed"),
        }})

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

        # ---- Step 5b: Persist the assistant message ----
        if body.conversation_id:
            save_conversation_message(body.conversation_id, "assistant", validated_response)

    # ---- Step 6: Log & Record Metrics ----
    # Real token usage is accumulated by TokenUsageHandler (attached to every
    # LLM in app.utils); no more word-count estimates.
    metrics.record_request(
        latency_ms=timer.elapsed_ms,
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
        thread_id=thread_id,
        conversation_id=body.conversation_id,
        retrieved_documents=retrieved_documents,
        citations=citations,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )


@app.post("/query/stream")
@limiter.limit(get_settings().rate_limit)
@traceable(name="query_stream_endpoint")
async def stream_query(request: Request, body: QueryRequest, _auth: None = Depends(require_api_key)):
    """
    Streaming multi-agent query endpoint via SSE.

    Yields real-time status events as the LangGraph workflow executes.
    """
    # ---- Security + Cache Lookup ----
    is_allowed, cleaned_message, _notes = security.check_input(body.query)
    if not is_allowed:
        metrics.record_request(latency_ms=0, error=True)
        raise HTTPException(
            status_code=400,
            detail="Your message was blocked by our security filters.",
        )

    # ---- Persist the user message ----
    if body.conversation_id:
        save_conversation_message(body.conversation_id, "user", body.query)

    if body.conversation_id:
        thread_id = body.thread_id or get_conversation_thread_id(body.conversation_id) or str(uuid.uuid4())
    else:
        thread_id = body.thread_id or str(uuid.uuid4())
    cache_scope = str(body.conversation_id) if body.conversation_id else body.thread_id or ""
    cache_key = f"{cache_scope}::{cleaned_message}" if cache_scope else cleaned_message

    cached_response = cache.get(cache_key)
    if cached_response is not None:
        metrics.record_request(latency_ms=0, cache_hit=True)
        if body.conversation_id:
            save_conversation_message(body.conversation_id, "assistant", cached_response)
        event = StreamEvent(type="done", response=cached_response, cached=True)
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            iter([f"data: {event.model_dump_json()}\n\n"]),
            media_type="text/event-stream",
            headers=headers,
        )

    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = _build_initial_state(body, cleaned_message, thread_id)

    start_time = time.monotonic()
    MAX_GRAPH_SECONDS = 180.0
    HEARTBEAT_INTERVAL = 15.0
    SSE_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    async def _run_graph_to_queue(queue: asyncio.Queue):
        final_output = None
        seen_nodes = set()
        try:
            async for event in multi_agent_graph.astream_events(
                initial_state, config=graph_config, version="v2"
            ):
                await queue.put(event)
                etype = event.get("event")
                if etype in ("on_node_end", "on_chain_end"):
                    node = event.get("name")
                    seen_nodes.add(f"{etype}:{node}")
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict) and output.get("response"):
                        if final_output is None:
                            final_output = dict(output)
                        else:
                            merged = dict(final_output)
                            merged.update({
                                k: v for k, v in output.items() if v not in (None, [], {})
                            })
                            merged["response"] = output["response"]
                            final_output = merged
                        logger.info("stream_final_output_captured", extra={"extra_data": {
                            "event": etype,
                            "node": node,
                            "response_length": len(output.get("response", "")),
                        }})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("stream_graph_error", extra={"extra_data": {
                "query": cleaned_message[:200],
                "error": str(e),
            }})
            await queue.put({"type": "error", "message": str(e)})
        finally:
            if final_output is None:
                logger.warning("stream_no_final_output", extra={"extra_data": {
                    "query": cleaned_message[:200],
                    "events": list(seen_nodes),
                }})
            await asyncio.shield(queue.put({"__final__": True, "data": final_output}))
            await asyncio.shield(queue.put(None))

    async def _convert_and_yield(event: dict):
        event_type = event.get("event", "")
        name = event.get("name", "")

        # LangGraph's astream_events (version="v2") emits on_chain_start/on_chain_end
        # events (with name == node name) rather than on_node_start/on_node_end.
        if event_type in ("on_node_start", "on_chain_start") and name in AGENT_NODES:
            status_msg = STATUS_MESSAGES.get(name, f"Running {name}...")
            return StreamEvent(type="status", agent=name, message=status_msg)
        elif event_type in ("on_node_end", "on_chain_end") and name in AGENT_NODES:
            return StreamEvent(type="step", agent=name)
        elif event_type in ("on_node_error", "on_chain_error"):
            err_msg = str(event.get("data", {}).get("error", "Unknown error"))
            return StreamEvent(type="error", message=err_msg)
        elif event_type == "error":
            return StreamEvent(type="error", message=event.get("message", "Unknown error"))
        return None

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_run_graph_to_queue(queue))
        last_heartbeat = time.monotonic()
        graph_started = time.monotonic()
        final_output = None
        terminal_sent = False

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    now = time.monotonic()
                    if now - graph_started >= MAX_GRAPH_SECONDS:
                        terminal_sent = True
                        error_event = StreamEvent(
                            type="error",
                            message="Request timed out. Please try again.",
                        )
                        yield f"data: {error_event.model_dump_json()}\n\n"
                        task.cancel()
                        break
                    if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                        yield ": ping\n\n"
                        last_heartbeat = now
                    continue

                if event is None:
                    break

                if event.get("__final__"):
                    final_output = event.get("data")
                    continue

                last_heartbeat = time.monotonic()
                converted = await _convert_and_yield(event)
                if converted is not None:
                    if converted.type == "error":
                        terminal_sent = True
                    yield f"data: {converted.model_dump_json()}\n\n"

        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if final_output:
            response_text = final_output.get("response", "")
            visited = final_output.get("visited_agents", [])
            entry_agent = visited[0] if visited else "unknown"
            final_agent = final_output.get("current_agent", "unknown")
            route_trace = final_output.get("route", [])
            intents = final_output.get("intents", [])
            order_id = final_output.get("order_id")
            retrieved_documents = final_output.get("retrieved_documents", [])
            citations = final_output.get("citations", [])

            validated_response, _ = security.check_output(response_text)
            cache.set(cleaned_message, validated_response)

            if body.conversation_id:
                save_conversation_message(body.conversation_id, "assistant", validated_response)

            done_event = StreamEvent(
                type="done",
                response=validated_response,
                data={
                    "entry_agent": entry_agent,
                    "final_agent": final_agent,
                    "visited_agents": visited,
                    "route": route_trace,
                    "intents": intents,
                    "order_id": order_id,
                    "thread_id": thread_id,
                    "conversation_id": body.conversation_id,
                    "retrieved_documents": retrieved_documents,
                    "citations": citations,
                },
            )
            yield f"data: {done_event.model_dump_json()}\n\n"
            terminal_sent = True

            # Real token usage arrives via TokenUsageHandler on each LLM call.
            metrics.record_request(
                latency_ms=(time.monotonic() - start_time) * 1000,
                cache_hit=False,
            )
        elif not terminal_sent:
            error_event = StreamEvent(
                type="error",
                message="The assistant couldn't complete the request. Please try again.",
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes — probes DB and cache, not just wiring."""
    settings = get_settings()

    checks = {
        "agent": multi_agent_graph is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    # Database connectivity (previously a dead DB still reported healthy).
    try:
        from sqlalchemy import text

        from app.db import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        checks["database"] = False
        logger.warning("health_db_check_failed", extra={"extra_data": {"error": str(e)}})

    # Redis reachability when it's the active backend.
    if cache is not None and getattr(cache, "_client", None) is not None:
        try:
            checks["cache_backend_reachable"] = bool(cache._client.ping())
        except Exception:
            checks["cache_backend_reachable"] = False

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
