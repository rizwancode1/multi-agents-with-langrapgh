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
import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
    StreamEvent, STATUS_MESSAGES,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.checkpoints import get_checkpointer
from app.api_conversations import router as conversations_router
from app.api_conversations import save_conversation_message, get_conversation_thread_id, get_conversation_messages
from pathlib import Path

load_dotenv()


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
    """
    history: list[dict] = []
    if body.conversation_id:
        history = get_conversation_messages(body.conversation_id)
        # The current user message is persisted before the graph runs; drop it so
        # the history shows only PRIOR turns (the current query is passed separately).
        if history and history[-1]["role"] == "user" and history[-1]["text"] == body.query:
            history = history[:-1]

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

    checkpointer = await get_checkpointer()
    multi_agent_graph = build_graph(checkpointer=checkpointer)

    # Initialize database and seed data
    from app.db import init_db
    from app.db_init import seed_orders
    init_db()
    seed_orders()

    output_path = Path.cwd() / "multi_agent_graph.png"

    try:
        output_path.write_bytes(
            multi_agent_graph.get_graph().draw_mermaid_png()
        )
        print(f"Graph Image saved to: {output_path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")

    logger.info("All components initialized. Ready to serve requests.", extra={"extra_data": {
        "checkpoint_storage": settings.checkpoint_storage,
        "checkpoint_path": settings.checkpoint_path,
    }})

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
app.include_router(conversations_router)

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

        # ---- Step 1b: Persist the user message ----
        if body.conversation_id:
            save_conversation_message(body.conversation_id, "user", body.query)

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
        if body.conversation_id:
            thread_id = body.thread_id or get_conversation_thread_id(body.conversation_id) or str(uuid.uuid4())
        else:
            thread_id = body.thread_id or str(uuid.uuid4())
        graph_config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        try:
            initial_state = _build_initial_state(body, cleaned_message, thread_id)
            result = multi_agent_graph.invoke(initial_state, config=graph_config)
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
async def stream_query(request: Request, body: QueryRequest):
    """
    Streaming multi-agent query endpoint via SSE.

    Yields real-time status events as the LangGraph workflow executes.
    """
    # ---- Security + Cache Lookup ----
    is_allowed, cleaned_message, notes = security.check_input(body.query)
    if not is_allowed:
        metrics.record_request(latency_ms=0, error=True)
        raise HTTPException(
            status_code=400,
            detail="Your message was blocked by our security filters.",
        )

    # ---- Persist the user message ----
    if body.conversation_id:
        save_conversation_message(body.conversation_id, "user", body.query)

    cached_response = cache.get(cleaned_message)
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

    if body.conversation_id:
        thread_id = body.thread_id or get_conversation_thread_id(body.conversation_id) or str(uuid.uuid4())
    else:
        thread_id = body.thread_id or str(uuid.uuid4())
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
                except asyncio.TimeoutError:
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

            input_tokens = int(len(cleaned_message.split()) * 1.3)
            output_tokens = int(len(validated_response.split()) * 1.3)
            metrics.record_request(
                latency_ms=(time.monotonic() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
