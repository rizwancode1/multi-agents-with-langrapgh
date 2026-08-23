"""
Monitoring & Structured Logging
Production-grade metrics collection and JSON logging.
"""

import json
import logging
import time
from datetime import UTC, datetime

# === Structured JSON Logger ===

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for log aggregation (ELK, Datadog, etc.)."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        # Merge any extra data attached to the record
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)
    
def get_logger(name: str = "production-api") -> logging.Logger:
    """Create a structured JSON logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
    


# === Metrics Collector ===

class MetricsCollector:
    """
    Collects and aggregates application metrics.

    In production, replace with Prometheus client:
        from prometheus_client import Counter, Histogram
    """

    def __init__(self):
        self._requests_total = 0
        self._errors_total = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._tokens_input = 0
        self._tokens_output = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: bool = False,
        cache_hit: bool = False,
    ):
        """Record a single request's metrics."""
        self._requests_total += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        if input_tokens or output_tokens:
            self.add_tokens(input_tokens, output_tokens)

        if error:
            self._errors_total += 1
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate real LLM token usage reported by provider callbacks."""
        self._tokens_input += max(0, input_tokens)
        self._tokens_output += max(0, output_tokens)

    @property
    def summary(self) -> dict:
        """Compute summary metrics."""
        avg_latency = (
            self._latency_sum / self._latency_count
            if self._latency_count > 0 else 0.0
        )
        error_rate = (
            self._errors_total / self._requests_total
            if self._requests_total > 0 else 0.0
        )
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / cache_total
            if cache_total > 0 else 0.0
        )

        return {
            "total_requests": self._requests_total,
            "total_errors": self._errors_total,
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "total_input_tokens": self._tokens_input,
            "total_output_tokens": self._tokens_output,
        }


# === Shared metrics singleton ===
#
# A single process-wide MetricsCollector so LLM callbacks deep inside agent
# code can record real token usage without threading a collector through
# every call. The API binds this instance in its lifespan.

_metrics: MetricsCollector = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Return the process-wide metrics collector."""
    return _metrics


def set_metrics(collector: MetricsCollector) -> None:
    """Replace the process-wide collector (used by tests)."""
    global _metrics
    _metrics = collector


class TokenUsageHandler:
    """
    LangChain callback handler that accumulates REAL provider-reported token
    usage into the shared MetricsCollector instead of word-count estimates.

    Attach via ``get_chat_llm(..., callbacks=[TokenUsageHandler()])`` — done
    once inside app.utils so every LLM call is accounted for.
    """

    def __init__(self, collector: "MetricsCollector | None" = None):
        self.collector = collector or get_metrics()

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            usage = {}
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or {}
            if not usage:
                # Newer OpenAI-style usage_metadata on the sampled generation
                gens = getattr(response, "generations", None) or []
                for gen_list in gens:
                    for gen in gen_list:
                        meta = getattr(gen, "usage_metadata", None) or (
                            gen.get("usage_metadata") if isinstance(gen, dict) else None
                        ) or {}
                        if meta:
                            usage = {
                                "prompt_tokens": meta.get("input_tokens", 0),
                                "completion_tokens": meta.get("output_tokens", 0),
                            }
                            break
                    if usage:
                        break
            if usage:
                self.collector.add_tokens(
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                )
        except Exception:
            # Never let metrics accounting break an LLM call
            pass


# === Request Timer (utility) ===

class RequestTimer:
    """Context manager for timing requests."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000
        
        
# uv run python -c "
# from app.monitoring import get_logger, MetricsCollector, RequestTimer
# import time

# logger = get_logger()
# metrics = MetricsCollector()

# print('=== STRUCTURED LOGGING ===')
# print()
# logger.info('Application starting')
# logger.info('Processing request', extra={'extra_data': {'user_id': 'user-123', 'thread_id': 'thread-456'}})
# logger.warning('Rate limit approaching', extra={'extra_data': {'current_rate': 18, 'limit': 20}})

# print()
# print('=== METRICS COLLECTION ===')
# print()

# # Simulate some requests
# with RequestTimer() as timer:
#     time.sleep(0.1)  # Simulate work
# metrics.record_request(latency_ms=timer.elapsed_ms, input_tokens=50, output_tokens=100, cache_hit=False)
# print(f'Request 1: {timer.elapsed_ms:.1f}ms')

# with RequestTimer() as timer:
#     time.sleep(0.05)
# metrics.record_request(latency_ms=timer.elapsed_ms, input_tokens=30, output_tokens=80, cache_hit=True)
# print(f'Request 2: {timer.elapsed_ms:.1f}ms (cache hit)')

# metrics.record_request(latency_ms=5.0, error=True)
# print(f'Request 3: error')

# print()
# print('=== METRICS SUMMARY ===')
# import json
# print(json.dumps(metrics.summary, indent=2))
# "