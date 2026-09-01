"""
Prometheus-style metrics for the knowledge transfer agent.

Exposes:
  - HTTP request counts / latency (via middleware helper)
  - /ask success, errors, latency, guardrail blocks
  - Per-agent duration histograms (guardrails, shared_memory, retriever, ...)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

_PROM_AVAILABLE = False
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    CollectorRegistry = None  # type: ignore
    Counter = Histogram = None  # type: ignore
    generate_latest = None  # type: ignore


_registry: Any = None
_ask_total: Any = None
_ask_errors: Any = None
_ask_blocked: Any = None
_ask_latency: Any = None
_agent_duration: Any = None
_agent_runs: Any = None
_http_requests: Any = None
_http_latency: Any = None
_initialized = False

# Lightweight fallback when prometheus_client is missing (tests / minimal envs)
_fallback: dict[str, Any] = {
    "ask_total": 0,
    "ask_errors": 0,
    "ask_blocked": 0,
    "ask_latency_sum": 0.0,
    "ask_latency_count": 0,
    "agent_duration": {},  # name -> {sum, count}
    "http_requests": {},
}


def metrics_enabled() -> bool:
    try:
        return bool(getattr(get_settings(), "metrics_enabled", True))
    except Exception:
        return True


def init_metrics() -> None:
    """Idempotent metrics initialization."""
    global _initialized, _registry
    global _ask_total, _ask_errors, _ask_blocked, _ask_latency
    global _agent_duration, _agent_runs, _http_requests, _http_latency

    if _initialized:
        return
    _initialized = True

    if not _PROM_AVAILABLE:
        logger.warning("prometheus_client not installed; using in-memory metrics fallback")
        return

    _registry = CollectorRegistry()
    _ask_total = Counter(
        "kta_ask_total",
        "Total /ask invocations",
        ["status"],
        registry=_registry,
    )
    _ask_errors = Counter(
        "kta_ask_errors_total",
        "Total /ask failures",
        ["error_type"],
        registry=_registry,
    )
    _ask_blocked = Counter(
        "kta_ask_guardrails_blocked_total",
        "Asks blocked by input guardrails",
        registry=_registry,
    )
    _ask_latency = Histogram(
        "kta_ask_latency_seconds",
        "End-to-end /ask latency in seconds",
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0),
        registry=_registry,
    )
    _agent_duration = Histogram(
        "kta_agent_duration_seconds",
        "Per-agent node duration in seconds",
        ["agent"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0),
        registry=_registry,
    )
    _agent_runs = Counter(
        "kta_agent_runs_total",
        "Per-agent node executions",
        ["agent", "status"],
        registry=_registry,
    )
    _http_requests = Counter(
        "kta_http_requests_total",
        "HTTP requests",
        ["method", "path", "status"],
        registry=_registry,
    )
    _http_latency = Histogram(
        "kta_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=_registry,
    )
    logger.info("Prometheus metrics registry initialized")


def observe_ask(
    *,
    latency_seconds: float,
    success: bool,
    blocked: bool = False,
    error_type: str | None = None,
) -> None:
    if not metrics_enabled():
        return
    init_metrics()
    if _PROM_AVAILABLE and _ask_total is not None:
        status = "blocked" if blocked else ("success" if success else "error")
        _ask_total.labels(status=status).inc()
        _ask_latency.observe(max(0.0, latency_seconds))
        if blocked:
            _ask_blocked.inc()
        if not success and not blocked:
            _ask_errors.labels(error_type=error_type or "unknown").inc()
        return

    _fallback["ask_total"] += 1
    _fallback["ask_latency_sum"] += latency_seconds
    _fallback["ask_latency_count"] += 1
    if blocked:
        _fallback["ask_blocked"] += 1
    if not success and not blocked:
        _fallback["ask_errors"] += 1


def observe_agent(agent: str, duration_seconds: float, *, success: bool = True) -> None:
    if not metrics_enabled():
        return
    init_metrics()
    name = (agent or "unknown").strip() or "unknown"
    if _PROM_AVAILABLE and _agent_duration is not None:
        _agent_duration.labels(agent=name).observe(max(0.0, duration_seconds))
        _agent_runs.labels(agent=name, status="success" if success else "error").inc()
        return

    bucket = _fallback["agent_duration"].setdefault(name, {"sum": 0.0, "count": 0})
    bucket["sum"] += duration_seconds
    bucket["count"] += 1


def observe_http(method: str, path: str, status: int, duration_seconds: float) -> None:
    if not metrics_enabled():
        return
    init_metrics()
    # Keep cardinality low: strip ids from common paths
    norm = _normalize_path(path)
    if _PROM_AVAILABLE and _http_requests is not None:
        _http_requests.labels(method=method, path=norm, status=str(status)).inc()
        _http_latency.labels(method=method, path=norm).observe(max(0.0, duration_seconds))
        return
    key = f"{method}:{norm}:{status}"
    _fallback["http_requests"][key] = _fallback["http_requests"].get(key, 0) + 1


def _normalize_path(path: str) -> str:
    p = path or "/"
    # Collapse UUIDs / long ids
    parts = []
    for part in p.split("/"):
        if not part:
            continue
        if len(part) >= 32 and all(c in "0123456789abcdef-" for c in part.lower()):
            parts.append("{id}")
        elif part.isdigit():
            parts.append("{id}")
        else:
            parts.append(part)
    return "/" + "/".join(parts) if parts else "/"


@contextmanager
def time_agent(agent: str) -> Iterator[None]:
    start = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        observe_agent(agent, time.perf_counter() - start, success=ok)


def timed_node(agent_name: str, fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap a LangGraph node to record duration and accumulate agent_timings_ms."""

    @wraps(fn)
    def _wrapped(state: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        start = time.perf_counter()
        ok = True
        try:
            out = fn(state, *args, **kwargs) or {}
        except Exception:
            ok = False
            observe_agent(agent_name, time.perf_counter() - start, success=False)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        observe_agent(agent_name, elapsed_ms / 1000.0, success=ok)
        timings = dict(state.get("agent_timings_ms") or {})
        timings.update(dict(out.get("agent_timings_ms") or {}))
        # Accumulate if same agent runs twice (e.g. supervisor)
        prev = float(timings.get(agent_name) or 0.0)
        timings[agent_name] = round(prev + elapsed_ms, 2)
        out = dict(out)
        out["agent_timings_ms"] = timings
        return out

    return _wrapped


def render_metrics() -> tuple[bytes, str]:
    """Return (body, content_type) for GET /metrics."""
    init_metrics()
    if _PROM_AVAILABLE and _registry is not None and generate_latest is not None:
        return generate_latest(_registry), CONTENT_TYPE_LATEST

    # Fallback text exposition
    lines = [
        "# HELP kta_ask_total Total ask invocations (fallback)",
        "# TYPE kta_ask_total counter",
        f"kta_ask_total {_fallback['ask_total']}",
        "# TYPE kta_ask_errors_total counter",
        f"kta_ask_errors_total {_fallback['ask_errors']}",
        "# TYPE kta_ask_guardrails_blocked_total counter",
        f"kta_ask_guardrails_blocked_total {_fallback['ask_blocked']}",
    ]
    if _fallback["ask_latency_count"]:
        avg = _fallback["ask_latency_sum"] / _fallback["ask_latency_count"]
        lines.append(f"kta_ask_latency_seconds_avg {avg}")
    for agent, stats in _fallback["agent_duration"].items():
        if stats["count"]:
            lines.append(
                f'kta_agent_duration_seconds_avg{{agent="{agent}"}} '
                f'{stats["sum"] / stats["count"]}'
            )
    body = ("\n".join(lines) + "\n").encode("utf-8")
    return body, "text/plain; version=0.0.4; charset=utf-8"
