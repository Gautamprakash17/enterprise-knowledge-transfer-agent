"""Tests for Prometheus metrics helpers."""

from __future__ import annotations

from knowledge_transfer_agent.core import metrics as metrics_mod
from knowledge_transfer_agent.core.metrics import (
    init_metrics,
    observe_agent,
    observe_ask,
    observe_http,
    render_metrics,
    timed_node,
)


def setup_function() -> None:
    # Reset module state between tests
    metrics_mod._initialized = False
    metrics_mod._registry = None
    metrics_mod._ask_total = None
    metrics_mod._fallback = {
        "ask_total": 0,
        "ask_errors": 0,
        "ask_blocked": 0,
        "ask_latency_sum": 0.0,
        "ask_latency_count": 0,
        "agent_duration": {},
        "http_requests": {},
    }


def test_observe_and_render_metrics():
    init_metrics()
    observe_ask(latency_seconds=1.25, success=True)
    observe_agent("retriever", 0.4, success=True)
    observe_http("GET", "/api/v1/health", 200, 0.01)
    body, ctype = render_metrics()
    text = body.decode("utf-8")
    assert "kta_ask" in text
    assert "text/plain" in ctype or "prometheus" in ctype or "version=" in ctype


def test_timed_node_records_timings():
    def fake_node(state):
        return {"answer": "ok"}

    wrapped = timed_node("writer", fake_node)
    out = wrapped({"agent_timings_ms": {}})
    assert out["answer"] == "ok"
    assert "writer" in out["agent_timings_ms"]
    assert out["agent_timings_ms"]["writer"] >= 0
