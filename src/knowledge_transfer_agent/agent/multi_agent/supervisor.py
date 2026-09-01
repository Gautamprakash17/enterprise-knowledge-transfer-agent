"""
Supervisor agent: deterministic router for the multi-agent graph.

Decides next specialist without an extra LLM call (same control flow as the
legacy conditional edges, expressed as explicit agents).
"""

from __future__ import annotations

from typing import Any, Literal

from knowledge_transfer_agent.agent.multi_agent.trace import stamp_agent
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.config import get_settings

NextAgent = Literal["retriever", "writer", "critic", "finish"]


def decide_next_agent(state: AgentState) -> NextAgent:
    """
    Route based on which agent last ran and critic outcome.

    - guardrails blocked → finish
    - after guardrails → shared_memory (load) then retriever — handled in graph edges
    - after shared_memory → retriever
    - after retriever → writer
    - after writer → critic
    - after critic fail with retries left → retriever
    - after critic pass (or retries exhausted) → finish
    """
    if state.get("guardrails_blocked"):
        return "finish"

    active = (state.get("active_agent") or "").strip()

    if not active or active == "guardrails":
        return "retriever"

    if active == "shared_memory":
        return "retriever"

    if active == "retriever":
        return "writer"

    if active == "writer":
        return "critic"

    if active == "critic":
        settings = get_settings()
        is_valid = state.get("is_valid", True)
        retries = state.get("reflection_retries", 0)
        max_retries = state.get(
            "max_reflection_retries",
            settings.max_reflection_retries,
        )
        if not is_valid and retries < max_retries:
            return "retriever"
        return "finish"

    # Fallback (e.g. unexpected active_agent): restart retrieval
    return "retriever"


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Supervisor step: record visit and set next_agent."""
    nxt = decide_next_agent(state)
    return stamp_agent(state, "supervisor", {"next_agent": nxt})


def route_from_supervisor(state: AgentState) -> str:
    """LangGraph conditional edge target from supervisor."""
    nxt = state.get("next_agent") or decide_next_agent(state)
    if nxt == "finish":
        return "memory"
    if nxt in ("retriever", "writer", "critic"):
        return nxt
    return "retriever"
