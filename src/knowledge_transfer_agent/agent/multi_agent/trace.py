"""Helpers for multi-agent state updates."""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.agent.state import AgentState


def append_agent_trace(state: AgentState, agent_name: str) -> list[str]:
    """Return agent_trace with agent_name appended."""
    trace = list(state.get("agent_trace") or [])
    trace.append(agent_name)
    return trace


def stamp_agent(state: AgentState, agent_name: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge specialist output with active_agent + agent_trace stamps."""
    out: dict[str, Any] = dict(extra or {})
    out["active_agent"] = agent_name
    out["agent_trace"] = append_agent_trace(state, agent_name)
    return out
