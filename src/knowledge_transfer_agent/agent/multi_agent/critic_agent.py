"""Critic agent: citation + groundedness checks (legacy reflection_node)."""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.agent.multi_agent.trace import stamp_agent
from knowledge_transfer_agent.agent.nodes import reflection_node
from knowledge_transfer_agent.agent.state import AgentState


def critic_agent(state: AgentState) -> dict[str, Any]:
    """Validate answer groundedness and citation markers."""
    updates = reflection_node(state)
    return stamp_agent(state, "critic", updates)
