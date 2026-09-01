"""Writer agent: grounded answer generation with [N] citations."""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.agent.multi_agent.trace import stamp_agent
from knowledge_transfer_agent.agent.nodes import generate_node
from knowledge_transfer_agent.agent.state import AgentState


def writer_agent(state: AgentState) -> dict[str, Any]:
    """Generate a citation-backed answer from retrieved context."""
    updates = generate_node(state)
    return stamp_agent(state, "writer", updates)
