"""Shared memory agents: load before specialists, save after successful answers."""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.agent.multi_agent.trace import stamp_agent
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.shared_memory import (
    load_shared_memory_for_question,
    save_qa_memory,
)
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


def shared_memory_load_agent(state: AgentState) -> dict[str, Any]:
    """Load workspace-scoped shared memories into state for Writer context."""
    settings = get_settings()
    if not settings.shared_memory_enabled or state.get("guardrails_blocked"):
        return stamp_agent(
            state,
            "shared_memory",
            {
                "shared_memory_context": "",
                "shared_memories": [],
            },
        )

    workspace_id = (state.get("workspace_id") or "default").strip() or "default"
    question = state.get("question", "") or ""
    context, memories = load_shared_memory_for_question(
        workspace_id=workspace_id,
        question=question,
    )
    logger.debug(
        "Loaded %d shared memories for workspace=%s",
        len(memories),
        workspace_id,
    )
    return stamp_agent(
        state,
        "shared_memory",
        {
            "shared_memory_context": context,
            "shared_memories": memories,
        },
    )


def shared_memory_save_agent(state: AgentState) -> dict[str, Any]:
    """Persist episodic memory after confidence is assigned (successful answers only)."""
    settings = get_settings()
    extras: dict[str, Any] = {}
    if (
        settings.shared_memory_enabled
        and settings.shared_memory_write_enabled
        and not state.get("guardrails_blocked")
    ):
        workspace_id = (state.get("workspace_id") or "default").strip() or "default"
        saved = save_qa_memory(
            workspace_id=workspace_id,
            question=state.get("question", "") or "",
            answer=state.get("answer", "") or "",
            thread_id=state.get("thread_id"),
            confidence=float(state.get("confidence_score") or 0.0),
        )
        if saved:
            extras["shared_memory_saved_id"] = saved["id"]

    return stamp_agent(state, "shared_memory_save", extras)
