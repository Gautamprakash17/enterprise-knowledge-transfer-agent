"""
Shared long-term memory for the multi-agent knowledge transfer system.

Scoped by workspace_id so Project A never reads Project B memories.
Used by agents as durable context across turns (preferences + episodic facts).
"""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.database import (
    clear_shared_memories,
    delete_shared_memory,
    init_database,
    insert_shared_memory,
    list_shared_memories,
    search_shared_memories,
)
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


def _ensure_db() -> None:
    settings = get_settings()
    if settings.persist_to_database:
        init_database()


def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return ""
    lines = []
    for m in memories:
        mtype = m.get("memory_type") or "episodic"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"- ({mtype}) {content}")
    return "\n".join(lines)


def load_shared_memory_for_question(
    *,
    workspace_id: str,
    question: str,
    limit: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Load workspace-scoped memories relevant to the question.

    Returns (prompt_block, raw_memory_dicts).
    """
    settings = get_settings()
    if not getattr(settings, "shared_memory_enabled", True):
        return "", []
    if not workspace_id:
        return "", []

    _ensure_db()
    max_items = limit or int(getattr(settings, "shared_memory_max_items", 8))
    memories = search_shared_memories(workspace_id, question, limit=max_items)
    if not memories:
        # Fall back to most recent workspace memories so follow-ups still benefit.
        memories = list_shared_memories(workspace_id, limit=max_items)
    return format_memories_for_prompt(memories), memories


def save_qa_memory(
    *,
    workspace_id: str,
    question: str,
    answer: str,
    thread_id: str | None = None,
    confidence: float = 0.0,
) -> dict[str, Any] | None:
    """Persist a short episodic memory after a successful grounded answer."""
    settings = get_settings()
    if not getattr(settings, "shared_memory_enabled", True):
        return None
    if not getattr(settings, "shared_memory_write_enabled", True):
        return None
    if not workspace_id:
        return None

    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return None
    if a.lower().startswith("i can't process this request"):
        return None
    if a.lower() in {"no sufficient data", "no response generated."}:
        return None
    if confidence < float(getattr(settings, "shared_memory_min_confidence", 0.5)):
        return None

    _ensure_db()
    max_q = 240
    max_a = 400
    content = f"Q: {q[:max_q]}\nA: {a[:max_a]}"
    mem = insert_shared_memory(
        workspace_id=workspace_id,
        thread_id=thread_id,
        memory_type="episodic",
        content=content,
        metadata={"confidence": confidence, "source": "ask"},
    )
    # Keep store bounded per workspace
    _trim_workspace(workspace_id, int(getattr(settings, "shared_memory_max_store", 200)))
    logger.debug("Saved shared memory %s for workspace %s", mem["id"], workspace_id)
    return mem


def save_preference_memory(
    *,
    workspace_id: str,
    preference: str,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    settings = get_settings()
    if not getattr(settings, "shared_memory_enabled", True):
        return None
    pref = (preference or "").strip()
    if not pref or not workspace_id:
        return None
    _ensure_db()
    return insert_shared_memory(
        workspace_id=workspace_id,
        thread_id=thread_id,
        memory_type="preference",
        content=pref[:500],
        metadata={"source": "explicit"},
    )


def _trim_workspace(workspace_id: str, max_store: int) -> None:
    if max_store <= 0:
        return
    rows = list_shared_memories(workspace_id, limit=max_store + 50)
    if len(rows) <= max_store:
        return
    for old in rows[max_store:]:
        delete_shared_memory(old["id"], workspace_id)


def list_workspace_memories(workspace_id: str, limit: int = 50) -> list[dict[str, Any]]:
    _ensure_db()
    return list_shared_memories(workspace_id, limit=limit)


def clear_workspace_memories(workspace_id: str) -> int:
    _ensure_db()
    return clear_shared_memories(workspace_id)
