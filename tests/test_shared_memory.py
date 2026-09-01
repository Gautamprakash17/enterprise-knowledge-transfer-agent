"""Tests for workspace-scoped shared memory."""

from __future__ import annotations

import pytest

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.database import init_database
from knowledge_transfer_agent.core.shared_memory import (
    clear_workspace_memories,
    format_memories_for_prompt,
    list_workspace_memories,
    load_shared_memory_for_question,
    save_preference_memory,
    save_qa_memory,
)
from knowledge_transfer_agent.agent.multi_agent.shared_memory_agent import (
    shared_memory_load_agent,
)
from knowledge_transfer_agent.agent.multi_agent.supervisor import decide_next_agent


@pytest.fixture(autouse=True)
def _memory_db(tmp_path, monkeypatch):
    db = tmp_path / "test_memory.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    monkeypatch.setenv("PERSIST_TO_DATABASE", "true")
    monkeypatch.setenv("SHARED_MEMORY_ENABLED", "true")
    monkeypatch.setenv("SHARED_MEMORY_WRITE_ENABLED", "true")
    monkeypatch.setenv("SHARED_MEMORY_MIN_CONFIDENCE", "0.5")
    get_settings.cache_clear()
    init_database()
    yield
    get_settings.cache_clear()


def test_save_and_list_isolation():
    save_qa_memory(
        workspace_id="proj-a",
        question="How do we deploy?",
        answer="We deploy via ECS with health checks. [1]",
        confidence=0.9,
    )
    save_qa_memory(
        workspace_id="proj-b",
        question="What DB do we use?",
        answer="PostgreSQL with replicas. [1]",
        confidence=0.9,
    )
    a = list_workspace_memories("proj-a")
    b = list_workspace_memories("proj-b")
    assert len(a) == 1
    assert len(b) == 1
    assert "deploy" in a[0]["content"].lower()
    assert "postgres" in b[0]["content"].lower()


def test_search_relevant_memories():
    save_qa_memory(
        workspace_id="ws1",
        question="Deploy pipeline steps",
        answer="Build image, push ECR, update ECS. [1]",
        confidence=0.9,
    )
    save_preference_memory(workspace_id="ws1", preference="Prefer bullet-point answers")
    ctx, mems = load_shared_memory_for_question(
        workspace_id="ws1",
        question="How does deployment work on ECS?",
    )
    assert mems
    assert "Shared" not in ctx  # format uses dashes only
    assert "deploy" in ctx.lower() or "ecs" in ctx.lower()


def test_skip_low_confidence_and_abstain():
    assert (
        save_qa_memory(
            workspace_id="ws1",
            question="x",
            answer="No sufficient data",
            confidence=0.9,
        )
        is None
    )
    assert (
        save_qa_memory(
            workspace_id="ws1",
            question="x",
            answer="Something useful [1]",
            confidence=0.2,
        )
        is None
    )


def test_format_memories_for_prompt():
    text = format_memories_for_prompt(
        [{"memory_type": "episodic", "content": "Q: hi\nA: hello"}]
    )
    assert "(episodic)" in text
    assert "hello" in text


def test_load_agent_sets_context_and_routes():
    save_qa_memory(
        workspace_id="default",
        question="What is FAISS used for?",
        answer="Vector search index for RAG. [1]",
        confidence=0.9,
    )
    out = shared_memory_load_agent(
        {"question": "Explain FAISS in this project", "workspace_id": "default"}
    )
    assert out["active_agent"] == "shared_memory"
    assert out.get("shared_memory_context")
    assert decide_next_agent({**out}) == "retriever"


def test_clear_workspace_memories():
    save_qa_memory(
        workspace_id="clr",
        question="q",
        answer="a grounded answer [1]",
        confidence=0.9,
    )
    n = clear_workspace_memories("clr")
    assert n >= 1
    assert list_workspace_memories("clr") == []
