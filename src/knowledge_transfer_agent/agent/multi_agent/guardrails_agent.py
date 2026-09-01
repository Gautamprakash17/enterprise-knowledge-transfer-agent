"""Guardrails agents: input checks at entry, output sanitization before save/return."""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.agent.multi_agent.trace import stamp_agent
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.core.guardrails import (
    apply_input_guardrails_soft,
    apply_output_guardrails,
)


_BLOCKED_MESSAGE = (
    "I can't process this request because it violates safety guardrails "
    "(for example prompt-injection, blocked keywords, or disallowed content)."
)


def _merge_flags(*flag_lists: list[str]) -> list[str]:
    """Dedupe flags while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for flags in flag_lists:
        for f in flags:
            if f and f not in seen:
                seen.add(f)
                out.append(f)
    return out


def guardrails_agent(state: AgentState) -> dict[str, Any]:
    """
    Run input guardrails on the user question.

    On block: set a safe refusal answer and guardrails_blocked=True so the
    supervisor finishes without calling retriever/writer.
    On pass: may redact PII in the question before retrieval/generation.
    """
    question = state.get("question", "") or ""
    result = apply_input_guardrails_soft(question)

    extra: dict[str, Any] = {
        "question": result.text,
        "guardrail_flags": list(result.flags),
        "guardrails_blocked": bool(result.blocked),
    }
    if result.blocked:
        extra.update(
            {
                "answer": _BLOCKED_MESSAGE,
                "retrieved_docs": [],
                "is_valid": True,
                "validation_issues": [],
                "reflection": result.reason or "Blocked by input guardrails",
            }
        )
    return stamp_agent(state, "guardrails", extra)


def output_guardrails_agent(state: AgentState) -> dict[str, Any]:
    """
    Sanitize the final answer (PII / secrets) before shared-memory save and client return.

    Runs after confidence so durable memory and API responses both get the cleaned text.
    Does not block the request; redacts sensitive patterns in place.
    """
    answer = state.get("answer", "") or ""
    result = apply_output_guardrails(answer)
    existing = list(state.get("guardrail_flags") or [])
    # Tag output-only flags so traces show which layer fired (keep raw too for metrics).
    tagged = [f if f.startswith("output:") else f"output:{f}" for f in result.flags]
    return stamp_agent(
        state,
        "output_guardrails",
        {
            "answer": result.text,
            "guardrail_flags": _merge_flags(existing, result.flags, tagged),
        },
    )
