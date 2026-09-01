"""Suggested follow-up questions after an answer."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_transfer_agent.agent.llm_client import get_llm_with_retry
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM = """You suggest short follow-up questions for an enterprise documentation assistant.
Return exactly 3 questions, one per line, no numbering, no bullets, under 15 words each.
Questions must be answerable from internal docs and relate to the user's topic."""


def suggest_followups(question: str, answer: str, max_suggestions: int = 3) -> list[str]:
    if not answer.strip():
        return []
    try:
        llm = get_llm_with_retry()
        msg = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(
                    content=f"User question:\n{question[:500]}\n\nAssistant answer:\n{answer[:1500]}"
                ),
            ]
        )
        text = getattr(msg, "content", "") or ""
        lines = [ln.strip().lstrip("0123456789.-) ") for ln in text.splitlines() if ln.strip()]
        return lines[:max_suggestions]
    except Exception as e:
        logger.warning("Follow-up suggestion failed: %s", e)
        return []
