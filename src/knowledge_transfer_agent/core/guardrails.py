"""
Input/output guardrails for the knowledge transfer agent.

Rules (injection regex, keyword deny-list, PII detectors) are loaded from an
external policy pack — see config/guardrails_rules.json and GUARDRAILS_RULES_PATH.
This module is the enforcement engine only; it does not hardcode policy content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.exceptions import GuardrailsError
from knowledge_transfer_agent.core.guardrail_policy import (
    GuardrailPolicy,
    get_guardrail_policy,
)
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GuardrailResult:
    """Result of applying guardrails to text."""

    text: str
    blocked: bool = False
    flags: list[str] = field(default_factory=list)
    reason: str = ""


def _luhn_ok(digits: str) -> bool:
    """Basic Luhn check so we don't redact every long number."""
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _policy(settings: Any | None = None) -> GuardrailPolicy:
    return get_guardrail_policy(settings=settings or get_settings())


def redact_pii(text: str, *, settings: Any | None = None) -> tuple[str, list[str]]:
    """Redact PII / secrets using patterns from the policy pack."""
    flags: list[str] = []
    out = text or ""
    policy = _policy(settings)

    for rule in policy.pii_rules:
        if rule.validator == "luhn":

            def _card(
                m: re.Match[str],
                *,
                replacement: str = rule.replacement,
                flag: str = rule.flag,
            ) -> str:
                raw = re.sub(r"[^\d]", "", m.group(0))
                if _luhn_ok(raw):
                    if flag not in flags:
                        flags.append(flag)
                    return replacement
                return m.group(0)

            out = rule.pattern.sub(_card, out)
            continue

        def _repl(
            m: re.Match[str],
            *,
            replacement: str = rule.replacement,
            flag: str = rule.flag,
        ) -> str:
            if flag not in flags:
                flags.append(flag)
            return replacement

        out = rule.pattern.sub(_repl, out)

    return out, flags


def detect_prompt_injection(
    text: str,
    *,
    settings: Any | None = None,
) -> list[str]:
    """Return matched injection rule ids (empty if clean)."""
    hits: list[str] = []
    for rule in _policy(settings).injection_rules:
        if rule.pattern.search(text or ""):
            hits.append(f"injection:{rule.id}")
    return hits


def parse_blocked_keywords(
    raw: str | None = None,
    *,
    settings: Any | None = None,
) -> list[str]:
    """
    Resolve deny-list.

    Priority:
      1. Explicit `raw` argument
      2. Non-empty GUARDRAILS_BLOCKED_KEYWORDS (ops override)
      3. Policy file `blocked_keywords`
    """
    cfg = settings or get_settings()
    if raw is not None:
        return [k.strip() for k in str(raw).split(",") if k.strip()]

    env_raw = getattr(cfg, "guardrails_blocked_keywords", "") or ""
    env_keywords = [k.strip() for k in str(env_raw).split(",") if k.strip()]
    if env_keywords:
        return env_keywords

    return list(_policy(cfg).blocked_keywords)


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Word-boundary match for single tokens; literal substring for multi-word phrases."""
    escaped = re.escape(keyword.strip())
    if " " in keyword.strip():
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def detect_blocked_keywords(
    text: str,
    *,
    keywords: list[str] | None = None,
    settings: Any | None = None,
) -> list[str]:
    """Return flags like keyword:exfiltrate for each deny-list hit."""
    cfg = settings or get_settings()
    if not getattr(cfg, "guardrails_block_keywords", True):
        return []
    kws = keywords if keywords is not None else parse_blocked_keywords(settings=cfg)
    hits: list[str] = []
    body = text or ""
    for kw in kws:
        if _keyword_pattern(kw).search(body):
            hits.append(f"keyword:{kw.lower()}")
    return hits


def redact_blocked_keywords(
    text: str,
    *,
    keywords: list[str] | None = None,
    settings: Any | None = None,
) -> tuple[str, list[str]]:
    """Replace deny-list hits with [REDACTED_KEYWORD]. Returns (text, flags)."""
    cfg = settings or get_settings()
    if not getattr(cfg, "guardrails_block_keywords", True):
        return text or "", []
    kws = keywords if keywords is not None else parse_blocked_keywords(settings=cfg)
    flags: list[str] = []
    out = text or ""
    for kw in sorted(kws, key=len, reverse=True):
        pat = _keyword_pattern(kw)

        def _repl(_m: re.Match[str], *, label: str = kw.lower()) -> str:
            flag = f"keyword:{label}"
            if flag not in flags:
                flags.append(flag)
            return "[REDACTED_KEYWORD]"

        out = pat.sub(_repl, out)
    return out, flags


def apply_input_guardrails(
    question: str,
    *,
    settings: Any | None = None,
) -> GuardrailResult:
    """
    Validate and sanitize user question.

    Raises GuardrailsError when the request must be blocked (injection / empty / too long).
    When called with raise_on_block=False via apply_input_guardrails_soft, returns blocked=True.
    """
    return _apply_input(question, settings=settings, raise_on_block=True)


def apply_input_guardrails_soft(
    question: str,
    *,
    settings: Any | None = None,
) -> GuardrailResult:
    """Same checks as apply_input_guardrails but returns blocked=True instead of raising."""
    return _apply_input(question, settings=settings, raise_on_block=False)


def _apply_input(
    question: str,
    *,
    settings: Any | None,
    raise_on_block: bool,
) -> GuardrailResult:
    cfg = settings or get_settings()
    if not getattr(cfg, "guardrails_enabled", True):
        return GuardrailResult(text=question or "", flags=["guardrails_disabled"])

    flags: list[str] = []
    text = (question or "").strip()

    max_chars = int(getattr(cfg, "guardrails_max_question_chars", 2000))
    if not text:
        reason = "Question is empty"
        if raise_on_block:
            raise GuardrailsError(reason, details={"flags": ["empty"]})
        return GuardrailResult(text="", blocked=True, flags=["empty"], reason=reason)

    if len(text) > max_chars:
        reason = f"Question exceeds maximum length ({max_chars} characters)"
        if raise_on_block:
            raise GuardrailsError(
                reason,
                details={"flags": ["too_long"], "max_chars": max_chars, "length": len(text)},
            )
        return GuardrailResult(
            text=text[:max_chars],
            blocked=True,
            flags=["too_long"],
            reason=reason,
        )

    if getattr(cfg, "guardrails_block_prompt_injection", True):
        injections = detect_prompt_injection(text, settings=cfg)
        if injections:
            flags.extend(injections)
            reason = "Request blocked by prompt-injection guardrail"
            logger.warning("Prompt injection blocked: %s", injections[:3])
            if raise_on_block:
                raise GuardrailsError(reason, details={"flags": flags})
            return GuardrailResult(text=text, blocked=True, flags=flags, reason=reason)

    if getattr(cfg, "guardrails_block_keywords", True):
        keyword_hits = detect_blocked_keywords(text, settings=cfg)
        if keyword_hits:
            flags.extend(keyword_hits)
            reason = "Request blocked by keyword deny-list guardrail"
            logger.warning("Blocked keywords: %s", keyword_hits[:5])
            if raise_on_block:
                raise GuardrailsError(reason, details={"flags": flags})
            return GuardrailResult(text=text, blocked=True, flags=flags, reason=reason)

    if getattr(cfg, "guardrails_redact_pii", True):
        text, pii_flags = redact_pii(text, settings=cfg)
        flags.extend(pii_flags)

    if flags:
        logger.info("Input guardrails flags: %s", flags)
    return GuardrailResult(text=text, blocked=False, flags=flags)


def apply_output_guardrails(
    answer: str,
    *,
    settings: Any | None = None,
) -> GuardrailResult:
    """Sanitize model output (PII / secrets / blocked keywords) before returning to the client."""
    cfg = settings or get_settings()
    if not getattr(cfg, "guardrails_enabled", True):
        return GuardrailResult(text=answer or "", flags=["guardrails_disabled"])
    if not getattr(cfg, "guardrails_check_output", True):
        return GuardrailResult(text=answer or "")

    text = answer or ""
    flags: list[str] = []
    if getattr(cfg, "guardrails_block_keywords", True):
        text, kw_flags = redact_blocked_keywords(text, settings=cfg)
        flags.extend(kw_flags)
    if getattr(cfg, "guardrails_redact_pii", True):
        text, pii_flags = redact_pii(text, settings=cfg)
        flags.extend(pii_flags)

    if flags:
        logger.info("Output guardrails flags: %s", flags)
    return GuardrailResult(text=text, flags=flags)
