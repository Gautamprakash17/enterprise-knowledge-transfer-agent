"""
LLM client with retry, caching, and structured output.
"""

import time
from dataclasses import dataclass
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.cache import cache_key, get_cache
from knowledge_transfer_agent.core.exceptions import LLMError
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class _CircuitBreakerState:
    failures: int = 0
    open_until_ts: float = 0.0


_breaker = _CircuitBreakerState()


def _exponential_backoff(attempt: int, min_wait: float, max_wait: float) -> float:
    """Compute backoff seconds for attempt (0-indexed)."""
    import random
    wait = min(min_wait * (2 ** attempt), max_wait)
    jitter = random.uniform(0, wait * 0.1)
    return wait + jitter


def get_llm_with_retry() -> ChatOpenAI:
    """Get LLM configured with retry logic."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_retries=settings.llm_max_retries,
    )


def invoke_with_retry(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    inputs: dict[str, Any],
) -> str:
    """
    Invoke LLM with retry and exponential backoff.
    """
    settings = get_settings()
    last_error: Exception | None = None

    # Circuit breaker: fail fast when upstream is unhealthy.
    if settings.circuit_breaker_enabled and _breaker.open_until_ts > time.time():
        raise LLMError(
            "LLM temporarily unavailable (circuit breaker open)",
            details={
                "cooldown_seconds": settings.circuit_breaker_cooldown_seconds,
                "open_until_ts": _breaker.open_until_ts,
                "failures": _breaker.failures,
            },
        )

    for attempt in range(settings.llm_max_retries + 1):
        try:
            chain = prompt | llm
            result = chain.invoke(inputs)
            text = result.content.strip() if hasattr(result, "content") else str(result)
            # Success -> reset breaker
            if settings.circuit_breaker_enabled:
                _breaker.failures = 0
                _breaker.open_until_ts = 0.0
            return text
        except Exception as e:
            last_error = e
            logger.warning("LLM attempt %d failed: %s", attempt + 1, str(e))
            if settings.circuit_breaker_enabled:
                _breaker.failures += 1
                if _breaker.failures >= settings.circuit_breaker_failure_threshold:
                    _breaker.open_until_ts = time.time() + float(
                        settings.circuit_breaker_cooldown_seconds
                    )
                    logger.warning(
                        "Circuit breaker opened for %ss after %s failures",
                        settings.circuit_breaker_cooldown_seconds,
                        _breaker.failures,
                    )
            if attempt < settings.llm_max_retries:
                wait = _exponential_backoff(
                    attempt,
                    settings.llm_retry_min_wait,
                    settings.llm_retry_max_wait,
                )
                logger.info("Retrying in %.1fs", wait)
                time.sleep(wait)

    raise LLMError(
        f"LLM failed after {settings.llm_max_retries + 1} attempts",
        details={"error": str(last_error) if last_error else "unknown"},
    ) from last_error


def invoke_cached(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    inputs: dict[str, Any],
    cache_prefix: str = "llm",
) -> str:
    """Invoke LLM with optional response caching."""
    settings = get_settings()
    if not settings.cache_enabled:
        return invoke_with_retry(llm, prompt, inputs)

    key = cache_key(cache_prefix, str(inputs))
    cache = get_cache()
    cached = cache.get(key)
    if cached is not None:
        logger.debug("Cache hit for %s", cache_prefix)
        return cached

    result = invoke_with_retry(llm, prompt, inputs)
    cache.set(key, result, ttl_seconds=settings.cache_ttl_seconds)
    return result
