"""
Load and cache external guardrail policy packs (JSON).

Policy lives outside application logic so security/ops can update
injection patterns, keyword deny-lists, and PII detectors without code changes.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_RELATIVE_PATH = "config/guardrails_rules.json"
_lock = threading.RLock()
_cached: GuardrailPolicy | None = None
_cached_path: Path | None = None
_cached_mtime: float | None = None


def project_root() -> Path:
    """Repository root (…/Enterprise Knowledge Transfer Agent)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class InjectionRule:
    id: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class PiiRule:
    id: str
    pattern: re.Pattern[str]
    replacement: str
    flag: str
    validator: str | None = None


@dataclass(frozen=True)
class GuardrailPolicy:
    """Compiled policy snapshot loaded from JSON."""

    version: int
    source_path: Path
    injection_rules: tuple[InjectionRule, ...] = field(default_factory=tuple)
    blocked_keywords: tuple[str, ...] = field(default_factory=tuple)
    pii_rules: tuple[PiiRule, ...] = field(default_factory=tuple)

    @property
    def mtime(self) -> float:
        try:
            return self.source_path.stat().st_mtime
        except OSError:
            return 0.0


class GuardrailPolicyError(RuntimeError):
    """Raised when the policy file is missing or invalid."""


def resolve_rules_path(path: str | Path | None = None) -> Path:
    """Resolve absolute path for the policy pack."""
    raw = str(path or _DEFAULT_RELATIVE_PATH).strip() or _DEFAULT_RELATIVE_PATH
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise GuardrailPolicyError(f"Policy field '{key}' must be a list")
    return value


def load_guardrail_policy(path: str | Path | None = None) -> GuardrailPolicy:
    """
    Load and compile a policy JSON file.

    Expected shape:
      version, prompt_injection_patterns[{id, pattern}],
      blocked_keywords[str], pii_patterns[{id, pattern, replacement, flag, validator?}]
    """
    rules_path = resolve_rules_path(path)
    if not rules_path.is_file():
        raise GuardrailPolicyError(
            f"Guardrail policy file not found: {rules_path}. "
            "Set GUARDRAILS_RULES_PATH or create config/guardrails_rules.json"
        )

    try:
        raw = json.loads(rules_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GuardrailPolicyError(f"Invalid JSON in {rules_path}: {e}") from e

    if not isinstance(raw, dict):
        raise GuardrailPolicyError("Policy root must be a JSON object")

    version = int(raw.get("version") or 1)
    injection: list[InjectionRule] = []
    for i, item in enumerate(_require_list(raw, "prompt_injection_patterns")):
        if not isinstance(item, dict):
            raise GuardrailPolicyError(f"prompt_injection_patterns[{i}] must be an object")
        rid = str(item.get("id") or f"injection_{i}").strip()
        pattern = str(item.get("pattern") or "").strip()
        if not pattern:
            raise GuardrailPolicyError(f"prompt_injection_patterns[{i}] missing pattern")
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise GuardrailPolicyError(
                f"Invalid injection regex id={rid}: {e}"
            ) from e
        injection.append(InjectionRule(id=rid, pattern=compiled))

    keywords: list[str] = []
    for i, item in enumerate(_require_list(raw, "blocked_keywords")):
        if not isinstance(item, str) or not item.strip():
            raise GuardrailPolicyError(f"blocked_keywords[{i}] must be a non-empty string")
        keywords.append(item.strip())

    pii: list[PiiRule] = []
    for i, item in enumerate(_require_list(raw, "pii_patterns")):
        if not isinstance(item, dict):
            raise GuardrailPolicyError(f"pii_patterns[{i}] must be an object")
        rid = str(item.get("id") or f"pii_{i}").strip()
        pattern = str(item.get("pattern") or "").strip()
        replacement = str(item.get("replacement") or "[REDACTED]").strip()
        flag = str(item.get("flag") or f"pii_{rid}").strip()
        validator = item.get("validator")
        validator_s = str(validator).strip().lower() if validator else None
        if not pattern:
            raise GuardrailPolicyError(f"pii_patterns[{i}] missing pattern")
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise GuardrailPolicyError(f"Invalid PII regex id={rid}: {e}") from e
        pii.append(
            PiiRule(
                id=rid,
                pattern=compiled,
                replacement=replacement,
                flag=flag,
                validator=validator_s,
            )
        )

    policy = GuardrailPolicy(
        version=version,
        source_path=rules_path,
        injection_rules=tuple(injection),
        blocked_keywords=tuple(keywords),
        pii_rules=tuple(pii),
    )
    logger.info(
        "Loaded guardrail policy v%s from %s (%d injection, %d keywords, %d pii)",
        policy.version,
        rules_path,
        len(policy.injection_rules),
        len(policy.blocked_keywords),
        len(policy.pii_rules),
    )
    return policy


def get_guardrail_policy(
    *,
    path: str | Path | None = None,
    settings: Any | None = None,
    force_reload: bool = False,
) -> GuardrailPolicy:
    """
    Return cached policy; reload when file mtime changes or force_reload=True.
    """
    global _cached, _cached_path, _cached_mtime

    cfg_path = path
    if cfg_path is None and settings is not None:
        cfg_path = getattr(settings, "guardrails_rules_path", None)
    if cfg_path is None:
        try:
            from knowledge_transfer_agent.config import get_settings

            cfg_path = get_settings().guardrails_rules_path
        except Exception:
            cfg_path = _DEFAULT_RELATIVE_PATH

    rules_path = resolve_rules_path(cfg_path)

    with _lock:
        mtime = rules_path.stat().st_mtime if rules_path.is_file() else None
        if (
            not force_reload
            and _cached is not None
            and _cached_path == rules_path
            and _cached_mtime == mtime
        ):
            return _cached

        policy = load_guardrail_policy(rules_path)
        _cached = policy
        _cached_path = rules_path
        _cached_mtime = policy.mtime
        return policy


def clear_guardrail_policy_cache() -> None:
    """Clear cached policy (tests / hot-reload)."""
    global _cached, _cached_path, _cached_mtime
    with _lock:
        _cached = None
        _cached_path = None
        _cached_mtime = None
