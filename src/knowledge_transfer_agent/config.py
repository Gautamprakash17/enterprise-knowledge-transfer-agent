"""
Environment-based configuration management.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    app_name: str = Field(default="Knowledge Transfer Agent", alias="APP_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # OpenAI Configuration
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    # Vector Store Configuration
    vector_store_path: str = Field(default="./data/faiss_index", alias="VECTOR_STORE_PATH")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # Retrieval Configuration
    # Total chunks passed to the LLM ≈ top_k_semantic + top_k_keyword (after dedup).
    # Slightly favor semantic FAISS hits; keep MMR (keyword path) small to reduce off-topic diversity.
    top_k_semantic: int = Field(default=6, alias="TOP_K_SEMANTIC")
    top_k_keyword: int = Field(default=2, alias="TOP_K_KEYWORD")
    retrieval_score_threshold: float = Field(default=0.7, alias="RETRIEVAL_SCORE_THRESHOLD")

    # Reranking (FlashRank cross-encoder; runs after hybrid retrieval)
    rerank_enabled: bool = Field(default=False, alias="RERANK_ENABLED")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    rerank_model: str = Field(
        default="ms-marco-TinyBERT-L-2-v2",
        alias="RERANK_MODEL",
    )

    # Agent Confidence Configuration
    confidence_valid: float = Field(default=0.9, alias="CONFIDENCE_VALID")
    confidence_invalid: float = Field(default=0.2, alias="CONFIDENCE_INVALID")
    confidence_no_docs: float = Field(default=0.0, alias="CONFIDENCE_NO_DOCS")
    confidence_penalty_per_issue: float = Field(default=0.1, alias="CONFIDENCE_PENALTY_PER_ISSUE")

    # Confluence Configuration (optional)
    confluence_url: Optional[str] = Field(default=None, alias="CONFLUENCE_URL")
    confluence_token: Optional[str] = Field(default=None, alias="CONFLUENCE_TOKEN")
    # Atlassian Cloud: often your Atlassian account email (with API token as password in some clients).
    confluence_username: Optional[str] = Field(default=None, alias="CONFLUENCE_USERNAME")
    confluence_space_keys: Optional[str] = Field(default=None, alias="CONFLUENCE_SPACE_KEYS")
    confluence_cloud: bool = Field(default=True, alias="CONFLUENCE_CLOUD")
    confluence_page_batch_size: int = Field(default=100, alias="CONFLUENCE_PAGE_BATCH_SIZE")
    confluence_request_delay_seconds: float = Field(default=0.0, alias="CONFLUENCE_REQUEST_DELAY_SECONDS")

    # GitHub Configuration (optional)
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")
    github_repos: Optional[str] = Field(default=None, alias="GITHUB_REPOS")
    # Remote URLs in GITHUB_REPOS are cloned here, then scanned (git must be installed).
    github_clone_cache_dir: str = Field(default="./data/git_clones", alias="GITHUB_CLONE_CACHE_DIR")
    github_shallow_clone: bool = Field(default=True, alias="GITHUB_SHALLOW_CLONE")
    github_clone_branch: Optional[str] = Field(default=None, alias="GITHUB_CLONE_BRANCH")
    github_clone_timeout_seconds: int = Field(default=300, alias="GITHUB_CLONE_TIMEOUT_SECONDS")

    # Security
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    enable_rbac: bool = Field(default=True, alias="ENABLE_RBAC")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # LLM Retry
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_retry_min_wait: float = Field(default=1.0, alias="LLM_RETRY_MIN_WAIT")
    llm_retry_max_wait: float = Field(default=10.0, alias="LLM_RETRY_MAX_WAIT")

    # LLM runtime controls (production hardening)
    reflection_enabled: bool = Field(default=True, alias="REFLECTION_ENABLED")

    # Context compression (reduce prompt size before generation)
    context_compression_enabled: bool = Field(default=False, alias="CONTEXT_COMPRESSION_ENABLED")
    context_compression_max_chars_per_chunk: int = Field(
        default=600, alias="CONTEXT_COMPRESSION_MAX_CHARS_PER_CHUNK"
    )

    circuit_breaker_enabled: bool = Field(default=True, alias="CIRCUIT_BREAKER_ENABLED")
    circuit_breaker_failure_threshold: int = Field(
        default=5, alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD"
    )
    circuit_breaker_cooldown_seconds: int = Field(
        default=30, alias="CIRCUIT_BREAKER_COOLDOWN_SECONDS"
    )

    # Cache
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(default=1000, alias="CACHE_MAX_SIZE")
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")

    # Rate Limiting
    rate_limit_requests: int = Field(default=60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, alias="RATE_LIMIT_PERIOD")

    # Agentic orchestration
    max_hops: int = Field(default=3, alias="MAX_HOPS")
    max_reflection_retries: int = Field(default=1, alias="MAX_REFLECTION_RETRIES")

    # Persistence (SQLite)
    database_path: str = Field(default="./data/kta.db", alias="DATABASE_PATH")
    persist_to_database: bool = Field(default=True, alias="PERSIST_TO_DATABASE")

    # Upload limits
    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    max_upload_files_per_batch: int = Field(
        default=2000, alias="MAX_UPLOAD_FILES_PER_BATCH"
    )
    allow_local_path_ingest: bool = Field(
        default=False, alias="ALLOW_LOCAL_PATH_INGEST"
    )
    allow_git_clone_ingest: bool = Field(
        default=True, alias="ALLOW_GIT_CLONE_INGEST"
    )

    # Guardrails (input/output safety)
    guardrails_enabled: bool = Field(default=True, alias="GUARDRAILS_ENABLED")
    guardrails_block_prompt_injection: bool = Field(
        default=True, alias="GUARDRAILS_BLOCK_PROMPT_INJECTION"
    )
    guardrails_redact_pii: bool = Field(default=True, alias="GUARDRAILS_REDACT_PII")
    guardrails_check_output: bool = Field(default=True, alias="GUARDRAILS_CHECK_OUTPUT")
    guardrails_max_question_chars: int = Field(
        default=2000, alias="GUARDRAILS_MAX_QUESTION_CHARS"
    )
    guardrails_block_keywords: bool = Field(
        default=True, alias="GUARDRAILS_BLOCK_KEYWORDS"
    )
    # Optional ops override (comma-separated). Empty = use policy file keywords.
    guardrails_blocked_keywords: str = Field(
        default="",
        alias="GUARDRAILS_BLOCKED_KEYWORDS",
    )
    # External policy pack (injection regex, keywords, PII patterns).
    guardrails_rules_path: str = Field(
        default="config/guardrails_rules.json",
        alias="GUARDRAILS_RULES_PATH",
    )

    # Shared long-term memory (workspace-scoped)
    shared_memory_enabled: bool = Field(default=True, alias="SHARED_MEMORY_ENABLED")
    shared_memory_write_enabled: bool = Field(
        default=True, alias="SHARED_MEMORY_WRITE_ENABLED"
    )
    shared_memory_max_items: int = Field(default=8, alias="SHARED_MEMORY_MAX_ITEMS")
    shared_memory_max_store: int = Field(default=200, alias="SHARED_MEMORY_MAX_STORE")
    shared_memory_min_confidence: float = Field(
        default=0.5, alias="SHARED_MEMORY_MIN_CONFIDENCE"
    )

    # Monitoring / Prometheus metrics
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
