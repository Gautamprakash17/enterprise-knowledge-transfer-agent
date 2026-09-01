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
    top_k_semantic: int = Field(default=6, alias="TOP_K_SEMANTIC")
    top_k_keyword: int = Field(default=2, alias="TOP_K_KEYWORD")
    retrieval_score_threshold: float = Field(default=0.7, alias="RETRIEVAL_SCORE_THRESHOLD")

    # Agent Confidence Configuration
    confidence_valid: float = Field(default=0.9, alias="CONFIDENCE_VALID")
    confidence_invalid: float = Field(default=0.2, alias="CONFIDENCE_INVALID")
    confidence_no_docs: float = Field(default=0.0, alias="CONFIDENCE_NO_DOCS")
    confidence_penalty_per_issue: float = Field(default=0.1, alias="CONFIDENCE_PENALTY_PER_ISSUE")

    # Confluence Configuration (optional)
    confluence_url: Optional[str] = Field(default=None, alias="CONFLUENCE_URL")
    confluence_token: Optional[str] = Field(default=None, alias="CONFLUENCE_TOKEN")
    confluence_username: Optional[str] = Field(default=None, alias="CONFLUENCE_USERNAME")
    confluence_space_keys: Optional[str] = Field(default=None, alias="CONFLUENCE_SPACE_KEYS")
    confluence_cloud: bool = Field(default=True, alias="CONFLUENCE_CLOUD")
    confluence_page_batch_size: int = Field(default=100, alias="CONFLUENCE_PAGE_BATCH_SIZE")
    confluence_request_delay_seconds: float = Field(default=0.0, alias="CONFLUENCE_REQUEST_DELAY_SECONDS")

    # GitHub Configuration (optional)
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")
    github_repos: Optional[str] = Field(default=None, alias="GITHUB_REPOS")
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

    # Cache
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(default=1000, alias="CACHE_MAX_SIZE")
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")

    # Rate Limiting
    rate_limit_requests: int = Field(default=60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=60, alias="RATE_LIMIT_PERIOD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
