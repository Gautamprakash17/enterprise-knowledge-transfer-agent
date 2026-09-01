"""Pytest fixtures."""

import os
import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("VECTOR_STORE_PATH", "/tmp/test_faiss_index")
    monkeypatch.setenv("ENABLE_RBAC", "false")
