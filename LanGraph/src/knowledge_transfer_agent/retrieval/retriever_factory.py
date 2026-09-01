"""
Factory for creating retrievers with config.
"""

from typing import Any, Optional

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.retrieval.hybrid_retriever import HybridRetriever
from knowledge_transfer_agent.retrieval.vector_store import get_vector_store


def get_hybrid_retriever(vector_store: Any = None) -> HybridRetriever:
    """
    Create HybridRetriever with settings from config.
    When vector_store is provided (e.g. from DI), use it; otherwise load from disk.
    """
    settings = get_settings()
    vs = vector_store if vector_store is not None else get_vector_store()
    return HybridRetriever(
        vector_store=vs,
        top_k_semantic=settings.top_k_semantic,
        top_k_keyword=settings.top_k_keyword,
        score_threshold=settings.retrieval_score_threshold,
    )
