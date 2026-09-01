"""
Optional cross-encoder reranking after hybrid retrieval (FlashRank).
"""

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def _flashrank_compressor(model: str, top_n: int) -> Any:
    """Reuse one Ranker per (model, top_n) — model download happens once."""
    from langchain_community.document_compressors import FlashrankRerank

    return FlashrankRerank(top_n=top_n, model=model)


def maybe_rerank(query: str, documents: list[Document]) -> list[Document]:
    """
    Re-order and trim documents by relevance to the query.
    If reranking is disabled, flashrank is missing, or anything fails, returns input order.
    """
    settings = get_settings()
    if not settings.rerank_enabled or len(documents) <= 1:
        return documents

    try:
        compressor = _flashrank_compressor(settings.rerank_model, settings.rerank_top_n)
        out = compressor.compress_documents(documents, query)
        result = list(out)
        if result:
            logger.debug("Reranked %d docs to %d", len(documents), len(result))
            return result
    except ImportError:
        logger.warning(
            "Rerank enabled but flashrank is not installed; pip install flashrank. "
            "Using hybrid order."
        )
    except Exception as e:
        logger.warning("Rerank failed, using hybrid retrieval order: %s", e)

    return documents
