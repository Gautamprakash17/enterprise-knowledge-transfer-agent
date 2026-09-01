"""
FAISS vector store management with reusable save/load functions.
"""

from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.retrieval.embeddings import get_embeddings

logger = get_logger(__name__)

_vector_store: Optional[FAISS] = None


# --- Reusable FAISS functions ---


def load_faiss_index(
    path: str | Path,
    embeddings: Any | None = None,
) -> FAISS:
    """
    Load FAISS index from disk. Reusable function.

    Args:
        path: Directory containing index.faiss and index.pkl
        embeddings: Optional embeddings (default: from config)

    Returns:
        FAISS vector store
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at {path}")
    emb = embeddings or get_embeddings()
    store = FAISS.load_local(
        str(path),
        emb,
        allow_dangerous_deserialization=True,
    )
    logger.info("Loaded FAISS index from %s", path)
    return store


def save_faiss_index(store: FAISS, path: str | Path) -> None:
    """
    Save FAISS index to disk. Reusable function.

    Args:
        store: FAISS vector store
        path: Output directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    logger.info("Saved FAISS index to %s", path)


def create_faiss_index(
    documents: list[Document],
    embeddings: Any | None = None,
) -> FAISS:
    """
    Create FAISS index from documents (embeds in process).

    Args:
        documents: LangChain documents with metadata
        embeddings: Optional embeddings (default: from config)

    Returns:
        FAISS vector store
    """
    if not documents:
        raise ValueError("Cannot create index from empty document list")
    emb = embeddings or get_embeddings()
    store = FAISS.from_documents(documents, emb)
    logger.info("Created FAISS index with %d documents", len(documents))
    return store


def add_documents_to_store(store: FAISS, documents: list[Document]) -> None:
    """
    Add documents to an existing FAISS store.

    Args:
        store: FAISS vector store
        documents: Documents to add
    """
    if documents:
        store.add_documents(documents)
        logger.info("Added %d documents to FAISS index", len(documents))


# --- Application-level API (uses global cache) ---


def get_vector_store(reload: bool = False) -> FAISS:
    """
    Get or create the FAISS vector store.
    Loads from disk if index exists; otherwise returns None (caller should create from docs).

    Args:
        reload: If True, force reload from disk (if exists)

    Returns:
        FAISS vector store instance

    Raises:
        FileNotFoundError: If no index exists on disk (use create_vector_store_from_documents)
    """
    global _vector_store
    settings = get_settings()
    store_path = Path(settings.vector_store_path)

    if _vector_store is not None and not reload:
        return _vector_store

    if store_path.exists():
        try:
            _vector_store = load_faiss_index(store_path)
            return _vector_store
        except Exception as e:
            logger.warning("Failed to load existing index: %s", e)

    raise FileNotFoundError(
        f"No FAISS index at {store_path}. Run ingestion pipeline first."
    )


def create_vector_store_from_documents(documents: list[Document]) -> FAISS:
    """Create a new FAISS store from documents (for initial ingestion)."""
    global _vector_store
    _vector_store = create_faiss_index(documents)
    return _vector_store


def get_or_create_vector_store(
    documents: list[Document] | None = None,
    reload: bool = False,
) -> FAISS:
    """Get existing store or create from documents if none exists."""
    try:
        return get_vector_store(reload=reload)
    except FileNotFoundError:
        if documents:
            return create_vector_store_from_documents(documents)
        raise


def reset_vector_store() -> None:
    """Reset the in-memory vector store (e.g., after ingest or for testing)."""
    global _vector_store
    _vector_store = None
    try:
        from knowledge_transfer_agent.core.cache import invalidate_all_caches

        invalidate_all_caches()
    except Exception:
        logger.debug("Could not invalidate caches after vector store reset", exc_info=True)
