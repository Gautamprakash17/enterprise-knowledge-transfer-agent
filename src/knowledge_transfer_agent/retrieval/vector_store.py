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

_vector_stores: dict[str, FAISS] = {}


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
    *,
    on_embed_progress: Any | None = None,
    embed_batch_size: int = 32,
) -> FAISS:
    """
    Create FAISS index from documents (embeds in process).

    Args:
        documents: LangChain documents with metadata
        embeddings: Optional embeddings (default: from config)
        on_embed_progress: Optional callback(done: int, total: int, message: str)
        embed_batch_size: Chunks per embedding API batch when on_embed_progress is set

    Returns:
        FAISS vector store
    """
    if not documents:
        raise ValueError("Cannot create index from empty document list")
    emb = embeddings or get_embeddings()
    total = len(documents)
    if on_embed_progress and total > embed_batch_size:
        store: FAISS | None = None
        for start in range(0, total, embed_batch_size):
            batch = documents[start : start + embed_batch_size]
            done = min(start + len(batch), total)
            on_embed_progress(
                done,
                total,
                f"Embedding chunks {done}/{total}",
            )
            if store is None:
                store = FAISS.from_documents(batch, emb)
            else:
                store.add_documents(batch)
        if store is None:
            raise ValueError("Cannot create index from empty document list")
        logger.info("Created FAISS index with %d documents (batched)", total)
        return store
    if on_embed_progress:
        on_embed_progress(total, total, f"Embedding chunks {total}/{total}")
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


def get_vector_store(
    reload: bool = False,
    *,
    workspace_id: str | None = None,
) -> FAISS:
    """
    Get the FAISS vector store for a workspace (project).

    Each workspace has its own index directory so Project A never retrieves Project B chunks.

    Args:
        reload: Force reload from disk
        workspace_id: Project id (default workspace uses legacy path if present)

    Raises:
        FileNotFoundError: If no index exists for this workspace yet
    """
    from knowledge_transfer_agent.core.workspaces import normalize_workspace_id, workspace_index_path

    global _vector_stores
    ws = normalize_workspace_id(workspace_id)
    store_path = workspace_index_path(ws)

    if ws in _vector_stores and not reload:
        return _vector_stores[ws]

    if (store_path / "index.faiss").is_file():
        try:
            _vector_stores[ws] = load_faiss_index(store_path)
            return _vector_stores[ws]
        except Exception as e:
            logger.warning("Failed to load index for workspace %s: %s", ws, e)

    raise FileNotFoundError(
        f"No FAISS index for workspace '{ws}' at {store_path}. "
        "Add documents to this project first."
    )


def create_vector_store_from_documents(
    documents: list[Document],
    *,
    workspace_id: str | None = None,
) -> FAISS:
    """Create a new FAISS store from documents (for initial ingestion)."""
    from knowledge_transfer_agent.core.workspaces import normalize_workspace_id

    global _vector_stores
    ws = normalize_workspace_id(workspace_id)
    _vector_stores[ws] = create_faiss_index(documents)
    return _vector_stores[ws]


def get_or_create_vector_store(
    documents: list[Document] | None = None,
    reload: bool = False,
    *,
    workspace_id: str | None = None,
) -> FAISS:
    """Get existing store or create from documents if none exists."""
    try:
        return get_vector_store(reload=reload, workspace_id=workspace_id)
    except FileNotFoundError:
        if documents:
            return create_vector_store_from_documents(documents, workspace_id=workspace_id)
        raise


def reset_vector_store(workspace_id: str | None = None) -> None:
    """Reset cached vector store(s) after ingest."""
    global _vector_stores
    from knowledge_transfer_agent.core.workspaces import normalize_workspace_id

    if workspace_id is not None:
        ws = normalize_workspace_id(workspace_id)
        _vector_stores.pop(ws, None)
    else:
        _vector_stores.clear()
    try:
        from knowledge_transfer_agent.core.cache import invalidate_all_caches

        invalidate_all_caches()
    except Exception:
        logger.debug("Could not invalidate caches after vector store reset", exc_info=True)


def list_index_sources(*, workspace_id: str | None = None) -> dict[str, Any]:
    """
    Aggregate unique indexed sources for a workspace from FAISS chunk metadata.

    Returns total_chunks and a sources list sorted by chunk_count desc.
    """
    from knowledge_transfer_agent.core.workspaces import normalize_workspace_id

    ws = normalize_workspace_id(workspace_id)
    store = get_vector_store(workspace_id=ws)

    by_source: dict[str, dict[str, Any]] = {}
    total_chunks = 0

    doc_ids = list(getattr(store, "index_to_docstore_id", {}).values())
    docstore = getattr(store, "docstore", None)
    if docstore is None:
        return {"workspace_id": ws, "total_chunks": 0, "source_count": 0, "sources": []}

    for did in doc_ids:
        try:
            doc = docstore.search(did)
        except Exception:
            continue
        if not isinstance(doc, Document):
            continue
        total_chunks += 1
        meta = doc.metadata or {}
        source = str(meta.get("source") or "unknown")
        entry = by_source.get(source)
        if entry is None:
            file_name = meta.get("file_name")
            if not file_name:
                file_name = source.rstrip("/").split("/")[-1] if source else "unknown"
            entry = {
                "source": source,
                "source_type": str(meta.get("source_type") or "generic"),
                "file_name": str(file_name),
                "chunk_count": 0,
                "doc_ids": set(),
            }
            by_source[source] = entry
        entry["chunk_count"] += 1
        doc_id = str(meta.get("doc_id") or "").strip()
        if doc_id:
            entry["doc_ids"].add(doc_id)

    sources = []
    for entry in by_source.values():
        sources.append(
            {
                "source": entry["source"],
                "source_type": entry["source_type"],
                "file_name": entry["file_name"],
                "chunk_count": entry["chunk_count"],
                "doc_ids": sorted(entry["doc_ids"]),
            }
        )
    sources.sort(key=lambda s: (-s["chunk_count"], s["file_name"].lower()))

    return {
        "workspace_id": ws,
        "total_chunks": total_chunks,
        "source_count": len(sources),
        "sources": sources,
    }
