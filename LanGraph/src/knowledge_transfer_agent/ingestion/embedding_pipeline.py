"""
Embedding pipeline: embed documents and build FAISS index.
Reusable functions for load → chunk → embed → save.
"""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document as LangChainDocument

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.ingestion.base import Document
from knowledge_transfer_agent.ingestion.chunking import chunk_documents
from knowledge_transfer_agent.ingestion.document_loader import load_documents_list
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.retrieval.embeddings import get_embeddings
from knowledge_transfer_agent.retrieval.vector_store import (
    add_documents_to_store,
    create_faiss_index,
    load_faiss_index,
    reset_vector_store,
    save_faiss_index,
)

logger = get_logger(__name__)


def embed_and_index(
    documents: list[LangChainDocument],
    embeddings: Any | None = None,
) -> Any:
    """
    Embed documents and create FAISS index.

    Args:
        documents: Chunked LangChain documents with metadata
        embeddings: Optional embeddings model (default: from config)

    Returns:
        FAISS vector store instance
    """
    emb = embeddings or get_embeddings()
    return create_faiss_index(documents, embeddings=emb)


def run_embedding_pipeline(
    paths: list[Path | str],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    output_path: str | Path | None = None,
    recursive: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    """
    Full pipeline: load → chunk → embed → save.

    Reusable function that loads documents from paths, chunks with overlap,
    embeds, and saves to FAISS.

    Args:
        paths: File or directory paths (TXT, PDF)
        chunk_size: Max chunk size
        chunk_overlap: Overlap between chunks
        output_path: Where to save FAISS index
        recursive: Recurse into subdirs

    Returns:
        Dict with stats: documents_loaded, chunks_created, index_path
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    output_path = Path(output_path or settings.vector_store_path)

    # Load
    docs = load_documents_list(paths, recursive=recursive)
    if not docs:
        logger.warning("No documents loaded from paths: %s", paths)
        return {"documents_loaded": 0, "chunks_created": 0, "index_path": None}

    logger.info("Loaded %d documents", len(docs))

    # Chunk
    chunks = chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Created %d chunks", len(chunks))

    # Embed and index
    store = embed_and_index(chunks)

    # Save
    if persist:
        save_faiss_index(store, str(output_path))
        reset_vector_store()
        idx_path = str(output_path)
    else:
        idx_path = None

    return {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "index_path": idx_path,
    }


def add_to_existing_index(
    paths: list[Path | str],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    index_path: str | Path | None = None,
    recursive: bool = True,
) -> dict[str, Any]:
    """
    Load, chunk, embed, and add to existing FAISS index.
    Loads index from disk, adds new documents, saves back.

    Args:
        paths: File or directory paths
        chunk_size: Max chunk size
        chunk_overlap: Overlap
        index_path: Path to existing index
        recursive: Recurse into subdirs

    Returns:
        Dict with stats
    """
    settings = get_settings()
    index_path = Path(index_path or settings.vector_store_path)

    docs = load_documents_list(paths, recursive=recursive)
    if not docs:
        return {"documents_loaded": 0, "chunks_added": 0}

    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    store = load_faiss_index(str(index_path))
    add_documents_to_store(store, chunks)
    save_faiss_index(store, str(index_path))
    reset_vector_store()

    return {"documents_loaded": len(docs), "chunks_added": len(chunks)}
