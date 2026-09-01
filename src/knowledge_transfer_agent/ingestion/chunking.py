"""
Recursive chunking with overlap. Reusable functions for document splitting.
"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

from knowledge_transfer_agent.ingestion.base import Document
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def create_recursive_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter with configurable overlap.

    Args:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        separators: Split order (default: paragraph, line, sentence, word)

    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or DEFAULT_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def recursive_chunk(
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[LangChainDocument]:
    """
    Split text into overlapping chunks using recursive character splitting.

    Args:
        content: Text to split
        metadata: Metadata to attach to each chunk
        chunk_size: Max chars per chunk
        chunk_overlap: Overlap between chunks
        separators: Split precedence

    Returns:
        List of LangChain documents with chunk metadata
    """
    splitter = create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    doc = LangChainDocument(
        page_content=content,
        metadata=metadata or {},
    )
    chunks = splitter.split_documents([doc])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def chunk_document(
    doc: Document,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[LangChainDocument]:
    """
    Chunk a Document preserving metadata.

    Args:
        doc: Input document
        chunk_size: Max chars per chunk
        chunk_overlap: Overlap between chunks
        separators: Split precedence

    Returns:
        List of chunked LangChain documents
    """
    lc_doc = doc.to_langchain_document()
    meta = dict(lc_doc.metadata)
    return recursive_chunk(
        lc_doc.page_content,
        metadata=meta,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[LangChainDocument]:
    """
    Chunk multiple documents into a flat list of chunks.

    Args:
        documents: Input documents
        chunk_size: Max chars per chunk
        chunk_overlap: Overlap between chunks
        separators: Split precedence

    Returns:
        Flat list of all chunks
    """
    all_chunks: list[LangChainDocument] = []
    for doc in documents:
        chunks = chunk_document(
            doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        all_chunks.extend(chunks)

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
