"""
Document ingestion layer for processing various knowledge sources.
"""

from knowledge_transfer_agent.ingestion.base import Document, IngestionResult
from knowledge_transfer_agent.ingestion.chunking import (
    chunk_document,
    chunk_documents,
    create_recursive_splitter,
    recursive_chunk,
)
from knowledge_transfer_agent.ingestion.document_loader import (
    load_document,
    load_documents,
    load_documents_list,
    load_pdf,
    load_pdf_pages,
    load_txt,
)
from knowledge_transfer_agent.ingestion.document_processor import DocumentProcessor
from knowledge_transfer_agent.ingestion.embedding_pipeline import (
    add_to_existing_index,
    embed_and_index,
    run_embedding_pipeline,
)
from knowledge_transfer_agent.ingestion.pipeline import IngestionPipeline

__all__ = [
    "Document",
    "IngestionResult",
    "DocumentProcessor",
    "IngestionPipeline",
    "load_document",
    "load_documents",
    "load_documents_list",
    "load_txt",
    "load_pdf",
    "load_pdf_pages",
    "recursive_chunk",
    "chunk_document",
    "chunk_documents",
    "create_recursive_splitter",
    "embed_and_index",
    "run_embedding_pipeline",
    "add_to_existing_index",
]
