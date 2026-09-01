"""
Orchestrates ingestion from multiple sources and loads into vector store.
Uses document loader, recursive chunking, embedding pipeline, and FAISS save/load.
"""

from pathlib import Path

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.ingestion.base import BaseIngester, IngestionResult
from knowledge_transfer_agent.ingestion.confluence import ConfluenceIngester
from knowledge_transfer_agent.ingestion.document_processor import DocumentProcessor
from knowledge_transfer_agent.ingestion.file_ingester import FileIngester
from knowledge_transfer_agent.ingestion.github import GitHubIngester
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.retrieval.vector_store import (
    add_documents_to_store,
    create_faiss_index,
    load_faiss_index,
    reset_vector_store,
    save_faiss_index,
)

logger = get_logger(__name__)


class IngestionPipeline:
    """Coordinates ingestion from multiple sources into the vector store."""

    def __init__(
        self,
        ingestors: list[BaseIngester] | None = None,
        document_processor: DocumentProcessor | None = None,
    ) -> None:
        self.ingestors = ingestors or self._default_ingestors()
        self.processor = document_processor or DocumentProcessor()

    def _default_ingestors(self) -> list[BaseIngester]:
        """Create default ingestors from config."""
        ingestors: list[BaseIngester] = []
        settings = get_settings()

        if settings.confluence_url and settings.confluence_token:
            ingestors.append(ConfluenceIngester())

        if settings.github_repos:
            ingestors.append(GitHubIngester())

        return ingestors

    def run(
        self,
        additional_paths: list[str | Path] | None = None,
        persist: bool = True,
        *,
        replace_index: bool = False,
    ) -> dict[str, IngestionResult]:
        """
        Run the full ingestion pipeline.

        Args:
            additional_paths: Extra file/dir paths to ingest
            persist: Whether to save the FAISS index to disk
            replace_index: If True, create a new index from this run only (do not merge into existing FAISS).

        Returns:
            Dict mapping source_type to IngestionResult
        """
        all_documents: list = []
        results: dict[str, IngestionResult] = {}

        # Run configured ingestors
        for ingestor in self.ingestors:
            result = ingestor.ingest()
            results[ingestor.source_type] = result
            all_documents.extend(result.documents)

        # Add file ingester for additional paths
        if additional_paths:
            file_ingester = FileIngester(additional_paths)
            file_result = file_ingester.ingest()
            results["file"] = file_result
            all_documents.extend(file_result.documents)

        if not all_documents:
            logger.warning("No documents ingested from any source")
            return results

        # Process and chunk
        chunks = self.processor.process_documents(all_documents)

        # Embed and load into FAISS
        try:
            settings = get_settings()
            store_path = Path(settings.vector_store_path)
            if replace_index:
                vector_store = create_faiss_index(chunks)
            else:
                try:
                    vector_store = load_faiss_index(store_path)
                    add_documents_to_store(vector_store, chunks)
                except FileNotFoundError:
                    vector_store = create_faiss_index(chunks)

            if persist:
                save_faiss_index(vector_store, store_path)
                reset_vector_store()

        except Exception as e:
            logger.exception("Failed to add documents to vector store")
            for r in results.values():
                r.errors.append(str(e))

        return results
