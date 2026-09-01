"""
Local file and directory ingester for runbooks, incident reports, etc.
Uses document loader for TXT and PDF support.
"""

from pathlib import Path

from knowledge_transfer_agent.ingestion.base import BaseIngester, Document, IngestionResult
from knowledge_transfer_agent.ingestion.document_loader import load_documents_list
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


class FileIngester(BaseIngester):
    """Ingests documents from local files and directories. Supports TXT and PDF."""

    source_type = "file"

    def __init__(self, paths: list[str | Path], recursive: bool = True) -> None:
        self.paths = [Path(p) for p in paths]
        self.recursive = recursive

    def ingest(self) -> IngestionResult:
        """Ingest documents from configured paths using document loader."""
        documents: list[Document] = []
        errors: list[str] = []

        for path in self.paths:
            path = path.resolve()
            if not path.exists():
                errors.append(f"Path not found: {path}")
                continue
            docs = load_documents_list([path], recursive=self.recursive)
            documents.extend(docs)

        return IngestionResult(
            success=len(documents) > 0,
            documents_processed=len(documents),
            documents_failed=len(errors),
            source="file",
            source_type=self.source_type,
            documents=documents,
            errors=errors,
        )
