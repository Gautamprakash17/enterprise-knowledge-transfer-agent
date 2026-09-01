"""
Document processing - delegates to chunking module for recursive chunking.
"""

from langchain_core.documents import Document as LangChainDocument

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.ingestion.base import Document
from knowledge_transfer_agent.ingestion.chunking import chunk_document, chunk_documents
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Processes and chunks documents for vector storage. Uses reusable chunking functions."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = separators

    def process_document(self, doc: Document) -> list[LangChainDocument]:
        """Process a document: recursive chunking with overlap, preserve metadata."""
        return chunk_document(
            doc,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def process_documents(self, documents: list[Document]) -> list[LangChainDocument]:
        """Process multiple documents into chunks."""
        return chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
