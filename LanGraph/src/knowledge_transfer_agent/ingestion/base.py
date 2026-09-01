"""
Base types and interfaces for document ingestion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document as LangChainDocument


@dataclass
class Document:
    """Unified document representation with metadata for vector storage."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    source_type: str = "generic"
    doc_id: Optional[str] = None

    def to_langchain_document(self) -> LangChainDocument:
        """Convert to LangChain Document format."""
        meta = {
            "source": self.source,
            "source_type": self.source_type,
            "doc_id": self.doc_id or "",
            **self.metadata,
        }
        return LangChainDocument(page_content=self.content, metadata=meta)


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    success: bool
    documents_processed: int
    documents_failed: int
    source: str
    source_type: str
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseIngester(ABC):
    """Abstract base class for document ingestors."""

    source_type: str = "generic"

    @abstractmethod
    def ingest(self) -> IngestionResult:
        """Ingest documents from the source. Must be implemented by subclasses."""
        pass
