"""Tests for ingestion layer."""

from pathlib import Path

import pytest

from knowledge_transfer_agent.ingestion.base import Document
from knowledge_transfer_agent.ingestion.chunking import (
    chunk_document,
    create_recursive_splitter,
    recursive_chunk,
)
from knowledge_transfer_agent.ingestion.document_loader import (
    load_document,
    load_documents_list,
    load_txt,
)
from knowledge_transfer_agent.ingestion.document_processor import DocumentProcessor
from knowledge_transfer_agent.ingestion.file_ingester import FileIngester


def test_document_to_langchain():
    doc = Document(content="Hello world", source="test", source_type="file")
    lc = doc.to_langchain_document()
    assert lc.page_content == "Hello world"
    assert lc.metadata["source"] == "test"


def test_load_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world")
    doc = load_txt(f)
    assert doc is not None
    assert doc.content == "Hello world"
    assert doc.metadata["file_name"] == "test.txt"


def test_load_document_txt(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Content")
    doc = load_document(f)
    assert doc is not None
    assert doc.content == "Content"


def test_load_documents_list(tmp_path):
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    docs = load_documents_list([tmp_path])
    assert len(docs) >= 2


def test_recursive_chunk():
    chunks = recursive_chunk("A" * 200, {}, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert all("chunk_index" in c.metadata for c in chunks)
    assert all("total_chunks" in c.metadata for c in chunks)


def test_chunk_document():
    doc = Document(content="A" * 200, source="test")
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert chunks[0].metadata.get("source") == "test"


def test_create_recursive_splitter():
    splitter = create_recursive_splitter(chunk_size=100, chunk_overlap=20)
    assert splitter._chunk_size == 100
    assert splitter._chunk_overlap == 20


def test_document_processor_chunks():
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    doc = Document(content="A" * 200, source="test")
    chunks = processor.process_document(doc)
    assert len(chunks) > 1
    assert all("chunk_index" in c.metadata for c in chunks)


def test_file_ingester(tmp_path):
    (tmp_path / "readme.txt").write_text("# Test\nContent here")
    ingester = FileIngester([tmp_path])
    result = ingester.ingest()
    assert result.success
    assert result.documents_processed >= 1
    assert any("Content here" in d.content for d in result.documents)


def test_format_and_extract_citations_include_page():
    from langchain_core.documents import Document as LCD

    from knowledge_transfer_agent.agent.context_formatter import (
        extract_citations_from_documents,
        format_documents_for_context,
    )

    d = LCD(
        page_content="chunk text",
        metadata={
            "source": "/tmp/x.pdf",
            "source_type": "file",
            "doc_id": "/tmp/x.pdf#p2",
            "page_number": 2,
        },
    )
    assert "page 2" in format_documents_for_context([d])
    cites = extract_citations_from_documents([d])
    assert cites[0]["page_number"] == 2
    assert cites[0]["source"] == "/tmp/x.pdf"


def test_extract_citations_omits_page_when_absent():
    from langchain_core.documents import Document as LCD

    from knowledge_transfer_agent.agent.context_formatter import extract_citations_from_documents

    d = LCD(
        page_content="t",
        metadata={
            "source": "/tmp/a.txt",
            "source_type": "file",
            "doc_id": "/tmp/a.txt",
        },
    )
    cites = extract_citations_from_documents([d])
    assert "page_number" not in cites[0]


def test_extract_citations_page_from_doc_id_when_metadata_missing():
    """Chunks indexed before page_number in metadata may still carry #pN in doc_id."""
    from langchain_core.documents import Document as LCD

    from knowledge_transfer_agent.agent.context_formatter import extract_citations_from_documents

    d = LCD(
        page_content="t",
        metadata={
            "source": "/tmp/x.pdf",
            "source_type": "file",
            "doc_id": "/tmp/x.pdf#p7",
        },
    )
    cites = extract_citations_from_documents([d])
    assert cites[0]["page_number"] == 7
