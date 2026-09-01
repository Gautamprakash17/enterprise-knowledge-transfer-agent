"""Tests for document library index aggregation."""

from langchain_core.documents import Document

from knowledge_transfer_agent.retrieval import vector_store as vs


def test_list_index_sources_aggregates_chunks(monkeypatch):
    class FakeDocstore:
        def __init__(self, docs):
            self._docs = docs

        def search(self, doc_id):
            return self._docs[doc_id]

    docs = {
        "a": Document(
            page_content="one",
            metadata={
                "source": "/tmp/a.md",
                "source_type": "file",
                "file_name": "a.md",
                "doc_id": "file:a",
            },
        ),
        "b": Document(
            page_content="two",
            metadata={
                "source": "/tmp/a.md",
                "source_type": "file",
                "file_name": "a.md",
                "doc_id": "file:a",
            },
        ),
        "c": Document(
            page_content="three",
            metadata={
                "source": "/tmp/b.txt",
                "source_type": "file",
                "file_name": "b.txt",
                "doc_id": "file:b",
            },
        ),
    }

    class FakeStore:
        index_to_docstore_id = {0: "a", 1: "b", 2: "c"}
        docstore = FakeDocstore(docs)

    monkeypatch.setattr(vs, "get_vector_store", lambda workspace_id=None: FakeStore())

    data = vs.list_index_sources(workspace_id="default")
    assert data["total_chunks"] == 3
    assert data["source_count"] == 2
    by_name = {s["file_name"]: s for s in data["sources"]}
    assert by_name["a.md"]["chunk_count"] == 2
    assert by_name["b.txt"]["chunk_count"] == 1
