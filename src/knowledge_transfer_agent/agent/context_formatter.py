"""
Context and citation formatting. Clear separation from nodes.
"""

import re
from typing import Any

from langchain_core.documents import Document

_DOC_PAGE_SUFFIX = re.compile(r"#p(\d+)$")


def _resolved_page_number(doc: Document) -> int | None:
    """Prefer metadata; fall back to doc_id suffix ...#pN (PDF page-aware ids)."""
    pn = doc.metadata.get("page_number")
    if pn is not None:
        try:
            return int(pn)
        except (TypeError, ValueError):
            pass
    did = str(doc.metadata.get("doc_id", "") or "")
    m = _DOC_PAGE_SUFFIX.search(did)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def format_documents_for_context(docs: list[Document]) -> str:
    """Format retrieved documents into context string with citation markers."""
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "generic")
        page = _resolved_page_number(doc)
        page_part = f", page {page}" if page is not None else ""
        parts.append(
            f"[{i}] Source: {source}{page_part} ({source_type})\n{doc.page_content}\n"
        )
    return "\n---\n".join(parts)


def extract_citations_from_documents(docs: list[Document]) -> list[dict[str, Any]]:
    """Extract citation metadata from retrieved documents (includes page_number for PDF chunks)."""
    out: list[dict[str, Any]] = []
    for doc in docs:
        item: dict[str, Any] = {
            "source": doc.metadata.get("source", "unknown"),
            "source_type": doc.metadata.get("source_type", "generic"),
            "doc_id": str(doc.metadata.get("doc_id", "")),
        }
        pn = _resolved_page_number(doc)
        if pn is not None:
            item["page_number"] = pn
        snippet = (doc.page_content or "").strip()
        if snippet:
            item["snippet"] = snippet[:400] + ("…" if len(snippet) > 400 else "")
        out.append(item)
    return out
