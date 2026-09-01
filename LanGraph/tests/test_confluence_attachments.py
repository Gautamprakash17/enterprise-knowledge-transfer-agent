from __future__ import annotations

from io import BytesIO

import pytest


def test_attachment_download_url_builds_absolute():
    from knowledge_transfer_agent.ingestion.confluence import _attachment_download_url

    att = {"_links": {"download": "/download/attachments/123/file.pdf"}}
    assert (
        _attachment_download_url("https://example.atlassian.net/wiki", att)
        == "https://example.atlassian.net/wiki/download/attachments/123/file.pdf"
    )


def test_extract_text_from_txt_attachment():
    from knowledge_transfer_agent.ingestion.confluence import _extract_text_from_attachment

    out = _extract_text_from_attachment(b"hello\\nworld", filename="note.txt")
    assert len(out) == 1
    text, meta = out[0]
    assert "hello" in text
    assert meta["file_type"] == ".txt"


def test_extract_text_from_docx_attachment():
    from knowledge_transfer_agent.ingestion.confluence import _extract_text_from_attachment

    try:
        import docx
    except Exception as e:
        pytest.skip(f"python-docx not available: {e}")

    d = docx.Document()
    d.add_paragraph("Hello DOCX")
    buf = BytesIO()
    d.save(buf)
    out = _extract_text_from_attachment(buf.getvalue(), filename="a.docx")
    assert len(out) == 1
    text, meta = out[0]
    assert "Hello DOCX" in text
    assert meta["file_type"] == ".docx"

