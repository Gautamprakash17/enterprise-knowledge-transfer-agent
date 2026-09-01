"""
Document loaders for TXT and PDF with metadata support.
Reusable functions for loading from file paths.

PDFs are expanded to one Document per page (with extractable text) so chunks
retain page_number for citations.
"""

from pathlib import Path
from typing import Iterator

from knowledge_transfer_agent.ingestion.base import Document
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".rst"}


def load_txt(path: Path, encoding: str = "utf-8") -> Document | None:
    """
    Load a TXT file into a Document with metadata.

    Args:
        path: Path to the text file
        encoding: File encoding (default: utf-8)

    Returns:
        Document or None on failure
    """
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding=encoding, errors="replace")
        return Document(
            content=content,
            metadata={
                "file_path": str(path),
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                "encoding": encoding,
            },
            source=str(path.resolve()),
            source_type="file",
            doc_id=str(path.resolve()),
        )
    except Exception as e:
        logger.warning("Failed to load TXT %s: %s", path, e)
        return None


def load_pdf_pages(path: Path) -> list[Document]:
    """
    Load each PDF page with text as its own Document (1-based page_number in metadata).

    Pages with no extractable text are skipped (same as treating the whole file as empty).

    Args:
        path: Path to the PDF file

    Returns:
        List of Documents (possibly empty)
    """
    if not path.is_file():
        return []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        resolved = str(path.resolve())
        out: list[Document] = []
        n = len(reader.pages)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            pnum = i + 1
            out.append(
                Document(
                    content=text.strip(),
                    metadata={
                        "file_path": str(path),
                        "file_name": path.name,
                        "file_type": ".pdf",
                        "page_number": pnum,
                        "page_count": n,
                    },
                    source=resolved,
                    source_type="file",
                    doc_id=f"{resolved}#p{pnum}",
                ),
            )
        if not out:
            logger.warning("PDF %s has no extractable text on any page", path)
        return out
    except ImportError:
        logger.error("pypdf required for PDF: pip install pypdf")
        return []
    except Exception as e:
        logger.warning("Failed to load PDF %s: %s", path, e)
        return []


def load_pdf(path: Path) -> Document | None:
    """
    Load a PDF as a single merged Document (all pages concatenated).

    Prefer :func:`load_pdf_pages` for ingestion so citations can include page numbers.
    Kept for callers that expect one Document per file.

    Args:
        path: Path to the PDF file

    Returns:
        Document or None on failure
    """
    pages = load_pdf_pages(path)
    if not pages:
        return None
    content = "\n\n".join(p.content for p in pages)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception:
        page_count = len(pages)

    return Document(
        content=content,
        metadata={
            "file_path": str(pages[0].metadata["file_path"]),
            "file_name": pages[0].metadata["file_name"],
            "file_type": ".pdf",
            "page_count": page_count,
        },
        source=str(Path(pages[0].metadata["file_path"]).resolve()),
        source_type="file",
        doc_id=str(Path(pages[0].metadata["file_path"]).resolve()),
    )


def load_document(path: Path) -> Document | None:
    """
    Load a single document by path. Supports TXT and PDF.

    Args:
        path: File path

    Returns:
        Document or None if unsupported format or failure
    """
    path = Path(path).resolve()
    suffix = path.suffix.lower()

    if suffix == ".txt" or suffix == ".md" or suffix == ".rst":
        return load_txt(path)
    if suffix == ".pdf":
        return load_pdf(path)
    return None


def _yield_from_path(path: Path, seen: set[str]) -> Iterator[Document]:
    """Yield documents from a single file path (PDFs expand to multiple)."""
    path = path.resolve()
    if not path.is_file():
        return
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        for doc in load_pdf_pages(path):
            did = doc.doc_id or ""
            if did and did not in seen:
                seen.add(did)
                yield doc
        return
    doc = load_document(path)
    if doc:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            yield doc


def load_documents(
    paths: list[Path | str],
    recursive: bool = True,
) -> Iterator[Document]:
    """
    Load documents from paths. Supports files and directories.

    Args:
        paths: File or directory paths
        recursive: If True, recurse into subdirectories

    Yields:
        Document instances
    """
    seen: set[str] = set()
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            logger.warning("Path does not exist: %s", path)
            continue

        if path.is_file():
            yield from _yield_from_path(path, seen)
        else:
            pattern = "**/*" if recursive else "*"
            for fp in path.glob(pattern):
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield from _yield_from_path(fp, seen)


def load_documents_list(
    paths: list[Path | str],
    recursive: bool = True,
) -> list[Document]:
    """Load documents and return as list. Wrapper around load_documents."""
    return list(load_documents(paths, recursive=recursive))
