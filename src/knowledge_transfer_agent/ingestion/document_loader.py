"""
Document loaders for TXT and PDF with metadata support.
Reusable functions for loading from file paths.

PDFs are expanded to one Document per page (with extractable text) so chunks
retain page_number for citations.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Iterator

from knowledge_transfer_agent.ingestion.base import Document
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

# Text-like sources (docs + common code) and PDF
TEXT_LIKE_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".markdown",
    ".pdf",
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".java",
    ".kt",
    ".kts",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".less",
    ".vue",
    ".svelte",
    ".xml",
    ".gradle",
    ".properties",
    ".ipynb",
}

SUPPORTED_EXTENSIONS = TEXT_LIKE_EXTENSIONS

_SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "target",
    ".next",
    ".nuxt",
    "coverage",
    ".eggs",
}


def _should_skip_path(path: Path) -> bool:
    parts = path.parts
    if any(p.lower() in _SKIP_DIR_NAMES for p in parts):
        return True
    return any(p.endswith(".egg-info") for p in parts)


def iter_indexable_files(
    paths: list[Path | str],
    recursive: bool = True,
) -> list[Path]:
    """List supported files under paths (skips vendor/cache dirs)."""
    found: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            continue
        candidates: list[Path] = []
        if path.is_file():
            candidates = [path]
        else:
            pattern = "**/*" if recursive else "*"
            candidates = [fp for fp in path.glob(pattern) if fp.is_file()]
        for fp in candidates:
            if _should_skip_path(fp):
                continue
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            key = str(fp.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(fp)
    return found


def count_indexable_files(paths: list[Path | str], recursive: bool = True) -> int:
    return len(iter_indexable_files(paths, recursive))


def _stable_content_hash(text: str) -> str:
    # Normalize whitespace so trivial formatting doesn't change identity too often.
    norm = " ".join((text or "").split())
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def _stable_file_hash(path: Path) -> str:
    # Hash raw bytes so identical files across machines share an ID.
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        content_hash = _stable_content_hash(content)
        return Document(
            content=content,
            metadata={
                "file_path": str(path),
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                "encoding": encoding,
                "content_hash": content_hash,
            },
            source=str(path.resolve()),
            source_type="file",
            doc_id=f"file:{content_hash}",
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
        file_hash = _stable_file_hash(path)
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
                        "file_hash": file_hash,
                    },
                    source=resolved,
                    source_type="file",
                    doc_id=f"file:{file_hash}#p{pnum}",
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
            "file_hash": pages[0].metadata.get("file_hash", ""),
        },
        source=str(Path(pages[0].metadata["file_path"]).resolve()),
        source_type="file",
        doc_id=f"file:{pages[0].metadata.get('file_hash','')}",
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

    if suffix in TEXT_LIKE_EXTENSIONS and suffix != ".pdf":
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
        key = doc.doc_id or str(path.resolve())
        if key and key not in seen:
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
    on_file_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    """
    Load documents and return as list.

    If on_file_progress is set, it is called as (files_done, files_total, label).
    """
    if not on_file_progress:
        return list(load_documents(paths, recursive=recursive))

    files = iter_indexable_files(paths, recursive)
    total = len(files)
    documents: list[Document] = []
    seen: set[str] = set()
    for i, fp in enumerate(files, start=1):
        on_file_progress(i, total, f"Reading {fp.name}")
        for doc in _yield_from_path(fp, seen):
            documents.append(doc)
    return documents


def load_documents_list_legacy(
    paths: list[Path | str],
    recursive: bool = True,
) -> list[Document]:
    """Load documents without per-file progress (legacy glob walk)."""
    return list(load_documents(paths, recursive=recursive))
