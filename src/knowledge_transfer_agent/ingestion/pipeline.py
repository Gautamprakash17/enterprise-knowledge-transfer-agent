"""
Orchestrates ingestion from multiple sources and loads into vector store.
Uses document loader, recursive chunking, embedding pipeline, and FAISS save/load.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

_MANIFEST_FILENAME = "ingestion_manifest.json"


def _doc_fingerprint(doc: Any) -> str:
    """
    Fingerprint for change detection.
    Prefer content-based hashing so identical docs across machines don't duplicate.
    """
    content = getattr(doc, "content", "") or ""
    # Normalize whitespace slightly so trivial formatting doesn't trigger rebuilds.
    norm = " ".join(str(content).split())
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def _doc_key(doc: Any) -> str:
    """
    Stable manifest key for a document.
    - If doc_id exists (Confluence pageId, github:path, file:hash), use it.
    - Otherwise fall back to content hash.
    """
    doc_id = getattr(doc, "doc_id", None)
    if doc_id:
        return str(doc_id)
    return _doc_fingerprint(doc)


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _classify_manifest_change(
    prev: dict[str, str],
    current: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """
    Classify the change between manifests.

    Returns:
      (mode, added_docs)
      - mode:
          - "no_change": exact match
          - "add_only": only new keys were added (existing keys unchanged)
          - "modified_or_removed": anything else (updates and/or deletions)
      - added_docs: {doc_key: fingerprint} subset (only meaningful for add_only)
    """
    if prev == current:
        return "no_change", {}

    prev_keys = set(prev.keys())
    cur_keys = set(current.keys())

    added = cur_keys - prev_keys
    removed = prev_keys - cur_keys
    common = prev_keys & cur_keys

    modified_common = any(prev[k] != current[k] for k in common)
    if removed or modified_common:
        return "modified_or_removed", {}

    if added:
        return "add_only", {k: current[k] for k in added}

    # Fallback (shouldn't happen, but keep safe)
    return "modified_or_removed", {}


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
        incremental_update: bool = True,
        workspace_id: str | None = None,
        include_configured_sources: bool = False,
        on_progress: Callable[[str, int, int, str], None] | None = None,
    ) -> dict[str, IngestionResult]:
        """
        Run the full ingestion pipeline.

        Args:
            additional_paths: Extra file/dir paths to ingest
            persist: Whether to save the FAISS index to disk
            replace_index: If True, create a new index from this run only (do not merge into existing FAISS).
            incremental_update: If True, detect doc changes via a manifest and rebuild index when needed.
            workspace_id: Project id — indexes into an isolated FAISS directory per workspace.
            include_configured_sources: If True, also run Confluence/GitHub from .env (global config).
                Default False for per-project uploads so other projects' sources are not mixed in.

        Returns:
            Dict mapping source_type to IngestionResult
        """
        from knowledge_transfer_agent.core.workspaces import normalize_workspace_id, workspace_index_path

        ws = normalize_workspace_id(workspace_id)
        all_documents: list = []
        results: dict[str, IngestionResult] = {}

        def report(phase: str, current: int, total: int, message: str) -> None:
            if on_progress:
                on_progress(phase, current, total, message)

        # Optional: Confluence / GitHub from env (shared config — use only when intended)
        if include_configured_sources:
            report("scan", 0, 1, "Loading Confluence/GitHub…")
            for ingestor in self.ingestors:
                result = ingestor.ingest()
                results[ingestor.source_type] = result
                all_documents.extend(result.documents)

        # Load files from paths (upload batch or local directory)
        if additional_paths:
            from knowledge_transfer_agent.ingestion.document_loader import load_documents_list

            file_total = 0

            def on_file(done: int, total: int, label: str) -> None:
                nonlocal file_total
                file_total = total
                report("scan", done, max(total, 1), label)

            report("scan", 0, 1, "Scanning files…")
            file_docs = load_documents_list(
                additional_paths,
                recursive=True,
                on_file_progress=on_file,
            )
            file_result = IngestionResult(
                success=len(file_docs) > 0,
                documents_processed=len(file_docs),
                documents_failed=0,
                source="file",
                source_type="file",
                documents=file_docs,
                errors=[],
            )
            results["file"] = file_result
            all_documents.extend(file_docs)
            report(
                "scan",
                file_total or len(file_docs),
                max(file_total, len(file_docs), 1),
                f"Loaded {len(file_docs)} document(s) from {file_total or '?'} file(s)",
            )

        if not all_documents:
            logger.warning("No documents ingested from any source")
            return results

        for doc in all_documents:
            doc.metadata["workspace_id"] = ws

        report("chunk", 0, 1, "Splitting into chunks…")
        chunks = self.processor.process_documents(all_documents)
        for chunk in chunks:
            chunk.metadata["workspace_id"] = ws
        report("chunk", 1, 1, f"Created {len(chunks)} chunk(s)")

        # Embed and load into FAISS (per-workspace path)
        try:
            store_path = workspace_index_path(ws)
            manifest_path = store_path / _MANIFEST_FILENAME

            current_manifest = {_doc_key(d): _doc_fingerprint(d) for d in all_documents}

            # Production-friendly behavior:
            # - If no changes -> skip
            # - If only new docs added -> incremental add
            # - If any doc modified/removed -> rebuild for correctness (FAISS deletions are non-trivial)
            if incremental_update and not replace_index:
                prev = _load_manifest(manifest_path)
                if prev and store_path.exists():
                    mode, _added = _classify_manifest_change(prev, current_manifest)
                    if mode == "no_change":
                        logger.info("No document changes detected; skipping reindex.")
                        return results
                    if mode == "modified_or_removed":
                        replace_index = True
                    # mode == "add_only" -> keep replace_index False (incremental add)
                else:
                    # No previous manifest (or no index) -> create fresh
                    replace_index = True

            def on_embed(done: int, total: int, message: str) -> None:
                report("embed", done, max(total, 1), message)

            if replace_index:
                vector_store = create_faiss_index(
                    chunks,
                    on_embed_progress=on_embed if on_progress else None,
                )
            else:
                try:
                    vector_store = load_faiss_index(store_path)
                    report("embed", 0, len(chunks), "Adding to existing index…")
                    add_documents_to_store(vector_store, chunks)
                    if on_progress:
                        on_embed(len(chunks), len(chunks), f"Added {len(chunks)} chunk(s)")
                except FileNotFoundError:
                    vector_store = create_faiss_index(
                        chunks,
                        on_embed_progress=on_embed if on_progress else None,
                    )

            if persist:
                report("save", 0, 1, "Saving index to disk…")
                save_faiss_index(vector_store, store_path)
                report("save", 1, 1, "Index saved")
                # Drop in-memory cache so FastAPI uses the updated index on disk
                reset_vector_store(ws)
                if incremental_update:
                    _write_manifest(manifest_path, current_manifest)

        except Exception as e:
            logger.exception("Failed to add documents to vector store")
            for r in results.values():
                r.errors.append(str(e))

        return results
