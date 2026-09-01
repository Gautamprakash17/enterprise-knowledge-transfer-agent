"""
Background ingestion jobs (thread pool) with SQLite status tracking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from knowledge_transfer_agent.core.database import create_ingest_job, get_ingest_job, update_ingest_job
from knowledge_transfer_agent.ingestion.pipeline import IngestionPipeline
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.services.ingest_progress import make_job_progress_reporter

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest-job")


def submit_file_ingest_job(
    workspace_id: str,
    batch_dir: Path,
    *,
    replace_index: bool = False,
) -> dict[str, Any]:
    """Queue ingestion for uploaded files; returns job metadata immediately."""
    job = create_ingest_job(workspace_id)
    job_id = job["id"]

    def _run() -> None:
        progress = make_job_progress_reporter(job_id)
        progress("scan", 0, 1, "Starting indexing…")
        try:
            pipeline = IngestionPipeline()
            results = pipeline.run(
                additional_paths=[batch_dir],
                persist=True,
                replace_index=replace_index,
                workspace_id=workspace_id,
                include_configured_sources=False,
                on_progress=progress,
            )
            payload = {
                k: {
                    "success": v.success,
                    "documents_processed": v.documents_processed,
                    "documents_failed": v.documents_failed,
                    "errors": v.errors[:5],
                }
                for k, v in results.items()
            }
            success = any(r.success for r in results.values())
            update_ingest_job(
                job_id,
                "completed" if success else "failed",
                "Ingestion completed" if success else "Ingestion had errors",
                {"results": payload, "success": success},
            )
        except Exception as e:
            logger.exception("Ingest job %s failed", job_id)
            update_ingest_job(job_id, "failed", str(e), {"success": False, "error": str(e)})

    _executor.submit(_run)
    return get_ingest_job(job_id) or job


def submit_path_ingest_job(
    workspace_id: str,
    source_dir: Path,
    *,
    replace_index: bool = False,
) -> dict[str, Any]:
    """Queue ingestion for a directory on the server filesystem."""
    job = create_ingest_job(workspace_id)
    job_id = job["id"]
    root = source_dir.resolve()

    def _run() -> None:
        progress = make_job_progress_reporter(job_id)
        progress("scan", 0, 1, f"Scanning {root.name}…")
        try:
            pipeline = IngestionPipeline()
            results = pipeline.run(
                additional_paths=[root],
                persist=True,
                replace_index=replace_index,
                workspace_id=workspace_id,
                include_configured_sources=False,
                on_progress=progress,
            )
            payload = {
                k: {
                    "success": v.success,
                    "documents_processed": v.documents_processed,
                    "documents_failed": v.documents_failed,
                    "errors": v.errors[:5],
                }
                for k, v in results.items()
            }
            success = any(r.success for r in results.values())
            update_ingest_job(
                job_id,
                "completed" if success else "failed",
                "Ingestion completed" if success else "Ingestion had errors",
                {"results": payload, "success": success, "path": str(root)},
            )
        except Exception as e:
            logger.exception("Path ingest job %s failed", job_id)
            update_ingest_job(job_id, "failed", str(e), {"success": False, "error": str(e)})

    _executor.submit(_run)
    return get_ingest_job(job_id) or job


def submit_git_clone_ingest_job(
    workspace_id: str,
    repo_url: str,
    *,
    replace_index: bool = False,
    branch: str | None = None,
) -> dict[str, Any]:
    """Clone a remote Git repo on the API host, then queue indexing."""
    job = create_ingest_job(workspace_id)
    job_id = job["id"]
    url = repo_url.strip()

    def _run() -> None:
        progress = make_job_progress_reporter(job_id)
        progress("scan", 0, 1, "Cloning repository…")
        try:
            from knowledge_transfer_agent.ingestion.github import clone_remote_repository

            cloned = clone_remote_repository(url, branch=branch)
            progress("scan", 0, 1, f"Scanning {cloned.name}…")
            pipeline = IngestionPipeline()
            results = pipeline.run(
                additional_paths=[cloned],
                persist=True,
                replace_index=replace_index,
                workspace_id=workspace_id,
                include_configured_sources=False,
                on_progress=progress,
            )
            payload = {
                k: {
                    "success": v.success,
                    "documents_processed": v.documents_processed,
                    "documents_failed": v.documents_failed,
                    "errors": v.errors[:5],
                }
                for k, v in results.items()
            }
            success = any(r.success for r in results.values())
            update_ingest_job(
                job_id,
                "completed" if success else "failed",
                "Repository cloned and indexed"
                if success
                else "Clone succeeded but indexing had errors",
                {
                    "results": payload,
                    "success": success,
                    "repo_url": url,
                    "clone_path": str(cloned),
                },
            )
        except Exception as e:
            logger.exception("Git clone ingest job %s failed", job_id)
            update_ingest_job(job_id, "failed", str(e), {"success": False, "error": str(e)})

    _executor.submit(_run)
    return get_ingest_job(job_id) or job
