"""
GitHub / Git repository ingester.

Supports:
- Local repository paths (existing clones or any folder tree).
- Remote Git URLs (https/http or git@...) — clones or pulls into a cache directory, then scans files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.ingestion.base import BaseIngester, Document, IngestionResult
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

# File extensions to include for documentation
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".py", ".yaml", ".yml", ".json", ".toml"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def _is_remote_git_url(ref: str) -> bool:
    """True if ref should be cloned with git (not a local filesystem path)."""
    t = ref.strip()
    if not t:
        return False
    if t.startswith(("git@", "ssh://")):
        return True
    if t.startswith("http://") or t.startswith("https://"):
        try:
            p = urlparse(t)
            return bool(p.netloc)
        except Exception:
            return True
    return False


def _inject_github_https_token(url: str, token: Optional[str]) -> str:
    """Embed GITHUB_TOKEN for https://github.com/... clones (private repos)."""
    if not token:
        return url.strip()
    u = url.strip()
    for prefix in ("https://github.com/", "http://github.com/"):
        if u.startswith(prefix):
            rest = u[len(prefix) :]
            return f"https://{token}@github.com/{rest}"
    return u


def _git_clone_or_pull(
    remote_url: str,
    token: Optional[str],
    branch: Optional[str] = None,
) -> Path:
    """
    Ensure a working copy exists under GITHUB_CLONE_CACHE_DIR and return its path.
    Uses shallow clone by default; on subsequent runs runs git pull --ff-only.
    """
    settings = get_settings()
    cache_root = Path(settings.github_clone_cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(remote_url.strip().encode("utf-8")).hexdigest()[:20]
    dest = cache_root / f"repo_{key}"
    auth_url = _inject_github_https_token(remote_url, token)
    timeout = settings.github_clone_timeout_seconds
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")

    git_dir = dest / ".git"
    if git_dir.is_dir():
        r = subprocess.run(
            ["git", "-C", str(dest), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if r.returncode != 0:
            logger.warning(
                "git pull failed for cached repo (using existing files): %s",
                (r.stderr or r.stdout or "")[:500],
            )
        return dest

    if dest.exists() and not git_dir.is_dir():
        shutil.rmtree(dest, ignore_errors=True)

    shallow: list[str] = ["--depth", "1"] if settings.github_shallow_clone else []
    branch_args: list[str] = []
    branch_name = (branch or settings.github_clone_branch or "").strip()
    if branch_name:
        branch_args = ["-b", branch_name]

    cmd = ["git", "clone", *shallow, *branch_args, auth_url, str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed: {(r.stderr or r.stdout or 'unknown')[:800]}"
        )
    return dest


def clone_remote_repository(
    url: str,
    token: Optional[str] = None,
    branch: Optional[str] = None,
) -> Path:
    """Clone or update a remote Git URL; returns path to working tree."""
    ref = url.strip()
    if not ref:
        raise ValueError("Repository URL is required")
    if not _is_remote_git_url(ref):
        raise ValueError(
            "URL must be a remote Git repository (https://, http://, git@, or ssh://)"
        )
    settings = get_settings()
    return _git_clone_or_pull(ref, token or settings.github_token, branch=branch)


def _resolve_repo_ref_to_path(
    repo_ref: str,
    token: Optional[str],
    branch: Optional[str] = None,
) -> Path:
    """Local path as-is; remote URL -> clone cache path."""
    ref = repo_ref.strip()
    if not ref:
        raise ValueError("empty repo ref")
    if _is_remote_git_url(ref):
        return _git_clone_or_pull(ref, token, branch=branch)
    p = Path(ref).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {ref}")
    return p


class GitHubIngester(BaseIngester):
    """Ingests text from Git repos: remote URLs (git clone) or local paths."""

    source_type = "github"

    def __init__(
        self,
        token: str | None = None,
        repo_paths: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.token = token or settings.github_token
        self.repo_paths = repo_paths or (
            [x.strip() for x in settings.github_repos.split(",") if x.strip()]
            if settings.github_repos
            else []
        )

    def _ingest_from_path(self, base_path: Path) -> list[Document]:
        """Ingest from a local directory (clone or plain folder)."""
        documents: list[Document] = []
        base_path = base_path.resolve()

        if not base_path.exists():
            logger.warning("Path does not exist: %s", base_path)
            return documents

        for file_path in base_path.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in file_path.parts for part in SKIP_DIRS):
                continue
            if file_path.suffix.lower() not in DOC_EXTENSIONS:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = file_path.relative_to(base_path)

                doc = Document(
                    content=content,
                    metadata={
                        "file_path": str(rel_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                    },
                    source=str(file_path),
                    source_type=self.source_type,
                    doc_id=f"github:{base_path.name}:{rel_path}",
                )
                documents.append(doc)

            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)

        return documents

    def ingest(self) -> IngestionResult:
        """
        Ingest from GITHUB_REPOS entries: each may be a local path or a remote Git URL.
        Remote URLs are cloned (or pulled) into GITHUB_CLONE_CACHE_DIR before scanning.
        """
        if not self.repo_paths:
            logger.warning(
                "No Git repos configured. Set GITHUB_REPOS (comma-separated local paths or git URLs)."
            )
            return IngestionResult(
                success=False,
                documents_processed=0,
                documents_failed=0,
                source="github",
                source_type=self.source_type,
                documents=[],
                errors=["GITHUB_REPOS not configured"],
            )

        documents: list[Document] = []
        errors: list[str] = []
        resolved: list[str] = []

        for repo_ref in self.repo_paths:
            try:
                base = _resolve_repo_ref_to_path(repo_ref, self.token)
                resolved.append(str(base))
                documents.extend(self._ingest_from_path(base))
            except Exception as e:
                errors.append(f"{repo_ref}: {e}")
                logger.warning("Git ingest failed for %s: %s", repo_ref, e)

        return IngestionResult(
            success=len(documents) > 0,
            documents_processed=len(documents),
            documents_failed=len(errors),
            source="github",
            source_type=self.source_type,
            documents=documents,
            errors=errors,
            metadata={
                "repo_refs": self.repo_paths,
                "resolved_paths": resolved,
            },
        )
