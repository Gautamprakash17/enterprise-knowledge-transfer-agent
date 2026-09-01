"""
Workspace (project) registry — separate knowledge bases per project.

Each workspace has its own FAISS index under data/workspaces/{id}/faiss_index
so retrieval never mixes Project A docs with Project B.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_WORKSPACE_ID = "default"
_REGISTRY_FILENAME = "registry.json"


def _workspaces_root() -> Path:
    settings = get_settings()
    # Sibling to legacy VECTOR_STORE_PATH (e.g. data/faiss_index -> data/workspaces)
    base = Path(settings.vector_store_path).resolve().parent / "workspaces"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _registry_path() -> Path:
    return _workspaces_root() / _REGISTRY_FILENAME


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "project"


def _load_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.is_file():
        return {"workspaces": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"workspaces": []}


def _save_registry(data: dict[str, Any]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _ensure_default_workspace(registry: dict[str, Any]) -> None:
    workspaces = registry.setdefault("workspaces", [])
    if not any(w.get("id") == DEFAULT_WORKSPACE_ID for w in workspaces):
        workspaces.insert(
            0,
            {
                "id": DEFAULT_WORKSPACE_ID,
                "name": "Default project",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def list_workspaces() -> list[dict[str, Any]]:
    """Return all workspaces (creates default + registry file if missing)."""
    registry = _load_registry()
    _ensure_default_workspace(registry)
    _save_registry(registry)
    return list(registry["workspaces"])


def get_workspace(workspace_id: str) -> dict[str, Any] | None:
    for ws in list_workspaces():
        if ws.get("id") == workspace_id:
            return ws
    return None


def create_workspace(name: str) -> dict[str, Any]:
    """Create a new workspace with an isolated FAISS directory."""
    name = (name or "").strip() or "Untitled project"
    registry = _load_registry()
    _ensure_default_workspace(registry)

    base_id = _slugify(name)
    ws_id = base_id
    existing = {w["id"] for w in registry["workspaces"]}
    n = 2
    while ws_id in existing:
        ws_id = f"{base_id}-{n}"
        n += 1

    entry = {
        "id": ws_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["workspaces"].append(entry)
    _save_registry(registry)

    workspace_index_path(ws_id).mkdir(parents=True, exist_ok=True)
    logger.info("Created workspace %s (%s)", ws_id, name)
    return entry


def workspace_index_path(workspace_id: str) -> Path:
    """
    Directory for this workspace's FAISS index + ingestion manifest.

    Legacy: workspace 'default' uses VECTOR_STORE_PATH if that index already exists.
    """
    settings = get_settings()
    if workspace_id == DEFAULT_WORKSPACE_ID:
        legacy = Path(settings.vector_store_path).resolve()
        if (legacy / "index.faiss").is_file():
            return legacy
    path = _workspaces_root() / workspace_id / "faiss_index"
    path.mkdir(parents=True, exist_ok=True)
    return path


_INDEX_ARTIFACTS = ("index.faiss", "index.pkl", "ingestion_manifest.json")


def _workspace_data_root(workspace_id: str) -> Path:
    """Per-project folder under data/workspaces/{id} (excludes legacy default path)."""
    return _workspaces_root() / normalize_workspace_id(workspace_id)


def _web_uploads_root() -> Path:
    return Path(get_settings().vector_store_path).resolve().parent / "web_uploads"


def delete_workspace(workspace_id: str) -> dict[str, Any]:
    """
    Remove a project from the registry and delete its index, uploads, and data folder.

    The default project cannot be deleted.
    """
    ws = normalize_workspace_id(workspace_id)
    if ws == DEFAULT_WORKSPACE_ID:
        raise ValueError("The default project cannot be deleted")

    registry = _load_registry()
    _ensure_default_workspace(registry)
    workspaces = registry.get("workspaces", [])
    if not any(w.get("id") == ws for w in workspaces):
        raise ValueError(f"Project '{ws}' not found")

    index_result = clear_workspace_index(ws)

    uploads_dir = _web_uploads_root() / ws
    uploads_removed = False
    if uploads_dir.is_dir():
        shutil.rmtree(uploads_dir, ignore_errors=True)
        uploads_removed = True

    data_root = _workspace_data_root(ws)
    data_removed = False
    if data_root.is_dir():
        shutil.rmtree(data_root, ignore_errors=True)
        data_removed = True

    registry["workspaces"] = [w for w in workspaces if w.get("id") != ws]
    _save_registry(registry)

    logger.info("Deleted workspace %s (uploads=%s, data=%s)", ws, uploads_removed, data_removed)
    return {
        "workspace_id": ws,
        "deleted": True,
        "index": index_result,
        "uploads_removed": uploads_removed,
        "data_removed": data_removed,
    }


def clear_workspace_index(workspace_id: str) -> dict[str, Any]:
    """
    Remove FAISS index + manifest for one project (does not delete uploaded files).
    """
    from knowledge_transfer_agent.retrieval.vector_store import reset_vector_store

    ws = normalize_workspace_id(workspace_id)
    path = workspace_index_path(ws)
    removed: list[str] = []
    for name in _INDEX_ARTIFACTS:
        target = path / name
        if target.is_file():
            target.unlink()
            removed.append(name)
    reset_vector_store(ws)
    logger.info("Cleared index for workspace %s: %s", ws, removed)
    return {"workspace_id": ws, "path": str(path), "removed": removed}


def clear_all_workspace_indices() -> list[dict[str, Any]]:
    """Clear indexed data for every registered project + legacy default path if present."""
    from knowledge_transfer_agent.config import get_settings
    from knowledge_transfer_agent.retrieval.vector_store import reset_vector_store

    results: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for ws in list_workspaces():
        r = clear_workspace_index(ws["id"])
        seen_paths.add(r["path"])
        results.append(r)

    legacy = Path(get_settings().vector_store_path).resolve()
    if str(legacy) not in seen_paths and legacy.is_dir():
        removed: list[str] = []
        for name in _INDEX_ARTIFACTS:
            target = legacy / name
            if target.is_file():
                target.unlink()
                removed.append(name)
        if removed:
            reset_vector_store("default")
            results.append(
                {"workspace_id": "default", "path": str(legacy), "removed": removed}
            )

    reset_vector_store()
    return results


def normalize_workspace_id(workspace_id: str | None) -> str:
    ws = (workspace_id or "").strip() or DEFAULT_WORKSPACE_ID
    if not get_workspace(ws):
        # Auto-create unknown ids only if they look like slugs (avoid typos polluting registry)
        if ws != DEFAULT_WORKSPACE_ID and re.match(r"^[a-z0-9][a-z0-9\-]{0,63}$", ws):
            registry = _load_registry()
            _ensure_default_workspace(registry)
            registry["workspaces"].append(
                {
                    "id": ws,
                    "name": ws.replace("-", " ").title(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            _save_registry(registry)
            workspace_index_path(ws).mkdir(parents=True, exist_ok=True)
            return ws
        return DEFAULT_WORKSPACE_ID
    return ws
