"""
SQLite persistence for audit logs, feedback, chat threads, and ingest jobs.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    workspace_id TEXT,
    thread_id TEXT,
    user_id TEXT DEFAULT 'default',
    question TEXT NOT NULL,
    answer TEXT,
    confidence_score REAL,
    reflection_status TEXT,
    citations_count INTEGER,
    latency_ms REAL,
    success INTEGER NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    workspace_id TEXT,
    thread_id TEXT,
    query TEXT,
    was_helpful INTEGER NOT NULL,
    feedback_text TEXT
);

CREATE TABLE IF NOT EXISTS chat_threads (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    message TEXT,
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS shared_memory (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    thread_id TEXT,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_workspace ON query_audit(workspace_id);
CREATE INDEX IF NOT EXISTS idx_threads_workspace ON chat_threads(workspace_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON chat_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_shared_memory_workspace ON shared_memory(workspace_id);
CREATE INDEX IF NOT EXISTS idx_shared_memory_type ON shared_memory(workspace_id, memory_type);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_database_path() -> Path:
    settings = get_settings()
    path = Path(settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(get_database_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
    logger.info("SQLite database ready at %s", get_database_path())


def insert_query_audit(
    *,
    workspace_id: str | None,
    thread_id: str | None,
    user_id: str,
    question: str,
    answer: str,
    confidence_score: float,
    reflection_status: str,
    citations_count: int,
    latency_ms: float,
    success: bool,
    error: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO query_audit (
                created_at, workspace_id, thread_id, user_id, question, answer,
                confidence_score, reflection_status, citations_count, latency_ms,
                success, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                workspace_id,
                thread_id,
                user_id,
                question,
                answer,
                confidence_score,
                reflection_status,
                citations_count,
                latency_ms,
                1 if success else 0,
                error,
            ),
        )


def get_recent_audit(limit: int = 50, workspace_id: str | None = None) -> list[dict[str, Any]]:
    with get_connection() as conn:
        if workspace_id:
            rows = conn.execute(
                """
                SELECT * FROM query_audit WHERE workspace_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (workspace_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM query_audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def insert_feedback(
    *,
    workspace_id: str | None,
    thread_id: str,
    query: str,
    was_helpful: bool,
    feedback_text: str | None,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback (created_at, workspace_id, thread_id, query, was_helpful, feedback_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_utc_now(), workspace_id, thread_id, query, 1 if was_helpful else 0, feedback_text),
        )
        return int(cur.lastrowid)


def create_chat_thread(workspace_id: str, title: str | None = None) -> dict[str, Any]:
    thread_id = str(uuid.uuid4())
    now = _utc_now()
    title = (title or "New conversation").strip()[:200] or "New conversation"
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_threads (id, workspace_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, workspace_id, title, now, now),
        )
    return {"id": thread_id, "workspace_id": workspace_id, "title": title, "created_at": now, "updated_at": now}


def list_chat_threads(workspace_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT t.*, COUNT(m.id) AS message_count
            FROM chat_threads t
            LEFT JOIN chat_messages m ON m.thread_id = t.id
            WHERE t.workspace_id = ?
            GROUP BY t.id
            ORDER BY t.updated_at DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat_thread(thread_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT t.*, COUNT(m.id) AS message_count
            FROM chat_threads t
            LEFT JOIN chat_messages m ON m.thread_id = t.id
            WHERE t.id = ?
            GROUP BY t.id
            """,
            (thread_id,),
        ).fetchone()
    return dict(row) if row else None


def get_chat_messages(thread_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, thread_id, role, content, created_at FROM chat_messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def append_chat_message(thread_id: str, role: str, content: str) -> dict[str, Any]:
    now = _utc_now()
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM chat_threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            raise ValueError(f"Thread not found: {thread_id}")
        cur = conn.execute(
            """
            INSERT INTO chat_messages (thread_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (thread_id, role, content, now),
        )
        conn.execute("UPDATE chat_threads SET updated_at = ? WHERE id = ?", (now, thread_id))
        if role == "user":
            title = content.strip()[:80] or "Conversation"
            conn.execute(
                "UPDATE chat_threads SET title = ? WHERE id = ? AND title = 'New conversation'",
                (title, thread_id),
            )
    return {"id": int(cur.lastrowid), "thread_id": thread_id, "role": role, "content": content, "created_at": now}


def update_chat_thread_title(thread_id: str, title: str) -> bool:
    title = (title or "New conversation").strip()[:200] or "New conversation"
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE chat_threads SET title = ?, updated_at = ? WHERE id = ?",
            (title, _utc_now(), thread_id),
        )
        return cur.rowcount > 0


def delete_chat_thread(thread_id: str) -> bool:
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
        cur = conn.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
        return cur.rowcount > 0


def delete_workspace_data(workspace_id: str) -> None:
    """Remove server-side chat threads, ingest jobs, and shared memory for a deleted project."""
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE thread_id IN "
            "(SELECT id FROM chat_threads WHERE workspace_id = ?)",
            (workspace_id,),
        )
        conn.execute("DELETE FROM chat_threads WHERE workspace_id = ?", (workspace_id,))
        conn.execute("DELETE FROM ingest_jobs WHERE workspace_id = ?", (workspace_id,))
        conn.execute("DELETE FROM feedback WHERE workspace_id = ?", (workspace_id,))
        conn.execute("DELETE FROM shared_memory WHERE workspace_id = ?", (workspace_id,))


def insert_shared_memory(
    *,
    workspace_id: str,
    content: str,
    memory_type: str = "episodic",
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mem_id = str(uuid.uuid4())
    now = _utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO shared_memory
                (id, workspace_id, thread_id, memory_type, content, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id,
                workspace_id,
                thread_id,
                memory_type,
                content,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
    return {
        "id": mem_id,
        "workspace_id": workspace_id,
        "thread_id": thread_id,
        "memory_type": memory_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }


def list_shared_memories(
    workspace_id: str,
    *,
    limit: int = 20,
    memory_type: str | None = None,
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        if memory_type:
            rows = conn.execute(
                """
                SELECT * FROM shared_memory
                WHERE workspace_id = ? AND memory_type = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (workspace_id, memory_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM shared_memory
                WHERE workspace_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (workspace_id, limit),
            ).fetchall()
    return [_row_to_memory(r) for r in rows]


def search_shared_memories(
    workspace_id: str,
    query: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Simple keyword relevance over recent memories (workspace-scoped)."""
    tokens = [t.lower() for t in (query or "").split() if len(t) > 2][:12]
    candidates = list_shared_memories(workspace_id, limit=max(limit * 5, 40))
    if not tokens:
        return candidates[:limit]

    scored: list[tuple[int, dict[str, Any]]] = []
    for mem in candidates:
        text = (mem.get("content") or "").lower()
        score = sum(1 for t in tokens if t in text)
        if score > 0:
            scored.append((score, mem))
    scored.sort(key=lambda x: (-x[0], x[1].get("updated_at") or ""))
    return [m for _, m in scored[:limit]]


def delete_shared_memory(memory_id: str, workspace_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM shared_memory WHERE id = ? AND workspace_id = ?",
            (memory_id, workspace_id),
        )
        return cur.rowcount > 0


def clear_shared_memories(workspace_id: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM shared_memory WHERE workspace_id = ?",
            (workspace_id,),
        )
        return int(cur.rowcount)


def _row_to_memory(row: sqlite3.Row) -> dict[str, Any]:
    meta_raw = row["metadata_json"] if "metadata_json" in row.keys() else None
    try:
        meta = json.loads(meta_raw) if meta_raw else {}
    except Exception:
        meta = {}
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "thread_id": row["thread_id"],
        "memory_type": row["memory_type"],
        "content": row["content"],
        "metadata": meta,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_ingest_job(workspace_id: str) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = _utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO ingest_jobs (id, workspace_id, status, created_at, updated_at, message)
            VALUES (?, ?, 'pending', ?, ?, 'Queued')
            """,
            (job_id, workspace_id, now, now),
        )
    return {"id": job_id, "workspace_id": workspace_id, "status": "pending", "created_at": now}


def update_ingest_job(
    job_id: str,
    status: str,
    message: str,
    result: dict[str, Any] | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE ingest_jobs SET status = ?, message = ?, result_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, message, json.dumps(result) if result else None, _utc_now(), job_id),
        )


def get_ingest_job(job_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM ingest_jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    out = dict(row)
    if out.get("result_json"):
        try:
            out["result"] = json.loads(out["result_json"])
        except json.JSONDecodeError:
            out["result"] = None
    return out
