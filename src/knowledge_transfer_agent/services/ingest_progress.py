"""Map ingestion pipeline phases to overall job progress (0–100%)."""

from __future__ import annotations

from collections.abc import Callable

from knowledge_transfer_agent.core.database import update_ingest_job

# phase -> (start_percent, end_percent)
_PHASE_WEIGHTS: dict[str, tuple[int, int]] = {
    "scan": (0, 22),
    "chunk": (22, 38),
    "embed": (38, 92),
    "save": (92, 100),
}


def phase_percent(phase: str, current: int, total: int) -> int:
    lo, hi = _PHASE_WEIGHTS.get(phase, (0, 100))
    if total <= 0:
        inner = 1.0 if current > 0 else 0.0
    else:
        inner = min(1.0, max(0.0, current / total))
    return min(100, max(0, int(lo + (hi - lo) * inner)))


def make_job_progress_reporter(job_id: str) -> Callable[[str, int, int, str], None]:
    """Return a callback suitable for IngestionPipeline.run(on_progress=...)."""

    def report(phase: str, current: int, total: int, message: str) -> None:
        pct = phase_percent(phase, current, total)
        update_ingest_job(
            job_id,
            "running",
            message,
            {
                "progress": {
                    "phase": phase,
                    "percent": pct,
                    "current": current,
                    "total": total,
                    "message": message,
                },
            },
        )

    return report
