"""Lightweight in-process background job manager.

Long-running operations (bulk ingest, analysis, indexing, experiment runs) run
in a worker thread so HTTP requests return immediately; the UI polls job status.
This is intentionally simple (single-process) and suitable for a local research
platform.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Callable

from ..core.logging import get_logger

logger = get_logger(__name__)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobHandle:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def update(self, **kwargs):
        with _lock:
            _jobs[self.job_id].update(kwargs)

    def progress(self, processed: int, total: int, message: str | None = None):
        with _lock:
            _jobs[self.job_id]["processed"] = processed
            _jobs[self.job_id]["total"] = total
            if message:
                _jobs[self.job_id]["message"] = message


def start_job(name: str, target: Callable[[JobHandle], dict]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id, "name": name, "status": "running", "processed": 0,
            "total": 0, "message": "", "result": None, "error": None,
            "started_at": _now(), "finished_at": None,
        }
    handle = JobHandle(job_id)

    def _run():
        try:
            result = target(handle)
            with _lock:
                _jobs[job_id].update(status="completed", result=result, finished_at=_now())
        except Exception as exc:  # pragma: no cover
            logger.exception("Job %s (%s) failed", job_id, name)
            with _lock:
                _jobs[job_id].update(
                    status="failed", error=str(exc),
                    message=traceback.format_exc()[-2000:], finished_at=_now(),
                )

    threading.Thread(target=_run, daemon=True, name=f"job-{name}").start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def list_jobs() -> list[dict]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
