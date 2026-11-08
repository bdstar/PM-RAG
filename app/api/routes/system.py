"""System / health endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ...core.config import get_settings
from ...database import get_db
from ...rag.vector_store import get_vector_store
from ...services import jobs
from ...services.stats import dashboard_stats

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok", "app": get_settings().app_name}


@router.get("/system/status")
def system_status(db: Session = Depends(get_db)):
    settings = get_settings()
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    store = get_vector_store()
    return {
        "database": {"ok": db_ok, "backend": settings.db_backend, "url": settings.sqlalchemy_url.split("://", 1)[0]},
        "vector_store": {"backend": store.backend, "count": store.count()},
        "models": {
            "embed_model": settings.embed_model,
            "reranker_model": settings.reranker_model,
            "nli_model": settings.nli_model,
            "llm_backend": settings.llm_backend,
            "llm_model": settings.llm_model,
        },
        "paths": {
            "dataset_dir": str(settings.dataset_path),
            "storage_dir": str(settings.storage_path),
            "chroma_dir": str(settings.chroma_path),
        },
        "stats": dashboard_stats(db),
    }


@router.get("/jobs")
def jobs_list():
    return {"jobs": jobs.list_jobs()}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        return {"error": "not found"}
    return job
