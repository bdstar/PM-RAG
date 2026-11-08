"""Dataset discovery, inspection, ingestion, profiling + pipeline operations."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.config import get_settings
from ...database import get_db, session_scope
from ...database import models as m
from ...ingestion import csv_inspector, discovery, importer, quality, schema_mapper
from ...project_analysis import budget_analysis, schedule_analysis, scenario_detection
from ...rag import indexer
from ...services import jobs

router = APIRouter(prefix="/api", tags=["datasets"])


class InspectRequest(BaseModel):
    file_path: str
    dataset_type: Optional[str] = None


class IngestRequest(BaseModel):
    file_path: Optional[str] = None
    dataset_type: Optional[str] = None
    limit: Optional[int] = None
    all_discovered: bool = False


class IndexRequest(BaseModel):
    limit_per_type: Optional[int] = None
    rebuild: bool = True


class AnalysisRequest(BaseModel):
    limit: Optional[int] = None


@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    out = []
    for ds in db.scalars(select(m.Dataset)).all():
        files = db.scalars(select(m.DatasetFile).where(m.DatasetFile.dataset_id == ds.id)).all()
        out.append({
            "id": ds.id, "dataset_type": ds.dataset_type, "name": ds.name,
            "evidence_category": ds.evidence_category, "source_url": ds.source_url,
            "files": [{"id": f.id, "file_name": f.file_name, "row_count": f.row_count,
                       "column_count": f.column_count, "classified_type": f.classified_type,
                       "checksum": f.checksum_sha256[:12]} for f in files],
        })
    return {"datasets": out}


@router.post("/datasets/discover")
def discover(directory: Optional[str] = None):
    items = discovery.discover_and_classify(directory)
    return {"directory": directory or str(get_settings().dataset_path), "items": items}


@router.post("/datasets/inspect")
def inspect(req: InspectRequest):
    try:
        profile = csv_inspector.profile_file(req.file_path, sample_size=1000)
    except Exception as exc:
        raise HTTPException(400, f"Could not inspect file: {exc}")
    col_names = [c["name"] for c in profile["columns"]]
    classification = schema_mapper.classify(col_names)
    qr = quality.build_quality_report(profile, req.dataset_type or classification["dataset_type"])
    return {"profile": profile, "classification": classification, "quality": qr}


@router.post("/datasets/ingest")
def ingest(req: IngestRequest):
    settings = get_settings()

    def _target(handle):
        results = []
        with session_scope() as s:
            if req.all_discovered:
                items = discovery.discover_and_classify()
                targets = [(it["file_path"], it["dataset_type"]) for it in items if "error" not in it]
            else:
                if not req.file_path or not req.dataset_type:
                    raise ValueError("file_path and dataset_type required unless all_discovered=true")
                targets = [(req.file_path, req.dataset_type)]
            handle.update(total=len(targets))
            for i, (path, dtype) in enumerate(targets, 1):
                handle.progress(i - 1, len(targets), f"Importing {path}")
                stats = importer.import_file(
                    s, path, dtype, limit=req.limit,
                    progress_cb=lambda p, t, f: handle.update(message=f"{dtype}: {p}/{t} rows"),
                )
                # Quality report
                profile = csv_inspector.profile_file(path, sample_size=500)
                df = s.scalar(select(m.DatasetFile).where(
                    m.DatasetFile.checksum_sha256 == profile["checksum_sha256"]))
                if df:
                    quality.persist_quality_report(s, df.id, profile, dtype)
                    s.commit()
                results.append(stats)
                handle.progress(i, len(targets))
        return {"imports": results}

    job_id = jobs.start_job("ingest", _target)
    return {"job_id": job_id}


@router.post("/analysis/run")
def run_analysis(req: AnalysisRequest):
    def _target(handle):
        with session_scope() as s:
            handle.update(message="Schedule analysis")
            sched = schedule_analysis.analyze_all(s, limit=req.limit)
            handle.update(message="Budget analysis")
            budg = budget_analysis.analyze_all(s, limit=req.limit)
            handle.update(message="Scenario detection")
            scen = scenario_detection.detect_all(s, limit=req.limit)
        return {"schedule": sched, "budget": budg, "scenarios": scen}

    return {"job_id": jobs.start_job("analysis", _target)}


@router.post("/index/rebuild")
def rebuild_index(req: IndexRequest):
    def _target(handle):
        with session_scope() as s:
            stats = indexer.full_reindex(
                s, limit_per_type=req.limit_per_type, rebuild_vectors=req.rebuild,
                progress_cb=lambda d, t: handle.progress(d, t, "Indexing evidence"),
            )
        return stats

    return {"job_id": jobs.start_job("index", _target)}


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.get(m.Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    files = db.scalars(select(m.DatasetFile).where(m.DatasetFile.dataset_id == ds.id)).all()
    return {
        "id": ds.id, "dataset_type": ds.dataset_type, "name": ds.name,
        "source_url": ds.source_url, "evidence_category": ds.evidence_category,
        "files": [{"id": f.id, "file_name": f.file_name, "row_count": f.row_count,
                   "column_count": f.column_count, "checksum": f.checksum_sha256} for f in files],
    }


@router.get("/datasets/{dataset_id}/profile")
def dataset_profile(dataset_id: int, db: Session = Depends(get_db)):
    files = db.scalars(select(m.DatasetFile).where(m.DatasetFile.dataset_id == dataset_id)).all()
    out = []
    for f in files:
        cols = db.scalars(select(m.DatasetColumn).where(m.DatasetColumn.file_id == f.id)).all()
        qr = db.scalar(select(m.DatasetDataQualityReport).where(
            m.DatasetDataQualityReport.file_id == f.id).order_by(
            m.DatasetDataQualityReport.id.desc()))
        out.append({
            "file": f.file_name, "row_count": f.row_count,
            "columns": [{"name": c.name, "type": c.inferred_type, "null_pct": c.null_pct,
                         "unique": c.unique_count, "is_identifier": c.is_identifier,
                         "samples": (c.sample_values or {}).get("values", [])} for c in cols],
            "quality": {"report": qr.report, "warnings": (qr.warnings or {}).get("items", [])} if qr else None,
        })
    return {"dataset_id": dataset_id, "files": out}


@router.get("/datasets/ingestion-runs")
def ingestion_runs(db: Session = Depends(get_db)):
    runs = db.scalars(select(m.DatasetIngestionRun).order_by(m.DatasetIngestionRun.id.desc())).all()
    return {"runs": [{
        "id": r.id, "dataset_id": r.dataset_id, "status": r.status,
        "rows_total": r.rows_total, "rows_succeeded": r.rows_succeeded,
        "rows_failed": r.rows_failed, "rows_skipped": r.rows_skipped,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    } for r in runs]}


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    settings = get_settings()
    settings.dataset_path.mkdir(parents=True, exist_ok=True)
    dest = settings.dataset_path / (file.filename or "uploaded.csv")
    dest.write_bytes(await file.read())
    profile = csv_inspector.profile_file(dest, sample_size=500)
    classification = schema_mapper.classify([c["name"] for c in profile["columns"]])
    return {"file": str(dest), "classification": classification, "row_count": profile["row_count"]}
