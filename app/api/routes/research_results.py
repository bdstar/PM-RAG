"""Research-results aggregation and export endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...database import get_db
from ...database import models as m
from ...experiments import reporting

router = APIRouter(prefix="/api/research-results", tags=["research-results"])


def _all_run_ids(db: Session) -> list[int]:
    return [r.id for r in db.scalars(
        select(m.ExperimentRun).where(m.ExperimentRun.status == "completed")).all()]


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    run_ids = _all_run_ids(db)
    summaries = [reporting.run_summary(db, rid) for rid in run_ids]
    return {"num_runs": len(summaries), "runs": summaries}


@router.get("/retrieval")
def retrieval_results(db: Session = Depends(get_db)):
    rows = db.scalars(select(m.RetrievalMetric)).all()
    out = {}
    for r in rows:
        run = db.get(m.ExperimentRun, r.run_id)
        label = f"{run.baseline}#{r.run_id}" if run else str(r.run_id)
        name = f"{r.metric_name}@{r.k}" if r.k else r.metric_name
        out.setdefault(label, {})[name] = {"value": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high}
    return {"metrics": out}


@router.get("/grounding")
def grounding_results(db: Session = Depends(get_db)):
    rows = db.scalars(select(m.RecommendationMetric).where(
        m.RecommendationMetric.metric_name.in_(
            ["supported_rate", "unsupported_rate", "contradiction_rate",
             "attribution_precision", "attribution_recall", "attribution_f1"]))).all()
    out = {}
    for r in rows:
        run = db.get(m.ExperimentRun, r.run_id)
        label = f"{run.baseline}#{r.run_id}" if run else str(r.run_id)
        out.setdefault(label, {})[r.metric_name] = {"value": r.value, "sample_size": r.sample_size}
    return {"metrics": out}


@router.get("/actionability")
def actionability_results(db: Session = Depends(get_db)):
    """Actionability-style metrics from expert annotations (if any)."""
    anns = db.scalars(select(m.ExpertAnnotation)).all()
    if not anns:
        return {"metrics": {}, "note": "No expert annotations recorded yet. "
                "Actionability/feasibility scores require expert review (spec 14.D)."}
    from statistics import mean

    def avg(attr):
        vals = [getattr(a, attr) for a in anns if getattr(a, attr) is not None]
        return mean(vals) if vals else None
    return {"metrics": {
        "actionability": avg("actionability"), "feasibility": avg("feasibility"),
        "relevance": avg("relevance"), "factual_support": avg("factual_support"),
        "n": len(anns)}}


@router.get("/efficiency")
def efficiency_results(db: Session = Depends(get_db)):
    rows = db.scalars(select(m.PerformanceMetric)).all()
    out = {}
    for r in rows:
        run = db.get(m.ExperimentRun, r.run_id)
        label = f"{run.baseline}#{r.run_id}" if run else str(r.run_id)
        out.setdefault(label, {})[r.metric_name] = {"value": r.value, "unit": r.unit}
    return {"metrics": out}


@router.get("/abstention")
def abstention_results(db: Session = Depends(get_db)):
    rows = db.scalars(select(m.RecommendationMetric).where(
        m.RecommendationMetric.metric_name.like("abstention%"))).all()
    out = {}
    for r in rows:
        run = db.get(m.ExperimentRun, r.run_id)
        label = f"{run.baseline}#{r.run_id}" if run else str(r.run_id)
        out.setdefault(label, {})[r.metric_name] = {"value": r.value, "sample_size": r.sample_size}
    return {"metrics": out}


@router.get("/export")
def export(run_ids: str = Query(None), db: Session = Depends(get_db)):
    ids = [int(x) for x in run_ids.split(",")] if run_ids else None
    csv_text = reporting.export_runs_csv(db, ids)
    return PlainTextResponse(csv_text, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=pm_rag_results.csv"})
