"""Project explorer endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ...database import get_db
from ...database import models as m

router = APIRouter(prefix="/api", tags=["projects"])


@router.get("/projects")
def list_projects(q: Optional[str] = None, phase: Optional[str] = None,
                  borough: Optional[str] = None, limit: int = Query(50, le=500),
                  offset: int = 0, db: Session = Depends(get_db)):
    stmt = select(m.Project)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(
            m.Project.project_key.like(like), m.Project.pid.like(like),
            m.Project.project_name.like(like),
        ))
    if phase:
        stmt = stmt.where(m.Project.current_phase == phase)
    if borough:
        stmt = stmt.where(m.Project.borough == borough)
    total = len(db.scalars(stmt).all())
    rows = db.scalars(stmt.limit(limit).offset(offset)).all()
    return {"total": total, "projects": [{
        "id": p.id, "project_key": p.project_key, "pid": p.pid,
        "name": p.project_name, "managing_agency": p.managing_agency,
        "phase": p.current_phase, "borough": p.borough,
    } for p in rows]}


@router.get("/projects/{project_id}")
def get_project(project_id: int, db: Session = Depends(get_db)):
    p = db.get(m.Project, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    fms = db.scalars(select(m.ProjectFmsMapping).where(m.ProjectFmsMapping.project_id == project_id)).all()
    bs = db.scalars(select(m.ProjectBudgetSchedule).where(
        m.ProjectBudgetSchedule.project_id == project_id)).all()
    fin = db.scalar(select(m.ProjectFinancialSnapshot).where(
        m.ProjectFinancialSnapshot.project_id == project_id))
    return {
        "id": p.id, "project_key": p.project_key, "pid": p.pid,
        "name": p.project_name, "managing_agency": p.managing_agency,
        "sponsor_agency": p.sponsor_agency, "phase": p.current_phase, "borough": p.borough,
        "fms_ids": [f.fms_id for f in fms],
        "budget_schedule": [{
            "reporting_period": b.reporting_period, "total_budget": b.total_budget,
            "spend_to_date": b.spend_to_date, "current_phase": b.current_phase,
            "forecast_completion": b.forecast_completion.isoformat() if b.forecast_completion else None,
            "source_record_id": b.id,
        } for b in bs[:50]],
        "financial_snapshot": {
            "original_budget": fin.original_budget, "current_budget": fin.current_budget,
            "budget_change": fin.budget_change, "method": fin.method,
            "source_record_ids": fin.source_record_ids,
        } if fin else None,
    }


@router.get("/projects/{project_id}/timeline")
def project_timeline(project_id: int, db: Session = Depends(get_db)):
    events = db.scalars(select(m.ProjectExecutionEvent).where(
        m.ProjectExecutionEvent.project_id == project_id)).all()
    return {"events": [{
        "event_type": e.event_type, "reporting_period": e.reporting_period,
        "event_date": e.event_date.isoformat() if e.event_date else None,
        "value": e.value, "description": e.description,
        "source_record_ids": e.source_record_ids, "method": e.method,
    } for e in sorted(events, key=lambda x: (x.reporting_period or ""))]}


@router.get("/projects/{project_id}/schedule-history")
def project_schedule_history(project_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(m.ProjectScheduleHistory).where(
        m.ProjectScheduleHistory.project_id == project_id)).all()
    return {"schedule_history": [{
        "id": r.id, "reporting_period": r.reporting_period,
        "completion_date": r.completion_date.isoformat() if r.completion_date else None,
        "completion_date_type": r.completion_date_type, "variance_days": r.variance_days,
        "current_phase": r.current_phase, "reason_for_change": r.reason_for_change,
    } for r in sorted(rows, key=lambda x: (x.reporting_period or ""))]}


@router.get("/projects/{project_id}/budget-history")
def project_budget_history(project_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(m.ProjectBudgetHistory).where(
        m.ProjectBudgetHistory.project_id == project_id)).all()
    return {"budget_history": [{
        "id": r.id, "reporting_period": r.reporting_period, "fms_id": r.fms_id,
        "total_budget": r.total_budget, "spend_to_date": r.spend_to_date,
        "budget_variance": r.budget_variance, "budget_variance_pct": r.budget_variance_pct,
    } for r in sorted(rows, key=lambda x: (x.reporting_period or ""))]}
