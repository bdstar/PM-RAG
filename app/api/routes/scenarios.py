"""Execution-scenario management + benchmark building."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...database import get_db, session_scope
from ...database import models as m
from ...ingestion.normalizer import period_to_date
from ...project_analysis import scenario_detection
from ...experiments import benchmark
from ...services import jobs

router = APIRouter(prefix="/api", tags=["scenarios"])


class ScenarioCreate(BaseModel):
    project_id: Optional[int] = None
    problem_type: str
    title: str
    description: Optional[str] = None
    reporting_period: Optional[str] = None
    severity: str = "Medium"
    evidence_sufficient_label: Optional[bool] = None
    include_in_benchmark: bool = False
    current_state: Optional[dict] = None


class ScenarioUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    annotation_status: Optional[str] = None
    evidence_sufficient_label: Optional[bool] = None
    include_in_benchmark: Optional[bool] = None
    split: Optional[str] = None
    origin: Optional[str] = None


class BenchmarkBuild(BaseModel):
    max_scenarios: Optional[int] = None


def _ser(sc: m.ExecutionScenario) -> dict:
    return {
        "id": sc.id, "scenario_key": sc.scenario_key, "project_id": sc.project_id,
        "problem_type": sc.problem_type, "title": sc.title, "description": sc.description,
        "as_of_date": sc.as_of_date.isoformat() if sc.as_of_date else None,
        "reporting_period": sc.reporting_period, "severity": sc.severity, "origin": sc.origin,
        "evidence_sufficient_label": sc.evidence_sufficient_label,
        "annotation_status": sc.annotation_status, "include_in_benchmark": sc.include_in_benchmark,
        "split": sc.split, "current_state": sc.current_state,
    }


@router.get("/scenarios")
def list_scenarios(problem_type: Optional[str] = None, origin: Optional[str] = None,
                   benchmark_only: bool = False, limit: int = Query(100, le=1000),
                   offset: int = 0, db: Session = Depends(get_db)):
    stmt = select(m.ExecutionScenario)
    if problem_type:
        stmt = stmt.where(m.ExecutionScenario.problem_type == problem_type)
    if origin:
        stmt = stmt.where(m.ExecutionScenario.origin == origin)
    if benchmark_only:
        stmt = stmt.where(m.ExecutionScenario.include_in_benchmark.is_(True))
    total = len(db.scalars(stmt).all())
    rows = db.scalars(stmt.order_by(m.ExecutionScenario.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "scenarios": [_ser(s) for s in rows]}


@router.post("/scenarios")
def create_scenario(req: ScenarioCreate, db: Session = Depends(get_db)):
    key = f"MAN-{req.problem_type}-{db.scalar(select(m.ExecutionScenario.id).order_by(m.ExecutionScenario.id.desc())) or 0}"
    sc = m.ExecutionScenario(
        scenario_key=key, project_id=req.project_id, problem_type=req.problem_type,
        title=req.title, description=req.description, reporting_period=req.reporting_period,
        as_of_date=period_to_date(req.reporting_period), severity=req.severity,
        origin="manual", evidence_sufficient_label=req.evidence_sufficient_label,
        include_in_benchmark=req.include_in_benchmark, current_state=req.current_state,
    )
    db.add(sc)
    db.commit()
    return _ser(sc)


@router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: int, db: Session = Depends(get_db)):
    sc = db.get(m.ExecutionScenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")
    gold = db.scalars(select(m.ScenarioEvidence).where(
        m.ScenarioEvidence.scenario_id == scenario_id)).all()
    data = _ser(sc)
    data["source_record_ids"] = sc.source_record_ids
    data["gold_evidence"] = [{"chunk_id": g.chunk_id, "relevance": g.relevance,
                              "is_distractor": g.is_distractor, "note": g.note} for g in gold]
    return data


@router.put("/scenarios/{scenario_id}")
def update_scenario(scenario_id: int, req: ScenarioUpdate, db: Session = Depends(get_db)):
    sc = db.get(m.ExecutionScenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")
    for field, value in req.model_dump(exclude_unset=True).items():
        setattr(sc, field, value)
    db.commit()
    return _ser(sc)


@router.post("/scenarios/detect")
def detect_scenarios(limit: Optional[int] = None):
    def _target(handle):
        with session_scope() as s:
            return scenario_detection.detect_all(s, limit=limit)
    return {"job_id": jobs.start_job("scenario_detection", _target)}


@router.post("/scenarios/benchmark/build")
def build_benchmark_endpoint(req: BenchmarkBuild):
    def _target(handle):
        with session_scope() as s:
            return benchmark.build_benchmark(s, max_scenarios=req.max_scenarios)
    return {"job_id": jobs.start_job("benchmark_build", _target)}
