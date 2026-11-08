"""Experiment management endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...database import get_db, session_scope
from ...database import models as m
from ...experiments import reporting
from ...experiments.baselines import list_baselines
from ...experiments.runner import run_experiment_run
from ...services import jobs

router = APIRouter(prefix="/api", tags=["experiments"])


class ExperimentCreate(BaseModel):
    name: str
    research_question: Optional[str] = None
    hypothesis: Optional[str] = None
    description: Optional[str] = None


class RunRequest(BaseModel):
    baseline: str = "pm_rag_full"
    split: Optional[str] = "test"
    max_scenarios: Optional[int] = None
    seed: int = 42
    config_overrides: Optional[dict] = None


@router.get("/experiments/baselines")
def baselines():
    return {"baselines": list_baselines()}


@router.post("/experiments")
def create_experiment(req: ExperimentCreate, db: Session = Depends(get_db)):
    exp = m.Experiment(name=req.name, research_question=req.research_question,
                       hypothesis=req.hypothesis, description=req.description, status="created")
    db.add(exp)
    db.commit()
    return {"id": exp.id, "name": exp.name}


@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    exps = db.scalars(select(m.Experiment).order_by(m.Experiment.id.desc())).all()
    out = []
    for e in exps:
        runs = db.scalars(select(m.ExperimentRun).where(m.ExperimentRun.experiment_id == e.id)).all()
        out.append({"id": e.id, "name": e.name, "research_question": e.research_question,
                    "hypothesis": e.hypothesis, "status": e.status, "num_runs": len(runs),
                    "runs": [{"id": r.id, "baseline": r.baseline, "status": r.status,
                              "num_scenarios": r.num_scenarios} for r in runs]})
    return {"experiments": out}


@router.get("/experiments/compare")
def compare(run_ids: str = Query(..., description="Comma-separated run ids"),
            db: Session = Depends(get_db)):
    ids = [int(x) for x in run_ids.split(",") if x.strip()]
    return reporting.compare_runs(db, ids)


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    e = db.get(m.Experiment, experiment_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    runs = db.scalars(select(m.ExperimentRun).where(m.ExperimentRun.experiment_id == e.id)).all()
    return {"id": e.id, "name": e.name, "research_question": e.research_question,
            "hypothesis": e.hypothesis, "description": e.description, "status": e.status,
            "runs": [{"id": r.id, "baseline": r.baseline, "status": r.status,
                      "num_scenarios": r.num_scenarios,
                      "reproducibility": r.reproducibility} for r in runs]}


@router.post("/experiments/{experiment_id}/run")
def run_experiment(experiment_id: int, req: RunRequest, db: Session = Depends(get_db)):
    if not db.get(m.Experiment, experiment_id):
        raise HTTPException(404, "Experiment not found")

    def _target(handle):
        with session_scope() as s:
            return run_experiment_run(
                s, experiment_id=experiment_id, baseline=req.baseline, split=req.split,
                max_scenarios=req.max_scenarios, seed=req.seed,
                config_overrides=req.config_overrides,
            )
    return {"job_id": jobs.start_job(f"experiment_run:{req.baseline}", _target)}


@router.get("/experiments/{experiment_id}/results")
def experiment_results(experiment_id: int, db: Session = Depends(get_db)):
    runs = db.scalars(select(m.ExperimentRun).where(m.ExperimentRun.experiment_id == experiment_id)).all()
    return {"experiment_id": experiment_id,
            "runs": [reporting.run_summary(db, r.id) for r in runs]}


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    return reporting.run_summary(db, run_id)
