"""Experiment reporting and export.

Reads metrics from the database (never invents values) and produces summary
structures + CSV/Excel exports for the research-results dashboard.
"""
from __future__ import annotations

import csv
import io
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import models as m


def run_summary(session: Session, run_id: int) -> dict:
    run = session.get(m.ExperimentRun, run_id)
    if not run:
        return {}
    retrieval = session.scalars(select(m.RetrievalMetric).where(m.RetrievalMetric.run_id == run_id)).all()
    reco = session.scalars(select(m.RecommendationMetric).where(m.RecommendationMetric.run_id == run_id)).all()
    perf = session.scalars(select(m.PerformanceMetric).where(m.PerformanceMetric.run_id == run_id)).all()
    return {
        "run_id": run_id,
        "experiment_id": run.experiment_id,
        "baseline": run.baseline,
        "status": run.status,
        "num_scenarios": run.num_scenarios,
        "reproducibility": run.reproducibility,
        "retrieval_metrics": [
            {"name": r.metric_name, "k": r.k, "value": r.value,
             "ci_low": r.ci_low, "ci_high": r.ci_high} for r in retrieval
        ],
        "recommendation_metrics": [
            {"name": r.metric_name, "value": r.value, "ci_low": r.ci_low,
             "ci_high": r.ci_high, "sample_size": r.sample_size, "note": r.note} for r in reco
        ],
        "performance_metrics": [
            {"name": p.metric_name, "value": p.value, "unit": p.unit} for p in perf
        ],
    }


def compare_runs(session: Session, run_ids: list[int]) -> dict:
    summaries = [run_summary(session, rid) for rid in run_ids]
    summaries = [s for s in summaries if s]
    # Build a flat metric matrix keyed by metric name.
    metric_matrix: dict[str, dict] = {}
    for s in summaries:
        label = f"{s['baseline']}#{s['run_id']}"
        for r in s["retrieval_metrics"]:
            key = f"{r['name']}@{r['k']}" if r["k"] else r["name"]
            metric_matrix.setdefault(key, {})[label] = r["value"]
        for r in s["recommendation_metrics"]:
            metric_matrix.setdefault(r["name"], {})[label] = r["value"]
        for p in s["performance_metrics"]:
            metric_matrix.setdefault(p["name"], {})[label] = p["value"]
    return {"runs": [{"run_id": s["run_id"], "baseline": s["baseline"]} for s in summaries],
            "matrix": metric_matrix}


def export_runs_csv(session: Session, run_ids: Optional[list[int]] = None) -> str:
    if run_ids is None:
        run_ids = [r.id for r in session.scalars(select(m.ExperimentRun)).all()]
    rows = []
    for rid in run_ids:
        s = run_summary(session, rid)
        if not s:
            continue
        for r in s["retrieval_metrics"]:
            rows.append({"run_id": rid, "baseline": s["baseline"], "category": "retrieval",
                         "metric": f"{r['name']}@{r['k']}" if r["k"] else r["name"],
                         "value": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
        for r in s["recommendation_metrics"]:
            rows.append({"run_id": rid, "baseline": s["baseline"], "category": "recommendation",
                         "metric": r["name"], "value": r["value"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
        for p in s["performance_metrics"]:
            rows.append({"run_id": rid, "baseline": s["baseline"], "category": "performance",
                         "metric": p["name"], "value": p["value"], "ci_low": None, "ci_high": None})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["run_id", "baseline", "category", "metric",
                                             "value", "ci_low", "ci_high"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
