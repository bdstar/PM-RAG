"""Export all data needed for the IEEE paper into reports/paper_data.json.
Reads only from the live DB; computes nothing fabricated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlalchemy import func, select  # noqa: E402
from app.database import session_scope  # noqa: E402
from app.database import models as m  # noqa: E402
from app.experiments.reporting import run_summary  # noqa: E402

out = {}
with session_scope() as s:
    # ---- Dataset statistics ----
    ds = []
    for d in s.scalars(select(m.Dataset)).all():
        ds.append({"id": d.id, "type": d.dataset_type, "name": d.name,
                   "source_url": d.source_url, "evidence_category": d.evidence_category})
    out["datasets"] = ds

    def cnt(model):
        return s.scalar(select(func.count()).select_from(model))

    out["counts"] = {
        "budget_schedule": cnt(m.ProjectBudgetSchedule),
        "schedule_history": cnt(m.ProjectScheduleHistory),
        "budget_history": cnt(m.ProjectBudgetHistory),
        "projects": cnt(m.Project),
        "fms_mappings": cnt(m.ProjectFmsMapping),
        "execution_events": cnt(m.ProjectExecutionEvent),
        "scenarios": cnt(m.ExecutionScenario),
        "evidence_documents": cnt(m.EvidenceDocument),
        "evidence_chunks": cnt(m.EvidenceChunk),
        "scenario_evidence": cnt(m.ScenarioEvidence),
    }

    # Reporting period ranges per table
    def period_range(model, col):
        lo = s.scalar(select(func.min(col)))
        hi = s.scalar(select(func.max(col)))
        return [lo, hi]
    out["period_ranges"] = {
        "budget_schedule": period_range(m.ProjectBudgetSchedule, m.ProjectBudgetSchedule.reporting_period),
        "schedule_history": period_range(m.ProjectScheduleHistory, m.ProjectScheduleHistory.reporting_period),
        "budget_history": period_range(m.ProjectBudgetHistory, m.ProjectBudgetHistory.reporting_period),
    }

    # Unique identifiers
    out["unique"] = {
        "projects": cnt(m.Project),
        "fms_in_mappings": s.scalar(select(func.count(func.distinct(m.ProjectFmsMapping.fms_id)))),
    }

    # Scenario type distribution (all + benchmark test)
    out["scenario_types_all"] = dict(s.execute(
        select(m.ExecutionScenario.problem_type, func.count())
        .group_by(m.ExecutionScenario.problem_type)).all())
    out["scenario_types_test"] = dict(s.execute(
        select(m.ExecutionScenario.problem_type, func.count())
        .where(m.ExecutionScenario.include_in_benchmark.is_(True),
               m.ExecutionScenario.split == "test")
        .group_by(m.ExecutionScenario.problem_type)).all())
    out["split_counts"] = dict(s.execute(
        select(m.ExecutionScenario.split, func.count())
        .where(m.ExecutionScenario.include_in_benchmark.is_(True))
        .group_by(m.ExecutionScenario.split)).all())

    # Data quality reports if present
    dq = []
    for r in s.scalars(select(m.DatasetDataQualityReport)).all():
        dq.append({"file_id": r.file_id, "report": r.report, "warnings": r.warnings})
    out["data_quality"] = dq

    # ---- Experiment 2 runs + all metrics ----
    exp = s.scalar(select(m.Experiment).where(m.Experiment.name.like("%benchmark campaign%"))
                   .order_by(m.Experiment.id.desc()))
    out["experiment"] = {"id": exp.id, "name": exp.name,
                         "research_question": exp.research_question,
                         "hypothesis": exp.hypothesis}
    runs = s.scalars(select(m.ExperimentRun).where(m.ExperimentRun.experiment_id == exp.id)
                     .order_by(m.ExperimentRun.id)).all()
    out["runs"] = []
    repro = None
    for r in runs:
        summ = run_summary(s, r.id)
        out["runs"].append({"run_id": r.id, "baseline": r.baseline,
                            "num_scenarios": r.num_scenarios,
                            "retrieval": summ["retrieval_metrics"],
                            "recommendation": summ["recommendation_metrics"],
                            "performance": summ["performance_metrics"]})
        if repro is None:
            repro = r.reproducibility
    out["reproducibility"] = repro

    # ---- Example recommendations (traceable via ExperimentPrediction) ----
    run_ids = [r.id for r in runs]
    examples = []
    preds = s.scalars(select(m.ExperimentPrediction)
                      .where(m.ExperimentPrediction.run_id.in_(run_ids))).all()
    for p in preds:
        if len(examples) >= 3 or p.abstained or not p.result_id:
            continue
        res = s.get(m.RecommendationResult, p.result_id)
        if not res:
            continue
        actions = s.scalars(select(m.RecommendedAction)
                            .where(m.RecommendedAction.result_id == res.id)).all()
        if not actions:
            continue
        cites = s.scalars(select(m.RecommendationCitation)
                          .where(m.RecommendationCitation.result_id == res.id)).all()
        examples.append({
            "result_id": res.id, "run_id": p.run_id, "scenario_id": res.scenario_id,
            "problem_summary": (res.problem_summary or "")[:500],
            "evidence_summary": (res.evidence_summary or "")[:500],
            "abstained": res.abstained,
            "action": actions[0].action, "category": actions[0].category,
            "priority": actions[0].priority, "responsible_role": actions[0].responsible_role,
            "requires_human_approval": actions[0].requires_human_approval,
            "retrieval_latency_ms": res.retrieval_latency_ms,
            "total_latency_ms": res.total_latency_ms,
            "citations": [{"cited_id": c.cited_source_id,
                           "valid": bool(c.resolves and c.temporally_valid)} for c in cites],
        })
    # One abstention example
    for p in preds:
        if p.abstained and p.result_id:
            res = s.get(m.RecommendationResult, p.result_id)
            if res:
                examples.append({"result_id": res.id, "run_id": p.run_id,
                                 "scenario_id": res.scenario_id, "abstained": True,
                                 "abstention_reason": (res.abstention_reason or "")[:400]})
                break
    out["examples"] = examples

Path("reports").mkdir(exist_ok=True)
Path("reports/paper_data.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print("Wrote reports/paper_data.json")
print("counts:", out["counts"])
print("period_ranges:", out["period_ranges"])
print("test types:", out["scenario_types_test"])
print("runs:", len(out["runs"]), "examples:", len(out["examples"]))
