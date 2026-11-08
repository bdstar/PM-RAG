"""Dashboard statistics computed from actual database state."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import models as m


def dashboard_stats(session: Session) -> dict:
    def count(model) -> int:
        return session.scalar(select(func.count()).select_from(model)) or 0

    total_records = (
        count(m.ProjectBudgetSchedule)
        + count(m.ProjectScheduleHistory)
        + count(m.ProjectBudgetHistory)
    )
    indexed_chunks = session.scalar(
        select(func.count()).select_from(m.EvidenceChunk).where(m.EvidenceChunk.indexed.is_(True))
    ) or 0

    results = session.scalars(select(m.RecommendationResult)).all()
    total_recs = len(results)
    abstained = sum(1 for r in results if r.abstained)

    citations = session.scalars(select(m.RecommendationCitation)).all()
    cite_total = len(citations)
    cite_valid = sum(1 for c in citations if c.resolves and c.temporally_valid)

    grounding = session.scalars(select(m.GroundingValidationResult)).all()
    g_total = len(grounding)
    g_unsupported = sum(1 for g in grounding if not g.supported)

    return {
        "total_datasets": count(m.Dataset),
        "total_records": total_records,
        "budget_schedule_records": count(m.ProjectBudgetSchedule),
        "schedule_history_records": count(m.ProjectScheduleHistory),
        "budget_history_records": count(m.ProjectBudgetHistory),
        "unique_projects": count(m.Project),
        "unique_fms": session.scalar(select(func.count(func.distinct(m.ProjectFmsMapping.fms_id)))) or 0,
        "evidence_documents": count(m.EvidenceDocument),
        "evidence_chunks": count(m.EvidenceChunk),
        "indexed_chunks": indexed_chunks,
        "scenarios": count(m.ExecutionScenario),
        "benchmark_scenarios": session.scalar(
            select(func.count()).select_from(m.ExecutionScenario).where(
                m.ExecutionScenario.include_in_benchmark.is_(True))
        ) or 0,
        "recommendations": total_recs,
        "abstentions": abstained,
        "abstention_rate": round(abstained / total_recs, 3) if total_recs else None,
        "citation_validation_rate": round(cite_valid / cite_total, 3) if cite_total else None,
        "unsupported_claim_rate": round(g_unsupported / g_total, 3) if g_total else None,
        "experiments": count(m.Experiment),
        "experiment_runs": count(m.ExperimentRun),
    }


def scenario_distribution(session: Session) -> dict:
    rows = session.execute(
        select(m.ExecutionScenario.problem_type, func.count())
        .group_by(m.ExecutionScenario.problem_type)
    ).all()
    return {r[0]: r[1] for r in rows}


def phase_distribution(session: Session) -> dict:
    rows = session.execute(
        select(m.Project.current_phase, func.count())
        .group_by(m.Project.current_phase)
    ).all()
    return {(r[0] or "Unknown"): r[1] for r in rows}
