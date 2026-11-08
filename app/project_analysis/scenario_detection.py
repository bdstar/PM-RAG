"""Rule-based execution-scenario detection.

All scenarios produced here are labelled ``origin='rule_generated'`` and are NOT
expert-validated. Rules operate over derived execution events + history and
preserve the source record IDs. No causal claims are made.
"""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..database import models as m
from ..ingestion.normalizer import period_to_date

logger = get_logger(__name__)

# Thresholds (configurable constants; documented, rule-derived).
SCHEDULE_DELAY_DAYS = 30
FORECAST_DETERIORATION_DAYS = 30
BUDGET_CHANGE_PCT = 10.0
UNUSUAL_BUDGET_PCT = 25.0
REPEATED_REVISIONS = 3


def _mk_key(project_id: int, problem_type: str, period: str | None) -> str:
    return f"SCN-{project_id}-{problem_type}-{period or 'NA'}"


def _add_scenario(session, *, project_id, problem_type, title, description,
                  period, severity, current_state, source_ids,
                  evidence_sufficient=True) -> bool:
    key = _mk_key(project_id, problem_type, period)
    if session.scalar(select(m.ExecutionScenario).where(m.ExecutionScenario.scenario_key == key)):
        return False
    session.add(
        m.ExecutionScenario(
            scenario_key=key,
            project_id=project_id,
            problem_type=problem_type,
            title=title,
            description=description,
            reporting_period=period,
            as_of_date=period_to_date(period),
            severity=severity,
            origin="rule_generated",
            evidence_sufficient_label=evidence_sufficient,
            current_state=current_state,
            source_record_ids=source_ids,
        )
    )
    return True


def detect_all(session: Session, limit: int | None = None) -> dict:
    # Clear stale rule-generated scenarios that are unreviewed and not in a benchmark.
    session.execute(
        delete(m.ExecutionScenario).where(
            m.ExecutionScenario.origin == "rule_generated",
            m.ExecutionScenario.annotation_status == "unreviewed",
            m.ExecutionScenario.include_in_benchmark.is_(False),
        )
    )
    session.commit()

    events = session.scalars(select(m.ProjectExecutionEvent)).all()
    by_project: dict[int, list[m.ProjectExecutionEvent]] = {}
    for e in events:
        by_project.setdefault(e.project_id, []).append(e)

    counts: dict[str, int] = {}
    created = 0
    project_ids = list(by_project.keys())
    if limit:
        project_ids = project_ids[:limit]

    for pid in project_ids:
        evs = by_project[pid]

        # Rule 1: forecast deterioration (moved later).
        for e in evs:
            if e.event_type == "forecast_completion_movement" and (e.value or 0) >= FORECAST_DETERIORATION_DAYS:
                if _add_scenario(
                    session, project_id=pid, problem_type="forecast_deterioration",
                    title="Forecast completion date moved later",
                    description=e.description, period=e.reporting_period, severity="High",
                    current_state={"forecast_movement_days": e.value},
                    source_ids=e.source_record_ids,
                ):
                    created += 1
                    counts["forecast_deterioration"] = counts.get("forecast_deterioration", 0) + 1
                break

        # Rule 2: potential schedule delay (reported variance).
        for e in evs:
            if e.event_type == "schedule_variance" and (e.value or 0) >= SCHEDULE_DELAY_DAYS:
                if _add_scenario(
                    session, project_id=pid, problem_type="schedule_delay",
                    title="Potential schedule delay (reported variance)",
                    description=e.description, period=e.reporting_period, severity="High",
                    current_state={"reported_variance_days": e.value},
                    source_ids=e.source_record_ids,
                ):
                    created += 1
                    counts["schedule_delay"] = counts.get("schedule_delay", 0) + 1
                break

        # Rule 3 & 4: budget increase / unusual budget change.
        for e in evs:
            if e.event_type == "budget_change" and e.value:
                snap = session.scalar(
                    select(m.ProjectFinancialSnapshot).where(
                        m.ProjectFinancialSnapshot.project_id == pid
                    )
                )
                pct = None
                if snap and snap.original_budget:
                    pct = 100.0 * e.value / snap.original_budget
                if pct is not None and abs(pct) >= BUDGET_CHANGE_PCT:
                    unusual = abs(pct) >= UNUSUAL_BUDGET_PCT
                    ptype = "budget_variance"
                    if _add_scenario(
                        session, project_id=pid, problem_type=ptype,
                        title=("Unusual budget change" if unusual else "Budget change observed"),
                        description=f"{e.description} (~{pct:.1f}% vs original budget).",
                        period=e.reporting_period,
                        severity="High" if unusual else "Medium",
                        current_state={"budget_change": e.value, "budget_change_pct": round(pct, 2)},
                        source_ids=e.source_record_ids,
                    ):
                        created += 1
                        counts[ptype] = counts.get(ptype, 0) + 1
                break

        # Rule 6: repeated forecast revisions.
        for e in evs:
            if e.event_type == "forecast_revision_count" and (e.value or 0) >= REPEATED_REVISIONS:
                if _add_scenario(
                    session, project_id=pid, problem_type="repeated_forecast_revision",
                    title="Repeated schedule-forecast revisions",
                    description=e.description, period=e.reporting_period, severity="Medium",
                    current_state={"forecast_revisions": e.value},
                    source_ids=e.source_record_ids,
                ):
                    created += 1
                    counts["repeated_forecast_revision"] = counts.get("repeated_forecast_revision", 0) + 1
                break

        if len(by_project) and project_ids.index(pid) % 500 == 0:
            session.commit()

    session.commit()

    # Rule 5 & 7: missing / insufficient evidence (projects in construction-like
    # phase with no forecast completion recorded).
    insufficient = 0
    projects = session.scalars(select(m.Project)).all()
    for p in (projects[:limit] if limit else projects):
        bs = session.scalar(
            select(m.ProjectBudgetSchedule).where(m.ProjectBudgetSchedule.project_id == p.id)
        )
        if bs and bs.current_phase and "pre-design" not in (bs.current_phase or "").lower() \
                and bs.forecast_completion is None:
            if _add_scenario(
                session, project_id=p.id, problem_type="insufficient_evidence",
                title="Insufficient schedule evidence for a defensible recommendation",
                description=(
                    f"Project is in phase '{bs.current_phase}' but has no forecast "
                    f"completion date recorded."
                ),
                period=bs.reporting_period, severity="Low",
                current_state={"current_phase": bs.current_phase, "forecast_completion": None},
                source_ids={"budget_schedule_ids": [bs.id]},
                evidence_sufficient=False,
            ):
                insufficient += 1
                created += 1
    counts["insufficient_evidence"] = counts.get("insufficient_evidence", 0) + insufficient
    session.commit()

    return {"scenarios_created": created, "by_type": counts}
