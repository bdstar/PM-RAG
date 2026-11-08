"""Derive budget-execution indicators from budget-spend-history snapshots.

Preserves source record IDs and the calculation method. A budget change is a
signal, never asserted as a 'cost overrun' (spec 8, step 6).
"""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..database import models as m

logger = get_logger(__name__)

EVENT_METHODS = ("budget_change", "period_budget_change", "spend_trend")


def analyze_project(session: Session, project_id: int) -> int:
    rows = session.scalars(
        select(m.ProjectBudgetHistory).where(m.ProjectBudgetHistory.project_id == project_id)
    ).all()
    rows = [r for r in rows if r.total_budget is not None]
    if not rows:
        return 0
    rows = sorted(rows, key=lambda r: (r.reporting_period or "", r.id))

    created = 0
    original = rows[0]
    current = rows[-1]
    budget_change = (current.total_budget or 0) - (original.total_budget or 0)

    # Financial snapshot (original vs current) with traceability.
    session.add(
        m.ProjectFinancialSnapshot(
            project_id=project_id,
            reporting_period=current.reporting_period,
            original_budget=original.total_budget,
            current_budget=current.total_budget,
            budget_change=budget_change,
            spend_to_date=current.spend_to_date,
            source_record_ids={"budget_history_ids": [original.id, current.id]},
            method="original=earliest snapshot total budget; current=latest snapshot",
        )
    )

    if budget_change != 0:
        session.add(
            m.ProjectExecutionEvent(
                project_id=project_id,
                event_type="budget_change",
                reporting_period=current.reporting_period,
                value=float(budget_change),
                description=(
                    f"Total budget changed by {budget_change:,.0f} between "
                    f"{original.reporting_period} and {current.reporting_period}."
                ),
                source_record_ids={"budget_history_ids": [original.id, current.id]},
                method="budget_change: latest minus earliest total budget",
            )
        )
        created += 1

    # Period-over-period changes (uses reported Budget Variance where present).
    for prev, curr in zip(rows, rows[1:]):
        delta = None
        if curr.budget_variance is not None and curr.budget_variance != 0:
            delta = curr.budget_variance
            method = "period_budget_change: reported 'Budget Variance'"
        elif prev.total_budget is not None and curr.total_budget is not None:
            d = curr.total_budget - prev.total_budget
            if d != 0:
                delta = d
                method = "period_budget_change: diff of consecutive total budgets"
        if delta:
            session.add(
                m.ProjectExecutionEvent(
                    project_id=project_id,
                    event_type="period_budget_change",
                    reporting_period=curr.reporting_period,
                    value=float(delta),
                    description=(
                        f"Period-over-period budget change of {delta:,.0f} at "
                        f"{curr.reporting_period}."
                    ),
                    source_record_ids={"budget_history_ids": [prev.id, curr.id]},
                    method=method,
                )
            )
            created += 1
    return created


def analyze_all(session: Session, limit: int | None = None) -> dict:
    session.execute(
        delete(m.ProjectExecutionEvent).where(
            m.ProjectExecutionEvent.event_type.in_(EVENT_METHODS)
        )
    )
    session.execute(delete(m.ProjectFinancialSnapshot))
    session.commit()

    project_ids = session.scalars(
        select(m.ProjectBudgetHistory.project_id).distinct()
    ).all()
    if limit:
        project_ids = project_ids[:limit]

    total_events = 0
    for i, pid in enumerate(project_ids, 1):
        total_events += analyze_project(session, pid)
        if i % 500 == 0:
            session.commit()
            logger.info("Budget analysis: %d/%d projects", i, len(project_ids))
    session.commit()
    return {"projects_analyzed": len(project_ids), "events_created": total_events}
