"""Derive schedule-execution indicators from schedule-history snapshots.

Every derived indicator preserves the source record IDs and the calculation
method (spec 8, step 5). Nothing is asserted as a *cause*; indicators are
movement/variance signals only.
"""
from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..database import models as m

logger = get_logger(__name__)

EVENT_METHODS = ("schedule_variance", "forecast_completion_movement", "forecast_revision_count")


def _sort_key(r: m.ProjectScheduleHistory):
    return (r.reporting_period or "", r.data_date or r.completion_date or "")


def analyze_project(session: Session, project_id: int) -> int:
    rows = session.scalars(
        select(m.ProjectScheduleHistory).where(m.ProjectScheduleHistory.project_id == project_id)
    ).all()
    if not rows:
        return 0
    rows = sorted(rows, key=_sort_key)

    created = 0
    forecast_rows = [r for r in rows if (r.completion_date_type or "").lower() == "forecast"]

    # 1) Per-snapshot reported schedule variance (day).
    for r in rows:
        if r.variance_days is not None and r.variance_days != 0:
            session.add(
                m.ProjectExecutionEvent(
                    project_id=project_id,
                    event_type="schedule_variance",
                    reporting_period=r.reporting_period,
                    event_date=r.completion_date,
                    value=float(r.variance_days),
                    description=(
                        f"Reported schedule variance of {r.variance_days} day(s) at "
                        f"reporting period {r.reporting_period}."
                    ),
                    source_record_ids={"schedule_history_ids": [r.id]},
                    method="schedule_variance: source 'Variance (day)' field",
                )
            )
            created += 1

    # 2) Movement of forecast completion between consecutive forecast snapshots.
    for prev, curr in zip(forecast_rows, forecast_rows[1:]):
        if prev.completion_date and curr.completion_date:
            delta = (curr.completion_date - prev.completion_date).days
            if delta != 0:
                session.add(
                    m.ProjectExecutionEvent(
                        project_id=project_id,
                        event_type="forecast_completion_movement",
                        reporting_period=curr.reporting_period,
                        event_date=curr.completion_date,
                        value=float(delta),
                        description=(
                            f"Forecast completion moved {'later' if delta > 0 else 'earlier'} "
                            f"by {abs(delta)} day(s) between {prev.reporting_period} and "
                            f"{curr.reporting_period}."
                        ),
                        source_record_ids={"schedule_history_ids": [prev.id, curr.id]},
                        method="forecast_completion_movement: diff of consecutive forecast snapshots",
                    )
                )
                created += 1

    # 3) Count of forecast revisions (distinct forecast completion dates).
    distinct_forecasts = {r.completion_date for r in forecast_rows if r.completion_date}
    if len(distinct_forecasts) > 1:
        session.add(
            m.ProjectExecutionEvent(
                project_id=project_id,
                event_type="forecast_revision_count",
                reporting_period=forecast_rows[-1].reporting_period if forecast_rows else None,
                value=float(len(distinct_forecasts)),
                description=f"{len(distinct_forecasts)} distinct forecast completion dates observed.",
                source_record_ids={"schedule_history_ids": [r.id for r in forecast_rows]},
                method="forecast_revision_count: distinct forecast completion dates",
            )
        )
        created += 1
    return created


def analyze_all(session: Session, limit: int | None = None) -> dict:
    # Recompute: clear previously derived schedule events.
    session.execute(
        delete(m.ProjectExecutionEvent).where(
            m.ProjectExecutionEvent.event_type.in_(EVENT_METHODS)
        )
    )
    session.commit()

    project_ids = session.scalars(
        select(m.ProjectScheduleHistory.project_id).distinct()
    ).all()
    if limit:
        project_ids = project_ids[:limit]

    total_events = 0
    for i, pid in enumerate(project_ids, 1):
        total_events += analyze_project(session, pid)
        if i % 500 == 0:
            session.commit()
            logger.info("Schedule analysis: %d/%d projects", i, len(project_ids))
    session.commit()
    return {"projects_analyzed": len(project_ids), "events_created": total_events}
