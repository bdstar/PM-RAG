"""Chunked, idempotent CSV importer into the relational schema.

Responsibilities:
  * Register the Dataset / DatasetFile (with checksum, row/col counts).
  * Resolve-or-create projects, preserving ALL original identifiers
    (FMS ID, PID) and never assuming a 1:1 FMS<->project mapping.
  * Stream rows in batches into the correct history table.
  * Dedup via a per-file source-record hash so re-import is safe.
  * Record an ingestion run and a failed-record log.
  * Never modify the original source CSV.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..database import models as m
from . import csv_inspector, normalizer as norm, schema_mapper

logger = get_logger(__name__)

ProgressCb = Optional[Callable[[int, int, int], None]]  # processed, total, failed

DATASET_META = {
    "citywide_budget_schedule": {
        "name": "Capital Projects Dashboard - Citywide Budget and Schedule",
        "evidence_category": "observed",
        "source_url": "https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-budget-and-schedule",
    },
    "schedule_history": {
        "name": "Capital Projects Dashboard - Citywide Schedule History and Variance",
        "evidence_category": "observed",
        "source_url": "https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-schedule-history-and-variance",
    },
    "budget_spend_history": {
        "name": "Capital Projects Dashboard - Citywide Budget Spend History and Variance",
        "evidence_category": "observed",
        "source_url": "https://catalog.data.gov/dataset/capital-projects-dashboard-citywide-budget-spend-history-and-variance",
    },
}


def _get_or_create_dataset(session: Session, dataset_type: str) -> m.Dataset:
    ds = session.scalar(select(m.Dataset).where(m.Dataset.dataset_type == dataset_type))
    if ds:
        return ds
    meta = DATASET_META.get(dataset_type, {"name": dataset_type, "evidence_category": "observed"})
    ds = m.Dataset(
        dataset_type=dataset_type,
        name=meta["name"],
        source_url=meta.get("source_url"),
        evidence_category=meta.get("evidence_category", "observed"),
    )
    session.add(ds)
    session.flush()
    return ds


def _register_file(session: Session, dataset: m.Dataset, profile: dict, dataset_type: str) -> m.DatasetFile:
    existing = session.scalar(
        select(m.DatasetFile).where(m.DatasetFile.checksum_sha256 == profile["checksum_sha256"])
    )
    if existing:
        return existing
    df = m.DatasetFile(
        dataset_id=dataset.id,
        file_name=profile["file_name"],
        file_path=profile["file_path"],
        checksum_sha256=profile["checksum_sha256"],
        file_size=profile["file_size"],
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        delimiter=profile["delimiter"],
        encoding=profile["encoding"],
        classified_type=dataset_type,
    )
    session.add(df)
    session.flush()
    for col in profile["columns"]:
        session.add(
            m.DatasetColumn(
                file_id=df.id,
                name=col["name"],
                inferred_type=col["inferred_type"],
                null_pct=col["null_pct"],
                unique_count=col["unique_count"],
                sample_values={"values": col["sample_values"]},
                is_identifier=col["is_identifier"],
            )
        )
    session.flush()
    return df


class _ProjectResolver:
    """Resolve-or-create projects with an in-memory cache (fast bulk import)."""

    def __init__(self, session: Session):
        self.session = session
        self.cache: dict[str, int] = {}
        self.fms_registered: set[tuple[int, str]] = set()

    def resolve(self, *, fms_id: Optional[str], pid: Optional[str], name: Optional[str],
                managing_agency: Optional[str] = None, sponsor_agency: Optional[str] = None,
                phase: Optional[str] = None, borough: Optional[str] = None) -> int:
        if fms_id:
            key = f"FMS:{fms_id}"
        elif pid:
            key = f"PID:{pid}"
        elif name:
            key = f"NAME:{name[:80]}"
        else:
            key = "UNKNOWN"

        if key in self.cache:
            pid_id = self.cache[key]
        else:
            proj = self.session.scalar(select(m.Project).where(m.Project.project_key == key))
            if not proj:
                proj = m.Project(
                    project_key=key,
                    pid=pid,
                    project_name=name,
                    managing_agency=managing_agency,
                    sponsor_agency=sponsor_agency,
                    current_phase=phase,
                    borough=borough,
                )
                self.session.add(proj)
                self.session.flush()
            pid_id = proj.id
            self.cache[key] = pid_id

        if fms_id and (pid_id, fms_id) not in self.fms_registered:
            exists = self.session.scalar(
                select(m.ProjectFmsMapping).where(
                    m.ProjectFmsMapping.project_id == pid_id,
                    m.ProjectFmsMapping.fms_id == fms_id,
                )
            )
            if not exists:
                self.session.add(m.ProjectFmsMapping(project_id=pid_id, fms_id=fms_id))
            self.fms_registered.add((pid_id, fms_id))
        return pid_id


def _build_record(dataset_type: str, row: dict, project_id: int, file_id: int, rhash: str):
    """Return an ORM history object for the given dataset type, or raise."""
    g = lambda k: row.get(k)  # noqa: E731
    if dataset_type == "citywide_budget_schedule":
        return m.ProjectBudgetSchedule(
            project_id=project_id,
            source_file_id=file_id,
            source_record_hash=rhash,
            reporting_period=norm.normalize_period(g("Reporting Period")),
            fms_id=norm.normalize_id(g("FMS ID")),
            pid=norm.clean_str(g("PID")),
            total_budget=norm.normalize_money(g("Total Budget")),
            spend_to_date=norm.normalize_money(g("Spend to Date")),
            spend_pct=norm.normalize_pct(g("Spend to Date (%)")),
            current_phase=norm.clean_str(g("Current Phase")),
            forecast_completion=norm.normalize_date(g("Forecast Completion")),
            budget_line=norm.clean_str(g("Budget Line")),
            borough=norm.clean_str(g("Borough")),
            raw=row,
        )
    if dataset_type == "schedule_history":
        return m.ProjectScheduleHistory(
            project_id=project_id,
            source_file_id=file_id,
            source_record_hash=rhash,
            reporting_period=norm.normalize_period(g("Reporting Period")),
            pid=norm.clean_str(g("PID")),
            managing_agency=norm.clean_str(g("Managing Agency")),
            current_phase=norm.clean_str(g("Current Phase")),
            completion_date=norm.normalize_date(g("Completion Date")),
            completion_date_type=norm.clean_str(g("Completion Date Type")),
            variance_days=norm.normalize_int(g("Variance (day)")),
            reason_for_change=norm.clean_str(g("Reason for Forecast Completion Change")),
            data_date=norm.normalize_date(g("Data Date")),
            raw=row,
        )
    if dataset_type == "budget_spend_history":
        return m.ProjectBudgetHistory(
            project_id=project_id,
            source_file_id=file_id,
            source_record_hash=rhash,
            reporting_period=norm.normalize_period(g("Year-Month Reported")),
            fms_id=norm.normalize_id(g("FMS ID")),
            managing_agency=norm.clean_str(g("Managing Agency")),
            total_budget=norm.normalize_money(g("Total Budget")),
            spend_to_date=norm.normalize_money(g("Spend to Date")),
            spend_pct=norm.normalize_pct(g("Spend to Date %")),
            budget_variance=norm.normalize_money(g("Budget Variance")),
            budget_variance_pct=norm.normalize_pct(g("Budget Variance %")),
            raw=row,
        )
    raise ValueError(f"Unsupported dataset_type '{dataset_type}'")


_TARGET_TABLE = {
    "citywide_budget_schedule": m.ProjectBudgetSchedule,
    "schedule_history": m.ProjectScheduleHistory,
    "budget_spend_history": m.ProjectBudgetHistory,
}


def import_file(
    session: Session,
    file_path: str | Path,
    dataset_type: str,
    *,
    batch_size: int = 1000,
    limit: Optional[int] = None,
    progress_cb: ProgressCb = None,
) -> dict:
    """Import a single CSV file. Returns a stats dict; idempotent per file."""
    file_path = Path(file_path)
    if dataset_type not in _TARGET_TABLE:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    profile = csv_inspector.profile_file(file_path)
    dataset = _get_or_create_dataset(session, dataset_type)
    df = _register_file(session, dataset, profile, dataset_type)
    session.commit()

    target = _TARGET_TABLE[dataset_type]
    existing_hashes = set(
        session.scalars(
            select(target.source_record_hash).where(target.source_file_id == df.id)
        ).all()
    )

    run = m.DatasetIngestionRun(
        dataset_id=dataset.id,
        file_id=df.id,
        status="running",
        rows_total=profile["row_count"],
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.commit()

    resolver = _ProjectResolver(session)
    succeeded = failed = skipped = processed = 0
    total = profile["row_count"]
    batch = 0

    try:
        for row in csv_inspector.iter_rows(file_path, profile["encoding"], profile["delimiter"]):
            processed += 1
            if limit and processed > limit:
                processed -= 1
                break
            try:
                rhash = norm.row_hash(row)
                if rhash in existing_hashes:
                    skipped += 1
                else:
                    fms = norm.normalize_id(row.get("FMS ID"))
                    pid = norm.clean_str(row.get("PID"))
                    name = (
                        norm.clean_str(row.get("Agency Project Name"))
                        or norm.clean_str(row.get("FMS Project Name"))
                    )
                    project_id = resolver.resolve(
                        fms_id=fms,
                        pid=pid,
                        name=name,
                        managing_agency=norm.clean_str(row.get("Managing Agency")),
                        sponsor_agency=norm.clean_str(row.get("Sponsor Agency")),
                        phase=norm.clean_str(row.get("Current Phase")),
                        borough=norm.clean_str(row.get("Borough")),
                    )
                    obj = _build_record(dataset_type, row, project_id, df.id, rhash)
                    session.add(obj)
                    existing_hashes.add(rhash)
                    succeeded += 1
                    batch += 1
            except Exception as exc:  # per-row isolation
                failed += 1
                session.add(
                    m.DatasetIngestionError(
                        run_id=run.id,
                        row_number=processed,
                        error_type=type(exc).__name__,
                        message=str(exc)[:2000],
                        raw_row={k: (str(v)[:200] if v else v) for k, v in list(row.items())[:40]},
                    )
                )

            if batch >= batch_size:
                session.commit()
                batch = 0
                if progress_cb:
                    progress_cb(processed, total, failed)

        session.commit()
        run.status = "completed"
    except Exception as exc:  # pragma: no cover - catastrophic failure
        session.rollback()
        run = session.get(m.DatasetIngestionRun, run.id)
        run.status = "failed"
        run.message = str(exc)[:2000]
        logger.exception("Ingestion failed for %s", file_path)

    run.rows_succeeded = succeeded
    run.rows_failed = failed
    run.rows_skipped = skipped
    run.finished_at = datetime.now(timezone.utc)
    session.commit()

    if progress_cb:
        progress_cb(processed, total, failed)

    return {
        "dataset_type": dataset_type,
        "file": profile["file_name"],
        "file_id": df.id,
        "run_id": run.id,
        "rows_total": total,
        "rows_succeeded": succeeded,
        "rows_failed": failed,
        "rows_skipped": skipped,
        "status": run.status,
    }
