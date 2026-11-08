"""Data-quality reporting for ingested CSV files."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ..database import models as m
from . import csv_inspector, schema_mapper


def build_quality_report(profile: dict, dataset_type: str | None = None) -> dict[str, Any]:
    """Produce a data-quality report + warnings from a file profile."""
    columns = profile["columns"]
    col_names = [c["name"] for c in columns]
    high_null = [c["name"] for c in columns if c["null_pct"] >= 50.0]
    empty_cols = [c["name"] for c in columns if c["inferred_type"] == "empty"]
    id_cols = [c["name"] for c in columns if c["is_identifier"]]
    date_cols = [c["name"] for c in columns if c["inferred_type"] == "date"]
    numeric_cols = [c["name"] for c in columns if c["inferred_type"] in ("numeric", "money")]

    warnings = []
    if profile["row_count"] == 0:
        warnings.append("File has no data rows.")
    if high_null:
        warnings.append(f"{len(high_null)} column(s) are >=50% empty: {', '.join(high_null[:8])}")
    if not id_cols:
        warnings.append("No identifier column detected; project linkage may be unreliable.")

    if dataset_type:
        warnings.extend(schema_mapper.validate_assignment(col_names, dataset_type))

    report = {
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "sampled_rows": profile.get("sampled_rows"),
        "identifier_columns": id_cols,
        "date_columns": date_cols,
        "numeric_columns": numeric_cols,
        "high_null_columns": high_null,
        "empty_columns": empty_cols,
        "columns": columns,
        "checksum_sha256": profile["checksum_sha256"],
    }
    return {"report": report, "warnings": warnings}


def persist_quality_report(session: Session, file_id: int, profile: dict, dataset_type: str | None = None) -> m.DatasetDataQualityReport:
    qr = build_quality_report(profile, dataset_type)
    record = m.DatasetDataQualityReport(file_id=file_id, report=qr["report"], warnings={"items": qr["warnings"]})
    session.add(record)
    session.flush()
    return record
