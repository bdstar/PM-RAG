"""Convert normalized structured records and documents into evidence chunks.

Structured records become compact, field-labelled textual representations with
full field-level provenance in metadata (spec 9). Unstructured documents use
sentence-aware chunking.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from ..core.config import get_settings


@dataclass
class Chunk:
    chunk_key: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{int(value):,}"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def budget_schedule_to_text(rec) -> str:
    return (
        f"Record Type: Project Budget & Schedule Snapshot\n"
        f"Project Key: {rec.project_id}\n"
        f"FMS ID: {_fmt(rec.fms_id)}\n"
        f"PID: {_fmt(rec.pid)}\n"
        f"Reporting Period: {_fmt(rec.reporting_period)}\n"
        f"Current Phase: {_fmt(rec.current_phase)}\n"
        f"Total Budget: {_fmt(rec.total_budget)}\n"
        f"Spend To Date: {_fmt(rec.spend_to_date)} ({_fmt(rec.spend_pct)}%)\n"
        f"Forecast Completion: {_fmt(rec.forecast_completion)}\n"
        f"Budget Line: {_fmt(rec.budget_line)}\n"
        f"Borough: {_fmt(rec.borough)}"
    )


def schedule_history_to_text(rec) -> str:
    return (
        f"Record Type: Schedule History Snapshot\n"
        f"Project Key: {rec.project_id}\n"
        f"PID: {_fmt(rec.pid)}\n"
        f"Reporting Period: {_fmt(rec.reporting_period)}\n"
        f"Current Phase: {_fmt(rec.current_phase)}\n"
        f"Completion Date: {_fmt(rec.completion_date)}\n"
        f"Completion Date Type: {_fmt(rec.completion_date_type)}\n"
        f"Schedule Variance (days): {_fmt(rec.variance_days)}\n"
        f"Reason for Forecast Change: {_fmt(rec.reason_for_change)}\n"
        f"Data Date: {_fmt(rec.data_date)}"
    )


def budget_history_to_text(rec) -> str:
    return (
        f"Record Type: Budget Spend History Snapshot\n"
        f"Project Key: {rec.project_id}\n"
        f"FMS ID: {_fmt(rec.fms_id)}\n"
        f"Reporting Period: {_fmt(rec.reporting_period)}\n"
        f"Total Budget: {_fmt(rec.total_budget)}\n"
        f"Spend To Date: {_fmt(rec.spend_to_date)} ({_fmt(rec.spend_pct)}%)\n"
        f"Budget Variance: {_fmt(rec.budget_variance)} ({_fmt(rec.budget_variance_pct)}%)"
    )


def sentence_chunks(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Sentence-aware splitting with word-based size/overlap (no heavy deps)."""
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sent in sentences:
        words = sent.split()
        if length + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # overlap tail
            tail = current[-overlap:] if overlap else []
            current = list(tail)
            length = len(tail)
        current.extend(words)
        length += len(words)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]
