"""Citation validation.

Checks that every cited evidence ID (a) exists, (b) was part of the retrieved
evidence set presented to the generator, (c) resolves to an actual chunk/record,
and (d) was available at or before the scenario decision date (temporal check).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import models as m


@dataclass
class CitationCheck:
    cited_id: str
    exists: bool
    in_retrieved_set: bool
    temporally_valid: bool
    chunk_id: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.exists and self.in_retrieved_set and self.temporally_valid


def validate_citations(
    session: Session,
    cited_ids: list[str],
    retrieved_chunk_keys: set[str],
    as_of_period: Optional[str] = None,
) -> list[CitationCheck]:
    checks: list[CitationCheck] = []
    if not cited_ids:
        return checks
    rows = session.scalars(
        select(m.EvidenceChunk).where(m.EvidenceChunk.chunk_key.in_(cited_ids))
    ).all()
    by_key = {r.chunk_key: r for r in rows}
    for cid in cited_ids:
        rec = by_key.get(cid)
        exists = rec is not None
        in_set = cid in retrieved_chunk_keys
        temporally_valid = True
        if exists and as_of_period and rec.reporting_period:
            temporally_valid = rec.reporting_period <= as_of_period
        checks.append(
            CitationCheck(
                cited_id=cid,
                exists=exists,
                in_retrieved_set=in_set,
                temporally_valid=temporally_valid,
                chunk_id=rec.id if rec else None,
            )
        )
    return checks


def citation_summary(checks: list[CitationCheck]) -> dict:
    total = len(checks)
    valid = sum(1 for c in checks if c.valid)
    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "resolve_rate": round(valid / total, 3) if total else 0.0,
        "details": [
            {
                "cited_id": c.cited_id,
                "exists": c.exists,
                "in_retrieved_set": c.in_retrieved_set,
                "temporally_valid": c.temporally_valid,
                "valid": c.valid,
            }
            for c in checks
        ],
    }
