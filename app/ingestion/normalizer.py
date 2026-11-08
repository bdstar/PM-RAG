"""Value normalization helpers.

All functions are defensive: they never raise on malformed input, returning
``None`` instead so that one bad cell cannot break a whole import. Both the
original and normalized values are preserved by the importer for audit.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Any, Optional

_MONEY_RE = re.compile(r"[^0-9.\-]")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "n/a"}:
        return None
    return s


def normalize_money(value: Any) -> Optional[float]:
    """'$66,808,000' -> 66808000.0 ; '($5,000)' -> -5000.0."""
    s = clean_str(value)
    if s is None:
        return None
    negative = s.startswith("(") and s.endswith(")")
    cleaned = _MONEY_RE.sub("", s)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        num = float(cleaned)
    except ValueError:
        return None
    return -num if negative else num


def normalize_pct(value: Any) -> Optional[float]:
    """'0%' -> 0.0 ; '12.5%' -> 12.5."""
    s = clean_str(value)
    if s is None:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group(0)) if m else None


def normalize_int(value: Any) -> Optional[float]:
    s = clean_str(value)
    if s is None:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group(0)) if m else None


# Reporting period handling: NYC uses YYYYMM (e.g. "202305").
def normalize_period(value: Any) -> Optional[str]:
    s = clean_str(value)
    if s is None:
        return None
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 6:
        return digits  # YYYYMM
    if len(digits) == 4:  # just a year
        return digits + "00"
    return s


def period_to_date(period: Optional[str]) -> Optional[date]:
    """Convert YYYYMM reporting period to the first day of that month."""
    if not period:
        return None
    digits = re.sub(r"[^0-9]", "", period)
    if len(digits) >= 6:
        try:
            return date(int(digits[:4]), max(1, int(digits[4:6])), 1)
        except ValueError:
            return None
    if len(digits) == 4:
        try:
            return date(int(digits), 1, 1)
        except ValueError:
            return None
    return None


_DATE_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S %p",
    "%Y-%m-%dT%H:%M:%S",
)


def normalize_date(value: Any) -> Optional[date]:
    """Parse inconsistent date strings (NYC uses MM/DD/YYYY primarily)."""
    s = clean_str(value)
    if s is None:
        return None
    s = s.split("T")[0].strip() if "T" in s else s
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.split(" ")[0], fmt.split(" ")[0]).date()
        except (ValueError, IndexError):
            continue
    return None


def normalize_id(value: Any) -> Optional[str]:
    s = clean_str(value)
    return s.upper() if s else None


def row_hash(row: dict, keys: Optional[list[str]] = None) -> str:
    """Deterministic hash of a source row for idempotent import / dedup."""
    if keys:
        payload = {k: row.get(k) for k in keys}
    else:
        payload = row
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
