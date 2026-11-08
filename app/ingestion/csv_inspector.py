"""CSV inspection and profiling.

Detects delimiter/encoding, reads headers and a representative sample, and
produces per-column statistics used for data-quality reporting and automatic
dataset classification. Uses chunked reads so very large files are never fully
loaded into memory.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from . import normalizer as norm

logger = get_logger(__name__)

_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def detect_encoding(path: Path) -> str:
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(65536)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def detect_delimiter(path: Path, encoding: str) -> str:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(16384)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_header(path: Path, encoding: str, delimiter: str) -> list[str]:
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            return [c.strip() for c in row]
    return []


def iter_rows(path: Path, encoding: str, delimiter: str):
    """Yield dict rows lazily (streaming)."""
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield {(k.strip() if k else k): v for k, v in row.items()}


def _infer_type(values: list[str]) -> str:
    non_empty = [v for v in values if v not in (None, "")]
    if not non_empty:
        return "empty"
    money = date_like = numeric = 0
    for v in non_empty:
        if norm.normalize_date(v) is not None:
            date_like += 1
        elif "$" in v or norm.normalize_money(v) is not None and any(c.isdigit() for c in v):
            money += 1
            if norm.normalize_int(v) is not None:
                numeric += 1
        elif norm.normalize_int(v) is not None:
            numeric += 1
    n = len(non_empty)
    if date_like / n > 0.6:
        return "date"
    if money / n > 0.4:
        return "money"
    if numeric / n > 0.6:
        return "numeric"
    return "string"


def profile_file(path: Path, sample_size: int = 2000) -> dict[str, Any]:
    """Return a full inspection/profile dict for a CSV file."""
    path = Path(path)
    encoding = detect_encoding(path)
    delimiter = detect_delimiter(path, encoding)
    header = read_header(path, encoding, delimiter)

    col_values: dict[str, list[str]] = {h: [] for h in header}
    col_nonnull: dict[str, int] = {h: 0 for h in header}
    col_unique: dict[str, set] = {h: set() for h in header}
    total_rows = 0
    sampled = 0

    for row in iter_rows(path, encoding, delimiter):
        total_rows += 1
        if sampled < sample_size:
            for h in header:
                v = (row.get(h) or "").strip()
                if v:
                    col_nonnull[h] += 1
                    if len(col_unique[h]) < 10000:
                        col_unique[h].add(v)
                    if len(col_values[h]) < 50:
                        col_values[h].append(v)
            sampled += 1

    columns = []
    for h in header:
        nonnull = col_nonnull[h]
        null_pct = round(100 * (1 - nonnull / sampled), 2) if sampled else 100.0
        inferred = _infer_type(col_values[h])
        name_l = h.lower()
        is_id = (
            "id" in name_l
            or name_l in {"pid", "fms id", "fms_id"}
            or (len(col_unique[h]) > 0.8 * max(sampled, 1) and inferred == "string")
        )
        columns.append(
            {
                "name": h,
                "inferred_type": inferred,
                "null_pct": null_pct,
                "unique_count": len(col_unique[h]),
                "sample_values": col_values[h][:8],
                "is_identifier": bool(is_id),
            }
        )

    file_size = path.stat().st_size
    return {
        "file_name": path.name,
        "file_path": str(path),
        "encoding": encoding,
        "delimiter": delimiter,
        "row_count": total_rows,
        "column_count": len(header),
        "sampled_rows": sampled,
        "columns": columns,
        "file_size": file_size,
        "checksum_sha256": norm.file_sha256(path),
    }
