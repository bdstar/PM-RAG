"""Dataset discovery: find CSV files and auto-classify them."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import get_settings
from ..core.logging import get_logger
from . import csv_inspector, schema_mapper

logger = get_logger(__name__)


def discover_csv_files(directory: str | Path | None = None) -> list[Path]:
    settings = get_settings()
    directory = Path(directory) if directory else settings.dataset_path
    if not directory.exists():
        logger.warning("Dataset directory does not exist: %s", directory)
        return []
    return sorted(p for p in directory.glob("*.csv") if p.is_file())


def discover_and_classify(directory: str | Path | None = None, sample_size: int = 500) -> list[dict[str, Any]]:
    """Return a manifest-style list with classification for each CSV found."""
    results = []
    for path in discover_csv_files(directory):
        try:
            profile = csv_inspector.profile_file(path, sample_size=sample_size)
            col_names = [c["name"] for c in profile["columns"]]
            classification = schema_mapper.classify(col_names)
            results.append(
                {
                    "file": profile["file_name"],
                    "file_path": profile["file_path"],
                    "row_count": profile["row_count"],
                    "column_count": profile["column_count"],
                    "encoding": profile["encoding"],
                    "delimiter": profile["delimiter"],
                    "checksum_sha256": profile["checksum_sha256"],
                    "dataset_type": classification["dataset_type"],
                    "label": classification["label"],
                    "confidence": classification["confidence"],
                    "scores": classification["scores"],
                    "warnings": classification["warnings"],
                    "columns": col_names,
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to profile %s", path)
            results.append({"file": path.name, "file_path": str(path), "error": str(exc)})
    return results


def build_manifest(directory: str | Path | None = None) -> dict:
    """Build a dataset manifest (spec 3.4)."""
    items = discover_and_classify(directory)
    return {
        "datasets": [
            {"dataset_type": it.get("dataset_type"), "file": it.get("file"),
             "confidence": it.get("confidence"), "warnings": it.get("warnings", [])}
            for it in items if "error" not in it
        ]
    }
