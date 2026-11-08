"""Automatic dataset classification.

Classifies a CSV into one of the three known PM-RAG dataset types by comparing
its column signature against expected characteristics. Filenames are NOT trusted
as the sole signal (spec 3.4): classification is driven by column headers, and a
confidence score plus warnings are produced so the administrator can override.
"""
from __future__ import annotations

from typing import Any

DATASET_TYPES = {
    "citywide_budget_schedule": {
        "label": "Citywide Budget and Schedule",
        "required_any": ["forecast completion", "current phase", "budget line"],
        "signature": [
            "reporting period", "managing agency", "pid", "fms id", "total budget",
            "spend to date", "current phase", "forecast completion", "budget line",
            "fms project name", "borough", "ten year plan category",
        ],
        "distinctive": ["forecast completion", "ten year plan category", "fms project name"],
    },
    "schedule_history": {
        "label": "Schedule History and Variance",
        "required_any": ["completion date type", "variance (day)", "completion date"],
        "signature": [
            "reporting period", "managing agency", "pid", "agency project name",
            "current phase", "completion date", "completion date type",
            "variance (day)", "reason for forecast completion change", "data date",
        ],
        "distinctive": ["completion date type", "reason for forecast completion change"],
    },
    "budget_spend_history": {
        "label": "Budget Spend History and Variance",
        "required_any": ["year-month reported", "budget variance", "budget variance %"],
        "signature": [
            "managing agency", "fms id", "year-month reported", "total budget",
            "spend to date", "spend to date %", "budget variance", "budget variance %",
        ],
        "distinctive": ["year-month reported", "budget variance"],
    },
}


def _norm(col: str) -> str:
    return (col or "").strip().lower()


def classify(columns: list[str]) -> dict[str, Any]:
    """Return best dataset_type match with confidence + per-type scores."""
    cols = {_norm(c) for c in columns}
    scores: dict[str, float] = {}
    for dtype, spec in DATASET_TYPES.items():
        sig = set(spec["signature"])
        overlap = len(cols & sig)
        base = overlap / len(sig) if sig else 0.0
        # Distinctive columns strongly identify the type.
        distinct_hits = sum(1 for d in spec["distinctive"] if d in cols)
        distinct_bonus = 0.25 * (distinct_hits / max(1, len(spec["distinctive"])))
        # Penalise if no required-any column present.
        has_required = any(r in cols for r in spec["required_any"])
        penalty = 0.0 if has_required else 0.4
        scores[dtype] = round(max(0.0, min(1.0, base + distinct_bonus - penalty)), 4)

    best = max(scores, key=scores.get)
    best_conf = scores[best]
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    warnings = []
    if best_conf < 0.4:
        warnings.append(
            "Low classification confidence; this CSV may not match any known "
            "PM-RAG dataset. Manual mapping recommended."
        )
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.1 and best_conf > 0:
        warnings.append(
            f"Ambiguous: '{ranked[0][0]}' and '{ranked[1][0]}' score similarly."
        )

    return {
        "dataset_type": best,
        "label": DATASET_TYPES[best]["label"],
        "confidence": best_conf,
        "scores": scores,
        "warnings": warnings,
    }


def validate_assignment(columns: list[str], assigned_type: str) -> list[str]:
    """Return warnings if a (possibly manual) assignment looks wrong."""
    cols = {_norm(c) for c in columns}
    spec = DATASET_TYPES.get(assigned_type)
    if not spec:
        return [f"Unknown dataset_type '{assigned_type}'."]
    warnings = []
    if not any(r in cols for r in spec["required_any"]):
        warnings.append(
            f"Assigned '{assigned_type}' but none of its signature columns "
            f"({', '.join(spec['required_any'])}) were found."
        )
    auto = classify(list(columns))
    if auto["dataset_type"] != assigned_type and auto["confidence"] > 0.6:
        warnings.append(
            f"Automatic classification suggests '{auto['dataset_type']}' "
            f"(confidence {auto['confidence']}), not '{assigned_type}'."
        )
    return warnings
