"""Unit tests for content-based dataset classification (filenames not trusted)."""
from __future__ import annotations

from app.ingestion import schema_mapper as SM


def test_classify_citywide_budget_schedule():
    cols = [
        "Reporting Period", "Managing Agency", "PID", "FMS ID", "Total Budget",
        "Spend To Date", "Current Phase", "Forecast Completion", "Budget Line",
        "FMS Project Name", "Borough", "Ten Year Plan Category",
    ]
    res = SM.classify(cols)
    assert res["dataset_type"] == "citywide_budget_schedule"
    assert res["confidence"] > 0.6
    assert res["warnings"] == []


def test_classify_schedule_history():
    cols = [
        "Reporting Period", "Managing Agency", "PID", "Agency Project Name",
        "Current Phase", "Completion Date", "Completion Date Type",
        "Variance (Day)", "Reason For Forecast Completion Change", "Data Date",
    ]
    res = SM.classify(cols)
    assert res["dataset_type"] == "schedule_history"
    assert res["confidence"] > 0.6


def test_classify_budget_spend_history():
    cols = [
        "Managing Agency", "FMS ID", "Year-Month Reported", "Total Budget",
        "Spend To Date", "Spend To Date %", "Budget Variance", "Budget Variance %",
    ]
    res = SM.classify(cols)
    assert res["dataset_type"] == "budget_spend_history"
    assert res["confidence"] > 0.6


def test_classify_unknown_lowers_confidence():
    res = SM.classify(["foo", "bar", "baz"])
    assert res["confidence"] < 0.4
    assert res["warnings"]


def test_validate_assignment_flags_mismatch():
    cols = ["Managing Agency", "FMS ID", "Year-Month Reported", "Budget Variance"]
    warnings = SM.validate_assignment(cols, "schedule_history")
    assert warnings  # assigned type lacks its signature columns
