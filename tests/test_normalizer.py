"""Unit tests for value normalization (NYC CSV conventions)."""
from __future__ import annotations

from datetime import date

from app.ingestion import normalizer as N


def test_normalize_money_variants():
    assert N.normalize_money("$66,808,000") == 66808000.0
    assert N.normalize_money("($5,000)") == -5000.0
    assert N.normalize_money("1234.56") == 1234.56
    assert N.normalize_money("") is None
    assert N.normalize_money("N/A") is None
    assert N.normalize_money(None) is None


def test_normalize_pct():
    assert N.normalize_pct("0%") == 0.0
    assert N.normalize_pct("12.5%") == 12.5
    assert N.normalize_pct("-3%") == -3.0
    assert N.normalize_pct(None) is None


def test_normalize_period_yyyymm():
    assert N.normalize_period("202305") == "202305"
    assert N.normalize_period("2023") == "202300"
    assert N.normalize_period(" 2023-05 ") == "202305"


def test_period_to_date():
    assert N.period_to_date("202305") == date(2023, 5, 1)
    assert N.period_to_date("2023") == date(2023, 1, 1)
    assert N.period_to_date(None) is None


def test_normalize_date_mmddyyyy():
    assert N.normalize_date("05/14/2023") == date(2023, 5, 14)
    assert N.normalize_date("2023-05-14") == date(2023, 5, 14)
    assert N.normalize_date("not a date") is None


def test_normalize_id_uppercases():
    assert N.normalize_id("pid123") == "PID123"
    assert N.normalize_id(None) is None


def test_row_hash_is_deterministic_and_order_independent():
    a = N.row_hash({"b": 2, "a": 1})
    b = N.row_hash({"a": 1, "b": 2})
    assert a == b
    assert a != N.row_hash({"a": 1, "b": 3})
