"""Unit tests for configuration (no hard-coded credentials)."""
from __future__ import annotations

from app.core.config import Settings


def test_sqlite_url_default():
    s = Settings(db_backend="sqlite")
    assert s.sqlalchemy_url.startswith("sqlite:///")


def test_mysql_url_built_from_fields_not_hardcoded():
    s = Settings(db_backend="mysql", db_user="root", db_password="",
                 db_host="localhost", db_port=3306, db_name="pm_rag")
    url = s.sqlalchemy_url
    assert url.startswith("mysql+pymysql://root:@localhost:3306/pm_rag")
    # Credentials come from fields, so a custom password flows through.
    s2 = Settings(db_backend="mysql", db_user="u", db_password="secret",
                  db_host="h", db_port=3307, db_name="db")
    assert "u:secret@h:3307/db" in s2.sqlalchemy_url


def test_explicit_database_url_wins():
    s = Settings(db_backend="mysql", database_url="sqlite:///override.db")
    assert s.sqlalchemy_url == "sqlite:///override.db"


def test_unsupported_backend_raises():
    import pytest

    s = Settings(db_backend="oracle")
    with pytest.raises(ValueError):
        _ = s.sqlalchemy_url
