"""Shared pytest fixtures and test environment setup.

Tests run fully offline against a disposable SQLite database and the
deterministic ``stub`` LLM backend, so no DB server, GPU, or paid API is
required. Environment variables are set *before* any app module is imported so
the cached ``Settings`` picks them up.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ---- Force a safe, isolated test configuration (before importing the app) ----
_TMP = Path(tempfile.mkdtemp(prefix="pmrag_test_"))
os.environ.setdefault("DB_BACKEND", "sqlite")
os.environ.setdefault("LLM_BACKEND", "stub")
os.environ.setdefault("NLI_ENABLED", "false")
os.environ.setdefault("RERANKER_ENABLED", "false")
os.environ["STORAGE_DIR"] = str(_TMP / "storage")
os.environ["CHROMA_DIR"] = str(_TMP / "storage" / "chroma")
os.environ["REPORTS_DIR"] = str(_TMP / "reports")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    from app.core.config import get_settings

    s = get_settings()
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def client():
    """A FastAPI TestClient that triggers lifespan (init_db) on SQLite."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
