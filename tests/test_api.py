"""Integration tests for the FastAPI app (SQLite, stub LLM, no external services).

These boot the real application (lifespan runs ``init_db``) and verify that the
API responds and every server-rendered page renders (regression guard for the
Starlette ``TemplateResponse`` signature).
"""
from __future__ import annotations

import pytest

PAGES = [
    "/", "/datasets", "/projects", "/scenarios", "/rag-workbench",
    "/recommendations", "/experiments", "/research-results",
    "/evidence-audit", "/settings",
]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_system_status(client):
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    assert body["database"]["ok"] is True
    assert "stats" in body


@pytest.mark.parametrize("path", PAGES)
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()


def test_detail_pages_render(client):
    assert client.get("/projects/1").status_code == 200
    assert client.get("/recommendations/1").status_code == 200


def test_baselines_endpoint(client):
    r = client.get("/api/experiments/baselines")
    assert r.status_code == 200
    assert "pm_rag_full" in str(r.json())
