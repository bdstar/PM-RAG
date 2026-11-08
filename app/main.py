"""PM-RAG FastAPI application entry point.

Serves the REST API and the interactive research dashboard. The previous
AquaTech RAG modules (``app/rag_pipeline.py`` etc.) are retained in the repo for
reference but the application now boots the PM-RAG platform.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .core.config import get_settings
from .core.logging import get_logger, setup_logging
from .database import init_db

setup_logging()
logger = get_logger("pm_rag")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    logger.info("PM-RAG started. Dataset dir: %s | DB: %s",
                settings.dataset_path, settings.sqlalchemy_url.split("://", 1)[0])
    yield


app = FastAPI(
    title="PM-RAG",
    description="Evidence-Grounded AI Framework for Corrective-Action Decision "
                "Support in Industrial Capital Project Execution.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- API routers ----
from .api.routes import (  # noqa: E402
    datasets, experiments, pages, projects, recommendations,
    research_results, retrieval, scenarios, system,
)

for module in (system, datasets, projects, scenarios, retrieval,
               recommendations, experiments, research_results):
    app.include_router(module.router)

# ---- UI (HTML) ----
app.include_router(pages.router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
