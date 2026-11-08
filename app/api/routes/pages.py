"""Server-rendered UI pages (Jinja2 + Bootstrap). Data is loaded via the API."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ...core.config import get_settings

router = APIRouter(tags=["ui"])

TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

NAV = [
    {"route": "/", "icon": "speedometer2", "label": "Dashboard"},
    {"route": "/datasets", "icon": "database", "label": "Datasets"},
    {"route": "/projects", "icon": "diagram-3", "label": "Projects"},
    {"route": "/scenarios", "icon": "exclamation-triangle", "label": "Scenarios"},
    {"route": "/rag-workbench", "icon": "cpu", "label": "PM-RAG Workbench"},
    {"route": "/recommendations", "icon": "lightbulb", "label": "Recommendations"},
    {"route": "/experiments", "icon": "flask", "label": "Experiments"},
    {"route": "/research-results", "icon": "bar-chart", "label": "Research Results"},
    {"route": "/evidence-audit", "icon": "search", "label": "Evidence Audit"},
    {"route": "/settings", "icon": "gear", "label": "Settings"},
]


def _ctx(request: Request, active: str, title: str, **extra):
    ctx = {"request": request, "nav": NAV, "active": active, "page_title": title,
           "app_name": get_settings().app_name}
    ctx.update(extra)
    return ctx


def _page(name: str):
    def handler(request: Request):
        meta = {
            "index": ("/", "Research Dashboard"),
            "datasets": ("/datasets", "Dataset Management"),
            "projects": ("/projects", "Project Explorer"),
            "scenarios": ("/scenarios", "Execution Scenarios"),
            "workbench": ("/rag-workbench", "PM-RAG Workbench"),
            "recommendations": ("/recommendations", "Corrective-Action Recommendations"),
            "experiments": ("/experiments", "Experiment Management"),
            "research_results": ("/research-results", "Research Results"),
            "evidence_audit": ("/evidence-audit", "Evidence Audit"),
            "settings": ("/settings", "Settings"),
        }[name]
        return templates.TemplateResponse(request, f"{name}.html", _ctx(request, meta[0], meta[1]))
    return handler


router.add_api_route("/", _page("index"), response_class=HTMLResponse)
router.add_api_route("/datasets", _page("datasets"), response_class=HTMLResponse)
router.add_api_route("/projects", _page("projects"), response_class=HTMLResponse)
router.add_api_route("/scenarios", _page("scenarios"), response_class=HTMLResponse)
router.add_api_route("/rag-workbench", _page("workbench"), response_class=HTMLResponse)
router.add_api_route("/recommendations", _page("recommendations"), response_class=HTMLResponse)
router.add_api_route("/experiments", _page("experiments"), response_class=HTMLResponse)
router.add_api_route("/research-results", _page("research_results"), response_class=HTMLResponse)
router.add_api_route("/evidence-audit", _page("evidence_audit"), response_class=HTMLResponse)
router.add_api_route("/settings", _page("settings"), response_class=HTMLResponse)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(request: Request, project_id: int):
    return templates.TemplateResponse(
        request, "project_detail.html",
        _ctx(request, "/projects", "Project Detail", project_id=project_id),
    )


@router.get("/recommendations/{recommendation_id}", response_class=HTMLResponse)
def recommendation_detail(request: Request, recommendation_id: int):
    return templates.TemplateResponse(
        request, "recommendation_detail.html",
        _ctx(request, "/recommendations", "Recommendation Detail",
             recommendation_id=recommendation_id),
    )
