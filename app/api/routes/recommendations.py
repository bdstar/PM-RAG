"""Corrective-action recommendation endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...database import get_db
from ...database import models as m
from ...recommendations import service
from ...recommendations.schemas import GenerateRequest, ReviewRequest

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.post("/recommendations/generate")
def generate(req: GenerateRequest, db: Session = Depends(get_db)):
    try:
        return service.generate_recommendation(db, req)
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}")


@router.get("/recommendations")
def list_recommendations(abstained: Optional[bool] = None, review_status: Optional[str] = None,
                         limit: int = Query(50, le=500), offset: int = 0,
                         db: Session = Depends(get_db)):
    stmt = select(m.RecommendationResult)
    if abstained is not None:
        stmt = stmt.where(m.RecommendationResult.abstained.is_(abstained))
    if review_status:
        stmt = stmt.where(m.RecommendationResult.review_status == review_status)
    total = len(db.scalars(stmt).all())
    rows = db.scalars(stmt.order_by(m.RecommendationResult.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "recommendations": [{
        "id": r.id, "scenario_id": r.scenario_id, "problem_summary": r.problem_summary,
        "abstained": r.abstained, "evidence_sufficient": r.evidence_sufficient,
        "review_status": r.review_status, "generator_backend": r.generator_backend,
        "total_latency_ms": r.total_latency_ms,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    } for r in rows]}


@router.get("/recommendations/{recommendation_id}")
def get_recommendation(recommendation_id: int, db: Session = Depends(get_db)):
    r = db.get(m.RecommendationResult, recommendation_id)
    if not r:
        raise HTTPException(404, "Recommendation not found")
    actions = db.scalars(select(m.RecommendedAction).where(
        m.RecommendedAction.result_id == r.id)).all()
    citations = db.scalars(select(m.RecommendationCitation).where(
        m.RecommendationCitation.result_id == r.id)).all()
    grounding = db.scalars(select(m.GroundingValidationResult).where(
        m.GroundingValidationResult.result_id == r.id)).all()
    reviews = db.scalars(select(m.HumanReviewRecord).where(
        m.HumanReviewRecord.result_id == r.id)).all()
    return {
        "id": r.id, "scenario_id": r.scenario_id, "problem_summary": r.problem_summary,
        "evidence_summary": r.evidence_summary, "evidence_sufficient": r.evidence_sufficient,
        "abstained": r.abstained, "abstention_reason": r.abstention_reason,
        "limitations": (r.limitations or {}).get("items", []),
        "generator_backend": r.generator_backend,
        "latency": {"retrieval_ms": r.retrieval_latency_ms, "generation_ms": r.generation_latency_ms,
                    "total_ms": r.total_latency_ms},
        "review_status": r.review_status,
        "actions": [{
            "id": a.id, "action": a.action, "category": a.category, "priority": a.priority,
            "responsible_role": a.responsible_role, "rationale": a.rationale,
            "expected_impact": a.expected_impact,
            "assumptions": (a.assumptions or {}).get("items", []),
            "supporting_evidence_ids": (a.supporting_evidence_ids or {}).get("items", []),
            "requires_human_approval": a.requires_human_approval,
        } for a in actions],
        "citations": [{
            "cited_source_id": c.cited_source_id, "resolves": c.resolves,
            "temporally_valid": c.temporally_valid, "evidence_chunk_id": c.evidence_chunk_id,
        } for c in citations],
        "grounding": [{"claim": g.claim, "label": g.label, "score": g.score,
                       "supported": g.supported, "method": g.method} for g in grounding],
        "retrieved_evidence": (r.retrieved_evidence or {}).get("items", []),
        "reviews": [{"decision": rv.decision, "reviewer": rv.reviewer, "notes": rv.notes,
                     "created_at": rv.created_at.isoformat() if rv.created_at else None} for rv in reviews],
    }


@router.post("/recommendations/{recommendation_id}/validate")
def revalidate(recommendation_id: int, db: Session = Depends(get_db)):
    """Re-run validation over the stored retrieved evidence + parsed output."""
    from ...recommendations import validation

    r = db.get(m.RecommendationResult, recommendation_id)
    if not r:
        raise HTTPException(404, "Recommendation not found")
    evidence = (r.retrieved_evidence or {}).get("items", [])
    report = validation.validate(db, r.parsed_output or {}, evidence)
    return {"recommendation_id": recommendation_id,
            "citations": report["citations"], "grounding": report["grounding"],
            "rule_warnings": report["rule_warnings"], "abstention": report["abstention"]}


@router.post("/recommendations/{recommendation_id}/review")
def review(recommendation_id: int, req: ReviewRequest, db: Session = Depends(get_db)):
    try:
        return service.record_review(
            db, recommendation_id, decision=req.decision, reviewer=req.reviewer,
            edited_action=req.edited_action, notes=req.notes,
            review_time_seconds=req.review_time_seconds,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
