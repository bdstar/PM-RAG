"""Retrieval / evidence search endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...core.config import get_settings
from ...database import get_db
from ...rag.hybrid_retriever import RETRIEVAL_MODES, RetrievalParams, retrieve
from ...recommendations.schemas import SearchRequest

router = APIRouter(prefix="/api", tags=["retrieval"])


@router.get("/retrieval/modes")
def retrieval_modes():
    return {"modes": RETRIEVAL_MODES}


@router.post("/retrieval/search")
def search(req: SearchRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    params = RetrievalParams(
        mode=req.retrieval_mode, project_id=req.project_id, as_of_period=req.as_of_period,
        top_k=req.top_k, bm25_weight=req.bm25_weight if req.bm25_weight is not None else settings.bm25_weight,
        dense_weight=req.dense_weight if req.dense_weight is not None else settings.dense_weight,
        rrf_k=settings.rrf_k, similarity_threshold=settings.similarity_threshold,
        reranker_enabled=settings.reranker_enabled if req.reranker_enabled is None else req.reranker_enabled,
        dataset_type=req.dataset_type, evidence_category=req.evidence_category,
        enabled_retrievers=tuple(req.enabled_retrievers) if req.enabled_retrievers else ("bm25", "dense"),
    )
    results = retrieve(db, req.query, params)
    return {
        "query": req.query, "mode": req.retrieval_mode, "count": len(results),
        "evidence": [r.to_dict() for r in results],
        "note": "Scores are retrieval scores, not calibrated factual-confidence probabilities.",
    }
