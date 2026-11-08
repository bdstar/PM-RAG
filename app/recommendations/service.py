"""End-to-end corrective-action recommendation service.

Orchestrates: context assembly -> hybrid retrieval -> generation -> citation /
grounding / rule validation -> abstention -> persistence. Stores both raw and
parsed output for audit (spec 7.3, 19).
"""
from __future__ import annotations

import time
from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import models as m
from ..rag import generator
from ..rag.hybrid_retriever import RetrievalParams, retrieve
from ..rag.llm_provider import get_llm
from . import validation
from .schemas import GenerateRequest

logger = get_logger(__name__)


def _period_from_date(d: Optional[date]) -> Optional[str]:
    return f"{d.year:04d}{d.month:02d}" if d else None


def _scenario_context(session: Session, scenario: Optional[m.ExecutionScenario],
                      project: Optional[m.Project]) -> str:
    parts = []
    if project:
        parts.append(f"Project: {project.project_name or project.project_key} "
                     f"(key={project.project_key}, phase={project.current_phase or 'N/A'})")
    if scenario:
        parts.append(f"Scenario: {scenario.title} [{scenario.problem_type}]")
        if scenario.description:
            parts.append(f"Description: {scenario.description}")
        if scenario.as_of_date:
            parts.append(f"As-of date: {scenario.as_of_date.isoformat()}")
        if scenario.current_state:
            parts.append(f"Current state: {scenario.current_state}")
    return "\n".join(parts) if parts else "No structured scenario context provided."


def generate_recommendation(session: Session, req: GenerateRequest,
                            experiment_run_id: Optional[int] = None) -> dict:
    settings = get_settings()
    t_start = time.perf_counter()

    scenario = session.get(m.ExecutionScenario, req.scenario_id) if req.scenario_id else None
    project_id = req.project_id or (scenario.project_id if scenario else None)
    project = session.get(m.Project, project_id) if project_id else None

    problem_type = req.problem_type or (scenario.problem_type if scenario else "insufficient_evidence")
    as_of_period = (
        req.as_of_period
        or (scenario.reporting_period if scenario else None)
        or (_period_from_date(scenario.as_of_date) if scenario and scenario.as_of_date else None)
    )
    problem = (
        req.problem_description
        or (scenario.description if scenario else None)
        or (scenario.title if scenario else "Execution problem reported.")
    )
    scenario_context = _scenario_context(session, scenario, project)
    query = " ".join(filter(None, [
        scenario.title if scenario else None, problem,
        problem_type.replace("_", " "),
    ]))

    reranker_enabled = settings.reranker_enabled if req.reranker_enabled is None else req.reranker_enabled

    # Persist the request (reproducibility).
    request = m.RecommendationRequest(
        scenario_id=req.scenario_id, project_id=project_id,
        as_of_date=scenario.as_of_date if scenario else None,
        retrieval_mode=req.retrieval_mode, top_k=req.top_k,
        embed_model=settings.embed_model, reranker_model=settings.reranker_model,
        generator_model=settings.llm_model, problem_description=problem,
        experiment_run_id=experiment_run_id,
        config={
            "grounding_enabled": req.grounding_enabled,
            "citation_enabled": req.citation_enabled,
            "abstention_enabled": req.abstention_enabled,
            "reranker_enabled": reranker_enabled,
            "dataset_type": req.dataset_type,
            "evidence_category": req.evidence_category,
            "prompt_version": generator.PROMPT_VERSION,
        },
    )
    session.add(request)
    session.flush()

    # --- Retrieval ---
    t_retr = time.perf_counter()
    enabled_retrievers = (
        tuple(req.enabled_retrievers) if req.enabled_retrievers is not None
        else ("bm25", "dense")
    )
    params = RetrievalParams(
        mode=req.retrieval_mode, project_id=project_id, as_of_period=as_of_period,
        top_k=req.top_k, bm25_weight=settings.bm25_weight, dense_weight=settings.dense_weight,
        rrf_k=settings.rrf_k, similarity_threshold=settings.similarity_threshold,
        reranker_enabled=reranker_enabled, dataset_type=req.dataset_type,
        evidence_category=req.evidence_category, enabled_retrievers=enabled_retrievers,
    )
    evidence = retrieve(session, query, params)
    evidence_dicts = [e.to_dict() for e in evidence]
    retrieval_latency = int((time.perf_counter() - t_retr) * 1000)

    # --- Generation ---
    t_gen = time.perf_counter()
    if req.force_rule_based:
        parsed_rb = generator.rule_based_generate(problem_type, problem, scenario_context, evidence_dicts)
        gen = {"parsed": parsed_rb, "raw": "", "backend": "rule-based-template",
               "prompt_version": generator.PROMPT_VERSION}
    else:
        llm = get_llm(settings)
        gen = generator.generate(problem_type, problem, scenario_context, evidence_dicts, llm=llm)
    parsed = gen["parsed"]
    generation_latency = int((time.perf_counter() - t_gen) * 1000)

    # --- Validation ---
    report = validation.validate(
        session, parsed, evidence_dicts, as_of_period=as_of_period,
        citation_enabled=req.citation_enabled, grounding_enabled=req.grounding_enabled,
        abstention_enabled=req.abstention_enabled,
    )

    abstain = report["abstention"]["abstain"] or not parsed.get("evidence_sufficient", True)
    abstention_reason = (
        report["abstention"]["reason"]
        or parsed.get("abstention_reason")
        or ("Evidence insufficient." if abstain else None)
    )

    total_latency = int((time.perf_counter() - t_start) * 1000)

    # --- Persist result ---
    result = m.RecommendationResult(
        request_id=request.id, scenario_id=req.scenario_id,
        problem_summary=parsed.get("problem_summary"),
        evidence_summary=parsed.get("evidence_summary"),
        evidence_sufficient=not abstain,
        abstained=abstain, abstention_reason=abstention_reason,
        limitations={"items": parsed.get("limitations", [])},
        raw_output=gen["raw"], parsed_output=parsed,
        retrieved_evidence={"items": evidence_dicts},
        generator_backend=gen["backend"],
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
        total_latency_ms=total_latency,
        review_status="pending",
    )
    session.add(result)
    session.flush()

    # Actions + citations (skip emitting actions when abstaining).
    if not abstain:
        for action in parsed.get("recommended_actions", []) or []:
            act = m.RecommendedAction(
                result_id=result.id, action=action.get("action", ""),
                category=action.get("category"), priority=action.get("priority", "Medium"),
                responsible_role=action.get("responsible_role"),
                rationale=action.get("rationale"), expected_impact=action.get("expected_impact"),
                assumptions={"items": action.get("assumptions", [])},
                supporting_evidence_ids={"items": action.get("supporting_evidence_ids", [])},
                requires_human_approval=action.get("requires_human_approval", True),
            )
            session.add(act)
            session.flush()
            for cid in action.get("supporting_evidence_ids", []) or []:
                check = next((c for c in report["citation_checks"] if c.cited_id == cid), None)
                session.add(m.RecommendationCitation(
                    result_id=result.id, action_id=act.id,
                    evidence_chunk_id=check.chunk_id if check else None,
                    cited_source_id=cid, claim=action.get("action"),
                    resolves=check.valid if check else False,
                    temporally_valid=check.temporally_valid if check else True,
                ))

    # Grounding validation records.
    for gr in report["grounding"]["results"]:
        session.add(m.GroundingValidationResult(
            result_id=result.id, claim=gr["claim"], label=gr["label"],
            score=gr["score"], method=gr["method"], supported=gr["supported"],
        ))

    if abstain:
        session.add(m.AbstentionRecord(
            result_id=result.id, reason=abstention_reason or "Evidence insufficient.",
            reason_code=report["abstention"]["reason_code"] or "insufficient_evidence",
        ))

    session.commit()

    return {
        "request_id": request.id,
        "result_id": result.id,
        "problem_summary": parsed.get("problem_summary"),
        "evidence_summary": parsed.get("evidence_summary"),
        "recommended_actions": [] if abstain else parsed.get("recommended_actions", []),
        "evidence_sufficient": not abstain,
        "abstained": abstain,
        "abstention_reason": abstention_reason,
        "limitations": parsed.get("limitations", []),
        "retrieved_evidence": evidence_dicts,
        "validation": {
            "citations": report["citations"],
            "grounding": report["grounding"],
            "rule_warnings": report["rule_warnings"],
            "abstention": report["abstention"],
        },
        "generator_backend": gen["backend"],
        "generation_mode": parsed.get("_mode"),
        "latency": {
            "retrieval_ms": retrieval_latency,
            "generation_ms": generation_latency,
            "total_ms": total_latency,
        },
    }


def record_review(session: Session, result_id: int, *, decision: str, reviewer: str,
                  edited_action: Optional[str] = None, notes: Optional[str] = None,
                  review_time_seconds: Optional[float] = None) -> dict:
    result = session.get(m.RecommendationResult, result_id)
    if not result:
        raise ValueError(f"Recommendation {result_id} not found")
    review = m.HumanReviewRecord(
        result_id=result_id, reviewer=reviewer, decision=decision,
        edited_action=edited_action, notes=notes, review_time_seconds=review_time_seconds,
    )
    session.add(review)
    result.review_status = decision
    session.commit()
    return {"result_id": result_id, "review_status": decision, "review_id": review.id}
