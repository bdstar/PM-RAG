"""Pydantic schemas for corrective-action recommendations and API I/O."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendedActionModel(BaseModel):
    action: str
    category: Optional[str] = None
    priority: str = "Medium"
    responsible_role: Optional[str] = None
    rationale: Optional[str] = None
    expected_impact: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    requires_human_approval: bool = True


class RecommendationModel(BaseModel):
    scenario_id: Optional[str] = None
    problem_summary: str = ""
    evidence_summary: str = ""
    recommended_actions: List[RecommendedActionModel] = Field(default_factory=list)
    evidence_sufficient: bool = True
    abstention_reason: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


# ---- API request / response ----
class GenerateRequest(BaseModel):
    scenario_id: Optional[int] = Field(default=None, description="Benchmark scenario id.")
    project_id: Optional[int] = None
    problem_type: Optional[str] = None
    problem_description: Optional[str] = Field(default=None)
    as_of_period: Optional[str] = Field(default=None, description="YYYYMM decision cutoff.")
    retrieval_mode: str = "hybrid_temporal"
    top_k: int = Field(default=8, ge=1, le=50)
    reranker_enabled: Optional[bool] = None
    grounding_enabled: bool = True
    citation_enabled: bool = True
    abstention_enabled: bool = True
    dataset_type: Optional[str] = None
    evidence_category: Optional[str] = None
    enabled_retrievers: Optional[List[str]] = None
    force_rule_based: bool = False


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    project_id: Optional[int] = None
    as_of_period: Optional[str] = None
    retrieval_mode: str = "hybrid_temporal"
    top_k: int = Field(default=8, ge=1, le=50)
    bm25_weight: Optional[float] = None
    dense_weight: Optional[float] = None
    reranker_enabled: Optional[bool] = None
    enabled_retrievers: Optional[List[str]] = None
    dataset_type: Optional[str] = None
    evidence_category: Optional[str] = None


class ReviewRequest(BaseModel):
    decision: str = Field(description="accept|edit|reject|request_evidence|insufficient")
    reviewer: str = "local_user"
    edited_action: Optional[str] = None
    notes: Optional[str] = None
    review_time_seconds: Optional[float] = None
