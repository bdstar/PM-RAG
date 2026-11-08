"""SQLAlchemy ORM models for the PM-RAG relational schema.

The schema is organised into six areas mirroring the research spec:
  1. Dataset management
  2. Project data (state + history snapshots)
  3. RAG knowledge (evidence documents / chunks / index versions)
  4. Research benchmark (scenarios, gold evidence, expert annotations)
  5. Recommendation system (requests, results, actions, citations, validation)
  6. Experiment management (runs, predictions, metrics, comparisons)

Design notes:
  * Original source identifiers (FMS ID, PID, reporting period) are always
    preserved alongside canonical keys.
  * No one-to-one FMS<->project assumption is baked in: a project can carry
    multiple FMS mappings (``project_fms_mappings``).
  * created_at / updated_at / version fields are provided via ``TimestampMixin``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .connection import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )
    version: Mapped[int] = mapped_column(Integer, default=1)


# ===========================================================================
# 1. Dataset management
# ===========================================================================
class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_type: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_org: Mapped[str | None] = mapped_column(String(255), nullable=True)
    evidence_category: Mapped[str] = mapped_column(String(64), default="observed")
    data_version: Mapped[str] = mapped_column(String(64), default="v1")

    files: Mapped[list["DatasetFile"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetFile(Base, TimestampMixin):
    __tablename__ = "dataset_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    file_name: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    delimiter: Mapped[str | None] = mapped_column(String(8), nullable=True)
    encoding: Mapped[str | None] = mapped_column(String(32), nullable=True)
    classified_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classification_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    manual_override: Mapped[bool] = mapped_column(Boolean, default=False)

    dataset: Mapped[Dataset] = relationship(back_populates="files")
    columns: Mapped[list["DatasetColumn"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("dataset_files.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    inferred_type: Mapped[str] = mapped_column(String(32), default="string")
    null_pct: Mapped[float] = mapped_column(Float, default=0.0)
    unique_count: Mapped[int] = mapped_column(Integer, default=0)
    sample_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_identifier: Mapped[bool] = mapped_column(Boolean, default=False)

    file: Mapped[DatasetFile] = relationship(back_populates="columns")


class DatasetIngestionRun(Base, TimestampMixin):
    __tablename__ = "dataset_ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    file_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_files.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_succeeded: Mapped[int] = mapped_column(Integer, default=0)
    rows_failed: Mapped[int] = mapped_column(Integer, default=0)
    rows_skipped: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)


class DatasetIngestionError(Base):
    __tablename__ = "dataset_ingestion_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("dataset_ingestion_runs.id"), index=True)
    row_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text)
    raw_row: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class DatasetDataQualityReport(Base, TimestampMixin):
    __tablename__ = "dataset_data_quality_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("dataset_files.id"), index=True)
    report: Mapped[dict] = mapped_column(JSON)
    warnings: Mapped[dict | None] = mapped_column(JSON, nullable=True)


# ===========================================================================
# 2. Project data
# ===========================================================================
class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    pid: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    managing_agency: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sponsor_agency: Mapped[str | None] = mapped_column(String(128), nullable=True)
    project_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    borough: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ProjectFmsMapping(Base):
    __tablename__ = "project_fms_mappings"
    __table_args__ = (UniqueConstraint("project_id", "fms_id", name="uq_project_fms"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    fms_id: Mapped[str] = mapped_column(String(64), index=True)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)


class ProjectBudgetSchedule(Base, TimestampMixin):
    """Snapshot rows from the Citywide Budget and Schedule dataset."""

    __tablename__ = "project_budget_schedule"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    source_file_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_files.id"), nullable=True)
    source_record_hash: Mapped[str] = mapped_column(String(64), index=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    fms_id: Mapped[str | None] = mapped_column(String(64), index=True)
    pid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    spend_to_date: Mapped[float | None] = mapped_column(Float, nullable=True)
    spend_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    forecast_completion: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    budget_line: Mapped[str | None] = mapped_column(String(64), nullable=True)
    borough: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ProjectScheduleHistory(Base, TimestampMixin):
    """Schedule History and Variance snapshots."""

    __tablename__ = "project_schedule_history"
    __table_args__ = (
        Index("ix_sched_proj_period", "project_id", "reporting_period"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    source_file_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_files.id"), nullable=True)
    source_record_hash: Mapped[str] = mapped_column(String(64), index=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    pid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    managing_agency: Mapped[str | None] = mapped_column(String(128), nullable=True)
    current_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    completion_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    completion_date_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    variance_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_for_change: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ProjectBudgetHistory(Base, TimestampMixin):
    """Budget Spend History and Variance snapshots."""

    __tablename__ = "project_budget_history"
    __table_args__ = (
        Index("ix_budg_proj_period", "project_id", "reporting_period"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    source_file_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_files.id"), nullable=True)
    source_record_hash: Mapped[str] = mapped_column(String(64), index=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    fms_id: Mapped[str | None] = mapped_column(String(64), index=True)
    managing_agency: Mapped[str | None] = mapped_column(String(128), nullable=True)
    total_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    spend_to_date: Mapped[float | None] = mapped_column(Float, nullable=True)
    spend_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    budget_variance: Mapped[float | None] = mapped_column(Float, nullable=True)
    budget_variance_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ProjectFinancialSnapshot(Base, TimestampMixin):
    """Derived financial state per project/reporting period (traceable)."""

    __tablename__ = "project_financial_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    original_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_budget: Mapped[float | None] = mapped_column(Float, nullable=True)
    budget_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    spend_to_date: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_record_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    method: Mapped[str | None] = mapped_column(String(255), nullable=True)


class ProjectExecutionEvent(Base, TimestampMixin):
    """Derived execution indicators (schedule movement, variance, etc.)."""

    __tablename__ = "project_execution_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    event_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_record_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    method: Mapped[str | None] = mapped_column(String(255), nullable=True)


# ===========================================================================
# 3. RAG knowledge
# ===========================================================================
class EvidenceDocument(Base, TimestampMixin):
    __tablename__ = "evidence_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    dataset_type: Mapped[str | None] = mapped_column(String(64), index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    document_type: Mapped[str] = mapped_column(String(64), default="structured_record")
    evidence_category: Mapped[str] = mapped_column(String(64), default="observed")
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_record_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    event_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    access_level: Mapped[str] = mapped_column(String(32), default="public")
    source_org: Mapped[str | None] = mapped_column(String(255), nullable=True)
    publication_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)


class EvidenceChunk(Base, TimestampMixin):
    __tablename__ = "evidence_chunks"
    __table_args__ = (
        Index("ix_chunk_proj_period", "project_id", "reporting_period"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("evidence_documents.id"), index=True)
    source_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    dataset_type: Mapped[str | None] = mapped_column(String(64), index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    fms_id: Mapped[str | None] = mapped_column(String(64), index=True)
    record_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), index=True)
    event_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    project_phase: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_category: Mapped[str] = mapped_column(String(64), default="observed")
    source_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    original_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    access_level: Mapped[str] = mapped_column(String(32), default="public")
    indexed: Mapped[bool] = mapped_column(Boolean, default=False)


class EmbeddingMetadata(Base, TimestampMixin):
    __tablename__ = "embedding_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("evidence_chunks.id"), index=True)
    index_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("retrieval_index_versions.id"), nullable=True
    )
    embed_model: Mapped[str] = mapped_column(String(255))
    dim: Mapped[int] = mapped_column(Integer)
    vector_id: Mapped[str] = mapped_column(String(128))


class RetrievalIndexVersion(Base, TimestampMixin):
    __tablename__ = "retrieval_index_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    embed_model: Mapped[str] = mapped_column(String(255))
    dim: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_size: Mapped[int] = mapped_column(Integer, default=0)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=0)
    collection_name: Mapped[str] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvidenceRelationship(Base):
    __tablename__ = "evidence_relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    src_document_id: Mapped[int] = mapped_column(ForeignKey("evidence_documents.id"), index=True)
    dst_document_id: Mapped[int] = mapped_column(ForeignKey("evidence_documents.id"), index=True)
    relationship_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ===========================================================================
# 4. Research benchmark
# ===========================================================================
class ExecutionScenario(Base, TimestampMixin):
    __tablename__ = "execution_scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    problem_type: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    as_of_date: Mapped[datetime | None] = mapped_column(Date, index=True, nullable=True)
    reporting_period: Mapped[str | None] = mapped_column(String(16), nullable=True)
    severity: Mapped[str] = mapped_column(String(32), default="Medium")
    origin: Mapped[str] = mapped_column(String(32), default="rule_generated")  # rule_generated|manual|expert_validated
    evidence_sufficient_label: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    annotation_status: Mapped[str] = mapped_column(String(32), default="unreviewed")
    include_in_benchmark: Mapped[bool] = mapped_column(Boolean, default=False)
    split: Mapped[str | None] = mapped_column(String(16), nullable=True)  # train|val|test
    current_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_record_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ScenarioEvidence(Base):
    """Gold relevant/distractor evidence links for a benchmark scenario."""

    __tablename__ = "scenario_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("execution_scenarios.id"), index=True)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("evidence_chunks.id"), nullable=True)
    relevance: Mapped[int] = mapped_column(Integer, default=1)  # graded relevance (0..3)
    is_distractor: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)


class ReferenceCorrectiveAction(Base, TimestampMixin):
    __tablename__ = "reference_corrective_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("execution_scenarios.id"), index=True)
    action_text: Mapped[str] = mapped_column(Text)
    acceptable: Mapped[bool] = mapped_column(Boolean, default=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_type: Mapped[str] = mapped_column(String(32), default="expert")  # expert|pattern|simulated
    supporting_evidence_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ExpertAnnotation(Base, TimestampMixin):
    __tablename__ = "expert_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int | None] = mapped_column(ForeignKey("execution_scenarios.id"), nullable=True, index=True)
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendation_results.id"), nullable=True, index=True
    )
    reviewer: Mapped[str] = mapped_column(String(128))
    factual_support: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actionability: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feasibility: Mapped[int | None] = mapped_column(Integer, nullable=True)
    safety: Mapped[int | None] = mapped_column(Integer, nullable=True)
    accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    edits_required: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    review_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class AnnotationDisagreement(Base):
    __tablename__ = "annotation_disagreements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("execution_scenarios.id"), index=True)
    metric: Mapped[str] = mapped_column(String(64))
    reviewer_a: Mapped[str] = mapped_column(String(128))
    reviewer_b: Mapped[str] = mapped_column(String(128))
    delta: Mapped[float] = mapped_column(Float, default=0.0)
    adjudicated_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class BenchmarkSplit(Base, TimestampMixin):
    __tablename__ = "benchmark_splits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    strategy: Mapped[str] = mapped_column(String(64), default="temporal")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    assignments: Mapped[dict | None] = mapped_column(JSON, nullable=True)


# ===========================================================================
# 5. Recommendation system
# ===========================================================================
class RecommendationRequest(Base, TimestampMixin):
    __tablename__ = "recommendation_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int | None] = mapped_column(ForeignKey("execution_scenarios.id"), nullable=True, index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    as_of_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    retrieval_mode: Mapped[str] = mapped_column(String(32), default="current_project")
    top_k: Mapped[int] = mapped_column(Integer, default=8)
    embed_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reranker_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    generator_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    problem_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    experiment_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiment_runs.id"), nullable=True, index=True
    )


class RecommendationResult(Base, TimestampMixin):
    __tablename__ = "recommendation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("recommendation_requests.id"), index=True)
    scenario_id: Mapped[int | None] = mapped_column(ForeignKey("execution_scenarios.id"), nullable=True, index=True)
    problem_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_sufficient: Mapped[bool] = mapped_column(Boolean, default=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    abstention_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    limitations: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieved_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    generator_backend: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retrieval_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    generation_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    token_usage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending")

    actions: Mapped[list["RecommendedAction"]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )


class RecommendedAction(Base):
    __tablename__ = "recommended_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("recommendation_results.id"), index=True)
    action: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    priority: Mapped[str] = mapped_column(String(32), default="Medium")
    responsible_role: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_impact: Mapped[str | None] = mapped_column(Text, nullable=True)
    assumptions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    supporting_evidence_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    requires_human_approval: Mapped[bool] = mapped_column(Boolean, default=True)

    result: Mapped[RecommendationResult] = relationship(back_populates="actions")


class RecommendationCitation(Base):
    __tablename__ = "recommendation_citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("recommendation_results.id"), index=True)
    action_id: Mapped[int | None] = mapped_column(ForeignKey("recommended_actions.id"), nullable=True)
    evidence_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("evidence_chunks.id"), nullable=True)
    cited_source_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claim: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolves: Mapped[bool] = mapped_column(Boolean, default=False)
    temporally_valid: Mapped[bool] = mapped_column(Boolean, default=True)


class GroundingValidationResult(Base):
    __tablename__ = "grounding_validation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("recommendation_results.id"), index=True)
    claim: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(32))  # entailment|contradiction|neutral
    score: Mapped[float] = mapped_column(Float, default=0.0)
    method: Mapped[str] = mapped_column(String(64), default="nli")
    supported: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("evidence_chunks.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AbstentionRecord(Base):
    __tablename__ = "abstention_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("recommendation_results.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    reason_code: Mapped[str] = mapped_column(String(64), default="insufficient_evidence")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class HumanReviewRecord(Base, TimestampMixin):
    __tablename__ = "human_review_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("recommendation_results.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(128), default="local_user")
    decision: Mapped[str] = mapped_column(String(32))  # accept|edit|reject|request_evidence|insufficient
    edited_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


# ===========================================================================
# 6. Experiment management
# ===========================================================================
class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    research_question: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hypothesis: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="created")


class ExperimentConfiguration(Base, TimestampMixin):
    __tablename__ = "experiment_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    config: Mapped[dict] = mapped_column(JSON)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    benchmark_split: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    top_k: Mapped[int] = mapped_column(Integer, default=8)
    reranker_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    llm_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    grounding_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    abstention_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    citation_enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class ExperimentRun(Base, TimestampMixin):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    configuration_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiment_configurations.id"), nullable=True
    )
    baseline: Mapped[str] = mapped_column(String(64), default="pm_rag_full")
    status: Mapped[str] = mapped_column(String(32), default="created")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    num_scenarios: Mapped[int] = mapped_column(Integer, default=0)
    reproducibility: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ExperimentPrediction(Base):
    __tablename__ = "experiment_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("execution_scenarios.id"), index=True)
    result_id: Mapped[int | None] = mapped_column(ForeignKey("recommendation_results.id"), nullable=True)
    retrieved_chunk_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cited_chunk_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class RetrievalMetric(Base):
    __tablename__ = "retrieval_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    value: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)


class RecommendationMetric(Base):
    __tablename__ = "recommendation_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    note: Mapped[str | None] = mapped_column(String(255), nullable=True)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)


class ExperimentComparison(Base, TimestampMixin):
    __tablename__ = "experiment_comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    run_ids: Mapped[dict] = mapped_column(JSON)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
