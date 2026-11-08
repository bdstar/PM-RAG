"""Pydantic schemas for the API and the anti-hallucination guardrail.

The ``GroundedAnswer`` model is the heart of the guardrail: the LLM is asked
to return JSON matching this schema, and Pydantic validates/normalises it.
If the model cannot ground the answer in the retrieved context, the schema is
forced into a safe "I don't know" refusal.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

REFUSAL_TEXT = "I don't know. Information not provided in the AquaTech knowledge base."


class AnswerStatus(str, Enum):
    """Whether the answer is grounded in the retrieved context."""

    ANSWERED = "answered"
    NOT_FOUND = "not_found"


# ---------------------------------------------------------------------------
# Guardrail model: what we expect the LLM to emit (as JSON)
# ---------------------------------------------------------------------------
class GroundedAnswer(BaseModel):
    """Structured, validated LLM output enforcing grounded answers."""

    status: AnswerStatus = Field(
        description="'answered' only if the answer is fully supported by the context, "
        "otherwise 'not_found'."
    )
    answer: str = Field(description="The answer text, or a refusal if not found.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence that the answer is grounded in the context.",
    )
    supporting_quotes: List[str] = Field(
        default_factory=list,
        description="Verbatim snippets from the context that support the answer.",
    )

    @field_validator("answer")
    @classmethod
    def _strip_answer(cls, v: str) -> str:
        return v.strip()

    def as_refusal(self) -> "GroundedAnswer":
        """Return a safe 'I don't know' version of this answer."""
        return GroundedAnswer(
            status=AnswerStatus.NOT_FOUND,
            answer=REFUSAL_TEXT,
            confidence=self.confidence,
            supporting_quotes=[],
        )

    def enforce(self) -> "GroundedAnswer":
        """Apply the guardrail: if the model judged the answer ungrounded, refuse.

        A separate context-faithfulness check (see ``guardrails.verify_faithful``)
        catches the opposite failure - a model that claims it is grounded but
        actually hallucinated content not present in the context.
        """
        if self.status == AnswerStatus.NOT_FOUND or not self.answer:
            return self.as_refusal()
        return self


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="The user's question.")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override the number of chunks to retrieve."
    )


class SourceChunk(BaseModel):
    rank: int
    score: float
    text: str
    source: str


class QueryResponse(BaseModel):
    query: str
    status: AnswerStatus
    answer: str
    confidence: float
    grounded: bool
    sources: List[SourceChunk]
    guardrail_triggered: bool = Field(
        description="True when a guardrail forced an 'I don't know' response."
    )
    backend: str
    latency_ms: int


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
    collection: str
    embed_model: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    chunks_indexed: int
    embed_model: str
    llm_model: str
    llm_backend: str
