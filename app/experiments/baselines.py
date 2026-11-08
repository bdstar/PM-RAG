"""Baseline and PM-RAG configurations for the experiment matrix (spec 13.1)."""
from __future__ import annotations

# Each baseline maps to a configuration dict consumed by the runner. Keys mirror
# the retrieval/generation toggles so each component can be ablated.
BASELINES: dict[str, dict] = {
    "no_retrieval_llm": {
        "label": "No-retrieval LLM",
        "enabled_retrievers": [],
        "retrieval_mode": "hybrid_temporal",
        "reranker_enabled": False,
        "citation_enabled": False,
        "grounding_enabled": True,
        "abstention_enabled": False,
        "description": "LLM/SLM with no retrieved evidence (expected: more unsupported claims).",
    },
    "keyword_template": {
        "label": "Keyword search + template",
        "enabled_retrievers": ["bm25"],
        "retrieval_mode": "hybrid_temporal",
        "reranker_enabled": False,
        "citation_enabled": True,
        "grounding_enabled": True,
        "abstention_enabled": True,
        "force_rule_based": True,
        "description": "Conventional keyword retrieval with a fixed recommendation template.",
    },
    "bm25_only": {
        "label": "BM25-only RAG",
        "enabled_retrievers": ["bm25"],
        "retrieval_mode": "hybrid_temporal",
        "reranker_enabled": False,
        "citation_enabled": True,
        "grounding_enabled": True,
        "abstention_enabled": True,
        "description": "Sparse keyword retrieval only.",
    },
    "dense_only": {
        "label": "Dense-only RAG",
        "enabled_retrievers": ["dense"],
        "retrieval_mode": "hybrid_temporal",
        "reranker_enabled": False,
        "citation_enabled": True,
        "grounding_enabled": True,
        "abstention_enabled": True,
        "description": "Dense semantic retrieval only.",
    },
    "hybrid_no_rerank": {
        "label": "Hybrid RAG (no rerank)",
        "enabled_retrievers": ["bm25", "dense"],
        "retrieval_mode": "hybrid_temporal",
        "reranker_enabled": False,
        "citation_enabled": True,
        "grounding_enabled": True,
        "abstention_enabled": True,
        "description": "BM25 + dense fused with RRF, no reranking.",
    },
    "pm_rag_full": {
        "label": "PM-RAG (hybrid + rerank)",
        "enabled_retrievers": ["bm25", "dense"],
        "retrieval_mode": "hybrid_temporal",
        "reranker_enabled": True,
        "citation_enabled": True,
        "grounding_enabled": True,
        "abstention_enabled": True,
        "description": "Full PM-RAG: hybrid retrieval + cross-encoder reranking + grounding + abstention.",
    },
    "nearest_historical_case": {
        "label": "Nearest historical case",
        "enabled_retrievers": ["dense"],
        "retrieval_mode": "same_project_historical",
        "reranker_enabled": True,
        "citation_enabled": True,
        "grounding_enabled": True,
        "abstention_enabled": True,
        "top_k": 3,
        "description": "Retrieve nearest same-project historical snapshot(s).",
    },
    "rule_based_alerts": {
        "label": "Rule-based project-control alerts",
        "enabled_retrievers": ["bm25", "dense"],
        "retrieval_mode": "current_project",
        "reranker_enabled": False,
        "citation_enabled": True,
        "grounding_enabled": False,
        "abstention_enabled": True,
        "force_rule_based": True,
        "description": "Threshold alerts over derived signals with templated actions.",
    },
}

# Ablation axes (spec 13.2) expressed as config overrides applied on top of a base.
ABLATION_AXES = {
    "retrieval_method": {
        "bm25": {"enabled_retrievers": ["bm25"], "reranker_enabled": False},
        "dense": {"enabled_retrievers": ["dense"], "reranker_enabled": False},
        "hybrid": {"enabled_retrievers": ["bm25", "dense"], "reranker_enabled": False},
        "hybrid_rerank": {"enabled_retrievers": ["bm25", "dense"], "reranker_enabled": True},
    },
    "retrieval_scope": {
        "current": {"retrieval_mode": "current_project"},
        "same_historical": {"retrieval_mode": "same_project_historical"},
        "cross": {"retrieval_mode": "cross_project"},
        "hybrid_temporal": {"retrieval_mode": "hybrid_temporal"},
    },
    "citation": {"on": {"citation_enabled": True}, "off": {"citation_enabled": False}},
    "grounding": {"on": {"grounding_enabled": True}, "off": {"grounding_enabled": False}},
    "abstention": {"on": {"abstention_enabled": True}, "off": {"abstention_enabled": False}},
}


def get_baseline(name: str) -> dict:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline '{name}'. Options: {list(BASELINES)}")
    return dict(BASELINES[name])


def list_baselines() -> list[dict]:
    return [{"name": k, **{kk: v[kk] for kk in ("label", "description")}} for k, v in BASELINES.items()]
