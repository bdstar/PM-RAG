"""Evaluation metric implementations.

Covers retrieval quality, evidence attribution, factual consistency, abstention,
and operational efficiency (spec 14). Metrics requiring expert labels are
computed against system-derived relevance (the records that generated a
scenario), which is clearly labelled as NOT expert-validated by the caller.
Metrics that genuinely require unavailable data return ``None`` rather than a
fabricated value.
"""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Optional, Sequence


# --------------------------------------------------------------------------
# Retrieval quality
# --------------------------------------------------------------------------
def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> Optional[float]:
    if not gold:
        return None
    topk = set(retrieved[:k])
    return len(topk & gold) / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> Optional[float]:
    if not gold or k == 0:
        return None
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return len(set(topk) & gold) / len(topk)


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> Optional[float]:
    if not gold:
        return None
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def mrr(retrieved: Sequence[str], gold: set[str]) -> Optional[float]:
    if not gold:
        return None
    for i, cid in enumerate(retrieved):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


# --------------------------------------------------------------------------
# Evidence attribution
# --------------------------------------------------------------------------
def attribution(cited_ids: Sequence[str], valid_cited_ids: set[str], gold: set[str]) -> dict:
    """Citation precision/recall/F1.

    A citation is correct if it is valid (resolves + temporally valid + in-set)
    AND the source is relevant to the scenario (in gold).
    """
    cited = list(dict.fromkeys(cited_ids))
    correct = [c for c in cited if c in valid_cited_ids and c in gold]
    precision = len(correct) / len(cited) if cited else None
    recall = len(set(correct)) / len(gold) if gold else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------
# Factual consistency (aggregate over grounding results)
# --------------------------------------------------------------------------
def factual_consistency(grounding_rows: list[dict]) -> dict:
    total = len(grounding_rows)
    if total == 0:
        return {"supported_rate": None, "unsupported_rate": None, "contradiction_rate": None, "n_claims": 0}
    supported = sum(1 for r in grounding_rows if r.get("supported"))
    contradiction = sum(1 for r in grounding_rows if r.get("label") == "contradiction")
    return {
        "supported_rate": supported / total,
        "unsupported_rate": (total - supported) / total,
        "contradiction_rate": contradiction / total,
        "n_claims": total,
    }


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------
def abstention_metrics(records: list[dict]) -> dict:
    """records: list of {abstained: bool, should_abstain: Optional[bool]}."""
    labelled = [r for r in records if r.get("should_abstain") is not None]
    if not labelled:
        return {"precision": None, "recall": None,
                "evidence_insufficient_detection_rate": None,
                "inappropriate_recommendation_rate": None, "n": 0}
    tp = sum(1 for r in labelled if r["abstained"] and r["should_abstain"])
    fp = sum(1 for r in labelled if r["abstained"] and not r["should_abstain"])
    fn = sum(1 for r in labelled if not r["abstained"] and r["should_abstain"])
    should = [r for r in labelled if r["should_abstain"]]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    detect = tp / len(should) if should else None
    inappropriate = fn / len(should) if should else None
    return {
        "precision": precision, "recall": recall,
        "evidence_insufficient_detection_rate": detect,
        "inappropriate_recommendation_rate": inappropriate,
        "n": len(labelled),
    }


# --------------------------------------------------------------------------
# Efficiency
# --------------------------------------------------------------------------
def efficiency(latencies: list[dict]) -> dict:
    if not latencies:
        return {}
    retr = [l["retrieval_ms"] for l in latencies]
    gen = [l["generation_ms"] for l in latencies]
    tot = [l["total_ms"] for l in latencies]
    total_time_s = sum(tot) / 1000.0
    return {
        "mean_retrieval_ms": mean(retr),
        "mean_generation_ms": mean(gen),
        "mean_total_ms": mean(tot),
        "throughput_per_min": (len(tot) / total_time_s * 60) if total_time_s > 0 else None,
    }


# --------------------------------------------------------------------------
# Confidence intervals (percentile bootstrap)
# --------------------------------------------------------------------------
def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    rng = random.Random(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return mean(vals), lo, hi


def mean_or_none(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None
