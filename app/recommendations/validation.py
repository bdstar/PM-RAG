"""Multi-stage recommendation validation.

Combines (12.1) citation validation, (12.2) claim extraction, (12.3) NLI
grounding, (12.4) rule-based checks and (12.5) abstention into one report. Pure
computation; persistence happens in the service layer.
"""
from __future__ import annotations

import re
from typing import Optional

from sqlalchemy.orm import Session

from ..rag import abstention, citations, grounding

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def rule_based_checks(parsed: dict, evidence_items: list[dict], valid_citation_ids: set[str]) -> list[str]:
    """Deterministic validity checks (12.4)."""
    warnings: list[str] = []
    actions = parsed.get("recommended_actions", []) or []
    for i, action in enumerate(actions):
        if not action.get("action"):
            warnings.append(f"Action {i+1}: missing required 'action' field.")
        if not action.get("responsible_role"):
            warnings.append(f"Action {i+1}: missing responsible role.")
        # Citations must be among the retrieved/valid evidence set.
        for cid in action.get("supporting_evidence_ids", []) or []:
            if cid not in valid_citation_ids:
                warnings.append(f"Action {i+1}: cites unknown/unresolvable evidence '{cid}'.")
        # Numbers in rationale must appear in cited evidence text.
        cited_text = " ".join(
            it["text"] for it in evidence_items
            if it["chunk_key"] in (action.get("supporting_evidence_ids", []) or [])
        )
        cited_nums = {n.replace(",", "") for n in _NUM_RE.findall(cited_text)}
        for field in ("rationale", "action", "expected_impact"):
            for n in _NUM_RE.findall(str(action.get(field) or "")):
                if n.replace(",", "") not in cited_nums and len(n) > 2:
                    warnings.append(
                        f"Action {i+1}: numeric value '{n}' in {field} is not present "
                        f"in cited evidence (possible unsupported figure)."
                    )
    return warnings


def validate(
    session: Session,
    parsed: dict,
    evidence_items: list[dict],
    *,
    as_of_period: Optional[str] = None,
    citation_enabled: bool = True,
    grounding_enabled: bool = True,
    abstention_enabled: bool = True,
) -> dict:
    retrieved_keys = {it["chunk_key"] for it in evidence_items}

    # Gather all cited ids from all actions.
    cited_ids: list[str] = []
    for action in parsed.get("recommended_actions", []) or []:
        cited_ids.extend(action.get("supporting_evidence_ids", []) or [])
    cited_ids = list(dict.fromkeys(cited_ids))  # dedup, preserve order

    # 12.1 Citation validation.
    checks = citations.validate_citations(session, cited_ids, retrieved_keys, as_of_period)
    cite_summary = citations.citation_summary(checks)
    valid_citation_ids = {c.cited_id for c in checks if c.valid}

    # 12.2 + 12.3 Claim extraction and NLI grounding.
    checker = grounding.get_grounding_checker()
    grounding_results = []
    supported = contradictions = total_claims = 0
    evidence_by_key = {it["chunk_key"]: it["text"] for it in evidence_items}
    all_evidence_text = " ".join(evidence_by_key.values())

    def _check(claim: str, evidence_text: str):
        nonlocal supported, contradictions, total_claims
        res = checker.check(claim, evidence_text)
        total_claims += 1
        if res.supported:
            supported += 1
        if res.label == "contradiction":
            contradictions += 1
        grounding_results.append({
            "claim": res.claim, "label": res.label, "score": res.score,
            "supported": res.supported, "method": res.method,
        })

    if grounding_enabled:
        # Factual claims at the result level, grounded against all retrieved evidence.
        for field in ("problem_summary", "evidence_summary"):
            for claim in grounding.extract_claims(str(parsed.get(field) or "")):
                _check(claim, all_evidence_text)
        # Factual claims in each action's rationale, grounded against its citations.
        for action in parsed.get("recommended_actions", []) or []:
            cited = action.get("supporting_evidence_ids", []) or []
            evidence_text = " ".join(evidence_by_key.get(c, "") for c in cited) or all_evidence_text
            for claim in grounding.extract_claims(str(action.get("rationale") or "")):
                _check(claim, evidence_text)

    # 12.4 Rule-based checks.
    rule_warnings = rule_based_checks(parsed, evidence_items, retrieved_keys)

    # 12.5 Abstention decision.
    decision = abstention.decide(
        num_evidence=len(evidence_items),
        valid_citations=len(valid_citation_ids),
        supported_claims=supported,
        total_claims=total_claims,
        contradictions=contradictions,
        citation_enabled=citation_enabled,
        grounding_enabled=grounding_enabled,
    ) if abstention_enabled else abstention.AbstentionDecision(False)

    unsupported = total_claims - supported
    return {
        "citations": cite_summary,
        "citation_checks": checks,
        "valid_citation_ids": valid_citation_ids,
        "grounding": {
            "results": grounding_results,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "contradictions": contradictions,
            "total_claims": total_claims,
            "supported_rate": round(supported / total_claims, 3) if total_claims else None,
            "unsupported_rate": round(unsupported / total_claims, 3) if total_claims else None,
            "backend": checker.backend,
        },
        "rule_warnings": rule_warnings,
        "abstention": {
            "abstain": decision.abstain,
            "reason": decision.reason,
            "reason_code": decision.reason_code,
        },
    }
