"""Evidence-aware abstention policy.

Decides whether the system should abstain (return 'evidence insufficient')
rather than emit a recommendation. Triggers:
  * too few retrieved evidence items,
  * no citation resolves to a real, in-set, temporally-valid source,
  * grounding support below threshold,
  * conflicting/contradictory grounding signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.config import get_settings


@dataclass
class AbstentionDecision:
    abstain: bool
    reason: Optional[str] = None
    reason_code: Optional[str] = None


def decide(
    *,
    num_evidence: int,
    valid_citations: int,
    supported_claims: int,
    total_claims: int,
    contradictions: int,
    citation_enabled: bool = True,
    grounding_enabled: bool = True,
) -> AbstentionDecision:
    settings = get_settings()

    if num_evidence < settings.min_evidence_for_answer:
        return AbstentionDecision(
            True,
            "No relevant evidence was retrieved for this scenario at the decision date.",
            "no_evidence",
        )

    if citation_enabled and valid_citations == 0:
        return AbstentionDecision(
            True,
            "No citation resolved to a valid, temporally-available source record.",
            "no_valid_citation",
        )

    if grounding_enabled and contradictions > 0 and contradictions >= supported_claims:
        return AbstentionDecision(
            True,
            "Generated claims conflict with the retrieved evidence.",
            "contradictory_evidence",
        )

    if grounding_enabled and total_claims > 0:
        support_rate = supported_claims / total_claims
        if support_rate < settings.abstention_min_support_rate:
            return AbstentionDecision(
                True,
                (
                    f"Only {supported_claims}/{total_claims} claims are supported by "
                    f"evidence (below the {settings.abstention_min_support_rate:.0%} threshold)."
                ),
                "insufficient_grounding",
            )

    return AbstentionDecision(False)
