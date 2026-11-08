"""Unit tests for claim extraction and lexical grounding fallback."""
from __future__ import annotations

from app.rag import grounding as G


def test_is_factual_claim_filters_imperatives():
    assert G.is_factual_claim("The project is 45 days behind schedule.") is True
    assert G.is_factual_claim("Review the affected work package and re-baseline.") is False
    assert G.is_factual_claim("Escalate to the procurement lead.") is False
    assert G.is_factual_claim("Too short") is False


def test_extract_claims_keeps_only_factual_by_default():
    text = (
        "The budget variance is $5,000 for this project. "
        "Conduct a budget review with the finance team."
    )
    claims = G.extract_claims(text)
    assert any("budget variance" in c.lower() for c in claims)
    assert not any(c.lower().startswith("conduct") for c in claims)


def test_lexical_grounding_supports_covered_claim():
    checker = G.GroundingChecker()
    # Force lexical fallback regardless of NLI availability.
    checker._backend = "lexical-fallback"
    evidence = "Reporting period 202305. Budget variance is 5000. Managing agency DEP."
    supported = checker.check("budget variance is 5000", evidence)
    assert supported.supported is True
    # A claim with an unsupported number must not be marked supported.
    unsupported = checker.check("budget variance is 999999", evidence)
    assert unsupported.supported is False
