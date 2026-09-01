"""Competitors section agent. Reuses the Phase 2 pattern. Lowest-confidence section
by design (design principle #7): no free structured competitor-data API exists, so
this is a single primary-filing source (10-K "Competition" narrative) synthesized by
an LLM, with an explicit confidence ceiling rather than letting a single qualitative
source score as if it were as reliable as a filed financial figure.

No general web-search fallback is implemented -- there's no reliable free/no-key web
search API that fits this build's constraints, and a scraped fallback would be a
fragile, likely-ToS-violating source anyway. For a private company (no 10-K at all)
this section is honestly near-empty rather than guessed.
"""
from __future__ import annotations

from datetime import datetime

from app.agents.common import chunk_text, extract_and_verify, select_relevant_chunks, split_verified_claims
from app.llm.client import OpenRouterClient
from app.llm.schemas import ReportSection, ResolvedEntity, SourcePassage
from app.scoring.confidence import ClaimScoringInput, days_since, score_section
from app.sources import edgar

_COMPETITOR_KEYWORDS = [
    "competition", "competitors", "compete", "competitive", "market share",
    "industry", "rival",
]

# Single-source, qualitative-narrative sourcing caps out well below a filed number's
# ceiling even when entailment and recency are both strong -- see module docstring.
CONFIDENCE_CEILING = 0.55


async def run_competitors_agent(entity: ResolvedEntity, client: OpenRouterClient) -> ReportSection:
    if not entity.is_public or not entity.cik:
        return ReportSection(
            section="competitors",
            summary=(
                f"No competitor information could be sourced for {entity.company_name}: it is "
                "not SEC-registered, so no 10-K 'Competition' disclosure exists, and this build "
                "has no general web-search source."
            ),
            claims=[],
            confidence=0.0,
            confidence_rationale="No source passages available.",
            data_gaps=[
                "Company is not SEC-registered; no 10-K Competition section available.",
                "This build has no general web-search integration for private-company competitor research.",
            ],
            model_used="n/a",
        )

    try:
        meta = await edgar.get_latest_filing_meta(entity.cik, forms=("10-K", "10-K/A"))
    except Exception as exc:  # noqa: BLE001
        meta = None
        meta_error = str(exc)
    else:
        meta_error = None
    if not meta:
        gap = (
            f"EDGAR request failed while looking up a 10-K filing: {meta_error}"
            if meta_error
            else "Could not locate a 10-K filing to search for a Competition disclosure."
        )
        return ReportSection(
            section="competitors",
            summary=f"No 10-K filing was found for {entity.company_name}.",
            claims=[],
            confidence=0.0,
            confidence_rationale="No source passages available.",
            data_gaps=[gap],
            model_used="n/a",
        )

    try:
        text = await edgar.fetch_filing_text(entity.cik, meta["accession"], meta["primary_document"])
    except Exception as exc:  # noqa: BLE001
        return ReportSection(
            section="competitors",
            summary=f"Could not fetch the 10-K text for {entity.company_name}.",
            claims=[],
            confidence=0.0,
            confidence_rationale="No source passages available.",
            data_gaps=[f"Failed to fetch 10-K text: {exc}"],
            model_used="n/a",
        )

    chunks = chunk_text(text, chunk_size=1400, overlap=150)
    relevant = select_relevant_chunks(chunks, _COMPETITOR_KEYWORDS, top_k=4)
    if not relevant:
        # No competition-related passage exists in this filing. Say so, rather than
        # extracting from whatever text happened to be first -- see select_relevant_chunks.
        return ReportSection(
            section="competitors",
            summary=(
                f"The latest {meta['form']} for {entity.company_name} contains no passage "
                "discussing competition, competitors, or market position."
            ),
            claims=[],
            confidence=0.0,
            confidence_rationale="No competition-related passages found; nothing to ground claims in.",
            data_gaps=[
                f"No text matching competition keywords was found in the {meta['form']} "
                f"filed {meta.get('filing_date', 'unknown date')} (accession {meta['accession']}).",
                "This build has no general web-search integration to corroborate competitor "
                "research outside SEC filings.",
            ],
            model_used="n/a",
        )
    url = edgar.filing_document_url(entity.cik, meta["accession"], meta["primary_document"])
    published_at = datetime.fromisoformat(meta["filing_date"]) if meta.get("filing_date") else None

    passages = [
        SourcePassage(
            id=f"comp_{i + 1}",
            url=url,
            title=f"{meta['form']} competition excerpt",
            text=chunk[:3000],
            source_tier="primary_filing",
            published_at=published_at,
        )
        for i, chunk in enumerate(relevant)
    ]

    instruction = (
        "Extract factual claims about the company's competitors, competitive landscape, "
        "market position, or basis of competition (e.g. named rival companies, how the "
        "company says it competes, market share statements)."
    )
    claims, summary, data_gaps, model_used = await extract_and_verify(
        client, passages, instruction, max_claims=6
    )
    verified, drop_notes = split_verified_claims(claims)

    scoring_inputs = [
        ClaimScoringInput(
            entailment=c.entailment or "no",
            source_tier="primary_filing",
            recency_days=days_since(published_at) if published_at else 365.0,
            corroboration_count=1,  # always a single filing -- never inflate this
        )
        for c in verified
    ]
    section_score = score_section(scoring_inputs)
    capped_confidence = min(section_score.confidence, CONFIDENCE_CEILING)

    data_gaps = data_gaps + drop_notes
    data_gaps.append(
        "Competitor data is sourced from a single 10-K narrative section only -- no "
        "independent corroborating source exists in this build, so confidence is capped."
    )

    rationale = section_score.rationale
    if capped_confidence < section_score.confidence:
        rationale += f" Capped from {section_score.confidence} to {capped_confidence} (single-source qualitative narrative)."

    return ReportSection(
        section="competitors",
        summary=summary,
        claims=verified,
        confidence=capped_confidence,
        confidence_rationale=rationale,
        data_gaps=data_gaps,
        model_used=model_used,
    )
