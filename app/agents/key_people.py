"""Key-people section agent. Reuses the Phase 2 pattern.

Source: the company's latest DEF 14A (annual proxy statement), which is where SEC
registrants disclose directors, executive officers, and their compensation. Public
companies only -- no general web-search fallback in this build (see competitors.py's
docstring for why), so a private company is an honest, near-empty section rather than
a guessed one.
"""
from __future__ import annotations

from datetime import datetime

from app.agents.common import chunk_text, extract_and_verify, select_relevant_chunks, split_verified_claims
from app.llm.client import OpenRouterClient
from app.llm.schemas import ReportSection, ResolvedEntity, SourcePassage
from app.scoring.confidence import ClaimScoringInput, days_since, score_section
from app.sources import edgar

_KEY_PEOPLE_KEYWORDS = [
    "director", "executive officer", "chief executive", "chief financial",
    "board of directors", "age", "biography", "compensation", "president",
    "founder", "chairman",
]


async def run_key_people_agent(entity: ResolvedEntity, client: OpenRouterClient) -> ReportSection:
    if not entity.is_public or not entity.cik:
        return ReportSection(
            section="key_people",
            summary=(
                f"No key-people information could be sourced for {entity.company_name}: it is "
                "not SEC-registered, so no DEF 14A proxy statement exists, and this build has "
                "no general web-search source."
            ),
            claims=[],
            confidence=0.0,
            confidence_rationale="No source passages available.",
            data_gaps=[
                "Company is not SEC-registered; no DEF 14A proxy statement available.",
                "This build has no general web-search integration for private-company leadership research.",
            ],
            model_used="n/a",
        )

    try:
        meta = await edgar.get_latest_filing_meta(entity.cik, forms=("DEF 14A",))
    except Exception as exc:  # noqa: BLE001
        meta = None
        meta_error = str(exc)
    else:
        meta_error = None
    if not meta:
        gap = (
            f"EDGAR request failed while looking up a DEF 14A: {meta_error}"
            if meta_error
            else "Could not locate a DEF 14A proxy statement (may not have filed one recently, "
            "or filings.recent window doesn't reach far enough back)."
        )
        return ReportSection(
            section="key_people",
            summary=f"No DEF 14A proxy statement was found for {entity.company_name}.",
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
            section="key_people",
            summary=f"Could not fetch the DEF 14A text for {entity.company_name}.",
            claims=[],
            confidence=0.0,
            confidence_rationale="No source passages available.",
            data_gaps=[f"Failed to fetch DEF 14A text: {exc}"],
            model_used="n/a",
        )

    chunks = chunk_text(text, chunk_size=1400, overlap=150)
    relevant = select_relevant_chunks(chunks, _KEY_PEOPLE_KEYWORDS, top_k=6)
    url = edgar.filing_document_url(entity.cik, meta["accession"], meta["primary_document"])
    published_at = datetime.fromisoformat(meta["filing_date"]) if meta.get("filing_date") else None

    passages = [
        SourcePassage(
            id=f"proxy_{i + 1}",
            url=url,
            title=f"{meta['form']} excerpt",
            text=chunk[:3000],
            source_tier="primary_filing",
            published_at=published_at,
        )
        for i, chunk in enumerate(relevant)
    ]

    instruction = (
        "Extract factual claims naming the company's directors, executive officers, or other "
        "key people -- their names, titles/roles, ages, tenure, or compensation figures, as "
        "directly stated in the passages."
    )
    claims, summary, data_gaps, model_used = await extract_and_verify(
        client, passages, instruction, max_claims=10
    )
    verified, drop_notes = split_verified_claims(claims)

    scoring_inputs = [
        ClaimScoringInput(
            entailment=c.entailment or "no",
            source_tier="primary_filing",
            recency_days=days_since(published_at) if published_at else 365.0,
            corroboration_count=1,
        )
        for c in verified
    ]
    section_score = score_section(scoring_inputs)

    data_gaps = data_gaps + drop_notes

    return ReportSection(
        section="key_people",
        summary=summary,
        claims=verified,
        confidence=section_score.confidence,
        confidence_rationale=section_score.rationale,
        data_gaps=data_gaps,
        model_used=model_used,
    )
