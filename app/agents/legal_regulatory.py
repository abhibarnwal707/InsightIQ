"""Legal/regulatory section agent. Reuses the Phase 2 pattern from app/agents/common.py.

Two passage sources, combined into one extraction pool:
1. The company's own 10-K text (same document financials.py may have already
   fetched -- app.sources.edgar caches it in-process), keyword-filtered toward the
   Item 3 "Legal Proceedings" style disclosures.
2. CourtListener RECAP federal docket search by exact company-name phrase match.

CourtListener's free tier (5/min, 50/hr, 125/day) is tracked independently of the
OpenRouter budget in app.cache.store -- see app.sources.courtlistener.
"""
from __future__ import annotations

from datetime import datetime

from app.agents.common import chunk_text, extract_and_verify, select_relevant_chunks, split_verified_claims
from app.llm.client import OpenRouterClient
from app.llm.schemas import ReportSection, ResolvedEntity, SourcePassage
from app.scoring.confidence import ClaimScoringInput, days_since, score_section
from app.sources import courtlistener, edgar

_LEGAL_KEYWORDS = [
    "legal proceedings", "litigation", "lawsuit", "regulatory", "investigation",
    "settlement", "compliance", "class action", "sec investigation", "subpoena",
]


async def _filing_passages(entity: ResolvedEntity) -> list[SourcePassage]:
    if not entity.is_public or not entity.cik:
        return []
    try:
        meta = await edgar.get_latest_filing_meta(entity.cik, forms=("10-K", "10-K/A"))
    except Exception:  # noqa: BLE001 -- EDGAR downtime: degrade to no filing passages
        return []
    if not meta:
        return []
    try:
        text = await edgar.fetch_filing_text(entity.cik, meta["accession"], meta["primary_document"])
    except Exception:  # noqa: BLE001
        return []

    chunks = chunk_text(text, chunk_size=1400, overlap=150)
    relevant = select_relevant_chunks(chunks, _LEGAL_KEYWORDS, top_k=4)
    url = edgar.filing_document_url(entity.cik, meta["accession"], meta["primary_document"])
    published_at = datetime.fromisoformat(meta["filing_date"]) if meta.get("filing_date") else None
    return [
        SourcePassage(
            id=f"legalfil_{i + 1}",
            url=url,
            title=f"{meta['form']} legal/regulatory excerpt",
            text=chunk[:3000],
            source_tier="primary_filing",
            published_at=published_at,
        )
        for i, chunk in enumerate(relevant)
    ]


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


async def _docket_passages(company_name: str) -> tuple[list[SourcePassage], list[str]]:
    notes: list[str] = []
    try:
        dockets = await courtlistener.search_dockets(company_name, max_records=6)
    except courtlistener.CourtListenerBudgetExceededError as exc:
        return [], [str(exc)]
    except Exception as exc:  # noqa: BLE001
        return [], [f"CourtListener request failed: {exc}"]

    if not dockets:
        notes.append("No matching CourtListener federal court dockets found for this company name.")
        return [], notes

    passages = []
    for i, d in enumerate(dockets):
        case_name = d.get("caseName", "")
        court = d.get("court_citation_string") or d.get("court", "")
        date_filed = d.get("dateFiled")
        docket_number = d.get("docketNumber", "")
        docket_path = d.get("docket_absolute_url", "")
        url = f"https://www.courtlistener.com{docket_path}" if docket_path else "https://www.courtlistener.com"
        recap_docs = d.get("recap_documents") or []
        description = recap_docs[0].get("description", "") if recap_docs else ""

        text = f"Federal court docket: {case_name}. Court: {court}. Filed: {date_filed}. Docket number: {docket_number}."
        if description:
            text += f" Most relevant document: {description}"

        passages.append(
            SourcePassage(
                id=f"docket_{i + 1}",
                url=url,
                title=case_name,
                text=text,
                source_tier="regulatory",
                published_at=_parse_date(date_filed),
            )
        )
    return passages, notes


async def run_legal_regulatory_agent(entity: ResolvedEntity, client: OpenRouterClient) -> ReportSection:
    filing_passages = await _filing_passages(entity)
    docket_passages, docket_notes = await _docket_passages(entity.company_name)
    all_passages = filing_passages + docket_passages

    if not all_passages:
        data_gaps = list(docket_notes)
        if not entity.is_public:
            data_gaps.append(
                "Company is not SEC-registered; no 10-K Legal Proceedings disclosure is available."
            )
        elif not filing_passages:
            data_gaps.append("Could not retrieve or find a Legal Proceedings-relevant excerpt from the 10-K.")
        return ReportSection(
            section="legal_regulatory",
            summary=f"No legal or regulatory source material was retrieved for {entity.company_name}.",
            claims=[],
            confidence=0.0,
            confidence_rationale="No source passages retrieved; nothing to score.",
            data_gaps=data_gaps or ["No legal/regulatory sources available."],
            model_used="n/a",
        )

    instruction = (
        "Extract factual claims about legal proceedings, litigation, government "
        "investigations, or regulatory/compliance matters involving the company."
    )
    claims, summary, data_gaps, model_used = await extract_and_verify(
        client, all_passages, instruction, max_claims=8
    )
    verified, drop_notes = split_verified_claims(claims)

    passage_by_id = {p.id: p for p in all_passages}
    scoring_inputs = [
        ClaimScoringInput(
            entailment=c.entailment or "no",
            source_tier=passage_by_id[c.source_id].source_tier if c.source_id in passage_by_id else "web",
            recency_days=(
                days_since(passage_by_id[c.source_id].published_at)
                if passage_by_id.get(c.source_id) and passage_by_id[c.source_id].published_at
                else 365.0
            ),
            corroboration_count=1,
        )
        for c in verified
    ]
    section_score = score_section(scoring_inputs)

    data_gaps = data_gaps + drop_notes + docket_notes
    if not entity.is_public:
        data_gaps.append(
            "Company is not SEC-registered; legal/regulatory coverage relies on CourtListener "
            "docket search only, not a 10-K Legal Proceedings disclosure."
        )
    elif not filing_passages:
        data_gaps.append("Could not retrieve a Legal Proceedings-relevant excerpt from the 10-K.")

    return ReportSection(
        section="legal_regulatory",
        summary=summary,
        claims=verified,
        confidence=section_score.confidence,
        confidence_rationale=section_score.rationale,
        data_gaps=data_gaps,
        model_used=model_used,
    )
