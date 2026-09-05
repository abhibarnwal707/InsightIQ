"""Financials section agent -- the Phase 2 pattern-proving agent.

Two distinct grounding paths, matching design principle #4 exactly:

1. Headline numbers (revenue, net income, assets, ...) are parsed directly from SEC
   XBRL company-facts data IN CODE. No LLM ever sees or regenerates these values, so
   they need no entailment check -- there's no model output to verify against a
   passage, since no model produced them. entailment is fixed to "yes" because the
   citation is code-constructed to describe the exact fact it points at.
2. Narrative claims (trends, drivers, commentary) come from the latest 10-K's text
   via the shared extract_and_verify pattern in agents/common.py: schema-constrained
   extraction, then a separate entailment check per claim, same as every other
   section agent from here on.
"""
from __future__ import annotations

from datetime import date, datetime

from app.agents.common import (
    chunk_text,
    extract_and_verify,
    select_relevant_chunks,
    split_verified_claims,
)
from app.llm.client import OpenRouterClient
from app.llm.schemas import ExtractedClaim, ReportSection, ResolvedEntity, SourcePassage
from app.scoring.confidence import ClaimScoringInput, days_since, score_section
from app.sources import edgar

# (primary XBRL concept, friendly label, is_instant (balance-sheet point-in-time vs.
#  income-statement duration), fallback concept names to try if the primary is absent)
_XBRL_CONCEPTS: list[tuple[str, str, bool, list[str]]] = [
    ("Revenues", "revenue", False, ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]),
    ("NetIncomeLoss", "net income", False, []),
    ("OperatingIncomeLoss", "operating income", False, []),
    ("Assets", "total assets", True, []),
    ("Liabilities", "total liabilities", True, []),
    ("StockholdersEquity", "stockholders' equity", True, ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    ("CashAndCashEquivalentsAtCarryingValue", "cash and cash equivalents", True, []),
]

_NARRATIVE_KEYWORDS = [
    "revenue", "net income", "results of operations", "financial condition",
    "liquidity", "cash flow", "gross margin", "operating income", "net sales",
    "year over year", "compared to",
]


def _format_usd(val: float) -> str:
    sign = "-" if val < 0 else ""
    val = abs(val)
    if val >= 1_000_000_000:
        return f"{sign}${val / 1_000_000_000:.2f} billion"
    if val >= 1_000_000:
        return f"{sign}${val / 1_000_000:.2f} million"
    return f"{sign}${val:,.0f}"


def _pick_latest_annual(entries: list[dict], instant: bool) -> dict | None:
    candidates = [e for e in entries if str(e.get("form", "")).startswith("10-K") and e.get("fp") == "FY"]
    if not instant:
        filtered = []
        for e in candidates:
            try:
                start = date.fromisoformat(e["start"])
                end = date.fromisoformat(e["end"])
            except (KeyError, ValueError):
                continue
            if 300 <= (end - start).days <= 400:
                filtered.append(e)
        candidates = filtered
    if not candidates:
        return None
    candidates.sort(key=lambda e: (e["end"], e.get("filed", "")), reverse=True)
    return candidates[0]


async def _build_xbrl_claims(
    cik: str, company_name: str
) -> tuple[list[ExtractedClaim], list[SourcePassage], list[ClaimScoringInput], list[str]]:
    try:
        facts = await edgar.get_company_facts(cik)
    except Exception as exc:  # noqa: BLE001
        return [], [], [], [f"Could not retrieve XBRL company facts from EDGAR: {exc}"]

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    claims: list[ExtractedClaim] = []
    passages: list[SourcePassage] = []
    scoring: list[ClaimScoringInput] = []
    missing: list[str] = []

    for i, (concept, label, instant, fallbacks) in enumerate(_XBRL_CONCEPTS):
        entry = us_gaap.get(concept)
        used_concept = concept
        if not entry:
            for fb in fallbacks:
                if us_gaap.get(fb):
                    entry, used_concept = us_gaap[fb], fb
                    break
        if not entry:
            missing.append(f"{label}: no XBRL concept found")
            continue

        usd_entries = entry.get("units", {}).get("USD", [])
        picked = _pick_latest_annual(usd_entries, instant)
        if not picked:
            missing.append(f"{label}: no annual 10-K value found")
            continue

        src_id = f"xbrl_{i + 1}"
        period_desc = picked["end"] if instant else f"{picked['start']} to {picked['end']}"
        filed_date = picked.get("filed")
        url = edgar.filing_index_url(cik, picked["accn"])
        text = (
            f"{company_name}: {label} ({used_concept}) = {_format_usd(picked['val'])} "
            f"({picked['val']:,} USD) for period {period_desc}, per {picked.get('form')} "
            f"filed {filed_date} (accession {picked['accn']})."
        )
        published_at = datetime.fromisoformat(filed_date) if filed_date else None
        passages.append(
            SourcePassage(
                id=src_id,
                url=url,
                title=f"{picked.get('form')} XBRL: {label}",
                text=text,
                source_tier="primary_filing",
                published_at=published_at,
            )
        )
        # entailment is "yes" by construction: code parsed this value and built this
        # citation from it, so there is no LLM step whose output needs verifying.
        claims.append(ExtractedClaim(text=text, source_id=src_id, source_url=url, entailment="yes"))
        scoring.append(
            ClaimScoringInput(
                entailment="yes",
                source_tier="primary_filing",
                recency_days=days_since(published_at) if published_at else 3650,
                corroboration_count=1,
            )
        )

    return claims, passages, scoring, missing


async def _build_narrative_passages(cik: str) -> tuple[list[SourcePassage], dict | None]:
    try:
        meta = await edgar.get_latest_filing_meta(cik, forms=("10-K", "10-K/A"))
    except Exception:  # noqa: BLE001 -- EDGAR downtime: degrade to no narrative passages
        return [], None
    if not meta:
        return [], None
    try:
        text = await edgar.fetch_filing_text(cik, meta["accession"], meta["primary_document"])
    except Exception:  # noqa: BLE001
        return [], meta

    chunks = chunk_text(text, chunk_size=1400, overlap=150)
    relevant = select_relevant_chunks(chunks, _NARRATIVE_KEYWORDS, top_k=6)
    url = edgar.filing_document_url(cik, meta["accession"], meta["primary_document"])
    published_at = datetime.fromisoformat(meta["filing_date"]) if meta.get("filing_date") else None

    passages = [
        SourcePassage(
            id=f"narr_{i + 1}",
            url=url,
            title=f"{meta['form']} narrative excerpt",
            text=chunk[:3000],
            source_tier="primary_filing",
            published_at=published_at,
        )
        for i, chunk in enumerate(relevant)
    ]
    return passages, meta


async def run_financials_agent(entity: ResolvedEntity, client: OpenRouterClient) -> ReportSection:
    if not entity.is_public or not entity.cik:
        return ReportSection(
            section="financials",
            summary=(
                f"{entity.company_name} does not appear to be a U.S. SEC-registered public "
                "company, so no structured financial data is available from EDGAR."
            ),
            claims=[],
            confidence=0.0,
            confidence_rationale="No SEC filings exist for a private company; nothing to ground financials in.",
            data_gaps=[
                "Company is not SEC-registered (private company, or not a U.S. filer); "
                "financial statements are unavailable from EDGAR."
            ],
            model_used="n/a",
        )

    xbrl_claims, _xbrl_passages, xbrl_scoring, missing_concepts = await _build_xbrl_claims(
        entity.cik, entity.company_name
    )
    narrative_passages, filing_meta = await _build_narrative_passages(entity.cik)
    filing_published_at = (
        datetime.fromisoformat(filing_meta["filing_date"])
        if filing_meta and filing_meta.get("filing_date")
        else None
    )

    narrative_claims: list[ExtractedClaim] = []
    narrative_summary = ""
    narrative_data_gaps: list[str] = []
    model_used = "n/a (deterministic XBRL only -- no narrative filing text retrieved)"

    if narrative_passages:
        instruction = (
            "Extract factual claims about the company's financial performance, results of "
            "operations, financial condition, liquidity, or financial trends/outlook. Do not "
            "restate a bare headline total (revenue, net income, total assets) unless the "
            "passage adds context beyond the number itself (e.g. year-over-year change, a "
            "stated driver of the change, segment-level detail)."
        )
        narrative_claims, narrative_summary, narrative_data_gaps, model_used = await extract_and_verify(
            client, narrative_passages, instruction
        )

    verified_narrative, drop_notes = split_verified_claims(narrative_claims)
    narrative_scoring = [
        ClaimScoringInput(
            entailment=c.entailment or "no",
            source_tier="primary_filing",
            recency_days=days_since(filing_published_at) if filing_published_at else 365.0,
            corroboration_count=1,
        )
        for c in verified_narrative
    ]

    all_claims = xbrl_claims + verified_narrative
    section_score = score_section(xbrl_scoring + narrative_scoring)

    data_gaps = list(narrative_data_gaps) + drop_notes
    if missing_concepts:
        data_gaps.append("XBRL concepts unavailable: " + "; ".join(missing_concepts))
    if not narrative_passages:
        data_gaps.append(
            f"No narrative passage matching financial-discussion keywords was found in the "
            f"{filing_meta['form']} filed {filing_meta.get('filing_date', 'unknown date')}; "
            "financial claims are limited to structured XBRL figures."
            if filing_meta
            else "No 10-K narrative text could be retrieved; financial claims are limited to "
            "structured XBRL figures."
        )

    headline = (
        "; ".join(c.text for c in xbrl_claims[:4])
        if xbrl_claims
        else f"{entity.company_name}: no structured financial figures were available from XBRL."
    )
    summary = f"{headline} {narrative_summary}".strip()

    return ReportSection(
        section="financials",
        summary=summary,
        claims=all_claims,
        confidence=section_score.confidence,
        confidence_rationale=section_score.rationale,
        data_gaps=data_gaps,
        model_used=model_used,
    )
