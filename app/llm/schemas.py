"""Pydantic models shared across the pipeline. Every LLM-structured output is one of these."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, create_model

SectionName = Literal["financials", "news", "legal_regulatory", "competitors", "key_people"]


class SourcePassage(BaseModel):
    """A retrieved passage with a scoped ID. IDs are only unique within one LLM call."""

    id: str  # "src_1", "src_2" — scoped per LLM call
    url: str  # exact filing/article/docket URL, never a search page
    title: str = ""
    text: str  # the retrieved passage, stored verbatim
    source_tier: Literal["primary_filing", "regulatory", "news", "web"] = "web"
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractedClaim(BaseModel):
    text: str
    source_id: str  # constrained to a Literal enum of in-context passage IDs at call time
    # The scoped source_id alone is meaningless once separated from the LLM call that
    # produced it. source_url is filled in by code from the matching SourcePassage right
    # after extraction -- never by the model -- so the final report stays traceable to an
    # actual retrieved URL, not just an ID that only meant something mid-call.
    source_url: str = ""
    entailment: Literal["yes", "partial", "no"] | None = None  # filled by the verification pass


class ReportSection(BaseModel):
    section: SectionName
    summary: str
    claims: list[ExtractedClaim]
    confidence: float
    confidence_rationale: str
    data_gaps: list[str]
    model_used: str  # which OpenRouter model produced this section


class ResolvedEntity(BaseModel):
    company_name: str
    is_public: bool
    cik: str | None = None  # SEC Central Index Key, zero-padded to 10 digits
    ticker: str | None = None
    domain: str | None = None
    jurisdiction: str | None = None
    resolution_notes: str = ""
    candidates_considered: list[str] = Field(default_factory=list)


def build_extraction_model(passage_ids: list[str]) -> type[BaseModel]:
    """Dynamically build a claim-extraction response model whose `source_id` field is a
    Literal enum of exactly the passage IDs fed into this call. This is what makes a
    fabricated citation a schema violation rather than a hope — see design principle #2.
    """
    if not passage_ids:
        raise ValueError("passage_ids must be non-empty to build a constrained extraction model")

    id_literal = Literal[tuple(passage_ids)]  # type: ignore[valid-type]
    constrained_claim = create_model(
        "ConstrainedClaim",
        text=(str, ...),
        source_id=(id_literal, ...),
    )
    return create_model(
        "SectionExtraction",
        summary=(str, ...),
        claims=(list[constrained_claim], ...),
        data_gaps=(list[str], ...),
    )


class EntailmentCheck(BaseModel):
    """Output of the separate, cheap entailment-verification pass (design principle #3)."""

    entailment: Literal["yes", "partial", "no"]
    rationale: str = ""


class DueDiligenceReport(BaseModel):
    company: str
    resolved_entity: dict
    sections: list[ReportSection]
    contradictions: list[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_calls_used: int  # surface budget consumption against the daily cap
