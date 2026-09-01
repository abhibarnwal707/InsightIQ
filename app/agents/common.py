"""The extraction -> entailment-verification pipeline every section agent reuses.

This module is the concrete implementation of design principles #1-#3 from PLAN.md:
claims only come from passages already in context (never LLM recall), source_id is
schema-constrained to the passage IDs actually fed in, and every extracted claim gets
a second, separate, cheap LLM call checking it against its cited passage before it's
allowed to survive into the report.
"""
from __future__ import annotations

import asyncio
import logging

from app.llm.client import OpenRouterClient
from app.llm.schemas import EntailmentCheck, ExtractedClaim, SourcePassage, build_extraction_model

logger = logging.getLogger("insightiq.agents.common")

_EXTRACTION_SYSTEM_PROMPT = """You are a claims-extraction assistant for a due-diligence tool.
You will be given a numbered list of source passages, each with an id, url, and text.
Extract factual claims that are DIRECTLY STATED OR DIRECTLY SUPPORTED by the passage text.

Rules:
- Every claim's source_id MUST be exactly one of the passage ids you were given -- never
  invent one, never leave it blank.
- Do not use outside knowledge. If a passage doesn't support a claim, don't extract it.
- Prefer specific, checkable claims (numbers, dates, named entities, events) over vague ones.
- If the passages don't support any claims worth extracting, return an empty claims list
  and explain why in data_gaps -- do not stretch a weak passage into a claim.
"""

_ENTAILMENT_SYSTEM_PROMPT = """You are a fact-checking assistant. You will be given one
source passage and one claim that was supposedly extracted from it. Decide:
- "yes": the passage directly and clearly supports the claim.
- "partial": the passage is related but doesn't fully support the specific claim as stated
  (e.g. right topic but wrong number/date, or the claim overstates what the passage says).
- "no": the passage does not support the claim at all, or the claim isn't derivable from it.
Be strict -- when genuinely torn between yes and partial, choose partial.
"""


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to chunk_size, carrying a small
    tail of the previous chunk forward so claims near a chunk boundary aren't stranded.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    raw_chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > chunk_size:
            raw_chunks.append(current)
            current = ""
        if len(para) > chunk_size:
            if current:
                raw_chunks.append(current)
                current = ""
            step = max(1, chunk_size - overlap)
            for i in range(0, len(para), step):
                raw_chunks.append(para[i : i + chunk_size])
            continue
        current = f"{current}\n{para}" if current else para
    if current:
        raw_chunks.append(current)

    overlapped: list[str] = []
    prev_tail = ""
    for c in raw_chunks:
        overlapped.append(f"{prev_tail}\n{c}".strip() if prev_tail else c)
        prev_tail = c[-overlap:] if len(c) > overlap else c
    return overlapped


def select_relevant_chunks(chunks: list[str], keywords: list[str], top_k: int = 6) -> list[str]:
    """Cheap keyword-count relevance ranking -- no embeddings/vector DB (keep infra minimal).

    Returns ONLY chunks that actually match at least one keyword. If nothing matches,
    this returns an empty list so the caller reports an honest data gap.

    It used to fall back to "the first top_k chunks" when nothing scored, which quietly
    converted "this document has no relevant section" into "here are some arbitrary
    passages" -- and the extractor, doing its job, would dutifully pull well-grounded
    claims that had nothing to do with the section. Observed live: a Tesla competitors
    section whose passages contained no competitor content returned five confidently
    cited claims about a director's biography and the compensation committee, all
    passing entailment because they *were* supported by the (irrelevant) passages
    they cited. Entailment verifies a claim against its passage; it cannot catch a
    passage that should never have been selected, so the honest gap has to happen here.
    """
    keywords_l = [k.lower() for k in keywords]
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_l = chunk.lower()
        score = sum(chunk_l.count(k) for k in keywords_l)
        if score > 0:
            scored.append((score, i, chunk))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top_k]]


async def extract_claims(
    client: OpenRouterClient,
    passages: list[SourcePassage],
    instruction: str,
    max_claims: int = 8,
) -> tuple[list[ExtractedClaim], str, list[str], str]:
    """Extraction only -- no entailment check yet. Returns (claims, summary, data_gaps, model_used).

    max_claims caps how many claims come out of one extraction call. Each surviving claim
    costs one more entailment-check LLM call downstream, and a single narrative document can
    easily contain far more extractable factual statements than the free-tier daily budget
    can afford to verify across all five section agents in one run -- observed live: one
    unbounded 10-K extraction returned 34 claims and burned the entire day's budget on
    entailment checks alone. Capping here keeps one section from starving the rest of the run.
    """
    if not passages:
        return [], "", ["No source passages were retrieved for this section."], "n/a"

    extraction_model = build_extraction_model([p.id for p in passages])
    passage_block = "\n\n".join(f"[{p.id}] (url: {p.url})\n{p.text}" for p in passages)
    user_content = (
        f"{instruction}\n\nExtract at most {max_claims} claims -- pick the {max_claims} most "
        "decision-relevant ones, not every statement you can find.\n\n"
        f"Source passages:\n\n{passage_block}"
    )

    result, model_used = await client.structured(
        [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        extraction_model,
    )
    passage_by_id = {p.id: p for p in passages}
    claims = [
        ExtractedClaim(text=c.text, source_id=c.source_id, source_url=passage_by_id[c.source_id].url)
        for c in result.claims
    ][:max_claims]
    return claims, result.summary, list(result.data_gaps), model_used


async def verify_entailment(
    client: OpenRouterClient,
    claims: list[ExtractedClaim],
    passage_by_id: dict[str, SourcePassage],
) -> list[ExtractedClaim]:
    """One separate, cheap LLM call per claim. Never skipped (design principle #3)."""

    async def _check(claim: ExtractedClaim) -> ExtractedClaim:
        passage = passage_by_id[claim.source_id]
        user_content = f"Passage:\n{passage.text}\n\nClaim:\n{claim.text}"
        try:
            check, _ = await client.structured(
                [
                    {"role": "system", "content": _ENTAILMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                EntailmentCheck,
                temperature=0.0,
            )
            return claim.model_copy(update={"entailment": check.entailment})
        except Exception as exc:  # noqa: BLE001
            # fail closed: an unverifiable claim is treated as failed verification, not
            # silently kept -- never let an infra error masquerade as a grounded claim.
            logger.warning("entailment check failed for claim %r: %s", claim.text[:80], exc)
            return claim.model_copy(update={"entailment": "no"})

    return list(await asyncio.gather(*[_check(c) for c in claims]))


async def extract_and_verify(
    client: OpenRouterClient,
    passages: list[SourcePassage],
    instruction: str,
    max_claims: int = 8,
) -> tuple[list[ExtractedClaim], str, list[str], str]:
    """The full Phase-2 pattern: extract -> verify entailment. Reused by every section agent."""
    claims, summary, data_gaps, model_used = await extract_claims(
        client, passages, instruction, max_claims=max_claims
    )
    if not claims:
        return claims, summary, data_gaps, model_used
    passage_by_id = {p.id: p for p in passages}
    verified = await verify_entailment(client, claims, passage_by_id)
    return verified, summary, data_gaps, model_used


def split_verified_claims(claims: list[ExtractedClaim]) -> tuple[list[ExtractedClaim], list[str]]:
    """Drop claims that failed entailment; return survivors plus a data_gaps note about drops."""
    survivors = [c for c in claims if c.entailment != "no"]
    dropped = len(claims) - len(survivors)
    notes = []
    if dropped:
        notes.append(f"{dropped} extracted claim(s) failed entailment verification and were removed.")
    return survivors, notes
