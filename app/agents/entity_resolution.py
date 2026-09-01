"""Entity resolution: runs once, first. Everything downstream keys off ResolvedEntity.

Design (see design principle #5 and #4 in PLAN.md):
- EDGAR candidate search is deterministic code, not LLM recall.
- The LLM only *chooses among* candidates that were actually retrieved — its choice
  field is a Literal enum of the real candidate IDs plus "none", so it is structurally
  impossible for the model to hallucinate a CIK that doesn't come from EDGAR. If EDGAR
  returns zero candidates (private company, or a name EDGAR's exact/substring match
  can't find), "none" is the only option in the schema.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field, create_model

from app.llm.client import OpenRouterClient
from app.llm.schemas import ResolvedEntity
from app.sources import edgar

logger = logging.getLogger("insightiq.agents.entity_resolution")

_SYSTEM_PROMPT = """You are an entity-resolution assistant for a due-diligence tool.
You will be given a company name a user typed, and a list of candidate registrants
found in SEC EDGAR by a deterministic search (not by you). Your job is ONLY to decide
which candidate (if any) is the same real-world company the user means, or to say the
query does not match any of them (private company, or EDGAR's text search missed it).

Rules:
- Pick a candidate ONLY if you are reasonably confident it is the same real-world
  company as the query (accounting for common names, abbreviations, former/legal
  names you know of).
- If multiple candidates could plausibly match, pick the single most likely one (e.g.
  the flagship/most prominent entity) and explain the alternatives in resolution_notes.
- If nothing in the candidate list is a real match, or the candidate list is empty,
  choose "none" — do not invent a CIK or ticker.
- jurisdiction and domain: you may state a well-known public fact (e.g. state of
  incorporation, primary web domain) in resolution_notes if you're confident, but do
  NOT put anything in the structured jurisdiction/domain fields unless it's directly
  derivable from the candidate data given to you — leave them null otherwise. A later
  deterministic lookup fills jurisdiction in from EDGAR when a candidate is chosen.
"""


def _build_choice_model(candidate_ids: list[str]) -> type[BaseModel]:
    options = tuple(candidate_ids) + ("none",)
    from typing import Literal

    choice_type = Literal[options]  # type: ignore[valid-type]
    return create_model(
        "EntityChoice",
        chosen_candidate=(choice_type, ...),
        is_public=(bool, ...),
        domain=(str | None, None),
        resolution_notes=(str, ...),
    )


async def resolve_entity(company_name: str, client: OpenRouterClient) -> ResolvedEntity:
    edgar_down = False
    try:
        candidates = await edgar.search_candidates(company_name, limit=8)
    except Exception as exc:  # noqa: BLE001
        # EDGAR downtime degrades to the same path as "no candidates found": the choice
        # model's Literal enum only has "none" available, so the LLM can't hallucinate a
        # CIK either way. The distinction is only in the caveat appended to resolution_notes.
        logger.warning("EDGAR search_candidates failed for %r: %s", company_name, exc)
        candidates = []
        edgar_down = True

    candidate_ids = [f"cand_{i + 1}" for i in range(len(candidates))]
    id_to_candidate = dict(zip(candidate_ids, candidates))

    choice_model = _build_choice_model(candidate_ids)

    candidate_block = "\n".join(
        f"{cid}: ticker={c['ticker']!r} title={c['title']!r} cik={c['cik']}"
        for cid, c in id_to_candidate.items()
    ) or "(no candidates found by EDGAR search)"

    user_prompt = (
        f'User query: "{company_name}"\n\n'
        f"Candidates from SEC EDGAR search:\n{candidate_block}\n\n"
        'Respond with your choice among exactly these candidate IDs (or "none").'
    )

    choice, model_used = await client.structured(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        choice_model,
    )

    if choice.chosen_candidate == "none":
        notes = choice.resolution_notes or "No matching SEC-registered entity found."
        if edgar_down:
            notes += " (Note: the EDGAR candidate search itself failed, so this may reflect EDGAR downtime rather than a genuinely private company -- retry later to confirm.)"
        return ResolvedEntity(
            company_name=company_name,
            is_public=False,
            cik=None,
            ticker=None,
            domain=choice.domain,
            jurisdiction=None,
            resolution_notes=notes,
            candidates_considered=[c["title"] for c in candidates],
        )

    picked = id_to_candidate[choice.chosen_candidate]
    jurisdiction: str | None = None
    try:
        profile = await edgar.get_submission_profile(picked["cik"])
        jurisdiction = profile.get("stateOfIncorporation") or None
    except Exception:  # noqa: BLE001 — jurisdiction is best-effort, never blocks resolution
        jurisdiction = None

    return ResolvedEntity(
        company_name=company_name,
        is_public=True,
        cik=picked["cik"],
        ticker=picked["ticker"] or None,
        domain=choice.domain,
        jurisdiction=jurisdiction,
        resolution_notes=choice.resolution_notes,
        candidates_considered=[c["title"] for c in candidates],
    )
