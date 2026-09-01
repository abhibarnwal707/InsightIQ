"""Cross-section contradiction check: its own LLM pass over the already-extracted,
already-verified claims from every section (design principle from Phase 6 in PLAN.md).

This is a meta-analysis pass over claims that already survived extraction and
entailment verification, not a new grounding step -- it doesn't need the source_id
Literal-enum pattern, since it isn't citing a passage, it's comparing claims that
are already each individually cited.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.llm.client import OpenRouterClient
from app.llm.schemas import ReportSection

_SYSTEM_PROMPT = """You are reviewing a due-diligence report's already-verified claims,
grouped by section, for CONTRADICTIONS -- cases where two claims state mutually
inconsistent facts (e.g. two different revenue figures for the same period, conflicting
dates for the same event, conflicting counts or names for the same role).

Rules:
- Only flag a genuine contradiction: the same specific fact stated two incompatible ways.
- Do NOT flag claims that are simply about different topics, different time periods, or
  that are merely incomplete relative to each other.
- Each contradiction should be one sentence naming the two sections involved and briefly
  stating what conflicts.
- If you find no contradictions, return an empty list. Do not manufacture one to have
  something to report.
"""


class ConsistencyCheckResult(BaseModel):
    contradictions: list[str]


def _format_claims(sections: list[ReportSection]) -> str:
    lines = []
    for section in sections:
        for claim in section.claims:
            lines.append(f"[{section.section}] {claim.text}")
    return "\n".join(lines)


async def run_consistency_check(
    sections: list[ReportSection], client: OpenRouterClient
) -> list[str]:
    claims_block = _format_claims(sections)
    if not claims_block.strip():
        return []

    result, _model_used = await client.structured(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Claims by section:\n\n{claims_block}"},
        ],
        ConsistencyCheckResult,
    )
    return list(result.contradictions)
