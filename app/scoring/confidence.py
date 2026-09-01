"""Confidence scoring: source tier + corroboration + recency + entailment -> a formula.

Design principle #6: confidence is computed, never asked of the model. Every function
here is a pure function of plain data — no LLM call, no I/O — so it's unit-testable
with synthetic inputs (Phase 2 acceptance criterion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

SourceTier = Literal["primary_filing", "regulatory", "news", "web"]
Entailment = Literal["yes", "partial", "no"]

SOURCE_TIER_WEIGHT: dict[SourceTier, float] = {
    "primary_filing": 1.0,
    "regulatory": 0.9,
    "news": 0.6,
    "web": 0.4,
}

ENTAILMENT_WEIGHT: dict[Entailment, float] = {
    "yes": 1.0,
    "partial": 0.5,
    "no": 0.0,
}

# Recency decay: exponential half-life. Old filings/news aren't worthless (a founding
# date or a historical filing is still true) so decay to a floor rather than to zero.
_RECENCY_HALF_LIFE_DAYS = 365.0
_RECENCY_FLOOR = 0.3

# Confidence keeps climbing as more distinct sources corroborate a section, but with
# diminishing returns; it's saturated once CORROBORATION_SATURATION distinct sources back it.
CORROBORATION_SATURATION = 3


def recency_factor(days_old: float) -> float:
    if days_old <= 0:
        return 1.0
    decayed = math.pow(0.5, days_old / _RECENCY_HALF_LIFE_DAYS)
    return _RECENCY_FLOOR + (1.0 - _RECENCY_FLOOR) * decayed


def days_since(reference: datetime, now: datetime | None = None) -> float:
    now = now or datetime.utcnow()
    ref = reference.replace(tzinfo=None) if reference.tzinfo else reference
    return max(0.0, (now - ref).total_seconds() / 86400.0)


@dataclass
class ClaimScoringInput:
    entailment: Entailment
    source_tier: SourceTier
    recency_days: float
    corroboration_count: int = 1  # distinct sources backing this claim, >= 1


def score_claim(claim: ClaimScoringInput) -> float:
    """One claim's contribution to section confidence, in [0, 1]."""
    entailment_w = ENTAILMENT_WEIGHT[claim.entailment]
    if entailment_w == 0.0:
        return 0.0  # entailment is a gate, not just a weight: a wrong citation contributes nothing
    tier_w = SOURCE_TIER_WEIGHT[claim.source_tier]
    corroboration_w = min(1.0, claim.corroboration_count / CORROBORATION_SATURATION)
    recency_w = recency_factor(claim.recency_days)
    base = 0.5 * tier_w + 0.3 * corroboration_w + 0.2 * recency_w
    return entailment_w * base


@dataclass
class SectionScore:
    confidence: float
    rationale: str
    surviving_claims: int
    dropped_claims: int


def score_section(claims: list[ClaimScoringInput]) -> SectionScore:
    """Aggregate per-claim scores into one section confidence + a human rationale.

    Claims with entailment == "no" are excluded from the average entirely (design
    principle #3: a claim that failed verification is not evidence for anything) but
    counted so the rationale/data_gaps can say so.
    """
    if not claims:
        return SectionScore(
            confidence=0.0,
            rationale="No claims were extracted for this section.",
            surviving_claims=0,
            dropped_claims=0,
        )

    scored = [(c, score_claim(c)) for c in claims]
    surviving = [(c, s) for c, s in scored if c.entailment != "no"]
    dropped = len(scored) - len(surviving)

    if not surviving:
        return SectionScore(
            confidence=0.0,
            rationale=(
                f"All {len(scored)} extracted claim(s) failed entailment verification "
                "and were dropped."
            ),
            surviving_claims=0,
            dropped_claims=dropped,
        )

    avg = sum(s for _, s in surviving) / len(surviving)
    distinct_tiers = {c.source_tier for c, _ in surviving}
    total_corroboration = sum(c.corroboration_count for c, _ in surviving)

    rationale_parts = [
        f"{len(surviving)} claim(s) verified" + (f", {dropped} dropped for failed entailment" if dropped else ""),
        f"source tiers: {', '.join(sorted(distinct_tiers))}",
        f"total corroborating sources: {total_corroboration}",
    ]
    return SectionScore(
        confidence=round(avg, 3),
        rationale="; ".join(rationale_parts) + ".",
        surviving_claims=len(surviving),
        dropped_claims=dropped,
    )
