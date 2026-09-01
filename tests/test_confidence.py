from app.scoring.confidence import (
    ClaimScoringInput,
    recency_factor,
    score_claim,
    score_section,
)


def test_no_entailment_zeroes_the_claim():
    claim = ClaimScoringInput(
        entailment="no", source_tier="primary_filing", recency_days=0, corroboration_count=5
    )
    assert score_claim(claim) == 0.0


def test_yes_entailment_scores_higher_than_partial():
    yes = ClaimScoringInput(entailment="yes", source_tier="news", recency_days=10, corroboration_count=1)
    partial = ClaimScoringInput(entailment="partial", source_tier="news", recency_days=10, corroboration_count=1)
    assert score_claim(yes) > score_claim(partial) > 0.0


def test_primary_filing_scores_higher_than_web_all_else_equal():
    filing = ClaimScoringInput(entailment="yes", source_tier="primary_filing", recency_days=30, corroboration_count=1)
    web = ClaimScoringInput(entailment="yes", source_tier="web", recency_days=30, corroboration_count=1)
    assert score_claim(filing) > score_claim(web)


def test_more_corroboration_scores_higher_up_to_saturation():
    one = ClaimScoringInput(entailment="yes", source_tier="news", recency_days=10, corroboration_count=1)
    three = ClaimScoringInput(entailment="yes", source_tier="news", recency_days=10, corroboration_count=3)
    ten = ClaimScoringInput(entailment="yes", source_tier="news", recency_days=10, corroboration_count=10)
    assert score_claim(one) < score_claim(three)
    assert score_claim(three) == score_claim(ten)  # saturates at CORROBORATION_SATURATION


def test_recency_decays_toward_floor_but_never_below_it():
    assert recency_factor(0) == 1.0
    assert abs(recency_factor(365) - 0.65) < 0.02  # one half-life: floor + half of the remaining range
    assert recency_factor(10_000) > 0.3
    assert recency_factor(10_000) < 0.31


def test_score_section_empty_is_zero_with_no_claims():
    result = score_section([])
    assert result.confidence == 0.0
    assert result.surviving_claims == 0


def test_score_section_drops_no_entailment_claims_from_average():
    claims = [
        ClaimScoringInput(entailment="yes", source_tier="primary_filing", recency_days=5, corroboration_count=2),
        ClaimScoringInput(entailment="no", source_tier="primary_filing", recency_days=5, corroboration_count=2),
    ]
    result = score_section(claims)
    assert result.surviving_claims == 1
    assert result.dropped_claims == 1
    assert result.confidence > 0.0  # driven only by the surviving claim, not zeroed by the dropped one


def test_score_section_all_failed_entailment_is_zero_confidence():
    claims = [
        ClaimScoringInput(entailment="no", source_tier="primary_filing", recency_days=5, corroboration_count=2),
    ]
    result = score_section(claims)
    assert result.confidence == 0.0
    assert "failed entailment" in result.rationale


def test_thin_sourcing_caps_confidence_low():
    # a single, low-tier, low-corroboration, stale claim should score well below 0.5
    weak = ClaimScoringInput(
        entailment="partial", source_tier="web", recency_days=900, corroboration_count=1
    )
    result = score_section([weak])
    assert result.confidence < 0.35
