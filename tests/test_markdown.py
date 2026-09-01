from app.llm.schemas import DueDiligenceReport, ExtractedClaim, ReportSection
from app.render.markdown import render_markdown


def _report() -> DueDiligenceReport:
    return DueDiligenceReport(
        company="TestCo",
        resolved_entity={
            "company_name": "TestCo",
            "is_public": True,
            "cik": "0000000001",
            "ticker": "TST",
            "jurisdiction": "DE",
            "domain": None,
            "resolution_notes": "clear match",
        },
        sections=[
            ReportSection(
                section="financials",
                summary="TestCo reported strong revenue.",
                claims=[
                    ExtractedClaim(
                        text="Revenue was $1 billion in 2023",
                        source_id="xbrl_1",
                        source_url="https://www.sec.gov/example",
                        entailment="yes",
                    )
                ],
                confidence=0.85,
                confidence_rationale="1 claim verified; primary_filing tier.",
                data_gaps=[],
                model_used="n/a",
            ),
            ReportSection(
                section="competitors",
                summary="",
                claims=[],
                confidence=0.0,
                confidence_rationale="No claims extracted.",
                data_gaps=["No competitor data available."],
                model_used="n/a",
            ),
        ],
        contradictions=["financials vs news revenue mismatch"],
        llm_calls_used=7,
    )


def test_markdown_includes_company_and_confidence():
    md = render_markdown(_report())
    assert "# Due Diligence Report: TestCo" in md
    assert "Confidence: 0.85 (High)" in md
    assert "[[source]](https://www.sec.gov/example)" in md
    assert "7 LLM call(s) used" in md


def test_markdown_shows_contradictions_when_present():
    md = render_markdown(_report())
    assert "financials vs news revenue mismatch" in md
    assert "None detected." not in md


def test_markdown_shows_none_detected_when_no_contradictions():
    report = _report().model_copy(update={"contradictions": []})
    md = render_markdown(report)
    assert "*None detected.*" in md


def test_markdown_shows_data_gaps_for_empty_section():
    md = render_markdown(_report())
    assert "No competitor data available." in md
    assert "*No verified claims for this section.*" in md
