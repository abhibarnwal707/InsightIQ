import json

import pytest

from app.agents.common import select_relevant_chunks
from app.llm.schemas import DueDiligenceReport, ExtractedClaim, ReportSection
from app.render.export import default_filename, load_report, save_report, slugify
from app.render.html import render_html
from app.sources import gdelt

SEC_URL = "https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm"
DOCKET_URL = "https://www.courtlistener.com/docket/73228033/sipin-v-tesla-inc/"


def _report() -> DueDiligenceReport:
    return DueDiligenceReport(
        company="Tesla",
        resolved_entity={
            "company_name": "Tesla, Inc.",
            "is_public": True,
            "cik": "0001318605",
            "ticker": "TSLA",
            "jurisdiction": "TX",
            "domain": None,
            "resolution_notes": "Unambiguous match.",
            "candidates_considered": ["Tesla, Inc."],
        },
        sections=[
            ReportSection(
                section="financials",
                summary="Revenue grew.",
                claims=[
                    ExtractedClaim(text="Revenue was $94.83B", source_id="xbrl_1", source_url=SEC_URL, entailment="yes"),
                    ExtractedClaim(text="Net income was $3.79B", source_id="xbrl_2", source_url=SEC_URL, entailment="yes"),
                ],
                confidence=0.86,
                confidence_rationale="2 claims verified.",
                data_gaps=["No segment detail."],
                model_used="minimax/minimax-m3:free",
            ),
            ReportSection(
                section="legal_regulatory",
                summary="Ongoing litigation.",
                claims=[
                    ExtractedClaim(text="A complaint was filed.", source_id="docket_1", source_url=DOCKET_URL, entailment="partial"),
                ],
                confidence=0.42,
                confidence_rationale="1 claim verified.",
                data_gaps=[],
                model_used="minimax/minimax-m3:free",
            ),
        ],
        contradictions=["Section A says X, section B says Y."],
        llm_calls_used=12,
    )


def test_html_is_self_contained_and_well_formed():
    html = render_html(_report())
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html
    # no external assets: the file must render offline as a single attachment
    assert "src=" not in html
    assert "<link" not in html


def test_repeated_source_gets_one_deduplicated_reference():
    """Two financial claims cite the same filing -- that must be one reference, not two,
    so a reader can see how much of a section rests on a single document."""
    html = render_html(_report())
    assert html.count('id="ref-1"') == 1
    assert html.count('id="ref-2"') == 1
    assert 'id="ref-3"' not in html
    # both claims still point at reference 1
    assert html.count('href="#ref-1"') == 2


def test_reference_list_records_which_sections_cite_each_source():
    html = render_html(_report())
    assert "Cited in: Financials" in html
    assert "Cited in: Legal &amp; Regulatory" in html


def test_claim_text_is_escaped_not_injected():
    report = _report()
    report.sections[0].claims[0].text = '<script>alert("xss")</script> & more'
    html = render_html(report)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_entailment_is_surfaced_per_claim():
    html = render_html(_report())
    assert "Verified" in html
    assert "Partially supported" in html


def test_contradictions_are_rendered():
    html = render_html(_report())
    assert "Section A says X" in html


def test_empty_report_still_renders():
    report = DueDiligenceReport(
        company="Nobody", resolved_entity={"company_name": "Nobody", "is_public": False},
        sections=[], contradictions=[], llm_calls_used=0,
    )
    html = render_html(report)
    assert "No sources were cited" in html
    assert "None detected" in html


def test_save_report_writes_html_file(tmp_path):
    path = save_report(_report(), tmp_path / "out.html", fmt="html")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_save_report_defaults_to_a_named_file_in_out_dir(tmp_path):
    path = save_report(_report(), None, fmt="html", out_dir=tmp_path / "reports")
    assert path.parent.name == "reports"
    assert path.name.startswith("tesla-due-diligence-")
    assert path.suffix == ".html"


def test_round_trip_json_response_to_html(tmp_path):
    """The real workflow: a saved /research JSON response converts to a report file."""
    src = tmp_path / "response.json"
    src.write_text(_report().model_dump_json(), encoding="utf-8")
    report = load_report(src)
    out = save_report(report, tmp_path / "r.html")
    assert "Tesla" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "raw,expected",
    [("Tesla, Inc.", "tesla-inc"), ("AT&T", "att"), ("  ", "report"), ("Ford/GM", "fordgm")],
)
def test_slugify_is_filesystem_safe(raw, expected):
    slug = slugify(raw)
    assert slug == expected
    assert not (set(slug) & set('<>:"/\\|?*'))


def test_default_filename_has_extension_per_format():
    report = _report()
    assert default_filename(report, "html").endswith(".html")
    assert default_filename(report, "markdown").endswith(".md")


# ---- the two grounding fixes ----

def test_select_relevant_chunks_returns_nothing_when_no_keyword_matches():
    """Previously fell back to arbitrary chunks, which produced confidently-cited
    off-topic claims (Tesla 'competitors' section full of director biographies)."""
    chunks = ["Executive compensation philosophy.", "Director biography details."]
    assert select_relevant_chunks(chunks, ["competition", "competitors"], top_k=4) == []


def test_select_relevant_chunks_ranks_by_keyword_density():
    chunks = ["we mention competition once", "competition competition competition", "unrelated text"]
    top = select_relevant_chunks(chunks, ["competition"], top_k=2)
    assert top[0] == "competition competition competition"
    assert len(top) == 2


def test_gdelt_quotes_multiword_company_names():
    assert gdelt.format_query("Tesla, Inc.") == '"Tesla, Inc"'
    assert gdelt.format_query("SpaceX") == "SpaceX"
    assert gdelt.format_query("  ") == '""'
