import pytest

from app.sources import edgar


def _profile(rows: list[tuple[str, str, str, str]]) -> dict:
    """rows = [(form, filingDate, accession, primaryDocument)], EDGAR order (newest first)."""
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "accessionNumber": [r[2] for r in rows],
                "primaryDocument": [r[3] for r in rows],
                "reportDate": [r[1] for r in rows],
            }
        }
    }


@pytest.mark.asyncio
async def test_original_10k_beats_a_newer_10ka_amendment(monkeypatch):
    """The Tesla bug: a 10-K/A is a Part III compensation amendment with no MD&A,
    no Competition section and no Legal Proceedings. Being newer must not win."""

    async def fake_profile(cik):
        return _profile(
            [
                ("10-K/A", "2026-04-30", "0001104659-26-053166", "tm2611837d1_10ka.htm"),
                ("10-K", "2026-01-29", "0001628280-26-003952", "tsla-20251231.htm"),
            ]
        )

    monkeypatch.setattr(edgar, "get_submission_profile", fake_profile)
    meta = await edgar.get_latest_filing_meta("0001318605", forms=("10-K", "10-K/A"))

    assert meta["form"] == "10-K"
    assert meta["primary_document"] == "tsla-20251231.htm"


@pytest.mark.asyncio
async def test_amendment_is_used_when_no_original_exists(monkeypatch):
    async def fake_profile(cik):
        return _profile([("10-K/A", "2026-04-30", "0001104659-26-053166", "amend.htm")])

    monkeypatch.setattr(edgar, "get_submission_profile", fake_profile)
    meta = await edgar.get_latest_filing_meta("0001318605", forms=("10-K", "10-K/A"))

    assert meta["form"] == "10-K/A"


@pytest.mark.asyncio
async def test_picks_newest_within_the_preferred_form(monkeypatch):
    async def fake_profile(cik):
        return _profile(
            [
                ("10-K", "2024-02-01", "0000000000-24-000001", "new.htm"),
                ("10-K", "2023-02-01", "0000000000-23-000001", "old.htm"),
            ]
        )

    monkeypatch.setattr(edgar, "get_submission_profile", fake_profile)
    meta = await edgar.get_latest_filing_meta("0000000000", forms=("10-K",))

    assert meta["primary_document"] == "new.htm"


@pytest.mark.asyncio
async def test_returns_none_when_no_requested_form_is_present(monkeypatch):
    async def fake_profile(cik):
        return _profile([("8-K", "2026-01-01", "0000000000-26-000001", "eightk.htm")])

    monkeypatch.setattr(edgar, "get_submission_profile", fake_profile)
    assert await edgar.get_latest_filing_meta("0000000000", forms=("10-K",)) is None
