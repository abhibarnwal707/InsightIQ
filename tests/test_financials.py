import httpx
import pytest
import respx

from app.agents import financials
from app.config import settings
from app.llm.schemas import ResolvedEntity
from app.sources import edgar


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_private_company_short_circuits_with_no_claims(isolated_db):
    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="Cargill", is_public=False)
    async with OpenRouterClient() as client:
        section = await financials.run_financials_agent(entity, client)

    assert section.claims == []
    assert section.confidence == 0.0
    assert "not SEC-registered" in section.data_gaps[0]


def test_pick_latest_annual_prefers_most_recent_10k_full_year():
    entries = [
        {"start": "2021-01-01", "end": "2021-12-31", "val": 100, "accn": "a1", "form": "10-K", "fp": "FY", "filed": "2022-02-01"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 200, "accn": "a2", "form": "10-K", "fp": "FY", "filed": "2023-02-01"},
        {"start": "2022-01-01", "end": "2022-03-31", "val": 999, "accn": "a3", "form": "10-Q", "fp": "Q1", "filed": "2022-04-01"},
    ]
    picked = financials._pick_latest_annual(entries, instant=False)
    assert picked["accn"] == "a2"


def test_pick_latest_annual_returns_none_when_no_full_year_present():
    entries = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 999, "accn": "a3", "form": "10-Q", "fp": "Q1", "filed": "2022-04-01"},
    ]
    assert financials._pick_latest_annual(entries, instant=False) is None


@pytest.mark.asyncio
async def test_public_company_builds_deterministic_xbrl_claims(isolated_db, monkeypatch):
    fake_facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01", "end": "2022-12-31", "val": 1_000_000_000,
                                "accn": "0000000000-23-000001", "form": "10-K", "fp": "FY", "filed": "2023-02-01",
                            }
                        ]
                    }
                },
            }
        }
    }

    async def fake_get_company_facts(cik):
        return fake_facts

    async def fake_get_latest_filing_meta(cik, forms=("10-K",)):
        return None  # no narrative text in this test -- exercise the XBRL-only path

    monkeypatch.setattr(edgar, "get_company_facts", fake_get_company_facts)
    monkeypatch.setattr(edgar, "get_latest_filing_meta", fake_get_latest_filing_meta)

    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="TestCo", is_public=True, cik="0000000000", ticker="TST")
    with respx.mock(base_url=settings.openrouter_base_url, assert_all_called=False):
        async with OpenRouterClient() as client:
            section = await financials.run_financials_agent(entity, client)

    assert len(section.claims) == 1
    assert section.claims[0].entailment == "yes"
    assert section.claims[0].source_id == "xbrl_1"
    assert "1.00 billion" in section.claims[0].text
    assert section.confidence > 0.0
    assert any("narrative" in gap.lower() for gap in section.data_gaps)


@pytest.mark.asyncio
async def test_edgar_downtime_on_filing_lookup_does_not_crash_the_agent(isolated_db, monkeypatch):
    fake_facts = {"facts": {"us-gaap": {}}}

    async def fake_get_company_facts(cik):
        return fake_facts

    async def failing_get_latest_filing_meta(cik, forms=("10-K",)):
        raise httpx.ConnectError("EDGAR is down")

    monkeypatch.setattr(edgar, "get_company_facts", fake_get_company_facts)
    monkeypatch.setattr(edgar, "get_latest_filing_meta", failing_get_latest_filing_meta)

    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="TestCo", is_public=True, cik="0000000000", ticker="TST")
    with respx.mock(base_url=settings.openrouter_base_url, assert_all_called=False):
        async with OpenRouterClient() as client:
            section = await financials.run_financials_agent(entity, client)

    assert section.claims == []
    assert section.confidence == 0.0
    assert any("narrative" in gap.lower() for gap in section.data_gaps)
