import pytest

from app import orchestrator
from app.llm.schemas import ReportSection, ResolvedEntity


def _stub_section(name: str) -> ReportSection:
    return ReportSection(
        section=name,
        summary=f"{name} summary",
        claims=[],
        confidence=0.5,
        confidence_rationale="stub",
        data_gaps=[],
        model_used="stub-model",
    )


@pytest.mark.asyncio
async def test_second_call_serves_from_cache_with_zero_new_llm_calls(isolated_db, monkeypatch):
    resolve_calls = {"n": 0}
    agent_calls = {"n": 0}

    async def fake_resolve_entity(company_name, client):
        resolve_calls["n"] += 1
        from app.cache import store
        store.record_llm_call("fake-resolver-model")  # simulate the real cost of resolution
        return ResolvedEntity(company_name=company_name, is_public=True, cik="0000000001", ticker="TST")

    async def fake_agent(entity, client):
        agent_calls["n"] += 1
        from app.cache import store
        store.record_llm_call("fake-section-model")  # simulate the real cost of a section agent
        return _stub_section("financials")

    monkeypatch.setattr(orchestrator, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(orchestrator, "run_financials_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_news_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_legal_regulatory_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_competitors_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_key_people_agent", fake_agent)

    client = object()  # never touched directly by the orchestrator itself

    first = await orchestrator.run_due_diligence("TestCo", client)
    assert first.llm_calls_used == 6  # 1 resolution + 5 section agents
    assert resolve_calls["n"] == 1
    assert agent_calls["n"] == 5

    second = await orchestrator.run_due_diligence("TestCo", client)
    assert second.llm_calls_used == 0
    assert resolve_calls["n"] == 1  # not called again -- entity cache hit
    assert agent_calls["n"] == 5  # not called again -- report cache hit
    assert second.company == first.company
    assert len(second.sections) == len(first.sections)


@pytest.mark.asyncio
async def test_private_company_report_cache_key_uses_normalized_name(isolated_db, monkeypatch):
    async def fake_resolve_entity(company_name, client):
        return ResolvedEntity(company_name=company_name, is_public=False)

    async def fake_agent(entity, client):
        return _stub_section("financials")

    monkeypatch.setattr(orchestrator, "resolve_entity", fake_resolve_entity)
    monkeypatch.setattr(orchestrator, "run_financials_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_news_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_legal_regulatory_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_competitors_agent", fake_agent)
    monkeypatch.setattr(orchestrator, "run_key_people_agent", fake_agent)

    report = await orchestrator.run_due_diligence("Cargill", object())
    assert report.resolved_entity["is_public"] is False
    cached = await orchestrator.run_due_diligence("Cargill", object())
    assert cached.llm_calls_used == 0
