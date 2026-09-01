from fastapi.testclient import TestClient

from app import main
from app.llm.schemas import DueDiligenceReport


def _stub_report() -> DueDiligenceReport:
    return DueDiligenceReport(
        company="TestCo",
        resolved_entity={"company_name": "TestCo", "is_public": True, "cik": "0000000001"},
        sections=[],
        contradictions=[],
        llm_calls_used=3,
    )


def test_research_endpoint_returns_json_report(isolated_db, monkeypatch):
    async def fake_run_due_diligence(company_name, client):
        assert company_name == "TestCo"
        return _stub_report()

    monkeypatch.setattr(main, "run_due_diligence", fake_run_due_diligence)

    client = TestClient(main.app)
    resp = client.post("/research", json={"company_name": "TestCo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["company"] == "TestCo"
    assert body["llm_calls_used"] == 3


def test_research_endpoint_markdown_format(isolated_db, monkeypatch):
    async def fake_run_due_diligence(company_name, client):
        return _stub_report()

    monkeypatch.setattr(main, "run_due_diligence", fake_run_due_diligence)

    client = TestClient(main.app)
    resp = client.post("/research?format=markdown", json={"company_name": "TestCo"})
    assert resp.status_code == 200
    assert "text/markdown" in resp.headers["content-type"]
    assert "# Due Diligence Report: TestCo" in resp.text


def test_research_endpoint_html_format(isolated_db, monkeypatch):
    async def fake_run_due_diligence(company_name, client):
        return _stub_report()

    monkeypatch.setattr(main, "run_due_diligence", fake_run_due_diligence)

    client = TestClient(main.app)
    resp = client.post("/research?format=html", json={"company_name": "TestCo"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert resp.text.startswith("<!doctype html>")
    assert "TestCo" in resp.text


def test_research_endpoint_html_download_sets_attachment_header(isolated_db, monkeypatch):
    async def fake_run_due_diligence(company_name, client):
        return _stub_report()

    monkeypatch.setattr(main, "run_due_diligence", fake_run_due_diligence)

    client = TestClient(main.app)
    resp = client.post("/research?format=html&download=true", json={"company_name": "TestCo"})
    assert resp.status_code == 200
    disposition = resp.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert ".html" in disposition


def test_research_endpoint_save_writes_file(isolated_db, monkeypatch, tmp_path):
    async def fake_run_due_diligence(company_name, client):
        return _stub_report()

    written: dict = {}

    def fake_save(report, fmt="html"):
        path = tmp_path / "saved.html"
        path.write_text("saved", encoding="utf-8")
        written["path"] = path
        return path

    monkeypatch.setattr(main, "run_due_diligence", fake_run_due_diligence)
    monkeypatch.setattr(main, "save_report", fake_save)

    client = TestClient(main.app)
    resp = client.post("/research?save=true", json={"company_name": "TestCo"})
    assert resp.status_code == 200
    assert resp.json()["saved_to"] == str(written["path"])


def test_research_endpoint_rejects_empty_company_name(isolated_db):
    client = TestClient(main.app)
    resp = client.post("/research", json={"company_name": "   "})
    assert resp.status_code == 422


def test_research_endpoint_surfaces_budget_exceeded_as_429(isolated_db, monkeypatch):
    from app.llm.client import BudgetExceededError

    async def fake_run_due_diligence(company_name, client):
        raise BudgetExceededError("daily budget spent")

    monkeypatch.setattr(main, "run_due_diligence", fake_run_due_diligence)

    client = TestClient(main.app)
    resp = client.post("/research", json={"company_name": "TestCo"})
    assert resp.status_code == 429
