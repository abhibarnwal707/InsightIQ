"""Render a DueDiligenceReport to readable Markdown. JSON (the report's own
model_dump_json) is the canonical machine-readable form; this is the human one.
"""
from __future__ import annotations

from app.llm.schemas import DueDiligenceReport, ReportSection

_SECTION_TITLES: dict[str, str] = {
    "financials": "Financials",
    "news": "News",
    "legal_regulatory": "Legal & Regulatory",
    "competitors": "Competitors",
    "key_people": "Key People",
}

_SECTION_ORDER = ["financials", "news", "legal_regulatory", "competitors", "key_people"]


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "High"
    if confidence >= 0.4:
        return "Medium"
    if confidence > 0:
        return "Low"
    return "None"


def _render_section(section: ReportSection) -> str:
    title = _SECTION_TITLES.get(section.section, section.section)
    label = _confidence_label(section.confidence)
    lines = [f"## {title} — Confidence: {section.confidence:.2f} ({label})", ""]

    if section.summary:
        lines += [section.summary, ""]

    lines += [f"> {section.confidence_rationale}", ""]

    if section.claims:
        lines.append("**Claims:**")
        lines.append("")
        for i, claim in enumerate(section.claims, start=1):
            entailment = claim.entailment or "unverified"
            link = f" [[source]]({claim.source_url})" if claim.source_url else ""
            lines.append(f"{i}. {claim.text}{link} — entailment: {entailment}")
        lines.append("")
    else:
        lines += ["*No verified claims for this section.*", ""]

    if section.data_gaps:
        lines.append("**Data gaps:**")
        lines.append("")
        for gap in section.data_gaps:
            lines.append(f"- {gap}")
        lines.append("")

    lines.append(f"*Model: {section.model_used}*")
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: DueDiligenceReport) -> str:
    entity = report.resolved_entity
    lines = [
        f"# Due Diligence Report: {report.company}",
        "",
        f"*Generated {report.generated_at.isoformat()} UTC &nbsp;|&nbsp; "
        f"{report.llm_calls_used} LLM call(s) used*",
        "",
        "## Resolved Entity",
        "",
        f"- **Public company:** {'Yes' if entity.get('is_public') else 'No'}",
    ]
    if entity.get("is_public"):
        lines.append(f"- **CIK:** {entity.get('cik') or 'unknown'}")
        lines.append(f"- **Ticker:** {entity.get('ticker') or 'unknown'}")
        lines.append(f"- **Jurisdiction:** {entity.get('jurisdiction') or 'unknown'}")
    if entity.get("domain"):
        lines.append(f"- **Domain:** {entity['domain']}")
    if entity.get("resolution_notes"):
        lines.append(f"- **Resolution notes:** {entity['resolution_notes']}")
    lines.append("")

    lines.append("## Cross-Section Contradictions")
    lines.append("")
    if report.contradictions:
        for c in report.contradictions:
            lines.append(f"- {c}")
    else:
        lines.append("*None detected.*")
    lines.append("")

    sections_by_name = {s.section: s for s in report.sections}
    for name in _SECTION_ORDER:
        section = sections_by_name.get(name)
        if section is None:
            continue
        lines.append(_render_section(section))

    return "\n".join(lines)
