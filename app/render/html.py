"""Render a DueDiligenceReport to a self-contained HTML file.

Why HTML and not Markdown/PDF/DOCX: the deliverable here is a *file the user opens*,
and every claim has to stay traceable to the exact source it came from. HTML is the
only option that renders on double-click in any browser with zero tooling, keeps the
citation links clickable, and still prints to PDF (Ctrl+P) -- all with no new
dependency. Markdown needs a viewer (on Windows a .md opens as raw text), PDF needs
weasyprint/GTK, DOCX needs python-docx and renders links poorly.

Everything is inlined (CSS, no images, no external fetches) so the file works offline
and survives being emailed around as a single attachment.

Citations use an academic numbering scheme rather than a bare "[source]" link per
claim: sources are deduplicated across the whole report into a numbered reference
list, each claim carries a superscript [n] pointing at it, and each reference points
back at the claims that cite it. That way a reader can see at a glance that, say, 12
financial claims all rest on one filing -- which the per-claim link format actively
hides, and which is exactly the kind of thin-sourcing signal this project exists to
surface.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from urllib.parse import urlparse

from app.llm.schemas import DueDiligenceReport, ReportSection

_SECTION_TITLES: dict[str, str] = {
    "financials": "Financials",
    "news": "News",
    "legal_regulatory": "Legal &amp; Regulatory",
    "competitors": "Competitors",
    "key_people": "Key People",
}

_SECTION_ORDER = ["financials", "news", "legal_regulatory", "competitors", "key_people"]

_ENTAILMENT_LABEL = {
    "yes": "Verified",
    "partial": "Partially supported",
    "no": "Unsupported",
}

_CSS = """
:root {
  --bg: #ffffff;
  --surface: #f7f8fa;
  --surface-2: #eef0f4;
  --border: #d8dce4;
  --text: #14171f;
  --muted: #5b6472;
  --accent: #1f4d8f;
  --accent-soft: #e8effa;
  --high: #1a7f4b;
  --medium: #9a6206;
  --low: #a33a2a;
  --none: #6b7280;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12141a;
    --surface: #1a1d25;
    --surface-2: #232733;
    --border: #333846;
    --text: #e8eaf0;
    --muted: #9aa3b2;
    --accent: #7aa7e8;
    --accent-soft: #1c2637;
    --high: #4ac585;
    --medium: #e0a640;
    --low: #ef8070;
    --none: #8b93a3;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 48px 28px 96px; }

header.doc { border-bottom: 3px solid var(--accent); padding-bottom: 22px; margin-bottom: 8px; }
.eyebrow {
  font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; margin: 0 0 6px;
}
h1 { font-size: 34px; line-height: 1.2; margin: 0 0 14px; letter-spacing: -.02em; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 10px; margin: 0; padding: 0; list-style: none; }
.meta li {
  font-size: 12.5px; color: var(--muted);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 999px; padding: 3px 11px;
}
.meta li b { color: var(--text); font-weight: 600; }

h2 {
  font-size: 13px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); font-weight: 700;
  margin: 44px 0 14px; padding-bottom: 7px; border-bottom: 1px solid var(--border);
}

/* ---- confidence overview ---- */
.overview { display: grid; gap: 8px; }
.ov-row {
  display: grid; grid-template-columns: 150px 1fr 82px;
  align-items: center; gap: 12px;
  font-size: 14px; text-decoration: none; color: inherit;
  padding: 7px 10px; border-radius: 7px;
}
.ov-row:hover { background: var(--surface); }
.ov-name { font-weight: 600; }
.bar { height: 7px; border-radius: 4px; background: var(--surface-2); overflow: hidden; }
.bar span { display: block; height: 100%; border-radius: 4px; }
.ov-val { text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; color: var(--muted); }
.fill-high { background: var(--high); } .fill-medium { background: var(--medium); }
.fill-low { background: var(--low); }   .fill-none { background: var(--none); }

/* ---- entity ---- */
.entity { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 4px 18px; }
.entity dl { display: grid; grid-template-columns: 168px 1fr; gap: 0; margin: 0; }
.entity dt {
  font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); font-weight: 600; padding: 11px 0; border-bottom: 1px solid var(--border);
}
.entity dd { margin: 0; padding: 11px 0; border-bottom: 1px solid var(--border); font-size: 14.5px; }
.entity dt:last-of-type, .entity dd:last-of-type { border-bottom: 0; }
code.mono { font-family: var(--mono); font-size: 13px; background: var(--surface-2); padding: 1px 6px; border-radius: 4px; }

/* ---- sections ---- */
section.rep { margin-top: 40px; scroll-margin-top: 16px; }
.sec-head { display: flex; align-items: baseline; justify-content: space-between; gap: 14px; flex-wrap: wrap; }
.sec-head h3 { font-size: 23px; margin: 0; letter-spacing: -.01em; }
.chip {
  font-size: 12px; font-weight: 700; letter-spacing: .04em;
  border-radius: 999px; padding: 4px 12px; white-space: nowrap;
  border: 1px solid currentColor;
}
.chip-high { color: var(--high); } .chip-medium { color: var(--medium); }
.chip-low { color: var(--low); }   .chip-none { color: var(--none); }
.summary { margin: 14px 0 0; font-size: 15.5px; }
.rationale {
  margin: 12px 0 0; padding: 9px 14px; font-size: 13.5px; color: var(--muted);
  border-left: 3px solid var(--border); background: var(--surface);
  border-radius: 0 6px 6px 0;
}
.sublabel {
  font-size: 11.5px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); font-weight: 700; margin: 26px 0 10px;
}

/* ---- claims ---- */
ol.claims { list-style: none; counter-reset: c; margin: 0; padding: 0; }
ol.claims > li {
  counter-increment: c; position: relative;
  padding: 12px 0 12px 40px; border-bottom: 1px solid var(--border);
}
ol.claims > li:last-child { border-bottom: 0; }
ol.claims > li::before {
  content: counter(c); position: absolute; left: 0; top: 13px;
  width: 25px; height: 25px; border-radius: 50%;
  background: var(--surface-2); color: var(--muted);
  font-size: 12px; font-weight: 700; display: grid; place-items: center;
}
.claim-text { font-size: 15px; }
.cite {
  font-size: 11px; font-weight: 700; vertical-align: super; line-height: 0;
  text-decoration: none; color: var(--accent);
  background: var(--accent-soft); border-radius: 4px; padding: 2px 5px; margin-left: 4px;
}
.cite:hover { text-decoration: underline; }
.tag {
  display: inline-block; margin-top: 7px; font-size: 11px; font-weight: 600;
  letter-spacing: .05em; text-transform: uppercase; border-radius: 4px; padding: 2px 8px;
}
.tag-yes { color: var(--high); background: color-mix(in srgb, var(--high) 12%, transparent); }
.tag-partial { color: var(--medium); background: color-mix(in srgb, var(--medium) 14%, transparent); }
.tag-no { color: var(--low); background: color-mix(in srgb, var(--low) 12%, transparent); }

.empty { color: var(--muted); font-style: italic; font-size: 14.5px; margin: 18px 0; }
ul.gaps { margin: 0; padding-left: 20px; }
ul.gaps li { font-size: 14px; color: var(--muted); margin-bottom: 6px; }
.model { margin-top: 20px; font-size: 12px; color: var(--muted); font-family: var(--mono); }

.warn {
  border: 1px solid var(--low); border-left-width: 4px; border-radius: 0 8px 8px 0;
  background: color-mix(in srgb, var(--low) 7%, transparent);
  padding: 12px 16px; margin: 0;
}
.warn ul { margin: 0; padding-left: 20px; } .warn li { font-size: 14.5px; }
.ok { color: var(--muted); font-style: italic; font-size: 14.5px; margin: 0; }

/* ---- references ---- */
ol.refs { list-style: none; counter-reset: r; margin: 0; padding: 0; }
ol.refs > li {
  counter-increment: r; position: relative;
  padding: 13px 0 13px 44px; border-bottom: 1px solid var(--border); scroll-margin-top: 16px;
}
ol.refs > li:last-child { border-bottom: 0; }
ol.refs > li::before {
  content: "[" counter(r) "]"; position: absolute; left: 0; top: 13px;
  font-size: 12px; font-weight: 700; color: var(--accent); font-family: var(--mono);
}
ol.refs > li:target { background: var(--accent-soft); border-radius: 6px; }
.ref-title { font-weight: 600; font-size: 14.5px; }
.ref-url { display: block; font-family: var(--mono); font-size: 11.5px; word-break: break-all; margin-top: 3px; }
.ref-url, .ref-title a { color: var(--accent); }
.ref-back { margin-top: 5px; font-size: 12px; color: var(--muted); }

footer.doc {
  margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--border);
  font-size: 12.5px; color: var(--muted);
}

@media (max-width: 640px) {
  .wrap { padding: 30px 18px 64px; }
  h1 { font-size: 26px; }
  .entity dl, .ov-row { grid-template-columns: 1fr; }
  .entity dt { padding-bottom: 0; border-bottom: 0; }
  .bar { display: none; }
}

@media print {
  :root { --bg:#fff; --surface:#f7f8fa; --surface-2:#eee; --border:#ccc; --text:#000; --muted:#444; --accent:#14417a; }
  .wrap { max-width: none; padding: 0; }
  section.rep, ol.refs > li { break-inside: avoid; }
  h2 { break-after: avoid; }
  a { color: var(--accent) !important; }
  .ref-url::after { content: ""; }
}
"""


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "High"
    if confidence >= 0.4:
        return "Medium"
    if confidence > 0:
        return "Low"
    return "None"


class _References:
    """Deduplicates source URLs across the whole report into a numbered reference list.

    Numbering is assignment-ordered (first cite wins [1]) so the reference list reads in
    the same order a reader encounters the citations.
    """

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._citing: dict[str, list[str]] = {}

    def cite(self, url: str, section_title: str) -> int:
        if url not in self._index:
            self._index[url] = len(self._index) + 1
            self._citing[url] = []
        if section_title not in self._citing[url]:
            self._citing[url].append(section_title)
        return self._index[url]

    def entries(self) -> list[tuple[int, str, list[str]]]:
        return [(n, url, self._citing[url]) for url, n in self._index.items()]


def _describe_source(url: str) -> str:
    """A human label for a URL, so the reference list reads as sources, not just links."""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    path = urlparse(url).path
    if "sec.gov" in host:
        if "-index.htm" in path:
            accession = path.rsplit("/", 1)[-1].replace("-index.htm", "")
            return f"SEC EDGAR &mdash; filing index {escape(accession)}"
        doc = path.rsplit("/", 1)[-1]
        return f"SEC EDGAR &mdash; filing document {escape(doc)}" if doc else "SEC EDGAR filing"
    if "courtlistener" in host:
        slug = [p for p in path.split("/") if p]
        if slug and slug[0] == "docket":
            name = slug[-1].replace("-", " ").title() if len(slug) > 2 else "docket"
            return f"CourtListener &mdash; {escape(name)}"
        return "CourtListener docket"
    return escape(host) if host else "Source"


def _render_section(section: ReportSection, refs: _References) -> str:
    title = _SECTION_TITLES.get(section.section, escape(section.section))
    label = _confidence_label(section.confidence)
    cls = label.lower()
    pct = max(0.0, min(1.0, section.confidence)) * 100

    out = [
        f'<section class="rep" id="sec-{escape(section.section)}">',
        '  <div class="sec-head">',
        f"    <h3>{title}</h3>",
        f'    <span class="chip chip-{cls}">{label} confidence &middot; {section.confidence:.2f}</span>',
        "  </div>",
        f'  <div class="bar" aria-hidden="true"><span class="fill-{cls}" style="width:{pct:.1f}%"></span></div>',
    ]

    if section.summary:
        out.append(f'  <p class="summary">{escape(section.summary)}</p>')
    if section.confidence_rationale:
        out.append(f'  <p class="rationale">{escape(section.confidence_rationale)}</p>')

    plain_title = _SECTION_TITLES.get(section.section, section.section).replace("&amp;", "&")
    if section.claims:
        out.append(f'  <p class="sublabel">Claims ({len(section.claims)})</p>')
        out.append('  <ol class="claims">')
        for claim in section.claims:
            cite = ""
            if claim.source_url:
                n = refs.cite(claim.source_url, plain_title)
                cite = (
                    f'<a class="cite" href="#ref-{n}" '
                    f'title="{escape(claim.source_url)}">[{n}]</a>'
                )
            ent = claim.entailment or "no"
            ent_label = _ENTAILMENT_LABEL.get(ent, "Unverified")
            out.append(
                f'    <li><span class="claim-text">{escape(claim.text)}{cite}</span><br>'
                f'<span class="tag tag-{escape(ent)}">{ent_label}</span></li>'
            )
        out.append("  </ol>")
    else:
        out.append('  <p class="empty">No verified claims could be grounded for this section.</p>')

    if section.data_gaps:
        out.append('  <p class="sublabel">Data gaps</p>')
        out.append('  <ul class="gaps">')
        out += [f"    <li>{escape(g)}</li>" for g in section.data_gaps]
        out.append("  </ul>")

    out.append(f'  <p class="model">Model: {escape(section.model_used)}</p>')
    out.append("</section>")
    return "\n".join(out)


def render_html(report: DueDiligenceReport) -> str:
    """Full self-contained HTML document for `report`. No external assets."""
    entity = report.resolved_entity
    refs = _References()

    ordered = [s for name in _SECTION_ORDER for s in report.sections if s.section == name]
    ordered += [s for s in report.sections if s.section not in _SECTION_ORDER]
    # Sections are rendered first so every citation is registered before the reference
    # list is emitted -- the numbering has to reflect document order, not section order.
    section_html = "\n".join(_render_section(s, refs) for s in ordered)

    generated = report.generated_at
    generated_str = (
        generated.strftime("%d %b %Y, %H:%M UTC") if isinstance(generated, datetime) else str(generated)
    )
    total_claims = sum(len(s.claims) for s in report.sections)

    head = [
        '<div class="wrap">',
        '<header class="doc">',
        '  <p class="eyebrow">Due Diligence Report</p>',
        f"  <h1>{escape(report.company)}</h1>",
        '  <ul class="meta">',
        f"    <li>Generated <b>{escape(generated_str)}</b></li>",
        f"    <li><b>{len(report.sections)}</b> sections</li>",
        f"    <li><b>{total_claims}</b> claims</li>",
        f"    <li><b>{report.llm_calls_used}</b> LLM calls</li>",
        "  </ul>",
        "</header>",
    ]

    # ---- confidence overview ----
    head.append("<h2>Confidence overview</h2>")
    head.append('<div class="overview">')
    for s in ordered:
        lab = _confidence_label(s.confidence)
        pct = max(0.0, min(1.0, s.confidence)) * 100
        name = _SECTION_TITLES.get(s.section, escape(s.section))
        head.append(
            f'  <a class="ov-row" href="#sec-{escape(s.section)}">'
            f'<span class="ov-name">{name}</span>'
            f'<span class="bar"><span class="fill-{lab.lower()}" style="width:{pct:.1f}%"></span></span>'
            f'<span class="ov-val">{s.confidence:.2f} {lab}</span></a>'
        )
    head.append("</div>")

    # ---- resolved entity ----
    head.append("<h2>Resolved entity</h2>")
    head.append('<div class="entity"><dl>')
    head.append(f"<dt>Company</dt><dd>{escape(str(entity.get('company_name') or report.company))}</dd>")
    head.append(f"<dt>Public registrant</dt><dd>{'Yes' if entity.get('is_public') else 'No'}</dd>")
    if entity.get("cik"):
        head.append(f"<dt>SEC CIK</dt><dd><code class=\"mono\">{escape(str(entity['cik']))}</code></dd>")
    if entity.get("ticker"):
        head.append(f"<dt>Ticker</dt><dd><code class=\"mono\">{escape(str(entity['ticker']))}</code></dd>")
    if entity.get("jurisdiction"):
        head.append(f"<dt>Jurisdiction</dt><dd>{escape(str(entity['jurisdiction']))}</dd>")
    if entity.get("domain"):
        head.append(f"<dt>Domain</dt><dd>{escape(str(entity['domain']))}</dd>")
    if entity.get("candidates_considered"):
        cands = ", ".join(escape(str(c)) for c in entity["candidates_considered"])
        head.append(f"<dt>Candidates considered</dt><dd>{cands}</dd>")
    if entity.get("resolution_notes"):
        head.append(f"<dt>Resolution notes</dt><dd>{escape(str(entity['resolution_notes']))}</dd>")
    head.append("</dl></div>")

    # ---- contradictions ----
    head.append("<h2>Cross-section contradictions</h2>")
    if report.contradictions:
        items = "".join(f"<li>{escape(c)}</li>" for c in report.contradictions)
        head.append(f'<div class="warn"><ul>{items}</ul></div>')
    else:
        head.append('<p class="ok">None detected across the sections in this report.</p>')

    head.append("<h2>Findings</h2>")

    # ---- references (populated by the section pass above) ----
    ref_html = ["<h2>References</h2>"]
    entries = refs.entries()
    if entries:
        ref_html.append('<ol class="refs">')
        for n, url, citing in entries:
            safe = escape(url, quote=True)
            cited_by = ", ".join(escape(c) for c in citing)
            ref_html.append(
                f'  <li id="ref-{n}">'
                f'<span class="ref-title">{_describe_source(url)}</span>'
                f'<a class="ref-url" href="{safe}" rel="noopener noreferrer" target="_blank">{escape(url)}</a>'
                f'<div class="ref-back">Cited in: {cited_by}</div></li>'
            )
        ref_html.append("</ol>")
    else:
        ref_html.append('<p class="ok">No sources were cited in this report.</p>')

    footer = [
        '<footer class="doc">',
        "  <p>Every claim above is extracted from a specific retrieved passage and independently "
        "verified against that passage before inclusion. Confidence scores are computed from "
        "source tier, corroboration, recency and verification outcome &mdash; they are not "
        "model-generated. Sections with thin sourcing say so in their data gaps rather than "
        "writing around the gap.</p>",
        f"  <p>InsightIQ &middot; {escape(report.company)} &middot; {escape(generated_str)}</p>",
        "</footer>",
        "</div>",
    ]

    title = f"Due Diligence Report: {report.company}"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            *head,
            section_html,
            *ref_html,
            *footer,
            "</body>",
            "</html>",
        ]
    )
