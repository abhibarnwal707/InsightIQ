"""Write a DueDiligenceReport to a file on disk, and a CLI to convert a saved
/research JSON response into a rendered report.

The API returns a report; this turns one into an artifact the user actually keeps.
Kept separate from the renderers so they stay pure string functions (trivially
testable, no filesystem), with all path/IO concerns in one place.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from app.llm.schemas import DueDiligenceReport
from app.render.html import render_html
from app.render.markdown import render_markdown

Format = Literal["html", "markdown", "json"]

_EXTENSION: dict[str, str] = {"html": ".html", "markdown": ".md", "json": ".json"}


def slugify(value: str) -> str:
    """Filesystem-safe stem for a company name. Windows-safe: no <>:\"/\\|?* or trailing dots."""
    slug = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-.")
    return slug or "report"


def render(report: DueDiligenceReport, fmt: Format = "html") -> str:
    if fmt == "html":
        return render_html(report)
    if fmt == "markdown":
        return render_markdown(report)
    if fmt == "json":
        return report.model_dump_json(indent=2)
    raise ValueError(f"unsupported format: {fmt!r}")


def default_filename(report: DueDiligenceReport, fmt: Format = "html") -> str:
    stamp = report.generated_at.strftime("%Y%m%d-%H%M")
    return f"{slugify(report.company)}-due-diligence-{stamp}{_EXTENSION[fmt]}"


def save_report(
    report: DueDiligenceReport,
    path: str | Path | None = None,
    fmt: Format = "html",
    out_dir: str | Path = "reports",
) -> Path:
    """Write `report` to disk and return the path written.

    `path` wins if given (its parent is created); otherwise the file lands in `out_dir`
    under a name derived from the company and generation time.
    """
    target = Path(path) if path is not None else Path(out_dir) / default_filename(report, fmt)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(report, fmt), encoding="utf-8")
    return target


def load_report(path: str | Path) -> DueDiligenceReport:
    """Parse a saved /research JSON response back into a report."""
    raw = Path(path).read_text(encoding="utf-8")
    return DueDiligenceReport.model_validate(json.loads(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.render.export",
        description="Convert a saved /research JSON response into a readable report file.",
    )
    parser.add_argument("input", help="path to a saved /research JSON response")
    parser.add_argument("-o", "--output", default=None, help="output file path (default: ./reports/<company>-...)")
    parser.add_argument(
        "-f", "--format", default="html", choices=["html", "markdown", "json"],
        help="output format (default: html)",
    )
    args = parser.parse_args(argv)

    try:
        report = load_report(args.input)
    except FileNotFoundError:
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"error: {args.input} is not a valid /research response: {exc}", file=sys.stderr)
        return 1

    written = save_report(report, args.output, args.format)
    print(f"Wrote {written.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
