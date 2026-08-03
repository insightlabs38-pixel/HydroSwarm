"""Build the polished HydroSwarm technical report PDF from checked-in source and results."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#071827")
CYAN = colors.HexColor("#15C5D8")
BLUE = colors.HexColor("#3274D9")
INK = colors.HexColor("#172A3A")
MUTED = colors.HexColor("#5D7182")
PALE = colors.HexColor("#EAF4F7")
GREEN = colors.HexColor("#168568")


class ArchitectureFlow(Flowable):
    def __init__(self) -> None:
        super().__init__()
        self.width = 6.8 * inch
        self.height = 1.5 * inch

    def draw(self) -> None:
        canvas = self.canv
        labels = ["Evidence", "Hydraulics", "Hybrid fusion", "Uncertainty", "WNTR", "Human"]
        widths = [0.88, 1.0, 1.08, 1.0, 0.78, 0.82]
        x = 0.06 * inch
        y = 0.5 * inch
        for index, (label, width) in enumerate(zip(labels, widths, strict=True)):
            box_width = width * inch
            canvas.setFillColor(PALE if index not in {2, 4} else colors.HexColor("#D7F2F4"))
            canvas.setStrokeColor(CYAN if index in {2, 4} else BLUE)
            canvas.roundRect(x, y, box_width, 0.52 * inch, 7, fill=1, stroke=1)
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica-Bold", 7.6)
            canvas.drawCentredString(x + box_width / 2, y + 0.22 * inch, label)
            if index < len(labels) - 1:
                arrow_x = x + box_width
                canvas.setStrokeColor(MUTED)
                canvas.line(arrow_x + 2, y + 0.26 * inch, arrow_x + 10, y + 0.26 * inch)
                canvas.line(arrow_x + 6, y + 0.31 * inch, arrow_x + 10, y + 0.26 * inch)
                canvas.line(arrow_x + 6, y + 0.21 * inch, arrow_x + 10, y + 0.26 * inch)
                x = arrow_x + 12
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(0.08 * inch, 0.18 * inch, "Typed evidence and explicit uncertainty; no autonomous control path")


def _header_footer(canvas, document) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.34 * inch, width, 0.34 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(0.62 * inch, height - 0.22 * inch, "HYDROSWARM  |  TECHNICAL REPORT")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.62 * inch, 0.35 * inch, "Research decision support - simulation results are not field validation")
    canvas.drawRightString(width - 0.62 * inch, 0.35 * inch, f"{document.page}")
    canvas.restoreState()


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", escaped)
    return escaped


def _results_table(results: dict) -> Table:
    metrics = results.get("aggregate", results.get("metrics", {}))
    rows = [["Measured metric", "Value", "Interpretation"]]
    selected = [
        ("localization_top1_accuracy", "Top-1 localization"),
        ("true_source_probability", "True-source probability"),
        ("entropy_reduction_bits", "Entropy reduction"),
        ("exposure_reduction_mg", "Modeled exposure reduction"),
        ("latency_seconds", "Golden latency"),
        ("logical_cache_hit_rate", "Logical cache hit rate"),
    ]
    for key, label in selected:
        value = metrics.get(key, "see JSON")
        if isinstance(value, dict):
            value = value.get("mean", value)
        if isinstance(value, float):
            rendered = f"{value:,.4g}"
        else:
            rendered = str(value)
        rows.append([label, rendered, "Frozen WNTR regression"])
    table = Table(rows, colWidths=[2.35 * inch, 1.25 * inch, 2.55 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CDD5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def build(source: Path, results_path: Path, target: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=31,
        leading=34, textColor=colors.white, alignment=TA_LEFT, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle", parent=styles["Heading2"], fontSize=15, leading=19,
        textColor=CYAN, alignment=TA_LEFT, spaceAfter=22,
    ))
    styles.add(ParagraphStyle(
        "CoverBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.15,
        leading=13.2, textColor=colors.HexColor("#C6D7E3"), spaceAfter=7.5,
    ))
    styles.add(ParagraphStyle(
        "CoverHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11,
        leading=14, textColor=CYAN, spaceBefore=9, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        "H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17,
        leading=21, textColor=NAVY, spaceBefore=10, spaceAfter=6, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11,
        leading=14, textColor=BLUE, spaceBefore=9, spaceAfter=5, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5,
        leading=12.0, textColor=INK, spaceAfter=6.0, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "Bulletx", parent=styles["Bodyx"], leftIndent=14, firstLineIndent=-8,
        bulletIndent=3, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "Referencex", parent=styles["Bulletx"], fontSize=6.8, leading=8.0,
        spaceAfter=1.0,
    ))
    text = source.read_text(encoding="utf-8")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lines = text.splitlines()
    title = lines[0].removeprefix("# ")
    subtitle = lines[1].removeprefix("## ")

    target.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(
        str(target), pagesize=letter, leftMargin=0.68 * inch, rightMargin=0.68 * inch,
        topMargin=0.62 * inch, bottomMargin=0.62 * inch, title="HydroSwarm Technical Report",
        author="HydroSwarm project",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="main")
    document.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=_header_footer))
    story: list[Flowable] = []

    cover = Table([[Paragraph(_inline(title), styles["ReportTitle"])],
                   [Paragraph(_inline(subtitle), styles["ReportSubtitle"])],
                   [Paragraph("Physics-first source localization, active evidence selection, and exact response verification", styles["CoverBody"])],
                   [Spacer(1, 0.45 * inch)],
                   [ArchitectureFlow()],
                   [Spacer(1, 0.55 * inch)],
                   [Paragraph("REVERIE HACKS 2026  /  SOFTWARE DEVELOPMENT", styles["CoverHeading"])],
                   [Paragraph("Version 0.1.0 | August 2026", styles["CoverBody"])],
                   [Spacer(1, 0.4 * inch)],
                   [Paragraph("Safety boundary", styles["CoverHeading"])],
                   [Paragraph("Research decision support only. No chemistry identification, no autonomous control, and no field action without qualified human approval.", styles["CoverBody"])]],
                  colWidths=[6.8 * inch])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR", (0, 2), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 26),
        ("RIGHTPADDING", (0, 0), (-1, -1), 26),
        ("TOPPADDING", (0, 0), (-1, 0), 36),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 28),
    ]))
    story.extend([Spacer(1, 0.42 * inch), cover, PageBreak()])

    paragraph: list[str] = []
    current_heading = ""

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["Bodyx"]))
            paragraph.clear()

    abstract_index = lines.index("## Abstract")
    for line in lines[abstract_index:]:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("## "):
            flush()
            heading = stripped[3:]
            current_heading = heading
            story.append(Paragraph(_inline(heading), styles["H1x"]))
            if heading == "3. System architecture":
                story.extend([ArchitectureFlow(), Spacer(1, 8)])
            if heading == "12. Measured results":
                story.extend([_results_table(results), Spacer(1, 9)])
        elif stripped.startswith("### "):
            flush()
            story.append(Paragraph(_inline(stripped[4:]), styles["H2x"]))
        elif re.match(r"^\d+\. ", stripped):
            flush()
            story.append(Paragraph(_inline(stripped), styles["Bodyx"]))
        elif stripped.startswith("- "):
            flush()
            style = styles["Referencex"] if current_heading == "19. References" else styles["Bulletx"]
            story.append(Paragraph(_inline(stripped[2:]), style, bulletText="-"))
        else:
            paragraph.append(stripped)
    flush()
    document.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/technical_report.md"))
    parser.add_argument("--results", type=Path, default=Path("reports/results/evaluation_results.json"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/HydroSwarm_Technical_Report.pdf"))
    args = parser.parse_args()
    build(args.source, args.results, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
