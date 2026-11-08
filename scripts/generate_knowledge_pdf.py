"""(Optional) fallback generator for the AquaTech knowledge base.

The real knowledge base PDFs are already provided under ``knowledge/``
(Remote Work, Healthcare Benefits, Leave Policy). This script is only a
convenience: it regenerates an equivalent single consolidated PDF under
``samples/`` using the SAME fictional facts, so the project stays reproducible
if the provided files are ever lost. It deliberately writes to ``samples/`` and
NOT to ``knowledge/`` to avoid conflicting with / duplicating the provided docs.

The facts below mirror the provided documents exactly.

Run:  python scripts/generate_knowledge_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples"
OUT_FILE = OUT_DIR / "aquatech_employee_policies_sample.pdf"

# Facts mirror the provided knowledge/*.pdf documents exactly.
# (heading, [paragraphs / bullet-lists]) -- bullet lists are tuples ("bullets", [...])
SECTIONS = [
    (
        "AquaTech Corporation - Remote Work Policy",
        [
            "AquaTech Corporation allows employees to work remotely up to 3 days a week. Core hours "
            "are 10:00 AM to 3:00 PM EST, during which all employees must be online and available "
            "for meetings.",
            "Employees are provided a one-time stipend of $500 to set up their home office. Hardware "
            "(laptop, monitor) must be requested through the IT portal.",
        ],
    ),
    (
        "AquaTech Corporation - Healthcare Benefits",
        [
            "AquaTech Corporation provides comprehensive healthcare coverage through BlueCross "
            "BlueShield. The plan covers 90% of in-network medical expenses after a $1,000 "
            "deductible. Dental and vision are also fully covered.",
            "Mental health benefits include up to 12 free therapy sessions per year via our partner "
            "network, BetterMinds. Gym memberships are reimbursed up to $50 per month.",
        ],
    ),
    (
        "AquaTech Corporation - Leave and Time Off",
        [
            "Employees at AquaTech are entitled to 20 days of Paid Time Off (PTO) annually, which "
            "accrues proportionally each pay period. In addition, there are 10 paid company holidays.",
            "Sick leave is unlimited but requires a doctor's note if taking more than 3 consecutive "
            "days off. Maternity and paternity leave are fully paid for 16 weeks and 8 weeks, "
            "respectively.",
        ],
    ),
]


def build_pdf() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_FILE),
        pagesize=LETTER,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        title="AquaTech Corporation - Employee Policies",
        author="AquaTech HR Department",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=20, spaceAfter=4, textColor="#16607a"
    )
    subtitle_style = ParagraphStyle(
        "Sub", parent=styles["Normal"], fontSize=10, textColor="#666666", spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "H", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6,
        textColor="#0d3b4f",
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
        spaceAfter=8,
    )

    flow = [
        Paragraph("AquaTech Corporation", title_style),
        Paragraph("Employee Policies &mdash; Consolidated Sample (mirrors the provided knowledge base)", subtitle_style),
    ]

    for heading, blocks in SECTIONS:
        flow.append(Paragraph(heading, heading_style))
        for block in blocks:
            if isinstance(block, tuple) and block[0] == "bullets":
                items = [ListItem(Paragraph(b, body_style), leftIndent=10) for b in block[1]]
                flow.append(ListFlowable(items, bulletType="bullet", start="•"))
                flow.append(Spacer(1, 6))
            else:
                flow.append(Paragraph(block, body_style))

    doc.build(flow)
    return OUT_FILE


if __name__ == "__main__":
    path = build_pdf()
    size_kb = path.stat().st_size / 1024
    print(f"Created knowledge base PDF: {path} ({size_kb:.1f} KB)")
