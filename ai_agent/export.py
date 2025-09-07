from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

__all__ = [
    "sanitize_styling",
    "dataframe_to_markdown",
    "build_markdown_package",
]


# ---------------- Formatting helpers ----------------

def sanitize_styling(md: str) -> str:
    """
    Remove stray emphasis that causes weird italics/bold in Streamlit.
    - Strips single *word* and _word_ styles anywhere in the text.
    - Leaves bullet markers like "- " or "* " at line starts alone.
    - Also strips **bold** and __bold__.
    """
    if not md:
        return md

    # Remove **bold** and __bold__
    md = re.sub(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", r"\1", md)
    md = re.sub(r"__(?!\s)(.+?)(?<!\s)__", r"\1", md)

    # Remove *italics* and _italics_
    md = re.sub(r"\*(?!\s|\*)([^*\n]+)\*", r"\1", md)
    md = re.sub(r"_(?!\s|_)([^_\n]+)_", r"\1", md)

    return md


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a small DataFrame to a GitHub-flavored Markdown table.
    (Avoids extra dependencies.)
    """
    if df is None or df.empty:
        return "_No data_\n"

    # Ensure all values are strings
    df_str = df.astype(str)

    header = "| " + " | ".join(df_str.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df_str.columns) + " |"
    rows = ["| " + " | ".join(df_str.iloc[i].tolist()) + " |" for i in range(len(df_str))]

    return "\n".join([header, sep] + rows) + "\n"


# ---------------- Export builder ----------------

def build_markdown_package(
    *,
    report_markdown: str,
    metadata: Dict[str, str],
    facts_block: Optional[str] = None,
    options_table: Optional[pd.DataFrame] = None,
) -> Tuple[str, bytes]:
    """
    Compose a single .md file that includes:
      - Title & timestamp
      - Metadata (tickers, timeframe, risk)
      - Optional FACTS/OPTIONS sections
      - The model's report (sanitized to remove italics/bold markers)

    Returns: (filename, file_bytes)
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    tka = metadata.get("ticker_a", "TKA")
    tkb = metadata.get("ticker_b", "TKB")
    timeframe = metadata.get("timeframe", "")
    risk_profile = metadata.get("risk_profile", "")
    lookback = metadata.get("lookback_days", "")
    max_news = metadata.get("max_news", "")

    pieces = []

    pieces.append(f"# AI Investment Report: {tka} vs {tkb}\n")
    pieces.append(f"_Generated: {now}_\n")
    pieces.append("## Parameters\n")
    pieces.append(f"- **Timeframe:** {timeframe}")
    pieces.append(f"- **Risk profile:** {risk_profile}")
    if lookback:
        pieces.append(f"- **News lookback (days):** {lookback}")
    if max_news:
        pieces.append(f"- **Max news per ticker:** {max_news}")
    pieces.append("")

    if facts_block:
        pieces.append("## Facts & Context")
        # Keep as fenced block so it’s visually distinct
        pieces.append("```text")
        pieces.append(sanitize_styling(facts_block).strip())
        pieces.append("```")
        pieces.append("")

    if options_table is not None and not options_table.empty:
        pieces.append("## Options Snapshot")
        pieces.append(dataframe_to_markdown(options_table))
        pieces.append("")

    pieces.append("## Analysis")
    pieces.append(sanitize_styling(report_markdown).strip())
    pieces.append("")

    full_md = "\n".join(pieces)
    filename = f"{tka}_{tkb}_investment_report.md"

    return filename, full_md.encode("utf-8")


# === PDF export =====================================================
import io
import html
from typing import Optional, Dict, Any, List

import pandas as pd

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def _as_table(dataframe: Optional[pd.DataFrame], col_widths: Optional[List[int]] = None):
    if dataframe is None or dataframe.empty:
        return None
    data = [list(dataframe.columns)] + dataframe.astype(str).values.tolist()
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf6ea")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0b4f0b")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bcdcbc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fbf7")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _md_to_paras(md: str, styles) -> List[Paragraph]:
    """
    Very small markdown-to-Paragraph converter:
      - #, ##, ### headings
      - bullet lines starting with -, *, •
      - everything else as body text
    """
    paras: List[Paragraph] = []
    if not md:
        return paras
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
            paras.append(Spacer(1, 6))
            continue

        txt = html.escape(line.lstrip(" -•*"))
        if line.startswith("###"):
            paras.append(Paragraph(html.escape(line[3:].strip()), styles["Heading3"]))
        elif line.startswith("##"):
            paras.append(Paragraph(html.escape(line[2:].strip()), styles["Heading2"]))
        elif line.startswith("#"):
            paras.append(Paragraph(html.escape(line[1:].strip()), styles["Heading1"]))
        elif line.lstrip().startswith(("-", "*", "•")):
            # simple bullet
            paras.append(Paragraph(f"&#8226; {txt}", styles["Bullet"]))
        else:
            paras.append(Paragraph(txt, styles["BodyText"]))
    return paras


def build_pdf_report(
    *,
    title: str,
    report_markdown: str,
    metadata: Dict[str, Any],
    options_table: Optional[pd.DataFrame] = None,
    fundamentals_a: Optional[pd.DataFrame] = None,
    fundamentals_b: Optional[pd.DataFrame] = None,
    catalysts_md: Optional[str] = None,
    pair_text_md: Optional[str] = None,
    sizing_text_md: Optional[str] = None,
    # NEW: extras
    selected_strategies_md: Optional[str] = None,
    event_playbook_md: Optional[str] = None,
    scenarios_a_df: Optional[pd.DataFrame] = None,
    scenarios_b_df: Optional[pd.DataFrame] = None,
    tickets_df: Optional[pd.DataFrame] = None,
) -> tuple[str, bytes]:
    """
    Build a single PDF bytes object containing:
      - Title & metadata
      - Main Markdown report (converted to paragraphs)
      - Options table, Fundamentals tables
      - Catalysts, Pair notes, Sizing notes
      - (NEW) Selected Strategies, Event Playbook
      - (NEW) Scenarios tables (A & B) and Tickets preview (appendix)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=36, rightMargin=36, topMargin=48, bottomMargin=48,
        title=title,
        author="AI Investment Agent",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10, textColor=colors.HexColor("#333333")))
    story: List = []

    # Title
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 6))

    # Metadata table
    if metadata:
        meta_rows = [[k, str(v)] for k, v in metadata.items()]
        meta_df = pd.DataFrame(meta_rows, columns=["Key", "Value"])
        meta_tbl = _as_table(meta_df, col_widths=[120, 360])
        if meta_tbl:
            story.append(meta_tbl)
            story.append(Spacer(1, 12))

    # Main Report
    story.append(Paragraph("Investment Analysis Report", styles["Heading2"]))
    story += _md_to_paras(report_markdown or "", styles)
    story.append(Spacer(1, 12))

    # Options
    if options_table is not None and not options_table.empty:
        story.append(Paragraph("Options Snapshot", styles["Heading2"]))
        tbl = _as_table(options_table)
        if tbl: story.append(tbl)
        story.append(Spacer(1, 12))

    # Fundamentals
    if fundamentals_a is not None and not fundamentals_a.empty:
        story.append(Paragraph("Fundamentals — A", styles["Heading2"]))
        tbl_a = _as_table(fundamentals_a)
        if tbl_a: story.append(tbl_a)
        story.append(Spacer(1, 8))
    if fundamentals_b is not None and not fundamentals_b.empty:
        story.append(Paragraph("Fundamentals — B", styles["Heading2"]))
        tbl_b = _as_table(fundamentals_b)
        if tbl_b: story.append(tbl_b)
        story.append(Spacer(1, 12))

    # Catalysts
    if catalysts_md:
        story.append(Paragraph("Catalyst Radar", styles["Heading2"]))
        story += _md_to_paras(catalysts_md, styles)
        story.append(Spacer(1, 12))

    # Pair notes
    if pair_text_md:
        story.append(Paragraph("Pair Analyzer", styles["Heading2"]))
        story += _md_to_paras(pair_text_md, styles)
        story.append(Spacer(1, 12))

    # Sizing notes
    if sizing_text_md:
        story.append(Paragraph("Sizing Notes", styles["Heading2"]))
        story += _md_to_paras(sizing_text_md, styles)
        story.append(Spacer(1, 12))

    # --- NEW: Selected Strategies
    if selected_strategies_md:
        story.append(Paragraph("Selected Strategies", styles["Heading2"]))
        story += _md_to_paras(selected_strategies_md, styles)
        story.append(Spacer(1, 12))

    # --- NEW: Event Playbook
    if event_playbook_md:
        story.append(Paragraph("Event Playbook", styles["Heading2"]))
        story += _md_to_paras(event_playbook_md, styles)
        story.append(Spacer(1, 12))

    # Start appendices on a new page if we have any tables below
    if (scenarios_a_df is not None and not scenarios_a_df.empty) or \
       (scenarios_b_df is not None and not scenarios_b_df.empty) or \
       (tickets_df is not None and not tickets_df.empty):
        story.append(PageBreak())
        story.append(Paragraph("Appendix", styles["Heading1"]))
        story.append(Spacer(1, 8))

    # --- NEW: Scenarios (A & B)
    if scenarios_a_df is not None and not scenarios_a_df.empty:
        story.append(Paragraph("Scenarios — A", styles["Heading2"]))
        tbl_sca = _as_table(scenarios_a_df)
        if tbl_sca: story.append(tbl_sca)
        story.append(Spacer(1, 8))
    if scenarios_b_df is not None and not scenarios_b_df.empty:
        story.append(Paragraph("Scenarios — B", styles["Heading2"]))
        tbl_scb = _as_table(scenarios_b_df)
        if tbl_scb: story.append(tbl_scb)
        story.append(Spacer(1, 12))

    # --- NEW: Tickets preview
    if tickets_df is not None and not tickets_df.empty:
        story.append(Paragraph("Tickets Preview", styles["Heading2"]))
        tbl_t = _as_table(tickets_df)
        if tbl_t: story.append(tbl_t)
        story.append(Spacer(1, 12))

    # Build & return
    doc.build(story)
    fname = f"{title.lower().replace(' ', '_')}.pdf"
    return fname, buf.getvalue()
