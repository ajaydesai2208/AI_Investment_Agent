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
