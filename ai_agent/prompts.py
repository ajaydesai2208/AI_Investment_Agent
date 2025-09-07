from __future__ import annotations
from typing import Optional

__all__ = ["build_prompt"]


def build_prompt(
    ticker_a: str,
    ticker_b: str,
    timeframe: str,
    risk_profile: str,
    news_digest_a: str,
    news_digest_b: str,
    fact_pack: Optional[str] = None,
    hedge_fund_mode: bool = True,
    today_str: Optional[str] = None,
) -> str:
    """
    Build the final LLM instruction string.

    Temporal controls:
      - We pass TODAY into the prompt.
      - The model is instructed to USE ONLY dates present in FACTS/News
        and to never guess or use past dates as upcoming.
      - All dates must be ISO (YYYY-MM-DD).

    Styling controls:
      - No italics/bold; avoid '*' and '_' for emphasis.
    """
    today = today_str or "[TODAY-NOT-PROVIDED]"

    base = (
        f"Compare {ticker_a} vs {ticker_b} and produce a trader-ready report. "
        "Use the FACTS block provided (if any) and any tools you need. "
        "Return a well-structured markdown report."
    )

    style_rules = (
        "\n\n### STYLE RULES\n"
        "- Do NOT use italics or bold. Avoid * and _ characters for emphasis.\n"
        "- Write option types in UPPERCASE: CALL, PUT. Use plain spaces around slashes, e.g., CALL / PUT.\n"
        "- Write prices/levels as plain text numbers (no emphasis markers).\n"
        "- Prefer compact lists and tables; no decorative formatting.\n"
    )

    temporal_rules = (
        "\n### TEMPORAL RULES\n"
        f"- TODAY is {today}. Treat all time references relative to this date.\n"
        "- Use ONLY dates explicitly present in the FACTS or News sections.\n"
        "- Never guess exact dates. If a date is unknown, write 'TBD' or 'unknown'.\n"
        "- Do NOT present past dates as upcoming; explicitly mark items as past vs upcoming.\n"
        "- All dates must be ISO formatted (YYYY-MM-DD).\n"
    )

    if not hedge_fund_mode:
        return base + style_rules + temporal_rules

    hf = f"""
### CONTEXT
TODAY: {today}
Timeframe: {timeframe}
Risk profile: {risk_profile}

{ticker_a} news:
{news_digest_a}

{ticker_b} news:
{news_digest_b}

### FACTS (deterministic)
{fact_pack or "None provided"}

{temporal_rules}
{style_rules}
### STRICT OUTPUT FORMAT
1. **Executive Summary** – 5–8 bullets that capture edge and risk.
2. **Factor Dashboard** – table with Valuation (P/E, EV/EBITDA if available), Growth, Profitability,
   Quality (ROIC if available), Momentum (1M/3M/6M/1Y), Risk (beta/vol), Analyst View.
3. **Key News → Impact** – table: headline → positive / negative / unclear, plus 1-line rationale.
4. **Events & Catalysts** – upcoming earnings, product/regulatory, supply chain, M&A; how each affects the thesis.
   - Clearly label each catalyst as PAST or UPCOMING based on TODAY and the given dates.
5. **Scenarios** – bull / base / bear with drivers and rough probabilities.
6. **Trade Recommendations** – for EACH ticker:
   - **Directional:** LONG or SHORT with entry zone, stop, targets, invalidation. Tie to catalysts.
   - **Options examples:** e.g., call debit spread / put debit spread / calendar; specify example strikes and expiry
     (use logical nearby expiries/strikes if exact chain not provided) and why that structure fits the thesis.
   - **Why now / What kills this trade** – 2 lines.
7. **Position Framing** – how to think about sizing and risk in one short paragraph.
8. **Actionable Checklist** – levels, dates, and metrics to monitor next, using ISO dates from FACTS/News only.
"""
    return base + hf
