from __future__ import annotations

from textwrap import dedent


def build_prompt(
    *,
    ticker_a: str,
    ticker_b: str,
    timeframe: str,
    risk_profile: str,
    news_digest_a: str,
    news_digest_b: str,
    fact_pack: str,
    hedge_fund_mode: bool,
    today_str: str,
) -> str:
    """
    Returns a decision-forward analysis prompt.

    Inputs the app passes include:
      - fact_pack: combined metrics table, fundamental/momentum/vol summaries,
                   options snapshot (spot, ATM, straddle debit, implied move, IV),
                   and a sizing summary line per ticker.
      - news_digest_*: compact deduped news/filings lists (recent window).
      - timeframe: chart selection string (e.g., 1Y, YTD)
      - risk_profile: Conservative | Balanced | Aggressive
      - today_str: ISO date (UTC) for strict time awareness.

    Output spec below enforces a clean markdown structure with explicit trade plans.
    """

    sys_directives = dedent(f"""
    You are an elite hedge-fund portfolio manager writing an execution-ready note
    for two tickers: {ticker_a} and {ticker_b}. Today is {today_str} (UTC). Do NOT
    assume any other current date. When mentioning dates from news/filings, label
    them explicitly as PAST or UPCOMING based on {today_str}.

    Style principles (important):
    - Decision-centric. Provide concrete actions and levels (entries, stops, targets).
    - Be specific with numbers. If you compute a level from provided data, show the math.
    - Use clear section headers and markdown tables. Avoid fluff.
    - Do NOT use italic formatting anywhere. Do not use asterisks or underscores for emphasis.
      Use bold (**like this**) only for short labels, not for whole paragraphs.
    - Never include legal disclaimers.
    - Length is NOT constrained. Be thorough if the inputs warrant detail.

    Risk alignment:
    - Frame trade sizes/risk using the supplied "sizing" facts and the selected risk profile: {risk_profile}.
    - If data is missing for a calculation, state the assumption you make.

    Options guidance:
    - Use the options snapshot and implied move to suggest 1–2 options structures per ticker.
      Prefer simple structures first (debit/call/put spreads). Include estimated debit and worst-case loss.
    - Use the straddle debit and implied move percent to inform targets and invalidations.

    Pair logic:
    - Offer a relative-value / pairs idea if it makes sense (long one vs short the other),
      including rationale and risk triggers.

    Time awareness:
    - If the news digest mentions earnings or events, classify as PAST or UPCOMING and reflect it in the playbook.
    - Do not invent dates. Only use dates present in the inputs or derived logically from {today_str}.

    Output must be valid Markdown only (no HTML). Use headings (#, ##, ###) and pipe tables.
    Absolutely avoid italic markers.
    """)

    output_spec = dedent(f"""
    =========================
    OUTPUT SPEC (use exactly)
    =========================

    # Investment Analysis Report: {ticker_a} vs {ticker_b}

    ## 0. Trade Tickets (Top)
    Provide two concise, order-ready tickets FIRST — one for each ticker. Keep formatting consistent.

    ### {ticker_a} — Trade Ticket
    | Field | Value |
    |---|---|
    | Side | [LONG / SHORT / AVOID] |
    | Entry | [exact level] ([method: breakout/pullback/support/resistance]) |
    | Stop | [exact level] ([-% from entry]) |
    | Targets | T1 [level]; T2 [level] |
    | Size (shares approx.) | [from sizing facts; mention $ risk at stop] |
    | Options alt (if useful) | [structure; strikes & expiry; est. debit; max loss; breakeven] |
    | Time horizon | [days/weeks] |

    **Why now:** 1–2 bullets tied to catalysts/flows/levels.  
    **Invalidation:** exact triggers that kill the thesis.

    ### {ticker_b} — Trade Ticket
    | Field | Value |
    |---|---|
    | Side | [LONG / SHORT / AVOID] |
    | Entry | [exact level] ([method]) |
    | Stop | [exact level] ([-% from entry]) |
    | Targets | T1 [level]; T2 [level] |
    | Size (shares approx.) | [from sizing facts; mention $ risk at stop] |
    | Options alt (if useful) | [structure; strikes & expiry; est. debit; max loss; breakeven] |
    | Time horizon | [days/weeks] |

    **Why now:** 1–2 bullets.  
    **Invalidation:** exact triggers.

    ## 1. Execution Summary (fast read)
    - Primary stance for {ticker_a}: [BUY/LONG or SELL/SHORT or AVOID] with brief one-line rationale.
    - Primary stance for {ticker_b}: [BUY/LONG or SELL/SHORT or AVOID] with brief one-line rationale.
    - Pair idea (optional): [e.g., LONG {ticker_a} / SHORT {ticker_b}] with 1-line thesis.
    - Risk profile applied: {risk_profile}. Timeframe focus: {timeframe}.

    ## 2. Market Context Snapshot
    Provide 4–8 bullets on macro, sector, factor/flows that are relevant. Tie to both tickers.

    ## 3. Company Drilldown
    ### {ticker_a}
    - Business/segment quick view (2–4 bullets)
    - Key positives
    - Key negatives
    - Recent catalysts (PAST) and upcoming catalysts (UPCOMING) from the news digest

    ### {ticker_b}
    - Business/segment quick view (2–4 bullets)
    - Key positives
    - Key negatives
    - Recent catalysts (PAST) and upcoming catalysts (UPCOMING)

    ## 4. Quant & Tape Factors
    Use the provided metrics/returns/volatility from the facts pack.
    - Momentum & trend
    - Volatility regime (20d/annualized where provided)
    - Valuation quick take if present
    - Breadth/relative strength vs peer (infer from facts if available)

    ## 5. Options & Implied Move Readout
    Summarize for each ticker using the options snapshot (spot, ATM, straddle debit, implied move, IV).
    Provide a small table:

    | Ticker | Spot | Implied Move % | Straddle Debit | Approx ATM IV |
    |---|---:|---:|---:|---:|
    | {ticker_a} | [spot] | [imp%] | [debit] | [iv%] |
    | {ticker_b} | [spot] | [imp%] | [debit] | [iv%] |

    Explain what the implied move suggests for expected range into the selected expiry.

    ## 6. Trade Plans (per ticker)
    Provide BOTH a directional plan and an options plan per ticker. (These elaborate the tickets above.)

    ### {ticker_a} — Directional Plan
    - Entry: state price level and logic (breakout/pullback/support)
    - Stop: define numeric stop and logic (invalidations)
    - Targets: at least 2 targets
    - Sizing reference: use the provided sizing facts (stop %, shares approx., dollar risk)
    - Why now: link to news/flows/levels
    - Invalidation & what changes the thesis

    ### {ticker_a} — Options Plan
    - Preferred structure (e.g., Call Debit Spread, Put Debit Spread). State strikes and expiry.
    - Est. debit and max loss; payoff at target; breakeven
    - Why choose this over stock (IV, risk, capital efficiency)

    ### {ticker_b} — Directional Plan
    - Entry, Stop, Targets, Sizing reference, Why now, Invalidation

    ### {ticker_b} — Options Plan
    - Structure, strikes/expiry, debit/max loss, payoff, breakeven, rationale

    ## 7. Pair / Relative-Value Angle (optional)
    If appropriate, propose LONG one / SHORT the other with:
    - Trigger to enter (level or catalyst)
    - Stop on the spread or ratio
    - Target differential and expected drivers

    ## 8. Portfolio Framing & Risk
    - Suggested tilt (Growth vs Quality vs Value) and expected drawdown if invalidated
    - Correlation considerations between {ticker_a} and {ticker_b}
    - Hedging notes (index hedge or sector ETF if relevant)

    ## 9. Actionable Checklist (next 5–10 trading days)
    - Specific items to watch (prices, events, filings, news streams)
    - If-then reactions (e.g., “If {ticker_a} closes above X on volume, add Y%”)
    - Exit/roll rules for the options structures

    ## 10. Scenario Table
    Provide a simple table with three rows:

    | Scenario | {ticker_a} Plan | {ticker_b} Plan | Portfolio Action |
    |---|---|---|---|
    | Bull case | [concise plan] | [concise plan] | [add/trim/rotate] |
    | Base case | [concise plan] | [concise plan] | [hold/monitor]    |
    | Bear case | [concise plan] | [concise plan] | [hedge/exit]      |

    ## 11. Appendix — Source Facts
    Quote the key numbers you used (spot, implied move %, IV, straddle debit, vol/returns) so readers can audit the plan.
    """)

    # Everything the model needs to reason with
    data_block = dedent(f"""
    =========================
    INPUT DATA BLOCK (verbatim)
    =========================

    FACT PACK:
    {fact_pack}

    NEWS DIGEST — {ticker_a}:
    {news_digest_a}

    NEWS DIGEST — {ticker_b}:
    {news_digest_b}
    """)

    # Hedge-fund mode adds “go deeper” speed-lanes; otherwise still decision-forward
    depth_hint = (
        "Go as deep as needed; add microstructure or flow/color if the inputs suggest it."
        if hedge_fund_mode
        else "Be crisp and decision-forward; you can still use all inputs."
    )

    final_request = dedent(f"""
    =========================
    TASK
    =========================
    Using ONLY the information in the INPUT DATA BLOCK and your financial reasoning,
    produce the Markdown report exactly per the OUTPUT SPEC. {depth_hint}

    Calculation helpers you MAY use if levels are not explicit in inputs:
    - Expected move levels ≈ spot × (1 ± implied_move%).
    - Use straddle debit as a sanity check for the magnitude of a target.
    - If sizing lines are given (stop %, approximate shares/contracts), tie them to the plan and dollarize risk.
    - When suggesting strikes for spreads, anchor them around spot and the target/invalidations.
    - If dates are needed, only use dates you see in news/filings or today’s date {today_str}.
    - Never invent unknown numbers; state assumptions when necessary.

    Output valid Markdown only. Do not use italics. Keep headings, bullets, and tables clean.
    """)

    return "\n".join([sys_directives, output_spec, data_block, final_request])
