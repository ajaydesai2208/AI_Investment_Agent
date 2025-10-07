from __future__ import annotations

from typing import Tuple
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from ai_agent.metrics import to_fact_pack
from ai_agent.options import format_options_fact
from ai_agent.catalysts import format_catalysts_md
from ai_agent.prompts import build_prompt
from ai_agent.agent import build_agent
from ai_agent.settings import get_openai_key
from ai_agent.export import sanitize_styling, build_markdown_package
from ai_agent.news import get_recent_news, format_news_digest
from ai_agent.risk import sizing_summary_table
from ui.helpers import _sizing_fact_line, _fmt_num, _run_agent_with_retries


def run_analysis(
    *,
    ticker_a: str,
    ticker_b: str,
    a_snap: dict,
    b_snap: dict,
    opt_a,
    opt_b,
    cat_a,
    cat_b,
    pair,
    hf_mode: bool,
    lookback_days: int,
    max_news: int,
    risk_profile: str,
    account_equity: float,
    use_rss: bool,
    use_yf: bool,
    use_reuters: bool,
    use_sec: bool,
    status_box,
    progress_bar,
) -> Tuple[str, str, bytes]:
    if status_box:
        status_box.update(label="Fetching recent news & filings…", state="running")
    news_kwargs = dict(use_rss=use_rss, use_yf=use_yf, use_reuters=use_reuters, use_sec=use_sec)
    news_a_all = get_recent_news(ticker_a, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
    news_b_all = get_recent_news(ticker_b, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
    news_a = news_a_all["merged"]
    news_b = news_b_all["merged"]
    progress_bar.progress(30)

    if status_box:
        status_box.update(label="Compiling data, options, catalysts & pair stats…", state="running")
    fact_pack = to_fact_pack(a_snap, b_snap)
    options_fact = "OPTIONS:\n" + format_options_fact(opt_a) + "\n" + format_options_fact(opt_b)
    catalysts_block = "CATALYSTS:\n" + format_catalysts_md(ticker_a, cat_a) + "\n" + format_catalysts_md(ticker_b, cat_b)

    fund_md_a = st.session_state.get("fundamentals_md_a")  # optional precompute
    fund_md_b = st.session_state.get("fundamentals_md_b")
    if not fund_md_a or not fund_md_b:
        # keep identical behavior to original by building via metrics.format_fundamentals_md_extended
        from ai_agent.metrics import format_fundamentals_md_extended as format_fundamentals_md
        fund_md_a = format_fundamentals_md(ticker_a)
        fund_md_b = format_fundamentals_md(ticker_b)
    fundamentals_block = "FUNDAMENTALS:\n" + fund_md_a + "\n" + fund_md_b

    if pair and getattr(pair, "window_used", 0) > 0:
        ztxt = f"{pair.spread_zscore:.2f}" if pair.spread_zscore is not None else "—"
        suggestion = "Neutral; watch for ±2.0 extremes."
        if pair.spread_zscore is not None:
            if pair.spread_zscore <= -2.0:
                suggestion = f"Consider LONG {ticker_a} / SHORT {ticker_b} (z={pair.spread_zscore:.2f})."
            elif pair.spread_zscore >= 2.0:
                suggestion = f"Consider SHORT {ticker_a} / LONG {ticker_b} (z={pair.spread_zscore:.2f})."
        pair_block = (
            "PAIR:\n"
            f"- Beta(A on B): {pair.beta_ab:.2f} | Corr: {pair.corr_ab:.2f} | Hedge ratio: {pair.hedge_ratio:.2f}\n"
            f"- Spread z-score (60): {ztxt}\n"
            f"- Read: {suggestion}"
        )
    else:
        pair_block = "PAIR:\n- Insufficient overlapping history for pair analysis."

    sizing_fact_line_a = _sizing_fact_line(
        ticker_a,
        sizing_summary_table(
            ticker=ticker_a, account_equity=float(account_equity), profile_name=risk_profile,
            spot=a_snap.get("price"), vol20_ann_pct=a_snap.get("vol20_ann_pct"), opt=opt_a
        ),
    )
    sizing_fact_line_b = _sizing_fact_line(
        ticker_b,
        sizing_summary_table(
            ticker=ticker_b, account_equity=float(account_equity), profile_name=risk_profile,
            spot=b_snap.get("price"), vol20_ann_pct=b_snap.get("vol20_ann_pct"), opt=opt_b
        ),
    )
    sizing_fact = "SIZING (profile={}, equity=${:,}):\n{}\n{}".format(
        risk_profile, int(account_equity), sizing_fact_line_a, sizing_fact_line_b
    )

    fact_pack_ext = (
        fact_pack
        + "\n" + fundamentals_block
        + "\n" + options_fact
        + "\n" + catalysts_block
        + "\n" + pair_block
        + "\n" + sizing_fact
    )
    progress_bar.progress(65)

    if status_box:
        status_box.update(label="Building prompt…", state="running")
    today_iso = datetime.now(timezone.utc).date().isoformat()
    prompt = build_prompt(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        timeframe=st.session_state.get("timeframe_select", "1Y"),
        risk_profile=risk_profile,
        news_digest_a=format_news_digest(ticker_a, news_a),
        news_digest_b=format_news_digest(ticker_b, news_b),
        fact_pack=fact_pack_ext,
        hedge_fund_mode=hf_mode,
        today_str=today_iso,
    )
    progress_bar.progress(80)

    if status_box:
        status_box.update(label="Calling model…", state="running")
    agent = build_agent(get_openai_key()[0])
    result = _run_agent_with_retries(agent, prompt, max_retries=3)
    content = getattr(result, "content", None) or str(result)
    sanitized = sanitize_styling(content)
    progress_bar.progress(100)
    if status_box:
        status_box.update(label="Analysis complete.", state="complete")

    metadata = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "timeframe": st.session_state.get("timeframe_select", "1Y"),
        "risk_profile": risk_profile,
        "lookback_days": str(lookback_days),
        "max_news": str(max_news),
    }
    fname, fbytes = build_markdown_package(
        report_markdown=sanitized,
        metadata=metadata,
        facts_block=fact_pack_ext,
        options_table=pd.DataFrame(
            [
                ("Expiry (DTE)", f"{opt_a.expiry or '—'} ({opt_a.dte or '—'})", f"{opt_b.expiry or '—'} ({opt_b.dte or '—'})"),
                ("Spot", _fmt_num(opt_a.spot), _fmt_num(opt_b.spot)),
                ("ATM strike", _fmt_num(opt_a.atm_strike), _fmt_num(opt_b.atm_strike)),
                ("Call mid", _fmt_num(opt_a.call_mid), _fmt_num(opt_b.call_mid)),
                ("Put mid", _fmt_num(opt_a.put_mid), _fmt_num(opt_b.put_mid)),
                ("Straddle debit", _fmt_num(opt_a.straddle_debit)),
                ("Implied move", _fmt_num(opt_a.implied_move_pct, pct=True), _fmt_num(opt_b.implied_move_pct, pct=True)),
                ("ATM IV (approx.)", _fmt_num(opt_a.atm_iv_pct, pct=True), _fmt_num(opt_b.atm_iv_pct, pct=True)),
            ],
            columns=["Metric", ticker_a, ticker_b],
        ),
        theme="terminal",
    )
    return sanitized, fname, fbytes


# (imports consolidated at top)


