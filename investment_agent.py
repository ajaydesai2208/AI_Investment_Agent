from __future__ import annotations

from typing import Optional
from datetime import datetime, timezone, date

import pandas as pd
import streamlit as st

# Our package modules
from ai_agent.settings import get_openai_key
from ai_agent.news import get_recent_news, format_news_digest, clear_news_cache
from ai_agent.prices import merge_for_compare
from ai_agent.metrics import snapshot, compare_table, to_fact_pack
from ai_agent.agent import build_agent
from ai_agent.prompts import build_prompt
from ai_agent.options import (
    options_snapshot,
    format_options_fact,
    OptionsSnapshot,
    list_expiries,
)
from ai_agent.export import sanitize_styling, build_markdown_package
from ai_agent.risk import sizing_summary_table

# ---------------- UI Boot ----------------

st.set_page_config(page_title="AI Investment Agent", page_icon="📈")
st.title("AI Investment Agent")
st.caption(
    "Compare two tickers using Yahoo Finance data, options context, and a GPT model. "
    "If OPENAI_API_KEY exists in .env/env it is used automatically; otherwise paste a key in the sidebar."
)

# ---------------- Sidebar ----------------

with st.sidebar:
    st.subheader("Authentication")
    current_key, source = get_openai_key()

    if current_key:
        st.success(f"OpenAI key loaded from {source}.")
        if st.button("Use a different key"):
            st.session_state["openai_api_key"] = ""
            st.experimental_rerun()

    if not current_key:
        typed = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if typed:
            st.session_state["openai_api_key"] = typed
            st.success("Key saved for this session.")
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Strategy Controls")
    hf_mode = st.checkbox("Enable hedge-fund-style analysis", value=True)
    lookback_days = st.slider("News lookback (days)", 1, 60, 30)
    max_news = st.slider("Max news per ticker", 5, 50, 20, step=5)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    account_equity = st.number_input("Account equity ($)", min_value=1000, value=25000, step=1000)

    st.caption("News sources")
    use_rss = st.checkbox("Yahoo Finance RSS", value=True)
    use_yf = st.checkbox("yfinance API", value=True)
    use_reuters = st.checkbox("Reuters RSS", value=True)  # NEW
    use_sec = st.checkbox("SEC EDGAR filings", value=True)  # NEW

    st.markdown("---")
    st.subheader("Options Expiry")
    expiry_mode = st.radio(
        "How to choose expiry?",
        ["Auto (nearest 7–45 DTE)", "Pick specific expiry"],
        index=0,
    )

    if st.button("Refresh news cache"):
        clear_news_cache()
        st.toast("News cache cleared.")

# ---------------- Ticker Inputs ----------------

col1, col2 = st.columns(2)
ticker_a = col1.text_input("Ticker A", value="NVDA").strip().upper()
ticker_b = col2.text_input("Ticker B", value="ANET").strip().upper()

# Helper to pick default index = first future expiry
def _default_expiry_index(exps: list[str]) -> int:
    if not exps:
        return 0
    today_d = date.today()
    for i, s in enumerate(exps):
        try:
            if pd.to_datetime(s).date() > today_d:
                return i
        except Exception:
            continue
    return max(0, len(exps) - 1)

# If user wants to pick, show per-ticker expiry dropdowns (after we have tickers)
expiry_a = None
expiry_b = None
if expiry_mode.startswith("Pick"):
    exps_a = list_expiries(ticker_a) if ticker_a else []
    exps_b = list_expiries(ticker_b) if ticker_b else []
    col3, col4 = st.columns(2)
    if exps_a:
        expiry_a = col3.selectbox(f"{ticker_a} expiry", exps_a, index=_default_expiry_index(exps_a))
    else:
        col3.write(f"• No expiries found for {ticker_a}")
    if exps_b:
        expiry_b = col4.selectbox(f"{ticker_b} expiry", exps_b, index=_default_expiry_index(exps_b))
    else:
        col4.write(f"• No expiries found for {ticker_b}")

# Chart controls
cc1, cc2, cc3 = st.columns([1.1, 1.1, 1])
timeframe = cc1.selectbox(
    "Chart timeframe",
    ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"],
    index=5,
)
normalize = cc2.checkbox("Normalize to 100", value=True)
show_vol = cc3.checkbox("Show volume (load only)", value=False)

go = st.button("Compare & Analyze")

# ---------------- Charts ----------------

st.markdown("### Price Charts")
chart_df = merge_for_compare(ticker_a, ticker_b, timeframe, normalize=normalize, show_vol=show_vol)
if chart_df.empty:
    st.info("No price data available for the selected timeframe.")
else:
    if normalize:
        st.caption("Index = 100 at the start of the selected period.")
    st.line_chart(chart_df, height=320, use_container_width=True)

# ---------------- Snapshot & Factors ----------------

with st.spinner("Loading snapshots & factors..."):
    a_snap = snapshot(ticker_a)
    b_snap = snapshot(ticker_b)

st.markdown("### Snapshot & Factors")
st.dataframe(compare_table(a_snap, b_snap), use_container_width=True)

# ---------------- Options Snapshot ----------------

def _fmt_num(x: Optional[float], pct: bool = False) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    return f"{v:.2f}%" if pct else f"{v:.2f}"

with st.spinner("Fetching options snapshot..."):
    # If mode is "Pick", pass the chosen expiry; else leave None to auto-pick (7–45 DTE window)
    opt_a: OptionsSnapshot = options_snapshot(ticker_a, expiry=expiry_a if expiry_mode.startswith("Pick") else None)
    opt_b: OptionsSnapshot = options_snapshot(ticker_b, expiry=expiry_b if expiry_mode.startswith("Pick") else None)

opt_rows = [
    ("Expiry (DTE)", f"{opt_a.expiry or '—'} ({opt_a.dte or '—'})", f"{opt_b.expiry or '—'} ({opt_b.dte or '—'})"),
    ("Spot", _fmt_num(opt_a.spot), _fmt_num(opt_b.spot)),
    ("ATM strike", _fmt_num(opt_a.atm_strike), _fmt_num(opt_b.atm_strike)),
    ("Call mid", _fmt_num(opt_a.call_mid), _fmt_num(opt_b.call_mid)),
    ("Put mid", _fmt_num(opt_a.put_mid), _fmt_num(opt_b.put_mid)),
    ("Straddle debit", _fmt_num(opt_a.straddle_debit), _fmt_num(opt_b.straddle_debit)),
    ("Implied move", _fmt_num(opt_a.implied_move_pct, pct=True), _fmt_num(opt_b.implied_move_pct, pct=True)),
    ("ATM IV (approx.)", _fmt_num(opt_a.atm_iv_pct, pct=True), _fmt_num(opt_b.atm_iv_pct, pct=True)),
]
opt_table = pd.DataFrame(opt_rows, columns=["Metric", ticker_a, ticker_b])

st.markdown("### Options Snapshot")
if expiry_mode.startswith("Pick"):
    st.caption("Using your selected expiries.")
else:
    st.caption("Auto mode: nearest expiry in a ~7–45 DTE window.")
st.dataframe(opt_table, use_container_width=True)

# ---------------- Position Sizing ----------------

with st.spinner("Computing sizing..."):
    size_a_df = sizing_summary_table(
        ticker=ticker_a,
        account_equity=float(account_equity),
        profile_name=risk_profile,
        spot=a_snap.get("price"),
        vol20_ann_pct=a_snap.get("vol20_ann_pct"),
        opt=opt_a,
    )
    size_b_df = sizing_summary_table(
        ticker=ticker_b,
        account_equity=float(account_equity),
        profile_name=risk_profile,
        spot=b_snap.get("price"),
        vol20_ann_pct=b_snap.get("vol20_ann_pct"),
        opt=opt_b,
    )

st.markdown("### Position Sizing (based on implied move & recent vol)")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(size_a_df.iloc[:, :2], use_container_width=True)
with c2:
    st.dataframe(size_b_df.iloc[:, :2], use_container_width=True)

def _sizing_fact_line(ticker: str, df: pd.DataFrame) -> str:
    """Extract a compact line from the sizing table for the prompt."""
    if df is None or df.empty:
        return f"{ticker}: sizing unavailable."
    sdf = df.iloc[:, :2]
    s = sdf.set_index(sdf.columns[0])["Value"]
    stop = s.get("Baseline stop %", "—")
    shares = s.get("Shares (approx.)", "—")
    contracts = s.get("Options contracts (≈ debit)", "—")
    return f"{ticker}: stop {stop}, shares {shares}, option_contracts {contracts}."

# ---------------- Helper ----------------

def validate_tickers(a: str, b: str) -> Optional[str]:
    if not a or not b:
        return "Please enter both tickers."
    if a == b:
        return "Please enter two different tickers."
    return None

# ---------------- Main Analysis ----------------

if go:
    if not current_key:
        st.error("No OpenAI key found. Add it in the sidebar or create a .env file with OPENAI_API_KEY.")
        st.stop()

    err = validate_tickers(ticker_a, ticker_b)
    if err:
        st.error(err)
        st.stop()

    # News context (respects source toggles, now including Reuters + SEC)
    news_kwargs = dict(use_rss=use_rss, use_yf=use_yf, use_reuters=use_reuters, use_sec=use_sec)
    news_a_all = get_recent_news(ticker_a, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
    news_b_all = get_recent_news(ticker_b, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
    news_a = news_a_all["merged"]
    news_b = news_b_all["merged"]

    if hf_mode:
        st.markdown("#### Recent News (context sent to the model)")
        with st.expander(f"{ticker_a} news ({len(news_a)})", expanded=False):
            st.caption(
                "Sources — "
                f"RSS: {len(news_a_all['by_source'].get('yahoo_rss', []))} | "
                f"yfinance: {len(news_a_all['by_source'].get('yfinance', []))} | "
                f"Reuters: {len(news_a_all['by_source'].get('reuters_rss', []))} | "
                f"SEC: {len(news_a_all['by_source'].get('sec_edgar', []))}"
            )
            st.write(format_news_digest(ticker_a, news_a))
        with st.expander(f"{ticker_b} news ({len(news_b)})", expanded=False):
            st.caption(
                "Sources — "
                f"RSS: {len(news_b_all['by_source'].get('yahoo_rss', []))} | "
                f"yfinance: {len(news_b_all['by_source'].get('yfinance', []))} | "
                f"Reuters: {len(news_b_all['by_source'].get('reuters_rss', []))} | "
                f"SEC: {len(news_b_all['by_source'].get('sec_edgar', []))}"
            )
            st.write(format_news_digest(ticker_b, news_b))

    # Fact pack for LLM (+ options + sizing)
    fact_pack = to_fact_pack(a_snap, b_snap)
    options_fact = "OPTIONS:\n" + format_options_fact(opt_a) + "\n" + format_options_fact(opt_b)
    sizing_fact = "SIZING (profile={}, equity=${:,}):\n{}\n{}".format(
        risk_profile, int(account_equity), _sizing_fact_line(ticker_a, size_a_df), _sizing_fact_line(ticker_b, size_b_df)
    )
    fact_pack_ext = fact_pack + "\n" + options_fact + "\n" + sizing_fact

    # Pass today's date (UTC) so model is time-aware
    today_iso = datetime.now(timezone.utc).date().isoformat()

    # Build prompt
    prompt = build_prompt(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        timeframe=timeframe,
        risk_profile=risk_profile,
        news_digest_a=format_news_digest(ticker_a, news_a),
        news_digest_b=format_news_digest(ticker_b, news_b),
        fact_pack=fact_pack_ext,
        hedge_fund_mode=hf_mode,
        today_str=today_iso,
    )

    with st.spinner("Analyzing like a hedge fund PM..."):
        try:
            agent = build_agent(current_key)
            result = agent.run(prompt, stream=False)
        except Exception as e:
            st.exception(e)
            st.stop()

    # ----- Display (sanitized to remove stray italics/bold) -----
    content = getattr(result, "content", None)
    if content is None:
        content = str(result)
    sanitized = sanitize_styling(content)

    st.markdown("### Investment Analysis Report:")
    st.markdown(sanitized)

    # ----- Export to Markdown -----
    metadata = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "timeframe": timeframe,
        "risk_profile": risk_profile,
        "lookback_days": str(lookback_days),
        "max_news": str(max_news),
    }
    fname, fbytes = build_markdown_package(
        report_markdown=sanitized,
        metadata=metadata,
        facts_block=fact_pack_ext,
        options_table=opt_table,
    )
    st.download_button(
        label="⬇️ Download report (.md)",
        data=fbytes,
        file_name=fname,
        mime="text/markdown",
        use_container_width=True,
    )
