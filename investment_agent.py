from __future__ import annotations

from typing import Optional, Tuple
from datetime import datetime, timezone, date
import time
import re
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from ui.page import setup_page, inject_global_css, apply_high_contrast_from_state
from ui.background import ensure_bg_refresher_started
from ui.state import init_session_state

# Our package modules
from ai_agent.catalysts import clear_catalyst_cache
from ai_agent.settings import get_openai_key
from ai_agent.news import get_recent_news, format_news_digest, clear_news_cache
from ai_agent.prices import merge_for_compare, get_spark_series
from ai_agent.metrics import (
    snapshot,
    compare_table,
    to_fact_pack,
    fundamentals_table_extended as fundamentals_table,          # NEW
    format_fundamentals_md_extended as format_fundamentals_md,  # NEW
)
from ai_agent.agent import build_agent
from ai_agent.prompts import build_prompt
from ai_agent.options import (
    options_snapshot,
    format_options_fact,
    OptionsSnapshot,
    list_expiries,
)
from ai_agent.export import sanitize_styling, build_markdown_package, build_pdf_report
from ai_agent.risk import sizing_summary_table, build_trade_plan
from ai_agent.catalysts import get_catalysts, format_catalysts_md
from ai_agent.quant import compute_pair_stats, slope_percent
from ui.helpers import (
    _fmt_num,
    _is_valid_ticker,
    _default_expiry_index,
    _trend_label,
    _regime_label,
    _regime_css_class,
    _infer_direction_for,
    _parse_wait_secs,
    _sizing_fact_line,
)
from ai_agent.tickets import build_tickets_for_ticker, tickets_to_csv_bytes, build_tickets_from_strategy
from ai_agent.greeks import atm_greeks_table
from ai_agent.strategies import suggest_strategies, plans_to_markdown
from ai_agent.playbook import build_event_playbook_for_ticker, playbook_to_markdown
from ui.state import init_session_state
from ui.sidebar import render_sidebar
from ui.tickers import render_tickers_and_expiry
from ui.preload import preload_all
from ui.stat_bar import render_stat_bar, render_sparklines

# ---------------- UI Boot ----------------
setup_page()
inject_global_css()
apply_high_contrast_from_state()

# High-contrast handled via apply_high_contrast_from_state()


# ---------------- Helpers (shared) ----------------

# helpers imported from ui.helpers


# Start background cache refresher once per session
ensure_bg_refresher_started()


# moved to ui.helpers


# moved to ui.helpers


# moved to ui.helpers


# moved to ui.helpers


# moved to ui.helpers


# moved to ui.helpers


# moved to ui.helpers


    


# moved to ui.helpers


# ---------------- Title & Session ----------------

st.title("AI Investment Agent")
st.caption(
    "Compare two tickers using Yahoo Finance data, options context, catalysts, and a GPT model — with Reuters + SEC filings."
)

init_session_state()


# ---------------- Sidebar ----------------
current_key, source, hf_mode, lookback_days, max_news, risk_profile, account_equity, use_rss, use_yf, use_reuters, use_sec, expiry_mode = render_sidebar()


# ---------------- Tickers & Expiries ----------------
ticker_a, ticker_b, expiry_a, expiry_b, go, tickers_ok = render_tickers_and_expiry(expiry_mode)
if not tickers_ok:
    st.stop()

# A visible status/progress bar right under the button (global, works from any tab)
status_container = st.container()


# ---------------- Snapshots, Options, Catalysts, Pair (preload for tabs) ----------------
a_snap, b_snap, opt_a, opt_b, cat_a, cat_b, cat_md_a, cat_md_b, pair, spark_a, spark_b, trend_a, trend_b, regime_a, regime_b = preload_all(
    ticker_a, ticker_b, expiry_a, expiry_b, expiry_mode
)
st.session_state["cat_md_a"] = cat_md_a
st.session_state["cat_md_b"] = cat_md_b
st.session_state["account_equity"] = account_equity


# ---------------- Uniform Ticker Stat Bar ----------------
render_stat_bar(
    ticker_a=ticker_a,
    ticker_b=ticker_b,
    opt_a=opt_a,
    opt_b=opt_b,
    trend_a=trend_a,
    trend_b=trend_b,
    regime_a=regime_a,
    regime_b=regime_b,
    risk_profile=risk_profile,
    account_equity=account_equity,
)

# --- Tiny 3M Sparklines under the stat bar ---
render_sparklines(ticker_a=ticker_a, ticker_b=ticker_b, spark_a=spark_a, spark_b=spark_b)


# ---------------- Global Status Placeholder (ABOVE tabs) ----------------
# This placeholder reserves space above tabs for the status bar
status_placeholder = st.empty()

# Set analysis_in_progress state BEFORE tabs render so shimmer can show
# Only set if we're starting a NEW analysis (button clicked and no existing report)
if go and not st.session_state.get("report_markdown"):
    st.session_state["analysis_in_progress"] = True
elif not go:
    # Clear the flag if button is not clicked (prevents shimmer from sticking)
    st.session_state["analysis_in_progress"] = False


# ---------------- Tabs ----------------

tab_overview, tab_news, tab_options, tab_sizing, tab_scenarios, tab_report = st.tabs(
    ["Overview", "News", "Options", "Sizing", "Scenarios", "Report"]
)

# --- Overview tab ---
with tab_overview:
    from ui.tabs.overview import render_overview_tab
    render_overview_tab(ticker_a=ticker_a, ticker_b=ticker_b, a_snap=a_snap, b_snap=b_snap, cat_md_a=cat_md_a, cat_md_b=cat_md_b, pair=pair)

# --- News tab ---
with tab_news:
    from ui.tabs.news import render_news_tab
    render_news_tab(ticker_a=ticker_a, ticker_b=ticker_b, go=go, hf_mode=hf_mode, lookback_days=lookback_days, max_news=max_news, use_rss=use_rss, use_yf=use_yf, use_reuters=use_reuters, use_sec=use_sec)

with tab_options:
    from ui.tabs.options import render_options_tab
    render_options_tab(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        opt_a=opt_a,
        opt_b=opt_b,
        pair=pair,
        trend_a=trend_a,
        trend_b=trend_b,
        regime_a=regime_a,
        regime_b=regime_b,
            risk_profile=risk_profile,
        expiry_mode=expiry_mode,
    )

with tab_sizing:
    from ui.tabs.sizing import render_sizing_tab
    render_sizing_tab(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        a_snap=a_snap,
        b_snap=b_snap,
        opt_a=opt_a,
        opt_b=opt_b,
            risk_profile=risk_profile,
        account_equity=account_equity,
        pair=pair,
        trend_a=trend_a,
        trend_b=trend_b,
    )

# --- Scenarios tab (stock & ATM option P/L ladders) ---
with tab_scenarios:
    from ui.tabs.scenarios import render_scenarios_tab
    render_scenarios_tab(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        a_snap=a_snap,
        b_snap=b_snap,
        opt_a=opt_a,
        opt_b=opt_b,
        pair=pair,
        trend_a=trend_a,
        trend_b=trend_b,
    )


# --- Report tab (shows shimmer if running, or displays stored result) ---
with tab_report:
    # Check if analysis is currently running
    if st.session_state.get("analysis_in_progress", False):
        # Show shimmer loading effect
        from ui.components.shimmer import render_report_shimmer
        render_report_shimmer()
    else:
        # Render the report if it exists
        from ui.tabs.report import render_report_tab
        render_report_tab(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            a_snap=a_snap,
            b_snap=b_snap,
            opt_a=opt_a,
            opt_b=opt_b,
            cat_a=cat_a,
            cat_b=cat_b,
            trend_a=trend_a,
            trend_b=trend_b,
            regime_a=regime_a,
            regime_b=regime_b,
            risk_profile=risk_profile,
            lookback_days=lookback_days,
            max_news=max_news,
            pair=pair,
            size_a_df=locals().get("size_a_df"),
            size_b_df=locals().get("size_b_df"),
        )


# ---------------- Analysis Logic (OUTSIDE tabs, runs globally) ----------------

from ui.analysis_runner import run_analysis

if go:
    if not current_key:
        status_placeholder.error("No OpenAI key found. Add it in the sidebar or create a .env file with OPENAI_API_KEY.")
    else:
        # Use the placeholder that's ABOVE the tabs to show status
        with status_placeholder.container():
            try:
                status = st.status("Starting analysis…", expanded=True)
            except Exception:
                status = None
            progress = st.progress(0)
            try:
                report_md, fname, fbytes = run_analysis(
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    a_snap=a_snap,
                    b_snap=b_snap,
                    opt_a=opt_a,
                    opt_b=opt_b,
                    cat_a=cat_a,
                    cat_b=cat_b,
                    pair=pair,
                    hf_mode=hf_mode,
                    lookback_days=lookback_days,
                    max_news=max_news,
                    risk_profile=risk_profile,
                    account_equity=float(account_equity),
                    use_rss=use_rss,
                    use_yf=use_yf,
                    use_reuters=use_reuters,
                    use_sec=use_sec,
                    status_box=status,
                    progress_bar=progress,
                )
                st.session_state["report_markdown"] = report_md
                st.session_state["export_fname"] = fname
                st.session_state["export_bytes"] = fbytes
                st.session_state["analysis_in_progress"] = False
                st.toast("Analysis ready — open the Report tab ✅")
            except Exception as e:
                if status: status.update(label="Analysis failed.", state="error")
                st.exception(e)
                st.session_state["analysis_in_progress"] = False
