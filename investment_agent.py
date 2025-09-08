from __future__ import annotations

from typing import Optional, Tuple
from datetime import datetime, timezone, date
import time
import re
import threading
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

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
from ai_agent.tickets import build_tickets_for_ticker, tickets_to_csv_bytes, build_tickets_from_strategy
from ai_agent.scenario import build_scenarios, build_payoff_grid
from ai_agent.greeks import atm_greeks_table
from ai_agent.strategies import suggest_strategies, plans_to_markdown
from ai_agent.playbook import build_event_playbook_for_ticker, playbook_to_markdown

# ---------------- UI Boot ----------------

st.set_page_config(page_title="AI Investment Agent", page_icon="📈", layout="wide")

# Global CSS: green/black, sharp buttons, card polish, uniform stat cards
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px;}
      h1, h2, h3, h4 {letter-spacing: .2px;}

      /* Cards & tables */
      .card {
        background: var(--secondary-background-color, #11181F);
        border: 1px solid rgba(0,200,5,.12);
        border-radius: 14px;
        padding: 16px 16px;
        box-shadow: 0 0 0 1px rgba(0,200,5,.04), 0 10px 30px rgba(0,0,0,.35);
      }
      div[data-testid="stDataFrame"] thead tr th {
        background: rgba(0,200,5,.06);
        border-bottom: 1px solid rgba(0,200,5,.18);
      }

      /* Buttons: sharp, interactive */
      .stButton>button, .stDownloadButton>button {
        border-radius: 12px;
        border: 1px solid #00C805;
        background: linear-gradient(180deg, #00D60A 0%, #00B205 100%);
        color: #051207;
        font-weight: 700; letter-spacing: .2px;
        padding: 0.55rem 0.9rem;
        transition: transform .06s ease, box-shadow .2s ease, filter .2s ease;
        box-shadow: 0 8px 24px rgba(0,200,5,.15), 0 0 0 1px rgba(0,200,5,.18) inset;
        width: 100%;
        min-height: 90px; /* ensure uniform height across wrapped/one-line labels */
        display: inline-flex; align-items: center; justify-content: center; /* vertical centering */
        text-align: center; white-space: normal; line-height: 1.15; /* allow wrapping but keep height consistent */
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 30px rgba(0,200,5,.28), 0 0 0 1px rgba(0,200,5,.2) inset;
        filter: brightness(1.03);
      }
      .stButton>button:active, .stDownloadButton>button:active {
        transform: translateY(0);
        filter: brightness(.98);
      }

      /* Tabs */
      .stTabs [data-baseweb="tab"] {
        text-transform: uppercase; letter-spacing: .4px; font-weight: 700;
      }
      .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, rgba(0,200,5,.35), rgba(0,200,5,.08));
      }

      /* Uniform stat bar at top */
      .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 8px 0 10px 0;
      }
      .stat-card {
        background: var(--secondary-background-color, #11181F);
        border: 1px solid rgba(0,200,5,.14);
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 78px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 0 0 1px rgba(0,200,5,.05);
      }
      .stat-title { font-weight: 800; letter-spacing: .3px; margin-bottom: 6px; }
      .pills { display:flex; gap:.5rem; flex-wrap:wrap; }
      .pill {
        padding: .12rem .55rem;
        border-radius: 999px;
        border: 1px solid rgba(0,200,5,.25);
        background: rgba(0,200,5,.08);
        font-size: .86rem;
        color: #AEE8B1;
      }
      /* Regime-specific pill accents */
      .pill.neutral { border-color: rgba(180,180,180,.35); background: rgba(180,180,180,.10); color: #D9DEE3; }
      .pill.calm { border-color: rgba(0,200,5,.35); background: rgba(0,200,5,.10); color: #AEE8B1; }
      .pill.stressed { border-color: rgba(255,60,60,.35); background: rgba(255,60,60,.10); color: #FFC1C1; }
      /* Sticky stat bar (removed) */
      /* High-contrast overrides */
      body.high-contrast .card { border-color: rgba(255,255,255,.28); box-shadow: 0 0 0 1px rgba(255,255,255,.12); }
      body.high-contrast div[data-testid="stDataFrame"] thead tr th { border-bottom-color: rgba(255,255,255,.35); }
      body.high-contrast .stat-card { border-color: rgba(255,255,255,.28); box-shadow: 0 0 0 1px rgba(255,255,255,.10); }
      body.high-contrast .pill { border-color: rgba(255,255,255,.35); color: #F3F6F9; }
      /* Skeleton loaders */
      @keyframes shimmer { 0% { background-position: -400px 0 } 100% { background-position: 400px 0 } }
      .skel-table { border: 1px solid rgba(255,255,255,.06); border-radius: 10px; padding: 12px; background: rgba(255,255,255,.02); }
      .skel-row { height: 14px; margin: 10px 0; border-radius: 6px; background: #121820; background-image: linear-gradient(90deg, rgba(255,255,255,0.05) 0, rgba(255,255,255,0.15) 20%, rgba(255,255,255,0.05) 40%); background-size: 800px 100%; animation: shimmer 1.2s infinite; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply high-contrast class on body when toggled (reads from session state)
try:
    if bool(st.session_state.get("ui_high_contrast", False)):
        st.markdown(
            """
            <script>
            document.documentElement.classList.add('high-contrast');
            document.body.classList.add('high-contrast');
            </script>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <script>
            document.documentElement.classList.remove('high-contrast');
            document.body.classList.remove('high-contrast');
            </script>
            """,
            unsafe_allow_html=True,
        )
except Exception:
    pass


# ---------------- Helpers (shared) ----------------

def _fmt_num(x: Optional[float], pct: bool = False, money: bool = False) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    if pct:
        return f"{v:.2f}%"
    if money:
        return f"${v:,.2f}"
    return f"{v:.2f}"


# Background cache refresher: clears news/catalysts every 3 hours
def _background_cache_refresher():
    while True:
        try:
            clear_news_cache()
            clear_catalyst_cache()
        except Exception:
            pass
        time.sleep(3 * 60 * 60)

# Start once per process/session
if not st.session_state.get("bg_refresher_started"):
    try:
        threading.Thread(target=_background_cache_refresher, daemon=True).start()
    except Exception:
        pass
    st.session_state["bg_refresher_started"] = True


def _is_valid_ticker(s: Optional[str]) -> bool:
    """Basic client-side ticker validation: A–Z/0–9/./- up to 10 chars."""
    if not s:
        return False
    try:
        t = str(s).strip().upper()
    except Exception:
        return False
    return bool(re.fullmatch(r"[A-Z0-9.\-]{1,10}", t))


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


def _trend_label(series: pd.Series) -> str:
    s = slope_percent(series, lookback=60) if series is not None and len(series) >= 3 else None
    if s is None:
        return "—"
    if s > 0.02:
        return "Up"
    if s < -0.02:
        return "Down"
    return "Sideways"


def _regime_label(implied_move_pct: Optional[float], vol20_ann_pct: Optional[float], dte: Optional[int]) -> str:
    try:
        imp = float(implied_move_pct) if implied_move_pct is not None else None
        ann = float(vol20_ann_pct) / 100.0 if vol20_ann_pct is not None else None  # convert % -> decimal
        days = int(dte) if dte is not None and dte > 0 else 21  # fallback ~1m
        if imp is None or ann is None or ann <= 0:
            return "—"
        # realized move over DTE (as percent)
        daily = ann / np.sqrt(252.0)
        realized_move_pct = daily * np.sqrt(days) * 100.0
        if realized_move_pct <= 0:
            return "—"
        ratio = imp / realized_move_pct
        if ratio >= 1.4:
            return "Stressed"
        if ratio <= 0.8:
            return "Calm"
        return "Normal"
    except Exception:
        return "—"


def _regime_css_class(regime_label: Optional[str]) -> str:
    """Return CSS class for regime pill: 'calm' | 'stressed' | 'neutral' (default)."""
    s = (regime_label or "").strip().lower()
    if s.startswith("calm"):
        return "calm"
    if s.startswith("stress"):
        return "stressed"
    return "neutral"


def _infer_direction_for(
    *,
    this_ticker: str,
    other_ticker: str,
    report_text: Optional[str],
    pair_obj,
    trend_label: Optional[str],
    is_a: bool,
) -> str:
    """
    Heuristics to pick LONG/SHORT:
    1) If the report explicitly says LONG/SHORT THIS_TICKER → use that.
    2) Else, use pair z-score extremes (±2):  z<<0 => LONG A/SHORT B; z>>0 => SHORT A/LONG B.
    3) Else, use Trend badge (Up->long, Down->short).
    4) Fallback to long.
    """
    txt = (report_text or "").upper()
    t = (this_ticker or "").upper()

    # Explicit in report?
    if t and f"LONG {t}" in txt and f"SHORT {t}" not in txt:
        return "long"
    if t and f"SHORT {t}" in txt and f"LONG {t}" not in txt:
        return "short"

    # Pair z-score rule
    try:
        z = float(getattr(pair_obj, "spread_zscore", None))
    except Exception:
        z = None
    if z is not None:
        if z <= -2.0:
            return "long" if is_a else "short"
        if z >= 2.0:
            return "short" if is_a else "long"

    # Trend badge fallback
    if isinstance(trend_label, str):
        tl = trend_label.lower()
        if tl == "up":
            return "long"
        if tl == "down":
            return "short"

    return "long"


def _parse_wait_secs(msg: str) -> Optional[float]:
    m = re.search(r"try again in ([0-9.]+)s", msg, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _run_agent_with_retries(agent, prompt: str, max_retries: int = 3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return agent.run(prompt, stream=False)
        except Exception as e:
            last_err = e
            msg = str(e)
            wait = _parse_wait_secs(msg) or (2 * attempt)
            st.warning(f"Model rate-limited or transient error (attempt {attempt}/{max_retries}). Retrying in {wait:.1f}s…")
            time.sleep(wait)
    raise last_err


def _sizing_fact_line(ticker: str, df: pd.DataFrame) -> str:
    """
    Build a concise, model-friendly summary line that includes BOTH
    baseline (implied-move) and ATR-based sizing details.
    """
    if df is None or df.empty:
        return f"{ticker}: sizing unavailable."

    try:
        s = df.set_index("Metric")["Value"]
    except Exception:
        sdf = df.iloc[:, :2]
        s = sdf.set_index(sdf.columns[0])[sdf.columns[1]]

    base_stop = s.get("Baseline stop %", "—")
    base_shares = s.get("Shares (approx.)", "—")
    base_contracts = s.get("Options contracts (≈ debit)", "—")

    atr_stop = s.get("ATR stop %", "—")
    atr_shares = s.get("ATR shares (approx.)", "—")
    atr_risk = s.get("ATR dollar risk at stop", "—")

    return (
        f"{ticker}: baseline_stop {base_stop}, shares {base_shares}, option_contracts {base_contracts}; "
        f"atr_stop {atr_stop}, atr_shares {atr_shares}, atr_risk {atr_risk}."
    )


# ---------------- Title & Session ----------------

st.title("AI Investment Agent")
st.caption(
    "Compare two tickers using Yahoo Finance data, options context, catalysts, and a GPT model — with Reuters + SEC filings."
)

for key, default in [
    ("report_markdown", None),
    ("export_fname", None),
    ("export_bytes", None),
    ("selected_strategies_md", []),
    ("tickets_extra", []),
    ("ui_high_contrast", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- Sidebar ----------------

with st.sidebar:
    st.subheader("Authentication")
    current_key, source = get_openai_key()

    if current_key:
        st.success(f"OpenAI key loaded from {source}.")
        if st.button("Use a different key"):
            st.session_state["openai_api_key"] = ""
            st.rerun()

    if not current_key:
        typed = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if typed:
            st.session_state["openai_api_key"] = typed
            st.success("Key saved for this session.")
            st.rerun()

    st.markdown("---")
    st.subheader("Display")
    high_contrast = st.toggle(
        "High-contrast mode",
        value=bool(st.session_state.get("ui_high_contrast", False)),
        help="Brighter text and borders for dark theme",
    )
    st.session_state["ui_high_contrast"] = bool(high_contrast)
    st.subheader("Strategy Controls")
    hf_mode = st.checkbox("Enable hedge-fund-style analysis", value=True)
    lookback_days = st.slider("News lookback (days)", 1, 60, 30)
    max_news = st.slider("Max news per ticker", 5, 50, 20, step=5)
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
    account_equity = st.number_input("Account equity ($)", min_value=1000, value=25000, step=1000)

    st.caption("News sources")
    use_rss = st.checkbox("Yahoo Finance RSS", value=True)
    use_yf = st.checkbox("yfinance API", value=True)
    use_reuters = st.checkbox("Reuters RSS", value=True)
    use_sec = st.checkbox("SEC EDGAR filings", value=True)

    st.markdown("---")
    st.subheader("Options Expiry")
    expiry_mode = st.radio(
        "How to choose expiry?",
        ["Auto (nearest 7–45 DTE)", "Pick specific expiry"],
        index=0,
    )

    if st.button("Refresh caches"):
        clear_news_cache()
        clear_catalyst_cache()
        st.toast("News & catalysts caches cleared.")


# ---------------- Tickers & Expiries ----------------

col1, col2 = st.columns(2)
ticker_a = col1.text_input("Ticker A", value="NVDA").strip().upper()
ticker_b = col2.text_input("Ticker B", value="ANET").strip().upper()

# Basic validity check before enabling action
tickers_ok = _is_valid_ticker(ticker_a) and _is_valid_ticker(ticker_b)

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

# Action button (disabled until valid)
go = st.button("Compare & Analyze", use_container_width=True, disabled=not tickers_ok)

if not tickers_ok:
    st.caption("Enter two valid tickers (A–Z, 0–9, '.', '-'). Example: AAPL, MSFT")
    # Early guard to avoid downstream loaders when a ticker is empty/invalid
    st.info("Enter two valid tickers to load data.")
    st.stop()

# A visible status/progress bar right under the button (global, works from any tab)
status_container = st.container()


# ---------------- Snapshots, Options, Catalysts, Pair (preload for tabs) ----------------

with st.spinner("Loading snapshots & factors..."):
    try:
        a_snap = snapshot(ticker_a)
        b_snap = snapshot(ticker_b)
    except Exception:
        a_snap, b_snap = {}, {}
        st.markdown(
            """
            <div class="skel-table">
              <div class="skel-row" style="width:40%"></div>
              <div class="skel-row" style="width:85%"></div>
              <div class="skel-row" style="width:90%"></div>
              <div class="skel-row" style="width:70%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.spinner("Fetching options snapshot..."):
    try:
        opt_a: OptionsSnapshot = options_snapshot(ticker_a, expiry=expiry_a if expiry_mode.startswith("Pick") else None)
        opt_b: OptionsSnapshot = options_snapshot(ticker_b, expiry=expiry_b if expiry_mode.startswith("Pick") else None)
    except Exception:
        from types import SimpleNamespace
        opt_a = SimpleNamespace(spot=None, implied_move_pct=None, dte=None, expiry=None, atm_iv_pct=None, call_mid=None, put_mid=None, atm_strike=None, straddle_debit=None)
        opt_b = SimpleNamespace(spot=None, implied_move_pct=None, dte=None, expiry=None, atm_iv_pct=None, call_mid=None, put_mid=None, atm_strike=None, straddle_debit=None)
        st.markdown(
            """
            <div class=\"skel-table\">
              <div class=\"skel-row\" style=\"width:50%\"></div>
              <div class=\"skel-row\" style=\"width:92%\"></div>
              <div class=\"skel-row\" style=\"width:88%\"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.spinner("Scanning catalysts (earnings & SEC filings)..."):
    try:
        cat_a = get_catalysts(ticker_a)
        cat_b = get_catalysts(ticker_b)
        cat_md_a = format_catalysts_md(ticker_a, cat_a)
        cat_md_b = format_catalysts_md(ticker_b, cat_b)
    except Exception:
        cat_a = cat_b = {"sec_filings": [], "asof": None}
        cat_md_a = cat_md_b = ""
        st.markdown(
            """
            <div class=\"skel-table\">
              <div class=\"skel-row\" style=\"width:30%\"></div>
              <div class=\"skel-row\" style=\"width:70%\"></div>
              <div class=\"skel-row\" style=\"width:60%\"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Pair analyzer guarded so it never crashes the app
from types import SimpleNamespace
with st.spinner("Analyzing pair relationships…"):
    try:
        pair = compute_pair_stats(ticker_a, ticker_b, lookback_days=365, window=60)
    except Exception as e:
        st.warning(f"Pair analyzer temporarily unavailable: {e}")
        pair = SimpleNamespace(
            beta_ab=None, corr_ab=None, hedge_ratio=None,
            spread_last=None, spread_zscore=None,
            window_used=0, a_close_last=None, b_close_last=None
        )

# ---- Spark series & Trend/Regime badges ----
spark_a = get_spark_series(ticker_a, months=3, normalize=True)
spark_b = get_spark_series(ticker_b, months=3, normalize=True)

trend_a = _trend_label(spark_a)
trend_b = _trend_label(spark_b)
regime_a = _regime_label(opt_a.implied_move_pct, a_snap.get("vol20_ann_pct"), opt_a.dte)
regime_b = _regime_label(opt_b.implied_move_pct, b_snap.get("vol20_ann_pct"), opt_b.dte)


# ---------------- Uniform Ticker Stat Bar ----------------

st.markdown(
    f"""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-title">{ticker_a}</div>
        <div class="pills">
          <span class="pill">Spot {_fmt_num(opt_a.spot, money=True)}</span>
          <span class="pill">Imp Move {_fmt_num(opt_a.implied_move_pct, pct=True)}</span>
          <span class="pill">DTE {opt_a.dte or "—"}</span>
          <span class="pill">Trend {trend_a}</span>
          <span class="pill {_regime_css_class(regime_a)}">Regime {regime_a}</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">{ticker_b}</div>
        <div class="pills">
          <span class="pill">Spot {_fmt_num(opt_b.spot, money=True)}</span>
          <span class="pill">Imp Move {_fmt_num(opt_b.implied_move_pct, pct=True)}</span>
          <span class="pill">DTE {opt_b.dte or "—"}</span>
          <span class="pill">Trend {trend_b}</span>
          <span class="pill {_regime_css_class(regime_b)}">Regime {regime_b}</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Profile</div>
        <div class="pills"><span class="pill">{risk_profile}</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Equity</div>
        <div class="pills"><span class="pill">${int(account_equity):,}</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Tiny 3M Sparklines under the stat bar ---
sc1, sc2 = st.columns(2)
with sc1:
    if spark_a is not None and not spark_a.empty:
        st.caption(f"{ticker_a} — 3M spark (Index=100)")
        st.line_chart(pd.DataFrame(spark_a), height=90, use_container_width=True)
    else:
        st.caption(f"{ticker_a} — no data")
with sc2:
    if spark_b is not None and not spark_b.empty:
        st.caption(f"{ticker_b} — 3M spark (Index=100)")
        st.line_chart(pd.DataFrame(spark_b), height=90, use_container_width=True)
    else:
        st.caption(f"{ticker_b} — no data")


# ---------------- Tabs ----------------

tab_overview, tab_news, tab_options, tab_sizing, tab_scenarios, tab_report = st.tabs(
    ["Overview", "News", "Options", "Sizing", "Scenarios", "Report"]
)

# --- Overview tab ---
with tab_overview:
    st.markdown("#### Price Chart", help="Select timeframe and normalize from the controls below.")
    cc1, cc2, cc3 = st.columns([1.1, 1.1, 1])
    timeframe = cc1.selectbox(
        "Chart timeframe",
        ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"],
        index=5,
        key="timeframe_select",
    )
    normalize = cc2.checkbox("Normalize to 100", value=True, key="normalize_check")
    show_vol = cc3.checkbox("Show volume (load only)", value=False, key="showvol_check")

    chart_df = merge_for_compare(ticker_a, ticker_b, timeframe, normalize=normalize, show_vol=show_vol)
    if chart_df.empty:
        st.info("No price data available for the selected timeframe.")
    else:
        if normalize:
            st.caption("Index = 100 at the start of the selected period.")
        st.line_chart(chart_df, height=360, use_container_width=True)

    # (Timeframe chips removed)

    st.markdown("#### Snapshot & Factors")
    st.dataframe(compare_table(a_snap, b_snap), use_container_width=True)
    # NEW — Fundamentals (TTM) tables
    st.markdown("#### Fundamentals (TTM)")
    ff1, ff2 = st.columns(2)
    with ff1:
        st.dataframe(fundamentals_table(ticker_a), use_container_width=True)
    with ff2:
        st.dataframe(fundamentals_table(ticker_b), use_container_width=True)

    st.markdown("#### Catalyst Radar")
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.markdown(cat_md_a)
    with ccol2:
        st.markdown(cat_md_b)

    # --- Pair Analyzer ---
    st.markdown("#### Pair Analyzer (beta-hedged spread)")
    def _pair_suggestion(z: Optional[float]) -> str:
        if z is None:
            return "Insufficient data"
        if z <= -2.0:
            return f"**Consider LONG {ticker_a} / SHORT {ticker_b}** (z={z:.2f})"
        if z >= 2.0:
            return f"**Consider SHORT {ticker_a} / LONG {ticker_b}** (z={z:.2f})"
        return f"Neutral (z={z:.2f}); watch for ±2.0 extremes"

    if pair.window_used == 0:
        st.info("Not enough overlapping history to compute pair stats.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta (A on B)", f"{pair.beta_ab:.2f}" if pair.beta_ab is not None else "—")
        c2.metric("Correlation", f"{pair.corr_ab:.2f}" if pair.corr_ab is not None else "—")
        c3.metric("Hedge ratio", f"{pair.hedge_ratio:.2f}" if pair.hedge_ratio is not None else "—")
        c4.metric("Spread z-score", f"{pair.spread_zscore:.2f}" if pair.spread_zscore is not None else "—")
        st.caption(_pair_suggestion(pair.spread_zscore))

# --- News tab ---
with tab_news:
    st.markdown("#### Recent News & Filings (used as model context)")
    st.caption(
        "Enable/disable sources in the sidebar • Adjust lookback/max items • Click **Compare & Analyze** to fetch latest. (cached ~15m; SEC map ~24h)"
    )
    if go:
        news_kwargs = dict(use_rss=use_rss, use_yf=use_yf, use_reuters=use_reuters, use_sec=use_sec)
        news_a_all = get_recent_news(ticker_a, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
        news_b_all = get_recent_news(ticker_b, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
        news_a = news_a_all["merged"]
        news_b = news_b_all["merged"]

        # Compute a simple news digest hash to detect updates
        try:
            digest_str = str(news_a) + "|" + str(news_b)
            digest = hashlib.sha256(digest_str.encode("utf-8")).hexdigest()
            prev = st.session_state.get("news_digest_hash")
            st.session_state["news_digest_hash"] = digest
            if prev and prev != digest:
                st.info("News updated since your last analysis. Consider re-running the report to reflect changes.")
        except Exception:
            pass

        coln1, coln2 = st.columns(2)
        with coln1:
            with st.expander(f"{ticker_a} news ({len(news_a)})", expanded=True):
                st.caption(
                    "Sources — "
                    f"RSS: {len(news_a_all['by_source'].get('yahoo_rss', []))} | "
                    f"yfinance: {len(news_a_all['by_source'].get('yfinance', []))} | "
                    f"Reuters: {len(news_a_all['by_source'].get('reuters_rss', []))} | "
                    f"SEC: {len(news_a_all['by_source'].get('sec_edgar', []))}"
                )
                st.write(format_news_digest(ticker_a, news_a))
        with coln2:
            with st.expander(f"{ticker_b} news ({len(news_b)})", expanded=True):
                st.caption(
                    "Sources — "
                    f"RSS: {len(news_b_all['by_source'].get('yahoo_rss', []))} | "
                    f"yfinance: {len(news_b_all['by_source'].get('yfinance', []))} | "
                    f"Reuters: {len(news_b_all['by_source'].get('reuters_rss', []))} | "
                    f"SEC: {len(news_b_all['by_source'].get('sec_edgar', []))}"
                )
                st.write(format_news_digest(ticker_b, news_b))
    else:
        st.info("Press **Compare & Analyze** to fetch the latest news & filings with your current source toggles.")

# --- Options tab ---
with tab_options:
    st.markdown("#### Options Snapshot")
    if expiry_mode.startswith("Pick"):
        st.caption("Using your selected expiries.")
    else:
        st.caption("Auto mode: nearest expiry in a ~7–45 DTE window. (cached ~5m)")

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
    st.dataframe(opt_table, use_container_width=True)
    # --- NEW: ATM Greeks (per contract) ---
    st.markdown("#### ATM Greeks (per contract)")

    # Infer directions (same heuristic used elsewhere)
    _rep_txt = st.session_state.get("report_markdown")
    _dir_a = _infer_direction_for(
        this_ticker=ticker_a,
        other_ticker=ticker_b,
        report_text=_rep_txt,
        pair_obj=pair,
        trend_label=trend_a,
        is_a=True,
    )
    _dir_b = _infer_direction_for(
        this_ticker=ticker_b,
        other_ticker=ticker_a,
        report_text=_rep_txt,
        pair_obj=pair,
        trend_label=trend_b,
        is_a=False,
    )

    gcol1, gcol2 = st.columns(2)

    with gcol1:
        st.markdown(f"**{ticker_a}** — assumed `{_dir_a.upper()}`")
        try:
            g_a = atm_greeks_table(
                spot=opt_a.spot,
                atm_strike=opt_a.atm_strike,
                dte=opt_a.dte,
                atm_iv_pct=opt_a.atm_iv_pct,
                call_mid=opt_a.call_mid,
                put_mid=opt_a.put_mid,
                direction=_dir_a,
                per_contract=True,
            )
        except Exception:
            g_a = pd.DataFrame({"Metric": [], "Value": []})
        if g_a.empty:
            st.info("Greeks unavailable for this expiry.")
        else:
            st.dataframe(g_a, use_container_width=True)

    with gcol2:
        st.markdown(f"**{ticker_b}** — assumed `{_dir_b.upper()}`")
        try:
            g_b = atm_greeks_table(
                spot=opt_b.spot,
                atm_strike=opt_b.atm_strike,
                dte=opt_b.dte,
                atm_iv_pct=opt_b.atm_iv_pct,
                call_mid=opt_b.call_mid,
                put_mid=opt_b.put_mid,
                direction=_dir_b,
                per_contract=True,
            )
        except Exception:
            g_b = pd.DataFrame({"Metric": [], "Value": []})
        if g_b.empty:
            st.info("Greeks unavailable for this expiry.")
        else:
            st.dataframe(g_b, use_container_width=True)

    # --- NEW: Strategy Picker ---
    st.markdown("#### Strategy Picker")

    # We already computed these directions above for Greeks:
    # _dir_a, _dir_b; and we have trend_a/trend_b + regime_a/regime_b + risk_profile.

    # Init compare selection state
    if "strategy_compare" not in st.session_state:
        st.session_state["strategy_compare"] = []

    try:
        plans_a = suggest_strategies(
            ticker=ticker_a,
            direction=_dir_a,
            opt=opt_a,
            trend_label=trend_a,
            regime_label=regime_a,
            risk_profile=risk_profile,
        )
    except Exception:
        plans_a = []

    try:
        plans_b = suggest_strategies(
            ticker=ticker_b,
            direction=_dir_b,
            opt=opt_b,
            trend_label=trend_b,
            regime_label=regime_b,
            risk_profile=risk_profile,
        )
    except Exception:
        plans_b = []

    sa, sb = st.columns(2)

    def _render_plans(side_title: str, ticker: str, plans: list):
        st.markdown(f"**{side_title}**")
        if not plans:
            st.info("No strategy suggestions available for current inputs.")
            return
        for i, p in enumerate(plans, start=1):
            with st.expander(f"{i}. {p.name}", expanded=(i == 1)):
                st.caption(p.rationale)
                st.caption(f"_When to use:_ {p.when_to_use}")
                # Summary stats (DTE, Net, Max/Loss/Gain, BEs, RR, Capital)
                try:
                    st.dataframe(p.header_stats(), use_container_width=True)
                except Exception:
                    st.write(p.header_stats())
                # Legs table
                st.markdown("**Legs**")
                try:
                    st.dataframe(p.to_dataframe(), use_container_width=True)
                except Exception:
                    st.write(p.to_dataframe())
                # Actions
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Add to Report", key=f"add_plan_{ticker}_{i}"):
                        md = plans_to_markdown([p])
                        lst = st.session_state.get("selected_strategies_md") or []
                        lst.append(md)
                        st.session_state["selected_strategies_md"] = lst
                        st.success(f"Added to Report: {ticker} — {p.name} (see Report tab).")
                with c2:
                    if st.button("Send to Tickets", key=f"send_plan_{ticker}_{i}"):
                        try:
                            extra = build_tickets_from_strategy(p, fallback_ticker=ticker)
                        except Exception:
                            extra = []
                        if extra:
                            buf = list(st.session_state.get("tickets_extra") or [])
                            buf.extend(extra)
                            st.session_state["tickets_extra"] = buf
                            st.success(f"Added {len(extra)} leg(s) to Tickets: {ticker} — {p.name} (Report tab).")
                        else:
                            st.info("No legs could be parsed for this plan.")
                with c3:
                    sel: list = st.session_state.get("strategy_compare") or []
                    is_selected = any((x.get("ticker") == ticker and x.get("name") == p.name) for x in sel)
                    at_limit = (len(sel) >= 2) and not is_selected
                    if not is_selected:
                        if st.button("Select for Compare", key=f"sel_{ticker}_{i}", disabled=at_limit):
                            entry = {
                                "ticker": ticker,
                                "name": p.name,
                                "dte": getattr(p, "dte", None),
                                "debit_credit": getattr(p, "debit_credit", None),
                                "est_cost": getattr(p, "est_cost", None),
                                "est_credit": getattr(p, "est_credit", None),
                                "max_loss": getattr(p, "max_loss", None),
                                "max_gain": getattr(p, "max_gain", None),
                                "breakevens": list(getattr(p, "breakevens", []) or []),
                                "rr_ratio": getattr(p, "rr_ratio", None),
                                "capital_req": getattr(p, "capital_req", None),
                            }
                            st.session_state["strategy_compare"] = sel + [entry]
                            st.rerun()
                    else:
                        if st.button("Remove from Compare", key=f"unsel_{ticker}_{i}"):
                            st.session_state["strategy_compare"] = [x for x in sel if not (x.get("ticker") == ticker and x.get("name") == p.name)]
                            st.rerun()

    with sa:
        _render_plans(f"{ticker_a} — {_dir_a.upper()} (Trend: {trend_a}, Regime: {regime_a})", ticker_a, plans_a)
    with sb:
        _render_plans(f"{ticker_b} — {_dir_b.upper()} (Trend: {trend_b}, Regime: {regime_b})", ticker_b, plans_b)

    # Comparison card (up to 2 strategies)
    comp = st.session_state.get("strategy_compare") or []
    if comp:
        st.markdown("#### Strategy Comparison")

        def _fmt_money(v):
            try:
                return f"${float(v):,.2f}" if v is not None else "—"
            except Exception:
                return "—"

        def _net_text(x):
            dc = (x.get("debit_credit") or "").upper()
            if dc == "CREDIT":
                return f"CREDIT {_fmt_money(x.get('est_credit'))}"
            if dc == "DEBIT":
                return f"DEBIT {_fmt_money(x.get('est_cost'))}"
            return "—"

        def _to_col(series):
            return {
                "Ticker": series.get("ticker"),
                "Strategy": series.get("name"),
                "DTE": series.get("dte") if series.get("dte") is not None else "—",
                "Net": _net_text(series),
                "Max Loss": _fmt_money(series.get("max_loss")),
                "Max Gain": _fmt_money(series.get("max_gain")),
                "Breakeven(s)": ", ".join([f"{b:.2f}" for b in (series.get("breakevens") or [])]) if series.get("breakevens") else "—",
                "R:R": (f"{float(series.get('rr_ratio')):.2f}" if series.get("rr_ratio") is not None else "—"),
                "Capital": _fmt_money(series.get("capital_req")),
            }

        import pandas as _pd  # local alias to avoid top-level import
        cols = [_to_col(x) for x in comp[:2]]
        df = _pd.DataFrame(cols)
        st.dataframe(df, use_container_width=True)
        cclear, _ = st.columns([1, 4])
        with cclear:
            if st.button("Clear selection", key="clear_strategy_compare"):
                st.session_state["strategy_compare"] = []
                st.rerun()

# --- Sizing tab ---
with tab_sizing:
    st.markdown("#### Position Sizing (based on implied move & recent vol)")
    st.caption("Includes **Baseline** (implied-move–anchored) and **ATR-based** suggestions.")
    # Risk budget badge (profile % of equity)
    try:
        prof = (risk_profile or "Balanced").lower()
        frac = 0.005 if prof.startswith("cons") else (0.02 if prof.startswith("aggr") else 0.01)
        rb = float(account_equity) * float(frac)
        st.markdown(f"**Risk budget:** ${rb:,.0f} (at {int(frac*100)}%)")
    except Exception:
        st.caption("Risk budget unavailable.")
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
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(size_a_df.iloc[:, :2], use_container_width=True)
    with c2:
        st.dataframe(size_b_df.iloc[:, :2], use_container_width=True)

    sizing_fact_a = _sizing_fact_line(ticker_a, size_a_df)
    sizing_fact_b = _sizing_fact_line(ticker_b, size_b_df)

    # --- Trade Plan (stock: entry/stop/targets & R:R) ---
    st.markdown("#### Trade Plan (Stock)")

    # Infer directions for both tickers (same heuristic we use elsewhere)
    _rep_txt = st.session_state.get("report_markdown")
    _dir_a_sz = _infer_direction_for(
        this_ticker=ticker_a,
        other_ticker=ticker_b,
        report_text=_rep_txt,
        pair_obj=pair,
        trend_label=trend_a,
        is_a=True,
    )
    _dir_b_sz = _infer_direction_for(
        this_ticker=ticker_b,
        other_ticker=ticker_a,
        report_text=_rep_txt,
        pair_obj=pair,
        trend_label=trend_b,
        is_a=False,
    )

    # Build plans (uses spot, vol20_ann_pct, implied move; ATR% is inferred if needed)
    try:
        tp_a = build_trade_plan(
            ticker=ticker_a,
            direction=_dir_a_sz,
            spot=a_snap.get("price"),
            account_equity=float(account_equity),
            risk_profile=risk_profile,
            vol20_ann_pct=a_snap.get("vol20_ann_pct"),
            implied_move_pct=opt_a.implied_move_pct,
        )
    except Exception:
        tp_a = None

    try:
        tp_b = build_trade_plan(
            ticker=ticker_b,
            direction=_dir_b_sz,
            spot=b_snap.get("price"),
            account_equity=float(account_equity),
            risk_profile=risk_profile,
            vol20_ann_pct=b_snap.get("vol20_ann_pct"),
            implied_move_pct=opt_b.implied_move_pct,
        )
    except Exception:
        tp_b = None

    tpa, tpb = st.columns(2)
    with tpa:
        st.markdown(f"**{ticker_a}** — `{_dir_a_sz.upper()}`")
        if tp_a is None:
            st.info("Trade plan unavailable.")
        else:
            st.dataframe(tp_a.to_dataframe(), use_container_width=True)
    with tpb:
        st.markdown(f"**{ticker_b}** — `{_dir_b_sz.upper()}`")
        if tp_b is None:
            st.info("Trade plan unavailable.")
        else:
            st.dataframe(tp_b.to_dataframe(), use_container_width=True)

# --- Scenarios tab (stock & ATM option P/L ladders) ---
with tab_scenarios:
    st.markdown("#### Scenario Explorer")
    st.caption(
        "Five scenarios over your chosen DTE: −2σ / −1σ / base / +1σ / +2σ. "
        "σ is proxied by the **implied move** over that horizon. "
        "Stock P/L is per share; Option P/L is per contract (×100)."
    )

    # Infer directions (uses report text if present → else pair z-score → else Trend badge)
    report_text = st.session_state.get("report_markdown")
    dir_a = _infer_direction_for(
        this_ticker=ticker_a,
        other_ticker=ticker_b,
        report_text=report_text,
        pair_obj=pair,
        trend_label=trend_a,
        is_a=True,
    )
    dir_b = _infer_direction_for(
        this_ticker=ticker_b,
        other_ticker=ticker_a,
        report_text=report_text,
        pair_obj=pair,
        trend_label=trend_b,
        is_a=False,
    )

    csa, csb = st.columns(2)

    with csa:
        st.markdown(f"**{ticker_a}** — assumed direction: `{dir_a.upper()}`")
        try:
            df_a = build_scenarios(
                ticker=ticker_a,
                direction=dir_a,
                spot=a_snap.get("price"),
                implied_move_pct=opt_a.implied_move_pct,
                opt=opt_a,  # OptionsSnapshot works; builder uses getattr defensively
            )
        except Exception:
            df_a = pd.DataFrame(columns=["Scenario", "Spot'", "Stock P/L/share", "Option P/L/contract"])
        st.dataframe(df_a, use_container_width=True)

    with csb:
        st.markdown(f"**{ticker_b}** — assumed direction: `{dir_b.upper()}`")
        try:
            df_b = build_scenarios(
                ticker=ticker_b,
                direction=dir_b,
                spot=b_snap.get("price"),
                implied_move_pct=opt_b.implied_move_pct,
                opt=opt_b,
            )
        except Exception:
            df_b = pd.DataFrame(columns=["Scenario", "Spot'", "Stock P/L/share", "Option P/L/contract"])
        st.dataframe(df_b, use_container_width=True)

    # Download combined scenarios as CSV
    def _df_to_bytes(df: pd.DataFrame) -> bytes:
        try:
            return df.to_csv(index=False).encode("utf-8")
        except Exception:
            return b"Scenario,Spot',Stock P/L/share,Option P/L/contract\n"

    combined = pd.DataFrame()
    if df_a is not None and not df_a.empty:
        dfa = df_a.copy()
        dfa.insert(0, "Ticker", ticker_a)
        dfa.insert(1, "Direction", dir_a.upper())
        combined = pd.concat([combined, dfa], ignore_index=True)
    if df_b is not None and not df_b.empty:
        dfb = df_b.copy()
        dfb.insert(0, "Ticker", ticker_b)
        dfb.insert(1, "Direction", dir_b.upper())
        combined = pd.concat([combined, dfb], ignore_index=True)

    st.download_button(
        "⬇️ Download scenarios (.csv)",
        data=_df_to_bytes(combined if not combined.empty else pd.DataFrame(
            columns=["Ticker","Direction","Scenario","Spot'","Stock P/L/share","Option P/L/contract"]
        )),
        file_name=f"scenarios_{ticker_a}_{ticker_b}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown("#### Payoff Charts")

    pa, pb = st.columns(2)

    def _render_payoff(df_grid: pd.DataFrame, breakeven: Optional[float], title: str):
        """Try Altair for interactive chart; fallback to st.line_chart."""
        if df_grid is None or df_grid.empty:
            st.info("No payoff data to chart.")
            return
        try:
            import altair as alt  # type: ignore
            # Melt to long format for legend & tooltips
            df_long = df_grid.melt("Spot", var_name="Instrument", value_name="P/L")

            base = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("Spot:Q", title="Spot"),
                    y=alt.Y("P/L:Q", title="P/L"),
                    color=alt.Color("Instrument:N", legend=alt.Legend(title=None)),
                    tooltip=["Spot:Q", "Instrument:N", alt.Tooltip("P/L:Q", format=".2f")],
                )
                .properties(title=title, height=240)
            )

            if breakeven is not None:
                rule = (
                    alt.Chart(pd.DataFrame({"x": [breakeven]}))
                    .mark_rule(strokeDash=[5, 5])
                    .encode(x="x:Q")
                )
                label = (
                    alt.Chart(pd.DataFrame({"x": [breakeven], "txt": ["breakeven"]}))
                    .mark_text(dy=-8)
                    .encode(x="x:Q", text="txt:N")
                )
                chart = base + rule + label
            else:
                chart = base

            st.altair_chart(chart, use_container_width=True)
            if breakeven is not None:
                st.caption(f"Breakeven ≈ **{breakeven:.2f}**")
        except Exception:
            # Fallback: simple line chart
            try:
                st.line_chart(df_grid.set_index("Spot"), height=240, use_container_width=True)
                if breakeven is not None:
                    st.caption(f"Breakeven ≈ **{breakeven:.2f}**")
            except Exception:
                st.info("Could not render chart.")

    # Build dense payoff grids using your direction inference
    try:
        grid_a, be_a = build_payoff_grid(
            ticker=ticker_a,
            direction=dir_a,
            spot=a_snap.get("price"),
            implied_move_pct=opt_a.implied_move_pct,
            opt=opt_a,
        )
    except Exception:
        grid_a, be_a = pd.DataFrame(columns=["Spot", "Stock P/L/share", "Option P/L/contract"]), None

    try:
        grid_b, be_b = build_payoff_grid(
            ticker=ticker_b,
            direction=dir_b,
            spot=b_snap.get("price"),
            implied_move_pct=opt_b.implied_move_pct,
            opt=opt_b,
        )
    except Exception:
        grid_b, be_b = pd.DataFrame(columns=["Spot", "Stock P/L/share", "Option P/L/contract"]), None

    with pa:
        st.markdown(f"**{ticker_a}** — `{dir_a.upper()}`")
        _render_payoff(grid_a, be_a, title=f"{ticker_a} payoff (per share / per contract)")

    with pb:
        st.markdown(f"**{ticker_b}** — `{dir_b.upper()}`")
        _render_payoff(grid_b, be_b, title=f"{ticker_b} payoff (per share / per contract)")


# --- Central analysis runner: status/progress + stores results in session state ---

def run_analysis(*, status_box, progress_bar) -> Tuple[str, str, bytes]:
    """Runs the full analysis and returns (report_markdown, export_filename, export_bytes)."""
    if status_box: status_box.update(label="Fetching recent news & filings…", state="running")
    news_kwargs = dict(use_rss=use_rss, use_yf=use_yf, use_reuters=use_reuters, use_sec=use_sec)
    news_a_all = get_recent_news(ticker_a, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
    news_b_all = get_recent_news(ticker_b, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
    news_a = news_a_all["merged"]
    news_b = news_b_all["merged"]
    progress_bar.progress(30)

    if status_box: status_box.update(label="Compiling data, options, catalysts & pair stats…", state="running")
    fact_pack = to_fact_pack(a_snap, b_snap)
    options_fact = "OPTIONS:\n" + format_options_fact(opt_a) + "\n" + format_options_fact(opt_b)
    catalysts_block = "CATALYSTS:\n" + format_catalysts_md(ticker_a, cat_a) + "\n" + format_catalysts_md(ticker_b, cat_b)

    # Fundamentals block (compact markdown for model)
    fund_md_a = format_fundamentals_md(ticker_a)
    fund_md_b = format_fundamentals_md(ticker_b)
    fundamentals_block = "FUNDAMENTALS:\n" + fund_md_a + "\n" + fund_md_b

    # Pair block
    if pair and pair.window_used > 0:
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

    # Sizing facts (recompute for freshness)
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

    if status_box: status_box.update(label="Building prompt…", state="running")
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

    if status_box: status_box.update(label="Calling model…", state="running")
    agent = build_agent(get_openai_key()[0])
    result = _run_agent_with_retries(agent, prompt, max_retries=3)
    content = getattr(result, "content", None) or str(result)
    sanitized = sanitize_styling(content)
    progress_bar.progress(100)
    if status_box: status_box.update(label="Analysis complete.", state="complete")

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
    )
    return sanitized, fname, fbytes


# --- If the user pressed the button, run analysis NOW with visible status/progress ---

if go:
    if not current_key:
        st.error("No OpenAI key found. Add it in the sidebar or create a .env file with OPENAI_API_KEY.")
    else:
        with status_container:
            try:
                status = st.status("Starting analysis…", expanded=True)
            except Exception:
                status = None
            progress = st.progress(0)
            try:
                report_md, fname, fbytes = run_analysis(status_box=status, progress_bar=progress)
            except Exception as e:
                if status: status.update(label="Analysis failed.", state="error")
                st.exception(e)
            else:
                st.session_state["report_markdown"] = report_md
                st.session_state["export_fname"] = fname
                st.session_state["export_bytes"] = fbytes
                st.toast("Analysis ready — open the Report tab ✅")


# --- Report tab (renders stored result) ---
with tab_report:
    if st.session_state.get("report_markdown"):
        st.markdown("#### Investment Analysis Report")
        st.markdown(st.session_state["report_markdown"])
        # --- PDF export of the full report (now with Strategies, Playbook, Scenarios & Tickets) ---
        try:
            # Use extended fundamentals in PDF
            from ai_agent.metrics import fundamentals_table_extended as fundamentals_table
            from ai_agent.playbook import build_event_playbook_for_ticker, playbook_to_markdown

            # Fundamentals tables
            fund_a_tbl = fundamentals_table(ticker_a)
            fund_b_tbl = fundamentals_table(ticker_b)

            # Catalysts text (reuse already-built markdown)
            catalysts_all_md = f"#### {ticker_a}\n\n{cat_md_a}\n\n#### {ticker_b}\n\n{cat_md_b}"

            # Pair notes (compact)
            if pair and pair.window_used > 0:
                ztxt = f"{pair.spread_zscore:.2f}" if pair.spread_zscore is not None else "—"
                pair_md = (
                    f"- Beta(A on B): {pair.beta_ab:.2f} | Corr: {pair.corr_ab:.2f} | Hedge ratio: {pair.hedge_ratio:.2f}\n"
                    f"- Spread z-score (60): {ztxt}"
                )
            else:
                pair_md = "- Insufficient overlapping history for pair analysis."

            # Sizing notes (fallback to recompute if needed)
            try:
                sizing_notes_md = f"- {sizing_fact_a}\n- {sizing_fact_b}"
            except Exception:
                sfa = _sizing_fact_line(
                    ticker_a,
                    sizing_summary_table(
                        ticker=ticker_a, account_equity=float(account_equity), profile_name=risk_profile,
                        spot=a_snap.get("price"), vol20_ann_pct=a_snap.get("vol20_ann_pct"), opt=opt_a
                    ),
                )
                sfb = _sizing_fact_line(
                    ticker_b,
                    sizing_summary_table(
                        ticker=ticker_b, account_equity=float(account_equity), profile_name=risk_profile,
                        spot=b_snap.get("price"), vol20_ann_pct=b_snap.get("vol20_ann_pct"), opt=opt_b
                    ),
                )
                sizing_notes_md = f"- {sfa}\n- {sfb}"

            # Selected strategies (join all chosen markdown blocks)
            sel_list = st.session_state.get("selected_strategies_md") or []
            selected_strategies_md = "\n\n---\n\n".join(sel_list) if sel_list else None

            # Event Playbook (recompute here to avoid ordering issues)
            plays_a = build_event_playbook_for_ticker(
                ticker=ticker_a, catalysts=cat_a, opt_snapshot=opt_a,
                trend_label=trend_a, regime_label=regime_a, risk_profile=risk_profile,
            )
            plays_b = build_event_playbook_for_ticker(
                ticker=ticker_b, catalysts=cat_b, opt_snapshot=opt_b,
                trend_label=trend_b, regime_label=regime_b, risk_profile=risk_profile,
            )
            event_playbook_md = (
                playbook_to_markdown(ticker_a, plays_a)
                + "\n\n"
                + playbook_to_markdown(ticker_b, plays_b)
            )

            # Scenarios (recompute locally; independent of Scenarios tab)
            dir_a_r = _infer_direction_for(
                this_ticker=ticker_a, other_ticker=ticker_b,
                report_text=st.session_state.get("report_markdown"),
                pair_obj=pair, trend_label=trend_a, is_a=True,
            )
            dir_b_r = _infer_direction_for(
                this_ticker=ticker_b, other_ticker=ticker_a,
                report_text=st.session_state.get("report_markdown"),
                pair_obj=pair, trend_label=trend_b, is_a=False,
            )
            try:
                scn_a_df = build_scenarios(
                    ticker=ticker_a, direction=dir_a_r, spot=a_snap.get("price"),
                    implied_move_pct=opt_a.implied_move_pct, opt=opt_a,
                )
            except Exception:
                scn_a_df = pd.DataFrame()
            try:
                scn_b_df = build_scenarios(
                    ticker=ticker_b, direction=dir_b_r, spot=b_snap.get("price"),
                    implied_move_pct=opt_b.implied_move_pct, opt=opt_b,
                )
            except Exception:
                scn_b_df = pd.DataFrame()

            # Tickets preview (recompute base + merge strategy-added extras)
            base_tix = []
            try:
                base_tix += build_tickets_for_ticker(
                    ticker=ticker_a, direction=dir_a_r, sizing_df=size_a_df, opt=opt_a,
                    spot=a_snap.get("price"), implied_move_pct=opt_a.implied_move_pct, profile_name=risk_profile,
                )
            except Exception:
                pass
            try:
                base_tix += build_tickets_for_ticker(
                    ticker=ticker_b, direction=dir_b_r, sizing_df=size_b_df, opt=opt_b,
                    spot=b_snap.get("price"), implied_move_pct=opt_b.implied_move_pct, profile_name=risk_profile,
                )
            except Exception:
                pass
            extra_tix = st.session_state.get("tickets_extra") or []
            all_tix = list(base_tix) + list(extra_tix)
            rows = []
            for t in all_tix:
                if hasattr(t, "__dict__"):
                    rows.append({k: v for k, v in t.__dict__.items()})
                elif isinstance(t, dict):
                    rows.append(t)
            tickets_df = pd.DataFrame(rows)

            # Build the PDF
            pdf_name, pdf_bytes = build_pdf_report(
                title=f"AI Investment Report {ticker_a} vs {ticker_b}",
                report_markdown=st.session_state["report_markdown"],
                metadata={
                    "Ticker A": ticker_a,
                    "Ticker B": ticker_b,
                    "Timeframe": st.session_state.get("timeframe_select", "1Y"),
                    "Risk profile": risk_profile,
                    "Lookback days": str(lookback_days),
                    "Max news": str(max_news),
                },
                options_table=pd.DataFrame(
                    [
                        ("Expiry (DTE)", f"{opt_a.expiry or '—'} ({opt_a.dte or '—'})", f"{opt_b.expiry or '—'} ({opt_b.dte or '—'})"),
                        ("Spot", _fmt_num(opt_a.spot), _fmt_num(opt_b.spot)),
                        ("ATM strike", _fmt_num(opt_a.atm_strike), _fmt_num(opt_b.atm_strike)),
                        ("Call mid", _fmt_num(opt_a.call_mid), _fmt_num(opt_b.call_mid)),
                        ("Put mid", _fmt_num(opt_a.put_mid), _fmt_num(opt_b.put_mid)),
                        ("Straddle debit", _fmt_num(opt_a.straddle_debit), _fmt_num(opt_b.straddle_debit)),
                        ("Implied move", _fmt_num(opt_a.implied_move_pct, pct=True), _fmt_num(opt_b.implied_move_pct, pct=True)),
                        ("ATM IV (approx.)", _fmt_num(opt_a.atm_iv_pct, pct=True), _fmt_num(opt_b.atm_iv_pct, pct=True)),
                    ],
                    columns=["Metric", ticker_a, ticker_b],
                ),
                fundamentals_a=fund_a_tbl,
                fundamentals_b=fund_b_tbl,
                catalysts_md=catalysts_all_md,
                pair_text_md=pair_md,
                sizing_text_md=sizing_notes_md,
                selected_strategies_md=selected_strategies_md,   # NEW
                event_playbook_md=event_playbook_md,             # NEW
                scenarios_a_df=scn_a_df,                         # NEW
                scenarios_b_df=scn_b_df,                         # NEW
                tickets_df=tickets_df,                           # NEW
            )

            st.download_button(
                "⬇️ Download full report (.pdf)",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _pdf_err:
            st.caption(f"PDF export unavailable: {_pdf_err}")
        st.download_button(
            label="⬇️ Download report (.md)",
            data=st.session_state["export_bytes"],
            file_name=st.session_state["export_fname"],
            mime="text/markdown",
            use_container_width=True,
        )
        # --- Trade tickets CSV ---
        st.markdown("#### Trade Tickets")

        # Infer directions for both tickers
        dir_a = _infer_direction_for(
            this_ticker=ticker_a,
            other_ticker=ticker_b,
            report_text=st.session_state["report_markdown"],
            pair_obj=pair,
            trend_label=trend_a,
            is_a=True,
        )
        dir_b = _infer_direction_for(
            this_ticker=ticker_b,
            other_ticker=ticker_a,
            report_text=st.session_state["report_markdown"],
            pair_obj=pair,
            trend_label=trend_b,
            is_a=False,
        )


        # Build base tickets from sizing/options for both tickers
        # Re-infer directions (keeps this block self-contained)
        dir_a_r = _infer_direction_for(
            this_ticker=ticker_a, other_ticker=ticker_b,
            report_text=st.session_state.get("report_markdown"),
            pair_obj=pair, trend_label=trend_a, is_a=True,
        )
        dir_b_r = _infer_direction_for(
            this_ticker=ticker_b, other_ticker=ticker_a,
            report_text=st.session_state.get("report_markdown"),
            pair_obj=pair, trend_label=trend_b, is_a=False,
        )

        tickets = []
        try:
            tickets += build_tickets_for_ticker(
                ticker=ticker_a,
                direction=dir_a_r,
                sizing_df=size_a_df,
                opt=opt_a,
                spot=a_snap.get("price"),
                implied_move_pct=opt_a.implied_move_pct,
                profile_name=risk_profile,
            )
        except Exception:
            pass
        try:
            tickets += build_tickets_for_ticker(
                ticker=ticker_b,
                direction=dir_b_r,
                sizing_df=size_b_df,
                opt=opt_b,
                spot=b_snap.get("price"),
                implied_move_pct=opt_b.implied_move_pct,
                profile_name=risk_profile,
            )
        except Exception:
            pass  
        # Build tickets from sizing & options context
        # Merge in any extra tickets sent from Strategy Picker
        extra = st.session_state.get("tickets_extra") or []
        tickets_all = list(tickets) + list(extra)

        # Robust CSV: try library function first; fall back to manual if types differ
        try:
            fname_csv, csv_bytes = tickets_to_csv_bytes(tickets_all)
        except Exception:
            # Normalize to dicts then build CSV manually
            import pandas as _pd
            norm_rows = []
            for t in tickets_all:
                if hasattr(t, "__dict__"):
                    norm_rows.append({k: v for k, v in t.__dict__.items()})
                elif isinstance(t, dict):
                    norm_rows.append(t)
            if not norm_rows:
                norm_rows = [{"plan": "", "ticker": "", "asset": "", "side": "", "quantity": 0, "expiry": "", "strike": "", "right": "", "price_note": "", "notes": ""}]
            _df_csv = _pd.DataFrame(norm_rows)
            csv_bytes = _df_csv.to_csv(index=False).encode("utf-8")
            fname_csv = f"tickets_{ticker_a}_{ticker_b}.csv"

        st.download_button(
            label="⬇️ Download trade tickets (.csv)",
            data=csv_bytes,
            file_name=fname_csv,
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Preview tickets"):
            try:
                import pandas as _pd
                rows = []
                for t in tickets_all:
                    if hasattr(t, "__dict__"):
                        rows.append({k: v for k, v in t.__dict__.items()})
                    elif isinstance(t, dict):
                        rows.append(t)
                _df_prev = _pd.DataFrame(rows)
                st.dataframe(_df_prev, use_container_width=True)
            except Exception:
                st.write("No tickets to preview.")

        # Optional management for extras
        if extra:
            cclr1, cclr2 = st.columns([1, 3])
            with cclr1:
                if st.button("Clear strategy-added tickets"):
                    st.session_state["tickets_extra"] = []
                    st.rerun()

        # --- Event Playbook (earnings/filings → concrete tactics) ---
        st.markdown("---")
        st.markdown("### Event Playbook")

        try:
            plays_a = build_event_playbook_for_ticker(
                ticker=ticker_a,
                catalysts=cat_a,
                opt_snapshot=opt_a,
                trend_label=trend_a,
                regime_label=regime_a,
                risk_profile=risk_profile,
            )
            md_a = playbook_to_markdown(ticker_a, plays_a)
        except Exception:
            md_a = f"### {ticker_a}: Event Playbook\n\n_Unable to build playbook at this time._\n"

        try:
            plays_b = build_event_playbook_for_ticker(
                ticker=ticker_b,
                catalysts=cat_b,
                opt_snapshot=opt_b,
                trend_label=trend_b,
                regime_label=regime_b,
                risk_profile=risk_profile,
            )
            md_b = playbook_to_markdown(ticker_b, plays_b)
        except Exception:
            md_b = f"### {ticker_b}: Event Playbook\n\n_Unable to build playbook at this time._\n"

        # Show side-by-side for quick scanning
        pb1, pb2 = st.columns(2)
        with pb1:
            st.markdown(md_a)
        with pb2:
            st.markdown(md_b)


        # --- Selected Strategies (from Options → Strategy Picker) ---
        st.markdown("---")
        st.markdown("### Selected Strategies")
        sel = st.session_state.get("selected_strategies_md") or []
        if not sel:
            st.caption("Use the **Options → Strategy Picker** to add strategies here.")
        else:
            combined_md = "\n".join(sel)
            st.markdown(combined_md)
            st.download_button(
                "⬇️ Download selected strategies (.md)",
                data=combined_md.encode("utf-8"),
                file_name=f"strategies_{ticker_a}_{ticker_b}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            if st.button("Clear selected strategies"):
                st.session_state["selected_strategies_md"] = []
                st.rerun()

        # --- Trade Plan (Stock) summary ---
        st.markdown("---")
        st.markdown("### Trade Plan (Stock)")

        # Re-infer directions (safe & local to Report tab)
        _dir_a_r = _infer_direction_for(
            this_ticker=ticker_a, other_ticker=ticker_b,
            report_text=st.session_state.get("report_markdown"),
            pair_obj=pair, trend_label=trend_a, is_a=True,
        )
        _dir_b_r = _infer_direction_for(
            this_ticker=ticker_b, other_ticker=ticker_a,
            report_text=st.session_state.get("report_markdown"),
            pair_obj=pair, trend_label=trend_b, is_a=False,
        )

        try:
            tp_a_r = build_trade_plan(
                ticker=ticker_a, direction=_dir_a_r, spot=a_snap.get("price"),
                account_equity=float(account_equity), risk_profile=risk_profile,
                vol20_ann_pct=a_snap.get("vol20_ann_pct"), implied_move_pct=opt_a.implied_move_pct,
            )
        except Exception:
            tp_a_r = None
        try:
            tp_b_r = build_trade_plan(
                ticker=ticker_b, direction=_dir_b_r, spot=b_snap.get("price"),
                account_equity=float(account_equity), risk_profile=risk_profile,
                vol20_ann_pct=b_snap.get("vol20_ann_pct"), implied_move_pct=opt_b.implied_move_pct,
            )
        except Exception:
            tp_b_r = None

        rpa, rpb = st.columns(2)
        with rpa:
            st.markdown(f"**{ticker_a}** — `{_dir_a_r.upper()}`")
            if tp_a_r is None:
                st.info("Trade plan unavailable.")
            else:
                st.dataframe(tp_a_r.to_dataframe(), use_container_width=True)
        with rpb:
            st.markdown(f"**{ticker_b}** — `{_dir_b_r.upper()}`")
            if tp_b_r is None:
                st.info("Trade plan unavailable.")
            else:
                st.dataframe(tp_b_r.to_dataframe(), use_container_width=True)
    else:
        st.info("Press **Compare & Analyze** to generate a full report.")
