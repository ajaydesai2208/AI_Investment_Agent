from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

__all__ = [
    "snapshot",
    "compare_table",
    "to_fact_pack",
    # optionally useful:
    "load_history_1y",
]

# ---------- History & helpers ----------

@st.cache_data(ttl=600, show_spinner=False)
def load_history_1y(ticker: str) -> pd.DataFrame:
    """Daily auto-adjusted history ~1y for returns/volatility."""
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        df = pd.DataFrame()
    return df if not df.empty else pd.DataFrame()

def _close_series(hist: pd.DataFrame) -> pd.Series:
    """
    Coerce history to a single numeric Close series, robust across yfinance/pandas variants.
    """
    if hist is None or hist.empty:
        return pd.Series(dtype="float64")
    if "Adj Close" in hist.columns:
        s = hist["Adj Close"]
    elif "Close" in hist.columns:
        s = hist["Close"]
    else:
        # Fallback to the first numeric column if none of the above exists
        num_cols = [c for c in hist.columns if pd.api.types.is_numeric_dtype(hist[c])]
        if not num_cols:
            return pd.Series(dtype="float64")
        s = hist[num_cols[0]]
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return pd.Series(dtype="float64")
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna()

def _pct_return(series: pd.Series, bars: int) -> Optional[float]:
    """
    Simple percentage return over 'bars' trading days.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or len(s) < bars + 1:
        return None
    end = float(s.iloc[-1])
    start = float(s.iloc[-(bars + 1)])
    if start == 0.0 or math.isnan(start) or math.isnan(end):
        return None
    return (end / start - 1.0) * 100.0

def _ann_vol20(series: pd.Series) -> Optional[float]:
    """
    20-day annualized volatility (%) from daily pct changes.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    rets = s.pct_change().dropna()
    if len(rets) < 20:
        return None
    daily_std = float(rets.tail(20).std())
    if math.isnan(daily_std):
        return None
    return daily_std * math.sqrt(252) * 100.0

def _fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}%"

def _fmt_num(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    # format large numbers as 1.2B etc.
    val = float(x)
    absx = abs(val)
    if absx >= 1e12:
        return f"{val/1e12:.2f}T"
    if absx >= 1e9:
        return f"{val/1e9:.2f}B"
    if absx >= 1e6:
        return f"{val/1e6:.2f}M"
    if absx >= 1e3:
        return f"{val/1e3:.2f}K"
    return f"{val:.2f}"

def _try_get(d: Any, key: str) -> Optional[Any]:
    try:
        if isinstance(d, dict):
            return d.get(key)
        return getattr(d, key, None)
    except Exception:
        return None

# ---------- Snapshot & comparison ----------

@st.cache_data(ttl=600, show_spinner=False)
def snapshot(ticker: str) -> Dict[str, Any]:
    """
    Build a robust point-in-time snapshot for a ticker using:
      - Ticker.fast_info (preferred where available)
      - Ticker.get_info() / .info (fallbacks)
      - 1y history-derived metrics (returns & vol)

    Returns dict with keys:
      ticker, price, market_cap, pe_ttm, dividend_yield_pct, beta,
      next_earnings_date, recent_ex_div_date,
      ret_1m_pct, ret_3m_pct, ret_6m_pct, ret_1y_pct, vol20_ann_pct
    """
    out: Dict[str, Any] = {"ticker": ticker}

    t = yf.Ticker(ticker)

    # fast_info (newer, faster)
    fi = {}
    try:
        fi = dict(getattr(t, "fast_info", {}) or {})
    except Exception:
        fi = {}

    # legacy info
    info = {}
    try:
        get_info = getattr(t, "get_info", None)
        info = get_info() if callable(get_info) else dict(getattr(t, "info", {}) or {})
    except Exception:
        info = {}

    # price (prefer fast_info)
    price = _try_get(fi, "last_price") or _try_get(fi, "lastPrice") or _try_get(fi, "regularMarketPrice") or info.get("currentPrice")
    out["price"] = float(price) if price not in (None, "None") else None

    # market cap
    mcap = _try_get(fi, "market_cap") or _try_get(fi, "marketCap") or info.get("marketCap")
    out["market_cap"] = float(mcap) if mcap not in (None, "None") else None

    # trailing P/E
    pe = _try_get(fi, "trailing_pe") or _try_get(fi, "trailingPe") or info.get("trailingPE")
    out["pe_ttm"] = float(pe) if pe not in (None, "None") else None

    # dividend yield (%)
    dy = _try_get(fi, "dividend_yield") or _try_get(fi, "dividendYield") or info.get("dividendYield")
    if dy not in (None, "None"):
        dy = float(dy)
        out["dividend_yield_pct"] = dy * 100.0 if dy < 1 else dy
    else:
        out["dividend_yield_pct"] = None

    # beta
    beta = _try_get(fi, "beta") or info.get("beta")
    out["beta"] = float(beta) if beta not in (None, "None") else None

    # events: next earnings date (best-effort)
    out["next_earnings_date"] = None
    try:
        edf = t.get_earnings_dates(limit=1)
        if edf is not None and len(edf) > 0:
            if "Earnings Date" in edf.columns:
                dt = edf["Earnings Date"].iloc[0]
                out["next_earnings_date"] = str(pd.to_datetime(dt).date())
            else:
                out["next_earnings_date"] = str(pd.to_datetime(edf.index[0]).date())
    except Exception:
        pass

    # recent ex-div date (not guaranteed future)
    out["recent_ex_div_date"] = None
    try:
        actions = t.get_actions()
        if actions is not None and "Dividends" in actions.columns:
            divs = actions[actions["Dividends"] > 0.0]
            if len(divs) > 0:
                out["recent_ex_div_date"] = str(pd.to_datetime(divs.index[-1]).date())
    except Exception:
        pass

    # 1y history-derived metrics
    hist = load_history_1y(ticker)
    close = _close_series(hist)
    out["ret_1m_pct"] = _pct_return(close, 21)    # ~21 trading days
    out["ret_3m_pct"] = _pct_return(close, 63)
    out["ret_6m_pct"] = _pct_return(close, 126)
    bars_1y = 252 if len(close) >= 300 else max(1, len(close) - 1)
    out["ret_1y_pct"] = _pct_return(close, bars_1y)
    out["vol20_ann_pct"] = _ann_vol20(close)

    return out

def compare_table(a_snap: Dict[str, Any], b_snap: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a readable comparison DataFrame for the UI.
    Values are formatted strings (e.g., 1.23B, 12.3%).
    """
    rows = [
        ("Price", _fmt_num(a_snap["price"]), _fmt_num(b_snap["price"])),
        ("Market Cap", _fmt_num(a_snap["market_cap"]), _fmt_num(b_snap["market_cap"])),
        ("P/E (TTM)", _fmt_num(a_snap["pe_ttm"]), _fmt_num(b_snap["pe_ttm"])),
        ("Dividend Yield", _fmt_pct(a_snap["dividend_yield_pct"]), _fmt_pct(b_snap["dividend_yield_pct"])),
        ("Beta", _fmt_num(a_snap["beta"]), _fmt_num(b_snap["beta"])),
        ("Return 1M", _fmt_pct(a_snap["ret_1m_pct"]), _fmt_pct(b_snap["ret_1m_pct"])),
        ("Return 3M", _fmt_pct(a_snap["ret_3m_pct"]), _fmt_pct(b_snap["ret_3m_pct"])),
        ("Return 6M", _fmt_pct(a_snap["ret_6m_pct"]), _fmt_pct(b_snap["ret_6m_pct"])),
        ("Return 1Y", _fmt_pct(a_snap["ret_1y_pct"]), _fmt_pct(b_snap["ret_1y_pct"])),
        ("Vol (20d ann.)", _fmt_pct(a_snap["vol20_ann_pct"]), _fmt_pct(b_snap["vol20_ann_pct"])),
        ("Next Earnings", a_snap.get("next_earnings_date") or "—", b_snap.get("next_earnings_date") or "—"),
        ("Recent Ex-Div", a_snap.get("recent_ex_div_date") or "—", b_snap.get("recent_ex_div_date") or "—"),
    ]
    return pd.DataFrame(rows, columns=["Metric", a_snap["ticker"], b_snap["ticker"]])

# ---------- LLM fact pack ----------

def to_fact_pack(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """
    Produce a compact, LLM-friendly block of deterministic facts.
    """
    def one(d: Dict[str, Any]) -> str:
        return (
            f"- ticker: {d['ticker']}\n"
            f"  price: {d.get('price')}\n"
            f"  market_cap: {d.get('market_cap')}\n"
            f"  pe_ttm: {d.get('pe_ttm')}\n"
            f"  dividend_yield_pct: {d.get('dividend_yield_pct')}\n"
            f"  beta: {d.get('beta')}\n"
            f"  returns_pct: {{1M: {d.get('ret_1m_pct')}, 3M: {d.get('ret_3m_pct')}, 6M: {d.get('ret_6m_pct')}, 1Y: {d.get('ret_1y_pct')}}}\n"
            f"  vol20_ann_pct: {d.get('vol20_ann_pct')}\n"
            f"  next_earnings_date: {d.get('next_earnings_date')}\n"
            f"  recent_ex_div_date: {d.get('recent_ex_div_date')}\n"
        )
    return "FACTS:\n" + one(a) + one(b)
