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


# === Fundamentals (TTM & ratios) ===================================

import math
from typing import Optional, Dict, Any, Tuple

import pandas as _pd

try:
    import yfinance as _yf  # already a dependency
except Exception:
    _yf = None


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return v


def _latest_col(df: _pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    try:
        # yfinance uses columns as datelike; pick the right-most
        return df.columns[0] if isinstance(df.columns[0], str) else df.columns[-1]
    except Exception:
        return None


def _sum_last_n(qdf: _pd.DataFrame, row: str, n: int = 4) -> Optional[float]:
    try:
        s = qdf.loc[row].dropna().astype(float)
        if s.empty:
            return None
        return float(s.iloc[:n].sum())
    except Exception:
        return None


def _get_ttm_or_annual(qdf: _pd.DataFrame, adf: _pd.DataFrame, row: str) -> Optional[float]:
    # Prefer TTM from quarterly; else last annual
    ttm = _sum_last_n(qdf, row, n=4) if qdf is not None and not qdf.empty else None
    if _safe_float(ttm) is not None and ttm != 0:
        return ttm
    try:
        col = _latest_col(adf)
        if col is None:
            return None
        v = adf.loc[row, col]
        return _safe_float(v)
    except Exception:
        return None


def _yf_frames(ticker: str) -> Tuple[_pd.DataFrame, _pd.DataFrame, _pd.DataFrame, _pd.DataFrame, _pd.DataFrame, _pd.DataFrame]:
    """
    Returns (income_annual, income_quarterly, bs_annual, bs_quarterly, cf_annual, cf_quarterly)
    All may be empty dataframes; call sites must guard.
    """
    if _yf is None:
        return (_pd.DataFrame(),)*6
    tk = _yf.Ticker(ticker)
    # yfinance 0.2.x names:
    # - income_stmt / quarterly_income_stmt
    # - balance_sheet / quarterly_balance_sheet
    # - cashflow / quarterly_cashflow
    try:
        ia = getattr(tk, "income_stmt", _pd.DataFrame())
        iq = getattr(tk, "quarterly_income_stmt", _pd.DataFrame())
        ba = getattr(tk, "balance_sheet", _pd.DataFrame())
        bq = getattr(tk, "quarterly_balance_sheet", _pd.DataFrame())
        ca = getattr(tk, "cashflow", _pd.DataFrame())
        cq = getattr(tk, "quarterly_cashflow", _pd.DataFrame())
    except Exception:
        # older yfinance aliases
        ia = getattr(tk, "financials", _pd.DataFrame())
        iq = getattr(tk, "quarterly_financials", _pd.DataFrame())
        ba = getattr(tk, "balance_sheet", _pd.DataFrame())
        bq = getattr(tk, "quarterly_balance_sheet", _pd.DataFrame())
        ca = getattr(tk, "cashflow", _pd.DataFrame())
        cq = getattr(tk, "quarterly_cashflow", _pd.DataFrame())
    # Ensure index is standardized (capitalize)
    def _norm(df):
        try:
            df = df.copy()
            df.index = [str(i).strip() for i in df.index]
            return df
        except Exception:
            return _pd.DataFrame()
    return tuple(_norm(x) for x in (ia, iq, ba, bq, ca, cq))  # type: ignore


def fundamentals_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Compute a compact set of fundamentals (TTM preferred, else latest annual).
    Keys (may be None):
      revenue_ttm, revenue_growth_1y, gross_margin_pct, op_margin_pct, net_margin_pct,
      fcf_ttm, fcf_margin_pct, net_debt, net_debt_to_ebitda, buyback_1y_pct,
      shares_out, dividend_yield_pct
    """
    ia, iq, ba, bq, ca, cq = _yf_frames(ticker)

    # Revenue & growth
    rev_ttm = _get_ttm_or_annual(iq, ia, "Total Revenue")
    rev_prev_ttm = None
    try:
        if iq is not None and not iq.empty:
            # previous 4 quarters (shift)
            s = iq.loc["Total Revenue"].dropna().astype(float)
            if len(s) >= 8:
                rev_prev_ttm = float(s.iloc[4:8].sum())
    except Exception:
        pass
    rev_growth = None
    if _safe_float(rev_ttm) and _safe_float(rev_prev_ttm) and rev_prev_ttm not in (None, 0):
        rev_growth = (rev_ttm - rev_prev_ttm) / abs(rev_prev_ttm) * 100.0

    # Margins (gross, operating, net)
    cogs_ttm = _get_ttm_or_annual(iq, ia, "Cost Of Revenue")
    gross_margin = None
    if _safe_float(rev_ttm) and _safe_float(cogs_ttm) and rev_ttm not in (None, 0):
        gross_margin = (rev_ttm - cogs_ttm) / rev_ttm * 100.0

    op_income_ttm = _get_ttm_or_annual(iq, ia, "Operating Income")
    op_margin = None
    if _safe_float(rev_ttm) and _safe_float(op_income_ttm) and rev_ttm not in (None, 0):
        op_margin = op_income_ttm / rev_ttm * 100.0

    net_income_ttm = _get_ttm_or_annual(iq, ia, "Net Income")
    net_margin = None
    if _safe_float(rev_ttm) and _safe_float(net_income_ttm) and rev_ttm not in (None, 0):
        net_margin = net_income_ttm / rev_ttm * 100.0

    # FCF = Operating Cash Flow - CapEx
    ocf_ttm = _get_ttm_or_annual(cq, ca, "Operating Cash Flow")
    capex_ttm = _get_ttm_or_annual(cq, ca, "Capital Expenditure")
    fcf_ttm = None
    if _safe_float(ocf_ttm) is not None and _safe_float(capex_ttm) is not None:
        fcf_ttm = float(ocf_ttm) - float(capex_ttm)

    fcf_margin = None
    if _safe_float(rev_ttm) and _safe_float(fcf_ttm) and rev_ttm not in (None, 0):
        fcf_margin = fcf_ttm / rev_ttm * 100.0

    # Net debt = (Total Debt) - Cash & ST Investments
    total_debt = _get_ttm_or_annual(bq, ba, "Total Debt") or _get_ttm_or_annual(bq, ba, "Short Long Term Debt Total")
    cash = _get_ttm_or_annual(bq, ba, "Cash And Cash Equivalents")
    st_inv = _get_ttm_or_annual(bq, ba, "Other Short Term Investments")
    net_debt = None
    if _safe_float(total_debt) is not None and (_safe_float(cash) is not None or _safe_float(st_inv) is not None):
        net_debt = float(total_debt) - float(_safe_float(cash) or 0.0) - float(_safe_float(st_inv) or 0.0)

    # EBITDA (best-effort), Net debt / EBITDA
    ebitda_ttm = _get_ttm_or_annual(iq, ia, "Ebitda") or _get_ttm_or_annual(iq, ia, "EBITDA")
    ndebitda = None
    if _safe_float(net_debt) and _safe_float(ebitda_ttm) and ebitda_ttm not in (None, 0):
        ndebitda = net_debt / ebitda_ttm

    # Share count & buyback %
    shares_out = None
    try:
        info = _yf.Ticker(ticker).info if _yf else {}
        shares_out = _safe_float(info.get("sharesOutstanding"))
        dividend_yield = _safe_float(info.get("dividendYield"))
        if dividend_yield is not None:
            dividend_yield *= 100.0  # convert to %
    except Exception:
        info = {}
        dividend_yield = None

    buyback_pct = None
    try:
        # Approx: change in "Common Stock" or "Ordinary Shares Number" on BS (quarterly)
        for key in ("Ordinary Shares Number", "Common Stock"):
            if key in bq.index:
                s = bq.loc[key].dropna().astype(float)
                if len(s) >= 5:
                    curr = s.iloc[0]
                    prev = s.iloc[4]
                    if prev != 0:
                        buyback_pct = (prev - curr) / abs(prev) * 100.0
                        break
    except Exception:
        pass

    return {
        "revenue_ttm": _safe_float(rev_ttm),
        "revenue_growth_1y_pct": _safe_float(rev_growth),
        "gross_margin_pct": _safe_float(gross_margin),
        "op_margin_pct": _safe_float(op_margin),
        "net_margin_pct": _safe_float(net_margin),
        "fcf_ttm": _safe_float(fcf_ttm),
        "fcf_margin_pct": _safe_float(fcf_margin),
        "capex_ttm": _safe_float(capex_ttm),
        "net_debt": _safe_float(net_debt),
        "net_debt_to_ebitda": _safe_float(ndebitda),
        "shares_out": _safe_float(shares_out),
        "buyback_1y_pct": _safe_float(buyback_pct),
        "dividend_yield_pct": _safe_float(dividend_yield),
    }


def fundamentals_table(ticker: str) -> _pd.DataFrame:
    f = fundamentals_snapshot(ticker)
    rows = [
        ("Revenue (TTM)", f"${f['revenue_ttm']:,.0f}" if f.get("revenue_ttm") is not None else "—"),
        ("Revenue growth (YoY, TTM)", f"{f['revenue_growth_1y_pct']:.1f}%" if f.get("revenue_growth_1y_pct") is not None else "—"),
        ("Gross margin", f"{f['gross_margin_pct']:.1f}%" if f.get("gross_margin_pct") is not None else "—"),
        ("Operating margin", f"{f['op_margin_pct']:.1f}%" if f.get("op_margin_pct") is not None else "—"),
        ("Net margin", f"{f['net_margin_pct']:.1f}%" if f.get("net_margin_pct") is not None else "—"),
        ("FCF (TTM)", f"${f['fcf_ttm']:,.0f}" if f.get("fcf_ttm") is not None else "—"),
        ("FCF margin", f"{f['fcf_margin_pct']:.1f}%" if f.get("fcf_margin_pct") is not None else "—"),
        ("CapEx (TTM)", f"${f['capex_ttm']:,.0f}" if f.get("capex_ttm") is not None else "—"),
        ("Net debt", f"${f['net_debt']:,.0f}" if f.get("net_debt") is not None else "—"),
        ("Net debt / EBITDA", f"{f['net_debt_to_ebitda']:.2f}×" if f.get("net_debt_to_ebitda") is not None else "—"),
        ("Shares outstanding", f"{f['shares_out']:,.0f}" if f.get("shares_out") is not None else "—"),
        ("Buyback (approx, 1y)", f"{f['buyback_1y_pct']:.1f}%" if f.get("buyback_1y_pct") is not None else "—"),
        ("Dividend yield", f"{f['dividend_yield_pct']:.2f}%" if f.get("dividend_yield_pct") is not None else "—"),
    ]
    return _pd.DataFrame(rows, columns=["Metric", ticker])


def format_fundamentals_md(ticker: str) -> str:
    f = fundamentals_snapshot(ticker)
    def _fmt(v, money=False, pct=False, mult=False):
        if v is None:
            return "—"
        if money:
            return f"${v:,.0f}"
        if pct:
            return f"{v:.1f}%"
        if mult:
            return f"{v:.2f}×"
        return f"{v:.2f}"

    lines = [
        f"### {ticker} — Fundamentals (TTM)",
        "",
        f"- Revenue: {_fmt(f.get('revenue_ttm'), money=True)} (YoY: {_fmt(f.get('revenue_growth_1y_pct'), pct=True)})",
        f"- Margins: Gross {_fmt(f.get('gross_margin_pct'), pct=True)}, Op {_fmt(f.get('op_margin_pct'), pct=True)}, Net {_fmt(f.get('net_margin_pct'), pct=True)}",
        f"- FCF: {_fmt(f.get('fcf_ttm'), money=True)} (FCF margin {_fmt(f.get('fcf_margin_pct'), pct=True)}); CapEx {_fmt(f.get('capex_ttm'), money=True)}",
        f"- Leverage: Net debt {_fmt(f.get('net_debt'), money=True)}, Net debt/EBITDA {_fmt(f.get('net_debt_to_ebitda'), mult=True)}",
        f"- Capital return: Shares out {_fmt(f.get('shares_out'))}, Buyback ~{_fmt(f.get('buyback_1y_pct'), pct=True)}, Dividend yield {_fmt(f.get('dividend_yield_pct'), pct=True)}",
        "",
    ]
    return "\n".join(lines)

# === Extra Fundamentals: Valuation & Returns =========================
# Adds P/E (TTM), P/S (TTM), EV/EBITDA (TTM), ROIC (approx) and
# table/markdown helpers that extend the existing fundamentals output.

from typing import Optional, Dict
import pandas as _pd

def _yf_info(ticker: str) -> Dict:
    try:
        return (_yf.Ticker(ticker).info) if _yf else {}
    except Exception:
        return {}

def _market_caps_from_info(info: Dict) -> (Optional[float], Optional[float]):
    mc = _safe_float(info.get("marketCap"))
    ev = _safe_float(info.get("enterpriseValue"))
    return mc, ev

def _ebit_ttm(iq: _pd.DataFrame, ia: _pd.DataFrame) -> Optional[float]:
    # EBIT ≈ Operating Income + (if needed) interest adjustments; we use Operating Income TTM.
    return _get_ttm_or_annual(iq, ia, "Operating Income")

def _total_equity_latest(ba: _pd.DataFrame, bq: _pd.DataFrame) -> Optional[float]:
    for key in ("Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Assets - Total Liabilities Net Minority Interest"):
        try:
            col = _latest_col(ba)
            if col and key in ba.index:
                return _safe_float(ba.loc[key, col])
        except Exception:
            pass
        try:
            colq = _latest_col(bq)
            if colq and key in bq.index:
                return _safe_float(bq.loc[key, colq])
        except Exception:
            pass
    return None

def extra_fundamentals_snapshot(ticker: str) -> Dict[str, Optional[float]]:
    """
    Compute robust valuation/return ratios with best-effort fallbacks:
      pe_ttm       ≈ Market Cap / Net Income (TTM)
      ps_ttm       ≈ Market Cap / Revenue (TTM)
      ev_to_ebitda ≈ Enterprise Value / EBITDA (TTM)
      roic_pct     ≈ NOPAT / Invested Capital, where
                     NOPAT ≈ EBIT * (1 - 21%); Invested ≈ Debt + Equity - (Cash + ST inv)
    Returns a dict; fields may be None if data unavailable.
    """
    ia, iq, ba, bq, ca, cq = _yf_frames(ticker)
    info = _yf_info(ticker)

    # Building blocks from your existing helpers
    rev_ttm = _get_ttm_or_annual(iq, ia, "Total Revenue")
    net_income_ttm = _get_ttm_or_annual(iq, ia, "Net Income")
    ebitda_ttm = _get_ttm_or_annual(iq, ia, "Ebitda") or _get_ttm_or_annual(iq, ia, "EBITDA")
    ebit_ttm = _ebit_ttm(iq, ia)

    total_debt = _get_ttm_or_annual(bq, ba, "Total Debt") or _get_ttm_or_annual(bq, ba, "Short Long Term Debt Total")
    cash = _get_ttm_or_annual(bq, ba, "Cash And Cash Equivalents")
    st_inv = _get_ttm_or_annual(bq, ba, "Other Short Term Investments")
    cash_like = (_safe_float(cash) or 0.0) + (_safe_float(st_inv) or 0.0)

    equity = _total_equity_latest(ba, bq)

    mkt_cap, enterprise_value = _market_caps_from_info(info)
    # If EV missing, approximate EV = Market Cap + Debt - Cash
    if enterprise_value is None and _safe_float(mkt_cap) is not None and _safe_float(total_debt) is not None:
        enterprise_value = float(mkt_cap) + float(total_debt) - float(cash_like)

    # Ratios
    pe_ttm = None
    if _safe_float(mkt_cap) is not None and _safe_float(net_income_ttm) not in (None, 0):
        pe_ttm = float(mkt_cap) / float(net_income_ttm)

    ps_ttm = None
    if _safe_float(mkt_cap) is not None and _safe_float(rev_ttm) not in (None, 0):
        ps_ttm = float(mkt_cap) / float(rev_ttm)

    ev_to_ebitda = None
    if _safe_float(enterprise_value) is not None and _safe_float(ebitda_ttm) not in (None, 0):
        ev_to_ebitda = float(enterprise_value) / float(ebitda_ttm)

    # ROIC (approx): NOPAT / InvestedCapital
    roic_pct = None
    try:
        if _safe_float(ebit_ttm) is not None:
            nopat = float(ebit_ttm) * (1.0 - 0.21)  # 21% flat tax proxy
            invested = None
            if _safe_float(total_debt) is not None and _safe_float(equity) is not None:
                invested = float(total_debt) + float(equity) - float(cash_like)
            if invested and invested != 0:
                roic_pct = nopat / invested * 100.0
    except Exception:
        pass

    return {
        "pe_ttm": _safe_float(pe_ttm),
        "ps_ttm": _safe_float(ps_ttm),
        "ev_to_ebitda": _safe_float(ev_to_ebitda),
        "roic_pct": _safe_float(roic_pct),
        "market_cap": _safe_float(mkt_cap),
        "enterprise_value": _safe_float(enterprise_value),
    }

def fundamentals_table_extended(ticker: str) -> _pd.DataFrame:
    """
    Returns your existing fundamentals table with an extra block of valuation ratios.
    """
    base = fundamentals_table(ticker)
    x = extra_fundamentals_snapshot(ticker)

    add_rows = [
        ("—", "—"),  # spacer
        ("P/E (TTM)", f"{x['pe_ttm']:.2f}" if x.get("pe_ttm") is not None else "—"),
        ("P/S (TTM)", f"{x['ps_ttm']:.2f}" if x.get("ps_ttm") is not None else "—"),
        ("EV/EBITDA (TTM)", f"{x['ev_to_ebitda']:.2f}" if x.get("ev_to_ebitda") is not None else "—"),
        ("ROIC (approx)", f"{x['roic_pct']:.1f}%" if x.get("roic_pct") is not None else "—"),
        ("Market cap", f"${x['market_cap']:,.0f}" if x.get("market_cap") is not None else "—"),
        ("Enterprise value", f"${x['enterprise_value']:,.0f}" if x.get("enterprise_value") is not None else "—"),
    ]
    add_df = _pd.DataFrame(add_rows, columns=["Metric", ticker])
    try:
        return _pd.concat([base, add_df], ignore_index=True)
    except Exception:
        return add_df

def format_fundamentals_md_extended(ticker: str) -> str:
    """
    Extends your model-friendly fundamentals markdown with valuation/return ratios.
    """
    base_md = format_fundamentals_md(ticker)
    x = extra_fundamentals_snapshot(ticker)

    def _fmt(v, pct=False):
        if v is None:
            return "—"
        return f"{v:.1f}%" if pct else f"{v:.2f}"

    lines = [
        base_md.rstrip(),
        f"- Valuation: P/E { _fmt(x.get('pe_ttm')) }, P/S { _fmt(x.get('ps_ttm')) }, EV/EBITDA { _fmt(x.get('ev_to_ebitda')) }",
        f"- Returns: ROIC ~{ _fmt(x.get('roic_pct'), pct=True) }",
        "",
    ]
    return "\n".join(lines)
