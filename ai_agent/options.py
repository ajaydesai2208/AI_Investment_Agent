from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st
import yfinance as yf


__all__ = [
    "OptionsSnapshot",
    "list_expiries",
    "options_snapshot",
    "format_options_fact",
]


# ---------------- Models ----------------

@dataclass
class OptionsSnapshot:
    ticker: str
    spot: Optional[float]
    expiry: Optional[str]              # ISO date string, e.g. "2025-01-17"
    dte: Optional[int]                 # days to expiry
    atm_strike: Optional[float]
    call_mid: Optional[float]
    put_mid: Optional[float]
    straddle_debit: Optional[float]
    implied_move_pct: Optional[float]  # straddle / spot * 100
    call_iv_pct: Optional[float]
    put_iv_pct: Optional[float]
    atm_iv_pct: Optional[float]

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------- Helpers ----------------

def _to_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).tz_localize(None)

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None

def _mid_from_row(row: pd.Series) -> Optional[float]:
    # Prefer (bid+ask)/2; fall back to lastPrice if needed
    bid = None
    ask = None
    for bkey in ("bid", "bidPrice", "Bid"):
        if bkey in row:
            bid = _safe_float(row[bkey])
            break
    for akey in ("ask", "askPrice", "Ask"):
        if akey in row:
            ask = _safe_float(row[akey])
            break
    if bid is not None and ask is not None and bid >= 0 and ask > 0:
        return (bid + ask) / 2.0

    for lkey in ("lastPrice", "Last Price", "lastTradePrice"):
        if lkey in row:
            v = _safe_float(row[lkey])
            if v is not None:
                return v
    return None

def _iv_from_row(row: pd.Series) -> Optional[float]:
    # yfinance returns iv as fraction (e.g., 0.45). Convert to %.
    for key in ("impliedVolatility", "impliedVol"):
        if key in row:
            v = _safe_float(row[key])
            if v is not None:
                return v * 100.0 if v < 1.0 else v
    return None

def _nearest_expiry(expiries: List[str], min_dte: int, max_dte: int) -> Optional[str]:
    """Pick the nearest expiry with DTE in [min_dte, max_dte], else the smallest positive DTE."""
    if not expiries:
        return None
    today = pd.Timestamp(datetime.now(tz=timezone.utc).date())
    dtes = []
    for ex in expiries:
        try:
            ts = _to_ts(ex)
            dte = int((ts - today).days)
            if dte > 0:
                dtes.append((ex, dte))
        except Exception:
            continue
    if not dtes:
        return None
    # first try range
    in_range = [e for e in dtes if min_dte <= e[1] <= max_dte]
    if in_range:
        return sorted(in_range, key=lambda x: x[1])[0][0]
    # otherwise closest positive
    return sorted(dtes, key=lambda x: x[1])[0][0]

def _atm_rows(spot: float, calls: pd.DataFrame, puts: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return the call and put row with strike nearest to spot."""
    def nearest(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype="float64")
        if "strike" not in df.columns:
            # legacy column name sometimes "Strike"
            if "Strike" in df.columns:
                df = df.rename(columns={"Strike": "strike"})
            else:
                return pd.Series(dtype="float64")
        idx = (df["strike"] - spot).abs().idxmin()
        return df.loc[idx]

    c = nearest(calls)
    p = nearest(puts)
    return c, p


# ---------------- Public API ----------------

@st.cache_data(ttl=900, show_spinner=False)
def list_expiries(ticker: str) -> List[str]:
    """
    Return a sorted list of available option expiry dates (ISO strings) for a ticker.
    """
    try:
        t = yf.Ticker(ticker)
        exps = list(getattr(t, "options", []) or [])
    except Exception:
        exps = []
    # sort ascending by date
    try:
        exps = sorted(exps, key=lambda s: _to_ts(s))
    except Exception:
        pass
    return exps


@st.cache_data(ttl=300, show_spinner=False)
def options_snapshot(
    ticker: str,
    min_dte: int = 7,
    max_dte: int = 45,
    *,
    expiry: Optional[str] = None,   # NEW: explicit expiry override
) -> OptionsSnapshot:
    """
    Build an options snapshot focused on either:
      - the provided `expiry`, OR
      - the nearest expiry in [min_dte, max_dte] (fallback to nearest positive DTE).

    Returns an OptionsSnapshot dataclass (use .as_dict()).
    """
    spot = None
    chosen_expiry = expiry
    dte = None
    atm_strike = None
    call_mid = None
    put_mid = None
    straddle = None
    call_iv = None
    put_iv = None
    atm_iv = None

    try:
        t = yf.Ticker(ticker)
        # spot
        finfo = getattr(t, "fast_info", {}) or {}
        spot = _safe_float(finfo.get("last_price") or finfo.get("lastPrice") or finfo.get("regularMarketPrice"))

        # expiries
        expiries = list_expiries(ticker)
        if not chosen_expiry:
            chosen_expiry = _nearest_expiry(expiries, min_dte=min_dte, max_dte=max_dte)
        if not chosen_expiry:
            return OptionsSnapshot(ticker, spot, None, None, None, None, None, None, None, None, None, None)

        # DTE
        try:
            ex_ts = _to_ts(chosen_expiry)
            dte = int((ex_ts - pd.Timestamp(datetime.now(tz=timezone.utc).date())).days)
        except Exception:
            dte = None

        # chain
        chain = t.option_chain(chosen_expiry)
        calls_df = getattr(chain, "calls", pd.DataFrame())
        puts_df = getattr(chain, "puts", pd.DataFrame())

        if spot is None and not calls_df.empty and "lastPrice" in calls_df.columns:
            # fallback: infer rough spot from atm strike if needed
            try:
                spot = float(calls_df.loc[(calls_df["inTheMoney"] == True), "strike"].max())  # noqa: E712
            except Exception:
                pass

        if spot is None or (calls_df.empty and puts_df.empty):
            return OptionsSnapshot(ticker, spot, chosen_expiry, dte, None, None, None, None, None, None, None, None)

        c_row, p_row = _atm_rows(spot, calls_df, puts_df)
        if not c_row.empty:
            call_mid = _mid_from_row(c_row)
            call_iv = _iv_from_row(c_row)
        if not p_row.empty:
            put_mid = _mid_from_row(p_row)
            put_iv = _iv_from_row(p_row)
        atm_strike = _safe_float(c_row.get("strike") if not c_row.empty else p_row.get("strike"))

        if call_mid is not None and put_mid is not None:
            straddle = call_mid + put_mid
        if straddle is not None and spot and spot > 0:
            implied_move_pct = (straddle / spot) * 100.0
        else:
            implied_move_pct = None

        if call_iv is not None and put_iv is not None:
            atm_iv = (call_iv + put_iv) / 2.0

        return OptionsSnapshot(
            ticker=ticker,
            spot=spot,
            expiry=str(chosen_expiry) if chosen_expiry else None,
            dte=dte,
            atm_strike=atm_strike,
            call_mid=call_mid,
            put_mid=put_mid,
            straddle_debit=straddle,
            implied_move_pct=implied_move_pct,
            call_iv_pct=call_iv,
            put_iv_pct=put_iv,
            atm_iv_pct=atm_iv,
        )
    except Exception:
        # Fail soft; return empty structure with whatever we have
        return OptionsSnapshot(ticker, spot, chosen_expiry, dte, atm_strike, call_mid, put_mid, straddle, None, None, None, None)


def format_options_fact(s: OptionsSnapshot) -> str:
    """
    Render a compact single-paragraph fact line for the prompt/report.
    """
    def fmt(x, pct=False):
        if x is None:
            return "—"
        try:
            v = float(x)
        except Exception:
            return "—"
        return f"{v:.2f}%" if pct else f"{v:.2f}"

    return (
        f"{s.ticker} options (expiry {s.expiry}, DTE {s.dte}): "
        f"spot {fmt(s.spot)}, ATM {fmt(s.atm_strike)}, "
        f"call mid {fmt(s.call_mid)}, put mid {fmt(s.put_mid)}, "
        f"straddle {fmt(s.straddle_debit)} ⇒ implied move {fmt(s.implied_move_pct, pct=True)}, "
        f"ATM IV ~ {fmt(s.atm_iv_pct, pct=True)}"
    )
