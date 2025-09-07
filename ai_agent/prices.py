from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, List

import pandas as pd
import streamlit as st
import yfinance as yf

__all__ = [
    "timeframe_params",
    "load_prices",
    "merge_for_compare",
]

# ---------- Timeframe mapping ----------

def timeframe_params(tf: str) -> Dict:
    """
    Map a human timeframe to yfinance download params.
    Supports: 1D, 5D, 1M, 3M, 6M, 1Y, 5Y, YTD
    """
    tf = (tf or "").upper()
    if tf == "1D":
        return {"period": "1d", "interval": "5m"}
    if tf == "5D":
        return {"period": "5d", "interval": "15m"}
    if tf == "1M":
        return {"period": "1mo", "interval": "1d"}
    if tf == "3M":
        return {"period": "3mo", "interval": "1d"}
    if tf == "6M":
        return {"period": "6mo", "interval": "1d"}
    if tf == "1Y":
        return {"period": "1y", "interval": "1d"}
    if tf == "5Y":
        return {"period": "5y", "interval": "1wk"}
    if tf == "YTD":
        start = datetime(datetime.now().year, 1, 1)
        return {"start": start.strftime("%Y-%m-%d"), "interval": "1d"}
    # default
    return {"period": "6mo", "interval": "1d"}

# ---------- Price loading ----------

@st.cache_data(ttl=300, show_spinner=False)
def load_prices(ticker: str, tf: str, show_vol: bool = False) -> pd.DataFrame:
    """
    Download OHLCV for a ticker & timeframe.
    Returns a DataFrame indexed by datetime with at least a 'Close' column
    (auto-adjusted). Optionally includes 'Volume'.
    """
    params = timeframe_params(tf)
    try:
        df = yf.download(ticker, **params, auto_adjust=True, progress=False)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return df

    # Prefer Adj Close if present
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = df[[close_col]].rename(columns={close_col: "Close"})
    if show_vol and "Volume" in df.columns:
        out["Volume"] = df["Volume"]
    return out

def _series_from_close(df: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
    """
    Coerce the 'Close' column to a 1-D numeric Series robustly, across
    pandas/yfinance variants where 'Close' might be a DataFrame.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s.name = col_name
    return s

def merge_for_compare(ticker_a: str, ticker_b: str, tf: str, normalize: bool = True, show_vol: bool = False) -> pd.DataFrame:
    """
    Load prices for two tickers and return a comparison DataFrame with columns
    named by the tickers. If normalize=True, each series is rebased to 100 at
    its first valid value in the selected window.
    """
    a = load_prices(ticker_a, tf, show_vol=show_vol)
    b = load_prices(ticker_b, tf, show_vol=show_vol)

    series_list: List[pd.Series] = []
    sa = _series_from_close(a, ticker_a)
    sb = _series_from_close(b, ticker_b)
    if sa is not None:
        series_list.append(sa)
    if sb is not None:
        series_list.append(sb)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1, join="outer").sort_index()

    if normalize and not df.empty:
        for col in df.columns:
            first_idx = df[col].first_valid_index()
            if first_idx is not None:
                base = df.at[first_idx, col]
                if pd.notna(base) and float(base) != 0.0:
                    df[col] = (df[col] / float(base)) * 100.0
    return df
