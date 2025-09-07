from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# ===============================
# Data access (robust helpers)
# ===============================

def _collapse_multiindex_cols(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    If yfinance returns MultiIndex columns (e.g., ('Close','NVDA')), pick the
    cross-section for the given ticker and return a single-level column frame.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # Try to slice on any level that contains the ticker
    for lvl in range(df.columns.nlevels - 1, -1, -1):
        try:
            lv = df.columns.get_level_values(lvl)
            if ticker in lv:
                df = df.xs(ticker, axis=1, level=lvl, drop_level=True)
                break
        except Exception:
            continue

    # If still MultiIndex (rare), flatten by joining levels
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns]

    return df


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to title case and ensure we have High/Low/Close.
    If only Adj Close exists, use it as Close fallback.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["High", "Low", "Close"])

    # Title-case columns to make selection robust
    new_cols = []
    for c in df.columns:
        s = str(c).strip()
        # Keep common names predictable
        if s.lower() in ("open", "high", "low", "close", "adj close", "volume"):
            s = s.title()
        new_cols.append(s)
    df = df.copy()
    df.columns = new_cols

    # Convert numerics
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If Close missing but Adj Close present, map it
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # Keep only OHLC we need
    cols = [c for c in ["High", "Low", "Close"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["High", "Low", "Close"])

    out = df[cols].dropna(how="any")
    return out


def fetch_ohlc(
    ticker: str,
    *,
    lookback_days: int = 180,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV history using yfinance.
    Returns a DataFrame with ['High','Low','Close'] indexed by DatetimeIndex (UTC-naive).
    Handles MultiIndex columns and missing Close gracefully.
    """
    period = f"{max(7, int(lookback_days))}d"
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["High", "Low", "Close"])

    # Normalize index to UTC-naive timestamps
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)  # if tz-aware
    except Exception:
        pass
    df.index = idx

    # Collapse multiindex and normalize columns
    df = _collapse_multiindex_cols(df, ticker)
    df = _normalize_ohlc_columns(df)
    return df


def fetch_close_series(
    ticker: str,
    *,
    lookback_days: int = 365,
    interval: str = "1d",
) -> pd.Series:
    """
    Convenience: return a clean 1-D Close series (float), aligned & sorted.
    """
    o = fetch_ohlc(ticker, lookback_days=lookback_days, interval=interval)
    if o.empty or "Close" not in o.columns:
        return pd.Series(dtype=float)

    s = o["Close"]
    # Ensure 1-D float series
    if isinstance(s, pd.DataFrame):
        # If a DataFrame slipped through, take the first column
        s = s.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


# ===============================
# Volatility / ATR computations
# ===============================

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Wilder's True Range:
      TR_t = max( high_t - low_t, |high_t - prev_close|, |low_t - prev_close| )
    """
    prev_close = close.shift(1)
    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()
    tr = pd.concat([range1, range2, range3], axis=1).max(axis=1)
    return tr


def atr_series(
    ohlc: pd.DataFrame,
    window: int = 14,
    method: str = "wilder",
) -> pd.Series:
    """
    Compute Average True Range (ATR) from OHLC data.
    method:
      - 'wilder' : Wilder's smoothing (EMA with alpha = 1/window)
      - 'sma'    : simple moving average of TR
    Returns a pd.Series aligned to the input index.
    """
    if ohlc is None or ohlc.empty:
        return pd.Series(dtype=float)

    for col in ["High", "Low", "Close"]:
        if col not in ohlc.columns:
            return pd.Series(dtype=float)

    tr = true_range(ohlc["High"], ohlc["Low"], ohlc["Close"])

    if method == "wilder":
        alpha = 1.0 / float(window)
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
    else:
        atr = tr.rolling(window=window, min_periods=window).mean()

    return atr


def atr_value_from_ticker(
    ticker: str,
    *,
    window: int = 14,
    lookback_days: int = 180,
    method: str = "wilder",
) -> Optional[float]:
    """
    Convenience: fetch OHLC and return the *latest* ATR value.
    """
    ohlc = fetch_ohlc(ticker, lookback_days=lookback_days, interval="1d")
    if ohlc.empty:
        return None
    atr = atr_series(ohlc, window=window, method=method)
    if atr is None or atr.empty:
        return None
    val = float(atr.iloc[-1])
    if np.isnan(val) or np.isinf(val):
        return None
    return val


def realized_vol(
    close: pd.Series,
    *,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> Optional[float]:
    """
    Rolling realized volatility (stdev of log returns).
    Returns the *latest* value (as a decimal, e.g., 0.32 means 32%).
    """
    if close is None or close.empty:
        return None
    rets = np.log(close / close.shift(1)).dropna()
    if rets.empty:
        return None
    rv = rets.rolling(window=window, min_periods=window).std()
    if rv is None or rv.empty or np.isnan(rv.iloc[-1]):
        return None
    val = float(rv.iloc[-1])
    if annualize:
        val *= np.sqrt(trading_days)
    return float(val)


# ===============================
# Trend helpers (for badges)
# ===============================

def slope_percent(
    series: pd.Series,
    *,
    lookback: int = 60,
) -> Optional[float]:
    """
    Simple linear slope (as % of price) over a lookback window.
    Useful for classifying trend Up/Down/Sideways.
    """
    if series is None or series.empty or len(series) < 3:
        return None
    s = series.dropna().iloc[-lookback:]
    if len(s) < 3:
        return None
    x = np.arange(len(s))
    y = s.values.astype(float)
    a = np.polyfit(x, y, 1)[0]  # slope
    ref = np.nanmean(y)
    if ref == 0 or np.isnan(ref):
        return None
    return float(a / ref)


# ===============================
# Pair / Relative-value analytics
# ===============================

@dataclass
class PairStats:
    """
    Summary for a market-neutral pair analysis (A vs B).
    """
    beta_ab: Optional[float]
    corr_ab: Optional[float]
    hedge_ratio: Optional[float]
    spread_last: Optional[float]
    spread_zscore: Optional[float]
    window_used: int
    a_close_last: Optional[float]
    b_close_last: Optional[float]


def _align_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two series on the intersection of their timestamps, drop NaNs.
    """
    if a is None or b is None or a.empty or b.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    j = j.dropna()
    return j["a"], j["b"]


def rolling_beta(a_ret: pd.Series, b_ret: pd.Series, window: int = 60) -> Optional[float]:
    """
    OLS beta of A on B over the last `window` points (returns).
    """
    a_ret, b_ret = _align_series(a_ret, b_ret)
    if len(a_ret) < max(10, window):
        return None
    x = b_ret.iloc[-window:].values.astype(float)
    y = a_ret.iloc[-window:].values.astype(float)
    if np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return None
    # beta = cov(A,B)/var(B)
    beta = float(np.cov(y, x)[0, 1] / np.var(x))
    return beta


def rolling_corr(a_ret: pd.Series, b_ret: pd.Series, window: int = 60) -> Optional[float]:
    """
    Pearson correlation of A vs B over the last `window` points.
    """
    a_ret, b_ret = _align_series(a_ret, b_ret)
    if len(a_ret) < max(10, window):
        return None
    r = float(pd.Series(a_ret.iloc[-window:]).corr(pd.Series(b_ret.iloc[-window:])))
    if np.isnan(r):
        return None
    return r


def zscore_series(s: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score of a series.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    m = s.rolling(window).mean()
    v = s.rolling(window).std()
    z = (s - m) / v
    return z


def compute_pair_stats(
    ticker_a: str,
    ticker_b: str,
    *,
    lookback_days: int = 365,
    window: int = 60,
) -> PairStats:
    """
    Compute a hedge ratio (beta of A on B), a beta-hedged spread, and its z-score.
    This supports a market-neutral LONG/SHORT idea:
      - If z << 0 (e.g., < -2), consider LONG A / SHORT B (expect spread mean reversion up)
      - If z >> 0 (e.g., > +2), consider SHORT A / LONG B
    """
    a_close = fetch_close_series(ticker_a, lookback_days=lookback_days)
    b_close = fetch_close_series(ticker_b, lookback_days=lookback_days)
    a_close, b_close = _align_series(a_close, b_close)

    if len(a_close) < max(60, window) or len(b_close) < max(60, window):
        return PairStats(
            beta_ab=None, corr_ab=None, hedge_ratio=None,
            spread_last=None, spread_zscore=None,
            window_used=0,
            a_close_last=float(a_close.iloc[-1]) if len(a_close) else None,
            b_close_last=float(b_close.iloc[-1]) if len(b_close) else None,
        )

    # Log returns for beta/corr stability
    a_ret = np.log(a_close / a_close.shift(1)).dropna()
    b_ret = np.log(b_close / b_close.shift(1)).dropna()
    a_ret, b_ret = _align_series(a_ret, b_ret)

    beta = rolling_beta(a_ret, b_ret, window=window)
    corr = rolling_corr(a_ret, b_ret, window=window)
    if beta is None or np.isnan(beta):
        beta = 1.0  # fallback

    # Beta-hedged spread on PRICE
    spread = a_close - beta * b_close
    z = zscore_series(spread, window=window)
    z_last = float(z.iloc[-1]) if not z.empty and not np.isnan(z.iloc[-1]) else None

    return PairStats(
        beta_ab=float(beta) if beta is not None else None,
        corr_ab=float(corr) if corr is not None else None,
        hedge_ratio=float(beta) if beta is not None else None,
        spread_last=float(spread.iloc[-1]) if not spread.empty else None,
        spread_zscore=z_last,
        window_used=min(len(spread), window),
        a_close_last=float(a_close.iloc[-1]) if len(a_close) else None,
        b_close_last=float(b_close.iloc[-1]) if len(b_close) else None,
    )


# ===============================
# Small container for ATR output
# ===============================

@dataclass
class AtrSummary:
    atr: Optional[float]
    spot: Optional[float]
    atr_pct_of_price: Optional[float]


def atr_summary_from_ticker(
    ticker: str,
    *,
    window: int = 14,
    lookback_days: int = 180,
) -> AtrSummary:
    """
    Returns ATR in dollars, the latest Close (spot proxy), and ATR as % of price.
    """
    ohlc = fetch_ohlc(ticker, lookback_days=lookback_days, interval="1d")
    if ohlc.empty:
        return AtrSummary(atr=None, spot=None, atr_pct_of_price=None)

    atr = atr_series(ohlc, window=window, method="wilder")
    if atr is None or atr.empty:
        return AtrSummary(atr=None, spot=None, atr_pct_of_price=None)

    spot = float(ohlc["Close"].iloc[-1])
    atr_val = float(atr.iloc[-1])
    if np.isnan(atr_val) or np.isinf(atr_val):
        return AtrSummary(atr=None, spot=spot, atr_pct_of_price=None)

    atr_pct = (atr_val / spot) if spot not in (None, 0) else None
    return AtrSummary(atr=atr_val, spot=spot, atr_pct_of_price=(float(atr_pct) if atr_pct is not None else None))
