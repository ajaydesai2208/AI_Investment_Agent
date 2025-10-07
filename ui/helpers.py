from __future__ import annotations

from typing import Optional
from datetime import date
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

from ai_agent.quant import slope_percent


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


