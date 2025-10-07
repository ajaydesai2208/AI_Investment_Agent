from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import streamlit as st

from ai_agent.metrics import snapshot
from ai_agent.options import options_snapshot, OptionsSnapshot
from ai_agent.catalysts import get_catalysts, format_catalysts_md
from ai_agent.quant import compute_pair_stats
from ai_agent.prices import get_spark_series
from ui.helpers import _trend_label, _regime_label


def preload_all(ticker_a: str, ticker_b: str, expiry_a, expiry_b, expiry_mode: str):
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

    spark_a = get_spark_series(ticker_a, months=3, normalize=True)
    spark_b = get_spark_series(ticker_b, months=3, normalize=True)

    trend_a = _trend_label(spark_a)
    trend_b = _trend_label(spark_b)
    regime_a = _regime_label(opt_a.implied_move_pct, a_snap.get("vol20_ann_pct"), opt_a.dte)
    regime_b = _regime_label(opt_b.implied_move_pct, b_snap.get("vol20_ann_pct"), opt_b.dte)

    return a_snap, b_snap, opt_a, opt_b, cat_a, cat_b, cat_md_a, cat_md_b, pair, spark_a, spark_b, trend_a, trend_b, regime_a, regime_b


