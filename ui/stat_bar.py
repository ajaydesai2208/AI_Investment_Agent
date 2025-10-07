from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.helpers import _fmt_num, _regime_css_class


def render_stat_bar(
    *,
    ticker_a: str,
    ticker_b: str,
    opt_a,
    opt_b,
    trend_a: str,
    trend_b: str,
    regime_a: str,
    regime_b: str,
    risk_profile: str,
    account_equity,
):
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


def render_sparklines(*, ticker_a: str, ticker_b: str, spark_a, spark_b) -> None:
    sc1, sc2 = st.columns(2)
    with sc1:
        if spark_a is not None and not getattr(spark_a, "empty", False):
            st.caption(f"{ticker_a} — 3M spark (Index=100)")
            st.line_chart(pd.DataFrame(spark_a), height=90, use_container_width=True)
        else:
            st.caption(f"{ticker_a} — no data")
    with sc2:
        if spark_b is not None and not getattr(spark_b, "empty", False):
            st.caption(f"{ticker_b} — 3M spark (Index=100)")
            st.line_chart(pd.DataFrame(spark_b), height=90, use_container_width=True)
        else:
            st.caption(f"{ticker_b} — no data")


