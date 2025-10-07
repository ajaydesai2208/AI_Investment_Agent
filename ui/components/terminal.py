from __future__ import annotations

from typing import Optional

import streamlit as st


def term_header(*, ticker_a: str, ticker_b: str, timeframe: str, risk_profile: str, z_hint: Optional[str] = None) -> None:
    st.markdown(
        f"""
        <div class="term-ribbon">
          <span>Report</span>
          <span class="pill">{ticker_a}</span>
          <span class="pill">vs</span>
          <span class="pill">{ticker_b}</span>
          <span class="pill">Timeframe: {timeframe}</span>
          <span class="pill">Profile: {risk_profile}</span>
          {f'<span class="pill">{z_hint}</span>' if z_hint else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def term_panel(title: Optional[str] = None):
    return st.container(border=False)


def term_kv(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="term-kv">
          <div class="term-k">{label}</div>
          <div class="term-v">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def term_meter(label: str, frac: float) -> None:
    frac = max(0.0, min(1.0, float(frac)))
    pct = int(round(frac * 100))
    st.markdown(
        f"""
        <div class="term-section">
          <div class="term-title">{label}</div>
          <div class="term-meter">
            <div class="bar"><div class="fill" style="width:{pct}%"></div></div>
            <div style="min-width:40px;text-align:right;">{pct}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


