from __future__ import annotations

import streamlit as st
from streamlit_searchbox import st_searchbox

from ai_agent.symbols import search_symbols, format_symbol_label
from ai_agent.options import list_expiries
from ui.helpers import _is_valid_ticker, _default_expiry_index


def _extract_ticker(selection: str) -> str:
    if not selection:
        return ""
    parts = str(selection).split(" — ", 1)
    return parts[0].strip().upper() if parts and parts[0] else str(selection).strip().upper()


def render_tickers_and_expiry(expiry_mode: str):
    col1, col2 = st.columns(2)
    with col1:
        sel_a = st_searchbox(
            search_function=lambda q: [format_symbol_label(m, q) for m in search_symbols(q, limit=100)],
            placeholder="e.g., AAPL or Apple",
            label="Ticker A",
            key="sb_tkr_a",
        )
    with col2:
        sel_b = st_searchbox(
            search_function=lambda q: [format_symbol_label(m, q) for m in search_symbols(q, limit=100)],
            placeholder="e.g., MSFT or Microsoft",
            label="Ticker B",
            key="sb_tkr_b",
        )

    ticker_a = _extract_ticker(sel_a)
    ticker_b = _extract_ticker(sel_b)

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

    # Primary CTA wrapper for full-width, sleek styling via CSS
    st.markdown('<div class="primary-cta">', unsafe_allow_html=True)
    go = st.button("Compare & Analyze", disabled=not tickers_ok)
    st.markdown('</div>', unsafe_allow_html=True)

    if not tickers_ok:
        st.caption("Enter two valid tickers (A–Z, 0–9, '.', '-'). Example: AAPL, MSFT")
        st.info("Enter two valid tickers to load data.")

    return ticker_a, ticker_b, expiry_a, expiry_b, go, tickers_ok


