from __future__ import annotations

import streamlit as st

from ai_agent.settings import get_openai_key
from ai_agent.catalysts import clear_catalyst_cache
from ai_agent.news import clear_news_cache
from ui.news_state import clear_news_snapshot


def render_sidebar():
    with st.sidebar:
        st.subheader("Authentication")
        current_key, source = get_openai_key()

        if current_key:
            st.success(f"OpenAI key loaded from {source}.")
            if st.button("Use a different key"):
                st.session_state["openai_api_key"] = ""
                st.rerun()

        if not current_key:
            typed = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            if typed:
                st.session_state["openai_api_key"] = typed
                st.success("Key saved for this session.")
                st.rerun()

        st.markdown("---")
        st.subheader("Display")
        high_contrast = st.toggle(
            "High-contrast mode",
            value=bool(st.session_state.get("ui_high_contrast", False)),
            help="Brighter text and borders for dark theme",
        )
        st.session_state["ui_high_contrast"] = bool(high_contrast)

        st.subheader("Strategy Controls")
        hf_mode = st.checkbox("Enable hedge-fund-style analysis", value=True)
        lookback_days = st.slider("News lookback (days)", 1, 60, 30)
        max_news = st.slider("Max news per ticker", 5, 50, 20, step=5)
        risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1)
        account_equity = st.number_input("Account equity ($)", min_value=1000, value=25000, step=1000)

        st.caption("News sources")
        use_rss = st.checkbox("Yahoo Finance RSS", value=True)
        use_yf = st.checkbox("yfinance API", value=True)
        use_reuters = st.checkbox("Reuters RSS", value=True)
        use_sec = st.checkbox("SEC EDGAR filings", value=True)

        st.markdown("---")
        st.subheader("Options Expiry")
        expiry_mode = st.radio(
            "How to choose expiry?",
            ["Auto (nearest 7–45 DTE)", "Pick specific expiry"],
            index=0,
        )

        if st.button("Refresh caches"):
            clear_news_cache()
            clear_catalyst_cache()
            clear_news_snapshot()
            st.toast("News & catalysts caches cleared.")

    return (
        current_key,
        source,
        hf_mode,
        lookback_days,
        max_news,
        risk_profile,
        account_equity,
        use_rss,
        use_yf,
        use_reuters,
        use_sec,
        expiry_mode,
    )


