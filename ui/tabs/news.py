from __future__ import annotations

import hashlib

import streamlit as st

from ai_agent.news import get_recent_news, format_news_digest


def render_news_tab(*, ticker_a: str, ticker_b: str, go: bool, hf_mode: bool, lookback_days: int, max_news: int, use_rss: bool, use_yf: bool, use_reuters: bool, use_sec: bool):
    st.markdown("#### Recent News & Filings (used as model context)")
    st.caption(
        "Enable/disable sources in the sidebar • Adjust lookback/max items • Click **Compare & Analyze** to fetch latest. (cached ~15m; SEC map ~24h)"
    )
    if go:
        news_kwargs = dict(use_rss=use_rss, use_yf=use_yf, use_reuters=use_reuters, use_sec=use_sec)
        news_a_all = get_recent_news(ticker_a, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
        news_b_all = get_recent_news(ticker_b, lookback_days, max_news, **news_kwargs) if hf_mode else {"merged": [], "by_source": {}}
        news_a = news_a_all["merged"]
        news_b = news_b_all["merged"]

        try:
            digest_str = str(news_a) + "|" + str(news_b)
            digest = hashlib.sha256(digest_str.encode("utf-8")).hexdigest()
            prev = st.session_state.get("news_digest_hash")
            st.session_state["news_digest_hash"] = digest
            if prev and prev != digest:
                st.info("News updated since your last analysis. Consider re-running the report to reflect changes.")
        except Exception:
            pass

        coln1, coln2 = st.columns(2)
        with coln1:
            with st.expander(f"{ticker_a} news ({len(news_a)})", expanded=True):
                st.caption(
                    "Sources — "
                    f"RSS: {len(news_a_all['by_source'].get('yahoo_rss', []))} | "
                    f"yfinance: {len(news_a_all['by_source'].get('yfinance', []))} | "
                    f"Reuters: {len(news_a_all['by_source'].get('reuters_rss', []))} | "
                    f"SEC: {len(news_a_all['by_source'].get('sec_edgar', []))}"
                )
                try:
                    from ai_agent.news import format_news_table
                    tbl_a = format_news_table(news_a)
                    st.dataframe(tbl_a, width='stretch', hide_index=True)
                except Exception:
                    st.write(format_news_digest(ticker_a, news_a))
        with coln2:
            with st.expander(f"{ticker_b} news ({len(news_b)})", expanded=True):
                st.caption(
                    "Sources — "
                    f"RSS: {len(news_b_all['by_source'].get('yahoo_rss', []))} | "
                    f"yfinance: {len(news_b_all['by_source'].get('yfinance', []))} | "
                    f"Reuters: {len(news_b_all['by_source'].get('reuters_rss', []))} | "
                    f"SEC: {len(news_b_all['by_source'].get('sec_edgar', []))}"
                )
                try:
                    from ai_agent.news import format_news_table
                    tbl_b = format_news_table(news_b)
                    st.dataframe(tbl_b, width='stretch', hide_index=True)
                except Exception:
                    st.write(format_news_digest(ticker_b, news_b))
    else:
        st.info("Press **Compare & Analyze** to fetch the latest news & filings with your current source toggles.")


