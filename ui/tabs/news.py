from __future__ import annotations

import streamlit as st

from ai_agent.news import format_news_digest
from ui.news_state import (
    build_news_params,
    clear_news_snapshot,
    ensure_news_snapshot,
    get_news_snapshot,
)


def render_news_tab(
    *,
    ticker_a: str,
    ticker_b: str,
    go: bool,
    hf_mode: bool,
    lookback_days: int,
    max_news: int,
    use_rss: bool,
    use_yf: bool,
    use_reuters: bool,
    use_sec: bool,
) -> None:
    st.markdown("#### Recent News & Filings (used as model context)")
    st.caption(
        "Enable/disable sources in the sidebar | Adjust lookback/max items |Click **Compare & Analyze** "
        "to fetch latest. (cached ~15m; SEC map ~24h)"
    )

    params = build_news_params(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        lookback_days=lookback_days,
        max_news=max_news,
        use_rss=use_rss,
        use_yf=use_yf,
        use_reuters=use_reuters,
        use_sec=use_sec,
        hf_mode=hf_mode,
    )

    snapshot = None
    refreshed = False

    if not params["hf_mode"]:
        clear_news_snapshot()
        st.info("Hedge-fund-style analysis is disabled, so news ingestion is off.")
        st.caption("Enable hedge-fund-style analysis in the sidebar to fetch news & filings.")
    else:
        if go:
            snapshot, refreshed = ensure_news_snapshot(params, force_refetch=True)
            if snapshot and refreshed and snapshot.get("digest_changed"):
                st.info("News updated since your last analysis. Consider re-running the report to reflect changes.")
        else:
            snapshot = get_news_snapshot(params)
            if snapshot is None:
                st.info("Press **Compare & Analyze** to fetch the latest news & filings with your current source toggles.")
            else:
                st.caption("Showing cached news snapshot from your last analysis run. Press Compare & Analyze to refresh.")

        if snapshot:
            news_a_all = snapshot.get("news_a_all") or {"merged": [], "by_source": {}}
            news_b_all = snapshot.get("news_b_all") or {"merged": [], "by_source": {}}
            news_a = news_a_all.get("merged", [])
            news_b = news_b_all.get("merged", [])

            if snapshot.get("digest"):
                st.session_state.setdefault("news_digest_hash", snapshot["digest"])

            coln1, coln2 = st.columns(2)
            with coln1:
                with st.expander(f"{ticker_a} news ({len(news_a)})", expanded=True):
                    st.caption(
                        "Sources �?� "
                        f"RSS: {len(news_a_all.get('by_source', {}).get('yahoo_rss', []))} | "
                        f"yfinance: {len(news_a_all.get('by_source', {}).get('yfinance', []))} | "
                        f"Reuters: {len(news_a_all.get('by_source', {}).get('reuters_rss', []))} | "
                        f"SEC: {len(news_a_all.get('by_source', {}).get('sec_edgar', []))}"
                    )
                    try:
                        from ai_agent.news import format_news_table

                        tbl_a = format_news_table(news_a)
                        st.dataframe(tbl_a, width="stretch", hide_index=True)
                    except Exception:
                        st.write(format_news_digest(ticker_a, news_a))

            with coln2:
                with st.expander(f"{ticker_b} news ({len(news_b)})", expanded=True):
                    st.caption(
                        "Sources"
                        f"RSS: {len(news_b_all.get('by_source', {}).get('yahoo_rss', []))} | "
                        f"yfinance: {len(news_b_all.get('by_source', {}).get('yfinance', []))} | "
                        f"Reuters: {len(news_b_all.get('by_source', {}).get('reuters_rss', []))} | "
                        f"SEC: {len(news_b_all.get('by_source', {}).get('sec_edgar', []))}"
                    )
                    try:
                        from ai_agent.news import format_news_table

                        tbl_b = format_news_table(news_b)
                        st.dataframe(tbl_b, width="stretch", hide_index=True)
                    except Exception:
                        st.write(format_news_digest(ticker_b, news_b))
        elif params["hf_mode"] and go:
            st.warning("Unable to load news for the current selection.")

    # End-cap guard to prevent ghost content bleeding (tab container adds another guard)
    st.markdown('<div class="tab-bleed-guard"></div>', unsafe_allow_html=True)
