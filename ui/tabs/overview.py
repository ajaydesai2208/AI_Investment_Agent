from __future__ import annotations

import streamlit as st

from ai_agent.prices import merge_for_compare
from ai_agent.metrics import compare_table, fundamentals_table_extended as fundamentals_table


def render_overview_tab(*, ticker_a: str, ticker_b: str, a_snap: dict, b_snap: dict, cat_md_a: str, cat_md_b: str, pair=None):
    st.markdown("#### Pair Analyzer (beta-hedged spread)")
    def _pair_suggestion(z):
        if z is None:
            return "Insufficient data"
        if z <= -2.0:
            return f"**Consider LONG {ticker_a} / SHORT {ticker_b}** (z={z:.2f})"
        if z >= 2.0:
            return f"**Consider SHORT {ticker_a} / LONG {ticker_b}** (z={z:.2f})"
        return f"Neutral (z={z:.2f}); watch for ±2.0 extremes"
    if pair is None or getattr(pair, "window_used", 0) == 0:
        st.info("Not enough overlapping history to compute pair stats.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beta (A on B)", f"{pair.beta_ab:.2f}" if pair.beta_ab is not None else "—")
        c2.metric("Correlation", f"{pair.corr_ab:.2f}" if pair.corr_ab is not None else "—")
        c3.metric("Hedge ratio", f"{pair.hedge_ratio:.2f}" if pair.hedge_ratio is not None else "—")
        c4.metric("Spread z-score", f"{pair.spread_zscore:.2f}" if pair.spread_zscore is not None else "—")
        st.caption(_pair_suggestion(getattr(pair, "spread_zscore", None)))
    st.markdown("#### Price Chart", help="Select timeframe and normalize from the controls below.")
    cc1, cc2, cc3 = st.columns([1.1, 1.1, 1])
    timeframe = cc1.selectbox(
        "Chart timeframe",
        ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y"],
        index=5,
        key="timeframe_select",
    )
    normalize = cc2.checkbox("Normalize to 100", value=True, key="normalize_check")
    show_vol = cc3.checkbox("Show volume (load only)", value=False, key="showvol_check")

    chart_df = merge_for_compare(ticker_a, ticker_b, timeframe, normalize=normalize, show_vol=show_vol)
    if chart_df.empty:
        st.info("No price data available for the selected timeframe.")
    else:
        if normalize:
            st.caption("Index = 100 at the start of the selected period.")
        st.line_chart(chart_df, height=360, use_container_width=True)

    st.markdown("#### Snapshot & Factors")
    st.dataframe(compare_table(a_snap, b_snap), width='stretch')
    st.markdown("#### Fundamentals (TTM)")
    ff1, ff2 = st.columns(2)
    with ff1:
        st.dataframe(fundamentals_table(ticker_a), width='stretch')
    with ff2:
        st.dataframe(fundamentals_table(ticker_b), width='stretch')

    st.markdown("#### Catalyst Radar")
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.markdown(cat_md_a)
    with ccol2:
        st.markdown(cat_md_b)


