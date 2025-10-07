from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_agent.risk import sizing_summary_table, build_trade_plan
from ui.helpers import _sizing_fact_line, _infer_direction_for


def render_sizing_tab(*, ticker_a, ticker_b, a_snap, b_snap, opt_a, opt_b, risk_profile, account_equity, pair, trend_a, trend_b):
    st.markdown("#### Position Sizing (based on implied move & recent vol)")
    st.caption("Includes **Baseline** (implied-move–anchored) and **ATR-based** suggestions.")
    try:
        prof = (risk_profile or "Balanced").lower()
        frac = 0.005 if prof.startswith("cons") else (0.02 if prof.startswith("aggr") else 0.01)
        rb = float(account_equity) * float(frac)
        st.markdown(f"**Risk budget:** ${rb:,.0f} (at {int(frac*100)}%)")
    except Exception:
        st.caption("Risk budget unavailable.")

    with st.spinner("Computing sizing..."):
        size_a_df = sizing_summary_table(
            ticker=ticker_a,
            account_equity=float(account_equity),
            profile_name=risk_profile,
            spot=a_snap.get("price"),
            vol20_ann_pct=a_snap.get("vol20_ann_pct"),
            opt=opt_a,
        )
        size_b_df = sizing_summary_table(
            ticker=ticker_b,
            account_equity=float(account_equity),
            profile_name=risk_profile,
            spot=b_snap.get("price"),
            vol20_ann_pct=b_snap.get("vol20_ann_pct"),
            opt=opt_b,
        )
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(size_a_df.iloc[:, :2], width='stretch')
    with c2:
        st.dataframe(size_b_df.iloc[:, :2], width='stretch')

    st.markdown("#### Trade Plan (Stock)")
    _rep_txt = st.session_state.get("report_markdown")
    _dir_a_sz = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=_rep_txt, pair_obj=pair, trend_label=trend_a, is_a=True)
    _dir_b_sz = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=_rep_txt, pair_obj=pair, trend_label=trend_b, is_a=False)

    try:
        tp_a = build_trade_plan(
            ticker=ticker_a,
            direction=_dir_a_sz,
            spot=a_snap.get("price"),
            account_equity=float(account_equity),
            risk_profile=risk_profile,
            vol20_ann_pct=a_snap.get("vol20_ann_pct"),
            implied_move_pct=opt_a.implied_move_pct,
        )
    except Exception:
        tp_a = None
    try:
        tp_b = build_trade_plan(
            ticker=ticker_b,
            direction=_dir_b_sz,
            spot=b_snap.get("price"),
            account_equity=float(account_equity),
            risk_profile=risk_profile,
            vol20_ann_pct=b_snap.get("vol20_ann_pct"),
            implied_move_pct=opt_b.implied_move_pct,
        )
    except Exception:
        tp_b = None

    tpa, tpb = st.columns(2)
    with tpa:
        st.markdown(f"**{ticker_a}** — `{_dir_a_sz.upper()}`")
        if tp_a is None:
            st.info("Trade plan unavailable.")
        else:
            st.dataframe(tp_a.to_dataframe(), width='stretch')
    with tpb:
        st.markdown(f"**{ticker_b}** — `{_dir_b_sz.upper()}`")
        if tp_b is None:
            st.info("Trade plan unavailable.")
        else:
            st.dataframe(tp_b.to_dataframe(), width='stretch')

    # return for report usage
    return size_a_df, size_b_df


