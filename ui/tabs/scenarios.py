from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from ai_agent.scenario import build_scenarios, build_payoff_grid
from ui.helpers import _infer_direction_for


def render_scenarios_tab(*, ticker_a, ticker_b, a_snap, b_snap, opt_a, opt_b, pair, trend_a, trend_b):
    st.markdown("#### Scenario Explorer")
    st.caption(
        "Five scenarios over your chosen DTE: −2σ / −1σ / base / +1σ / +2σ. "
        "σ is proxied by the **implied move** over that horizon. "
        "Stock P/L is per share; Option P/L is per contract (×100)."
    )

    report_text = st.session_state.get("report_markdown")
    dir_a = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=report_text, pair_obj=pair, trend_label=trend_a, is_a=True)
    dir_b = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=report_text, pair_obj=pair, trend_label=trend_b, is_a=False)

    csa, csb = st.columns(2)
    with csa:
        st.markdown(f"**{ticker_a}** — assumed direction: `{dir_a.upper()}`")
        try:
            df_a = build_scenarios(ticker=ticker_a, direction=dir_a, spot=a_snap.get("price"), implied_move_pct=opt_a.implied_move_pct, opt=opt_a)
        except Exception:
            df_a = pd.DataFrame(columns=["Scenario", "Spot'", "Stock P/L/share", "Option P/L/contract"])
        st.dataframe(df_a, width='stretch')
    with csb:
        st.markdown(f"**{ticker_b}** — assumed direction: `{dir_b.upper()}`")
        try:
            df_b = build_scenarios(ticker=ticker_b, direction=dir_b, spot=b_snap.get("price"), implied_move_pct=opt_b.implied_move_pct, opt=opt_b)
        except Exception:
            df_b = pd.DataFrame(columns=["Scenario", "Spot'", "Stock P/L/share", "Option P/L/contract"])
        st.dataframe(df_b, width='stretch')

    def _df_to_bytes(df: pd.DataFrame) -> bytes:
        try:
            return df.to_csv(index=False).encode("utf-8")
        except Exception:
            return b"Scenario,Spot',Stock P/L/share,Option P/L/contract\n"

    combined = pd.DataFrame()
    if df_a is not None and not df_a.empty:
        dfa = df_a.copy()
        dfa.insert(0, "Ticker", ticker_a)
        dfa.insert(1, "Direction", dir_a.upper())
        combined = pd.concat([combined, dfa], ignore_index=True)
    if df_b is not None and not df_b.empty:
        dfb = df_b.copy()
        dfb.insert(0, "Ticker", ticker_b)
        dfb.insert(1, "Direction", dir_b.upper())
        combined = pd.concat([combined, dfb], ignore_index=True)

    st.download_button(
        "⬇️ Download scenarios (.csv)",
        data=_df_to_bytes(combined if not combined.empty else pd.DataFrame(columns=["Ticker","Direction","Scenario","Spot'","Stock P/L/share","Option P/L/contract"])),
        file_name=f"scenarios_{ticker_a}_{ticker_b}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Payoff Charts")

    pa, pb = st.columns(2)
    def _render_payoff(df_grid: pd.DataFrame, breakeven: Optional[float], title: str):
        if df_grid is None or df_grid.empty:
            st.info("No payoff data to chart.")
            return
        try:
            import altair as alt  # type: ignore
            df_long = df_grid.melt("Spot", var_name="Instrument", value_name="P/L")
            base = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("Spot:Q", title="Spot"),
                    y=alt.Y("P/L:Q", title="P/L"),
                    color=alt.Color("Instrument:N", legend=alt.Legend(title=None)),
                    tooltip=["Spot:Q", "Instrument:N", alt.Tooltip("P/L:Q", format=".2f")],
                )
                .properties(title=title, height=240)
            )
            if breakeven is not None:
                rule = alt.Chart(pd.DataFrame({"x": [breakeven]})).mark_rule(strokeDash=[5, 5]).encode(x="x:Q")
                label = alt.Chart(pd.DataFrame({"x": [breakeven], "txt": ["breakeven"]})).mark_text(dy=-8).encode(x="x:Q", text="txt:N")
                chart = base + rule + label
            else:
                chart = base
            st.altair_chart(chart, use_container_width=True)
            if breakeven is not None:
                st.caption(f"Breakeven ≈ **{breakeven:.2f}**")
        except Exception:
            try:
                st.line_chart(df_grid.set_index("Spot"), height=240, use_container_width=True)
                if breakeven is not None:
                    st.caption(f"Breakeven ≈ **{breakeven:.2f}**")
            except Exception:
                st.info("Could not render chart.")

    try:
        grid_a, be_a = build_payoff_grid(ticker=ticker_a, direction=dir_a, spot=a_snap.get("price"), implied_move_pct=opt_a.implied_move_pct, opt=opt_a)
    except Exception:
        grid_a, be_a = pd.DataFrame(columns=["Spot", "Stock P/L/share", "Option P/L/contract"]), None
    try:
        grid_b, be_b = build_payoff_grid(ticker=ticker_b, direction=dir_b, spot=b_snap.get("price"), implied_move_pct=opt_b.implied_move_pct, opt=opt_b)
    except Exception:
        grid_b, be_b = pd.DataFrame(columns=["Spot", "Stock P/L/share", "Option P/L/contract"]), None

    with pa:
        st.markdown(f"**{ticker_a}** — `{dir_a.upper()}`")
        _render_payoff(grid_a, be_a, title=f"{ticker_a} payoff (per share / per contract)")
    with pb:
        st.markdown(f"**{ticker_b}** — `{dir_b.upper()}`")
        _render_payoff(grid_b, be_b, title=f"{ticker_b} payoff (per share / per contract)")


