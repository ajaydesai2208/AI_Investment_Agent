from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_agent.greeks import atm_greeks_table
from ai_agent.strategies import suggest_strategies, plans_to_markdown
from ai_agent.tickets import build_tickets_from_strategy
from ui.helpers import _fmt_num, _infer_direction_for


def render_options_tab(*, ticker_a, ticker_b, opt_a, opt_b, pair, trend_a, trend_b, regime_a, regime_b, risk_profile, expiry_mode: str | None = None):
    st.markdown("#### Options Snapshot")
    if expiry_mode and expiry_mode.startswith("Pick"):
        st.caption("Using your selected expiries.")
    else:
        st.caption("Auto mode: nearest expiry in a ~7–45 DTE window. (cached ~5m)")

    opt_rows = [
        ("Expiry (DTE)", f"{opt_a.expiry or '—'} ({opt_a.dte or '—'})", f"{opt_b.expiry or '—'} ({opt_b.dte or '—'})"),
        ("Spot", _fmt_num(opt_a.spot), _fmt_num(opt_b.spot)),
        ("ATM strike", _fmt_num(opt_a.atm_strike), _fmt_num(opt_b.atm_strike)),
        ("Call mid", _fmt_num(opt_a.call_mid), _fmt_num(opt_b.call_mid)),
        ("Put mid", _fmt_num(opt_a.put_mid), _fmt_num(opt_b.put_mid)),
        ("Straddle debit", _fmt_num(opt_a.straddle_debit), _fmt_num(opt_b.straddle_debit)),
        ("Implied move", _fmt_num(opt_a.implied_move_pct, pct=True), _fmt_num(opt_b.implied_move_pct, pct=True)),
        ("ATM IV (approx.)", _fmt_num(opt_a.atm_iv_pct, pct=True), _fmt_num(opt_b.atm_iv_pct, pct=True)),
    ]
    opt_table = pd.DataFrame(opt_rows, columns=["Metric", ticker_a, ticker_b])
    st.dataframe(opt_table, width='stretch')

    st.markdown("#### ATM Greeks (per contract)")
    _rep_txt = st.session_state.get("report_markdown")
    _dir_a = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=_rep_txt, pair_obj=pair, trend_label=trend_a, is_a=True)
    _dir_b = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=_rep_txt, pair_obj=pair, trend_label=trend_b, is_a=False)

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown(f"**{ticker_a}** — assumed `{_dir_a.upper()}`")
        try:
            g_a = atm_greeks_table(
                spot=opt_a.spot, atm_strike=opt_a.atm_strike, dte=opt_a.dte, atm_iv_pct=opt_a.atm_iv_pct,
                call_mid=opt_a.call_mid, put_mid=opt_a.put_mid, direction=_dir_a, per_contract=True,
            )
        except Exception:
            g_a = pd.DataFrame({"Metric": [], "Value": []})
        if g_a.empty:
            st.info("Greeks unavailable for this expiry.")
        else:
            try:
                if "Value" in g_a.columns:
                    g_a["Value"] = g_a["Value"].astype(str)
            except Exception:
                pass
            st.dataframe(g_a, width='stretch')

    with gcol2:
        st.markdown(f"**{ticker_b}** — assumed `{_dir_b.upper()}`")
        try:
            g_b = atm_greeks_table(
                spot=opt_b.spot, atm_strike=opt_b.atm_strike, dte=opt_b.dte, atm_iv_pct=opt_b.atm_iv_pct,
                call_mid=opt_b.call_mid, put_mid=opt_b.put_mid, direction=_dir_b, per_contract=True,
            )
        except Exception:
            g_b = pd.DataFrame({"Metric": [], "Value": []})
        if g_b.empty:
            st.info("Greeks unavailable for this expiry.")
        else:
            try:
                if "Value" in g_b.columns:
                    g_b["Value"] = g_b["Value"].astype(str)
            except Exception:
                pass
            st.dataframe(g_b, width='stretch')

    st.markdown("#### Strategy Picker")
    if "strategy_compare" not in st.session_state:
        st.session_state["strategy_compare"] = []

    try:
        plans_a = suggest_strategies(
            ticker=ticker_a, direction=_dir_a, opt=opt_a, trend_label=trend_a, regime_label=regime_a, risk_profile=risk_profile,
        )
    except Exception:
        plans_a = []
    try:
        plans_b = suggest_strategies(
            ticker=ticker_b, direction=_dir_b, opt=opt_b, trend_label=trend_b, regime_label=regime_b, risk_profile=risk_profile,
        )
    except Exception:
        plans_b = []

    sa, sb = st.columns(2)

    def _render_plans(side_title: str, ticker: str, plans: list):
        st.markdown(f"**{side_title}**")
        if not plans:
            st.info("No strategy suggestions available for current inputs.")
            return
        for i, p in enumerate(plans, start=1):
            with st.expander(f"{i}. {p.name}", expanded=(i == 1)):
                st.caption(p.rationale)
                st.caption(f"_When to use:_ {p.when_to_use}")
                try:
                    _hs = p.header_stats()
                    if isinstance(_hs, pd.DataFrame) and "Value" in _hs.columns:
                        _hs["Value"] = _hs["Value"].astype(str)
                    st.dataframe(_hs, width='stretch')
                except Exception:
                    st.write(p.header_stats())
                st.markdown("**Legs**")
                try:
                    _legs = p.to_dataframe()
                    if isinstance(_legs, pd.DataFrame) and "Value" in _legs.columns:
                        _legs["Value"] = _legs["Value"].astype(str)
                    st.dataframe(_legs, width='stretch')
                except Exception:
                    st.write(p.to_dataframe())
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button("Add to Report", key=f"add_plan_{ticker}_{i}"):
                        md = plans_to_markdown([p])
                        lst = st.session_state.get("selected_strategies_md") or []
                        lst.append(md)
                        st.session_state["selected_strategies_md"] = lst
                        st.success(f"Added to Report: {ticker} — {p.name} (see Report tab).")
                with c2:
                    if st.button("Send to Tickets", key=f"send_plan_{ticker}_{i}"):
                        try:
                            extra = build_tickets_from_strategy(p, fallback_ticker=ticker)
                        except Exception:
                            extra = []
                        if extra:
                            buf = list(st.session_state.get("tickets_extra") or [])
                            buf.extend(extra)
                            st.session_state["tickets_extra"] = buf
                            st.success(f"Added {len(extra)} leg(s) to Tickets: {ticker} — {p.name} (Report tab).")
                        else:
                            st.info("No legs could be parsed for this plan.")
                with c3:
                    sel: list = st.session_state.get("strategy_compare") or []
                    is_selected = any((x.get("ticker") == ticker and x.get("name") == p.name) for x in sel)
                    at_limit = (len(sel) >= 2) and not is_selected
                    if not is_selected:
                        if st.button("Select for Compare", key=f"sel_{ticker}_{i}", disabled=at_limit):
                            entry = {
                                "ticker": ticker,
                                "name": p.name,
                                "dte": getattr(p, "dte", None),
                                "debit_credit": getattr(p, "debit_credit", None),
                                "est_cost": getattr(p, "est_cost", None),
                                "est_credit": getattr(p, "est_credit", None),
                                "max_loss": getattr(p, "max_loss", None),
                                "max_gain": getattr(p, "max_gain", None),
                                "breakevens": list(getattr(p, "breakevens", []) or []),
                                "rr_ratio": getattr(p, "rr_ratio", None),
                                "capital_req": getattr(p, "capital_req", None),
                            }
                            st.session_state["strategy_compare"] = sel + [entry]
                            st.rerun()
                    else:
                        if st.button("Remove from Compare", key=f"unsel_{ticker}_{i}"):
                            st.session_state["strategy_compare"] = [x for x in sel if not (x.get("ticker") == ticker and x.get("name") == p.name)]
                            st.rerun()

    with sa:
        _render_plans(f"{ticker_a} — {_dir_a.upper()} (Trend: {trend_a}, Regime: {regime_a})", ticker_a, plans_a)
    with sb:
        _render_plans(f"{ticker_b} — {_dir_b.upper()} (Trend: {trend_b}, Regime: {regime_b})", ticker_b, plans_b)

    comp = st.session_state.get("strategy_compare") or []
    if comp:
        st.markdown("#### Strategy Comparison")
        def _fmt_money(v):
            try:
                return f"${float(v):,.2f}" if v is not None else "—"
            except Exception:
                return "—"
        def _net_text(x):
            dc = (x.get("debit_credit") or "").upper()
            if dc == "CREDIT":
                return f"CREDIT {_fmt_money(x.get('est_credit'))}"
            if dc == "DEBIT":
                return f"DEBIT {_fmt_money(x.get('est_cost'))}"
            return "—"
        def _to_col(series):
            return {
                "Ticker": series.get("ticker"),
                "Strategy": series.get("name"),
                "DTE": series.get("dte") if series.get("dte") is not None else "—",
                "Net": _net_text(series),
                "Max Loss": _fmt_money(series.get("max_loss")),
                "Max Gain": _fmt_money(series.get("max_gain")),
                "Breakeven(s)": ", ".join([f"{b:.2f}" for b in (series.get("breakevens") or [])]) if series.get("breakevens") else "—",
                "R:R": (f"{float(series.get('rr_ratio')):.2f}" if series.get("rr_ratio") is not None else "—"),
                "Capital": _fmt_money(series.get("capital_req")),
            }
        cols = [_to_col(x) for x in comp[:2]]
        df = pd.DataFrame(cols)
        st.dataframe(df, width='stretch')
        cclear, _ = st.columns([1, 4])
        with cclear:
            if st.button("Clear selection", key="clear_strategy_compare"):
                st.session_state["strategy_compare"] = []
                st.rerun()

    # End-cap guard to prevent ghost content bleeding
    st.markdown('<div class="tab-bleed-guard"></div>', unsafe_allow_html=True)


