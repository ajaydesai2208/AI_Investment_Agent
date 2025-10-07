from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_agent.metrics import fundamentals_table_extended as fundamentals_table
from ai_agent.playbook import build_event_playbook_for_ticker, playbook_to_markdown
from ai_agent.risk import sizing_summary_table
from ai_agent.export import build_pdf_report
from ai_agent.tickets import build_tickets_for_ticker, tickets_to_csv_bytes
from ui.helpers import _sizing_fact_line, _infer_direction_for, _fmt_num


def render_report_tab(*, ticker_a, ticker_b, a_snap, b_snap, opt_a, opt_b, cat_a, cat_b, trend_a, trend_b, regime_a, regime_b, risk_profile, lookback_days, max_news, pair, size_a_df, size_b_df):
    if st.session_state.get("report_markdown"):
        st.markdown("#### Investment Analysis Report")
        st.markdown(st.session_state["report_markdown"])
        try:
            fund_a_tbl = fundamentals_table(ticker_a)
            fund_b_tbl = fundamentals_table(ticker_b)
            catalysts_all_md = f"#### {ticker_a}\n\n" + st.session_state.get("cat_md_a", "") + "\n\n#### " + ticker_b + "\n\n" + st.session_state.get("cat_md_b", "")
            if pair and getattr(pair, "window_used", 0) > 0:
                ztxt = f"{pair.spread_zscore:.2f}" if pair.spread_zscore is not None else "—"
                pair_md = (
                    f"- Beta(A on B): {pair.beta_ab:.2f} | Corr: {pair.corr_ab:.2f} | Hedge ratio: {pair.hedge_ratio:.2f}\n"
                    f"- Spread z-score (60): {ztxt}"
                )
            else:
                pair_md = "- Insufficient overlapping history for pair analysis."

            try:
                sizing_notes_md = f"- {_sizing_fact_line(ticker_a, size_a_df)}\n- {_sizing_fact_line(ticker_b, size_b_df)}"
            except Exception:
                sfa = _sizing_fact_line(
                    ticker_a,
                    sizing_summary_table(
                        ticker=ticker_a, account_equity=float(st.session_state.get("account_equity", 0)), profile_name=risk_profile,
                        spot=a_snap.get("price"), vol20_ann_pct=a_snap.get("vol20_ann_pct"), opt=opt_a
                    ),
                )
                sfb = _sizing_fact_line(
                    ticker_b,
                    sizing_summary_table(
                        ticker=ticker_b, account_equity=float(st.session_state.get("account_equity", 0)), profile_name=risk_profile,
                        spot=b_snap.get("price"), vol20_ann_pct=b_snap.get("vol20_ann_pct"), opt=opt_b
                    ),
                )
                sizing_notes_md = f"- {sfa}\n- {sfb}"

            plays_a = build_event_playbook_for_ticker(ticker=ticker_a, catalysts=cat_a, opt_snapshot=opt_a, trend_label=trend_a, regime_label=regime_a, risk_profile=risk_profile)
            plays_b = build_event_playbook_for_ticker(ticker=ticker_b, catalysts=cat_b, opt_snapshot=opt_b, trend_label=trend_b, regime_label=regime_b, risk_profile=risk_profile)
            event_playbook_md = playbook_to_markdown(ticker_a, plays_a) + "\n\n" + playbook_to_markdown(ticker_b, plays_b)

            dir_a_r = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=st.session_state.get("report_markdown"), pair_obj=pair, trend_label=trend_a, is_a=True)
            dir_b_r = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=st.session_state.get("report_markdown"), pair_obj=pair, trend_label=trend_b, is_a=False)
            try:
                scn_a_df = build_scenarios(ticker=ticker_a, direction=dir_a_r, spot=a_snap.get("price"), implied_move_pct=opt_a.implied_move_pct, opt=opt_a)
            except Exception:
                scn_a_df = pd.DataFrame()
            try:
                scn_b_df = build_scenarios(ticker=ticker_b, direction=dir_b_r, spot=b_snap.get("price"), implied_move_pct=opt_b.implied_move_pct, opt=opt_b)
            except Exception:
                scn_b_df = pd.DataFrame()

            base_tix = []
            try:
                base_tix += build_tickets_for_ticker(ticker=ticker_a, direction=dir_a_r, sizing_df=size_a_df, opt=opt_a, spot=a_snap.get("price"), implied_move_pct=opt_a.implied_move_pct, profile_name=risk_profile)
            except Exception:
                pass
            try:
                base_tix += build_tickets_for_ticker(ticker=ticker_b, direction=dir_b_r, sizing_df=size_b_df, opt=opt_b, spot=b_snap.get("price"), implied_move_pct=opt_b.implied_move_pct, profile_name=risk_profile)
            except Exception:
                pass
            extra_tix = st.session_state.get("tickets_extra") or []
            all_tix = list(base_tix) + list(extra_tix)
            rows = []
            for t in all_tix:
                if hasattr(t, "__dict__"):
                    rows.append({k: v for k, v in t.__dict__.items()})
                elif isinstance(t, dict):
                    rows.append(t)
            tickets_df = pd.DataFrame(rows)

            pdf_name, pdf_bytes = build_pdf_report(
                title=f"AI Investment Report {ticker_a} vs {ticker_b}",
                report_markdown=st.session_state["report_markdown"],
                metadata={
                    "Ticker A": ticker_a,
                    "Ticker B": ticker_b,
                    "Timeframe": st.session_state.get("timeframe_select", "1Y"),
                    "Risk profile": risk_profile,
                    "Lookback days": str(lookback_days),
                    "Max news": str(max_news),
                },
                options_table=pd.DataFrame(
                    [
                        ("Expiry (DTE)", f"{opt_a.expiry or '—'} ({opt_a.dte or '—'})", f"{opt_b.expiry or '—'} ({opt_b.dte or '—'})"),
                        ("Spot", _fmt_num(opt_a.spot), _fmt_num(opt_b.spot)),
                        ("ATM strike", _fmt_num(opt_a.atm_strike), _fmt_num(opt_b.atm_strike)),
                        ("Call mid", _fmt_num(opt_a.call_mid), _fmt_num(opt_b.call_mid)),
                        ("Put mid", _fmt_num(opt_a.put_mid), _fmt_num(opt_b.put_mid)),
                        ("Straddle debit", _fmt_num(opt_a.straddle_debit), _fmt_num(opt_b.straddle_debit)),
                        ("Implied move", _fmt_num(opt_a.implied_move_pct, pct=True), _fmt_num(opt_b.implied_move_pct, pct=True)),
                        ("ATM IV (approx.)", _fmt_num(opt_a.atm_iv_pct, pct=True), _fmt_num(opt_b.atm_iv_pct, pct=True)),
                    ],
                    columns=["Metric", ticker_a, ticker_b],
                ),
                fundamentals_a=fund_a_tbl,
                fundamentals_b=fund_b_tbl,
                catalysts_md=catalysts_all_md,
                pair_text_md=pair_md,
                sizing_text_md=sizing_notes_md,
                selected_strategies_md=("\n\n---\n\n".join(st.session_state.get("selected_strategies_md") or []) or None),
                event_playbook_md=event_playbook_md,
                scenarios_a_df=scn_a_df,
                scenarios_b_df=scn_b_df,
                tickets_df=tickets_df,
            )
            st.download_button("⬇️ Download full report (.pdf)", data=pdf_bytes, file_name=pdf_name, mime="application/pdf", width='stretch')
        except Exception as _pdf_err:
            st.caption(f"PDF export unavailable: {_pdf_err}")
        st.download_button(label="⬇️ Download report (.md)", data=st.session_state["export_bytes"], file_name=st.session_state["export_fname"], mime="text/markdown", width='stretch')

        st.markdown("#### Trade Tickets")
        dir_a = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=st.session_state["report_markdown"], pair_obj=pair, trend_label=trend_a, is_a=True)
        dir_b = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=st.session_state["report_markdown"], pair_obj=pair, trend_label=trend_b, is_a=False)

        dir_a_r = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=st.session_state.get("report_markdown"), pair_obj=pair, trend_label=trend_a, is_a=True)
        dir_b_r = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=st.session_state.get("report_markdown"), pair_obj=pair, trend_label=trend_b, is_a=False)

        tickets = []
        try:
            tickets += build_tickets_for_ticker(ticker=ticker_a, direction=dir_a_r, sizing_df=size_a_df, opt=opt_a, spot=a_snap.get("price"), implied_move_pct=opt_a.implied_move_pct, profile_name=risk_profile)
        except Exception:
            pass
        try:
            tickets += build_tickets_for_ticker(ticker=ticker_b, direction=dir_b_r, sizing_df=size_b_df, opt=opt_b, spot=b_snap.get("price"), implied_move_pct=opt_b.implied_move_pct, profile_name=risk_profile)
        except Exception:
            pass
        extra = st.session_state.get("tickets_extra") or []
        tickets_all = list(tickets) + list(extra)
        try:
            fname_csv, csv_bytes = tickets_to_csv_bytes(tickets_all)
        except Exception:
            import pandas as _pd
            norm_rows = []
            for t in tickets_all:
                if hasattr(t, "__dict__"):
                    norm_rows.append({k: v for k, v in t.__dict__.items()})
                elif isinstance(t, dict):
                    norm_rows.append(t)
            if not norm_rows:
                norm_rows = [{"plan": "", "ticker": "", "asset": "", "side": "", "quantity": 0, "expiry": "", "strike": "", "right": "", "price_note": "", "notes": ""}]
            _df_csv = _pd.DataFrame(norm_rows)
            csv_bytes = _df_csv.to_csv(index=False).encode("utf-8")
            fname_csv = f"tickets_{ticker_a}_{ticker_b}.csv"
        st.download_button(label="⬇️ Download trade tickets (.csv)", data=csv_bytes, file_name=fname_csv, mime="text/csv", width='stretch')

        with st.expander("Preview tickets"):
            try:
                import pandas as _pd
                rows = []
                for t in tickets_all:
                    if hasattr(t, "__dict__"):
                        rows.append({k: v for k, v in t.__dict__.items()})
                    elif isinstance(t, dict):
                        rows.append(t)
                _df_prev = _pd.DataFrame(rows)
                st.dataframe(_df_prev, width='stretch')
            except Exception:
                st.write("No tickets to preview.")

        if extra:
            cclr1, cclr2 = st.columns([1, 3])
            with cclr1:
                if st.button("Clear strategy-added tickets"):
                    st.session_state["tickets_extra"] = []
                    st.rerun()

        st.markdown("---")
        st.markdown("### Event Playbook")
        try:
            plays_a = build_event_playbook_for_ticker(ticker=ticker_a, catalysts=cat_a, opt_snapshot=opt_a, trend_label=trend_a, regime_label=regime_a, risk_profile=risk_profile)
            md_a = playbook_to_markdown(ticker_a, plays_a)
        except Exception:
            md_a = f"### {ticker_a}: Event Playbook\n\n_Unable to build playbook at this time._\n"
        try:
            plays_b = build_event_playbook_for_ticker(ticker=ticker_b, catalysts=cat_b, opt_snapshot=opt_b, trend_label=trend_b, regime_label=regime_b, risk_profile=risk_profile)
            md_b = playbook_to_markdown(ticker_b, plays_b)
        except Exception:
            md_b = f"### {ticker_b}: Event Playbook\n\n_Unable to build playbook at this time._\n"
        pb1, pb2 = st.columns(2)
        with pb1:
            st.markdown(md_a)
        with pb2:
            st.markdown(md_b)

        st.markdown("---")
        st.markdown("### Selected Strategies")
        sel = st.session_state.get("selected_strategies_md") or []
        if not sel:
            st.caption("Use the **Options → Strategy Picker** to add strategies here.")
        else:
            combined_md = "\n".join(sel)
            st.markdown(combined_md)
            st.download_button("⬇️ Download selected strategies (.md)", data=combined_md.encode("utf-8"), file_name=f"strategies_{ticker_a}_{ticker_b}.md", mime="text/markdown", width='stretch')
            if st.button("Clear selected strategies"):
                st.session_state["selected_strategies_md"] = []
                st.rerun()

        st.markdown("---")
        st.markdown("### Trade Plan (Stock)")
        _dir_a_r = _infer_direction_for(this_ticker=ticker_a, other_ticker=ticker_b, report_text=st.session_state.get("report_markdown"), pair_obj=pair, trend_label=trend_a, is_a=True)
        _dir_b_r = _infer_direction_for(this_ticker=ticker_b, other_ticker=ticker_a, report_text=st.session_state.get("report_markdown"), pair_obj=pair, trend_label=trend_b, is_a=False)
        try:
            tp_a_r = build_trade_plan(ticker=ticker_a, direction=_dir_a_r, spot=a_snap.get("price"), account_equity=float(st.session_state.get("account_equity", 0)), risk_profile=risk_profile, vol20_ann_pct=a_snap.get("vol20_ann_pct"), implied_move_pct=opt_a.implied_move_pct)
        except Exception:
            tp_a_r = None
        try:
            tp_b_r = build_trade_plan(ticker=ticker_b, direction=_dir_b_r, spot=b_snap.get("price"), account_equity=float(st.session_state.get("account_equity", 0)), risk_profile=risk_profile, vol20_ann_pct=b_snap.get("vol20_ann_pct"), implied_move_pct=opt_b.implied_move_pct)
        except Exception:
            tp_b_r = None
        rpa, rpb = st.columns(2)
        with rpa:
            st.markdown(f"**{ticker_a}** — `{_dir_a_r.upper()}`")
            if tp_a_r is None:
                st.info("Trade plan unavailable.")
            else:
                st.dataframe(tp_a_r.to_dataframe(), width='stretch')
        with rpb:
            st.markdown(f"**{ticker_b}** — `{_dir_b_r.upper()}`")
            if tp_b_r is None:
                st.info("Trade plan unavailable.")
            else:
                st.dataframe(tp_b_r.to_dataframe(), width='stretch')
    else:
        st.info("Press **Compare & Analyze** to generate a full report.")

# late import to avoid circulars
from ai_agent.scenario import build_scenarios  # noqa: E402  # isort:skip


