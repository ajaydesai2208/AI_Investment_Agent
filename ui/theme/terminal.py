from __future__ import annotations

import streamlit as st


def inject_report_theme() -> None:
    """Inject CSS for the terminal-style Report tab.

    The styles are scoped under the `.term-report` container to avoid
    altering the rest of the app. Uses the global IBM Plex Mono font
    already configured at the app level.
    """
    st.markdown(
        """
        <style>
        /* Terminal palette and primitives */
        .term-report {
          --bg: #0A0F14;                 /* page background */
          --panel: #11181F;              /* cards */
          --grid: rgba(0,200,5,.12);     /* grid lines */
          --accent: #00C805;             /* neon green */
          --amber: #E0A458;              /* warm amber */
          --muted: #AEE8B1;              /* soft mint */
          --warn: #FF5C5C;               /* error red */
          --text: #E6F2E6;               /* primary text */
        }

        .term-report .term-ribbon {
          display:flex; align-items:center; gap:.75rem;
          border-top: 1px solid var(--grid);
          border-bottom: 1px solid var(--grid);
          background: linear-gradient(90deg, rgba(0,200,5,.10), rgba(0,200,5,.02));
          padding: .55rem .8rem; margin: .25rem 0 1rem 0;
          text-transform: uppercase; letter-spacing: .6px; font-weight: 700;
          color: var(--muted);
        }

        .term-report .term-ribbon .pill {
          display:inline-block; padding:.12rem .5rem; border-radius:999px;
          border:1px solid var(--grid); background: rgba(0,200,5,.08);
          color: var(--muted); font-weight: 600;
        }
        .term-report .pill.buy { border-color: rgba(0,200,5,.35); color:#9FF3A3; }
        .term-report .pill.sell { border-color: rgba(255,92,92,.35); background: rgba(255,92,92,.06); color:#FFC1C1; }
        .term-report .pill.neutral { border-color: rgba(160,160,160,.35); background: rgba(160,160,160,.08); color:#D9DEE3; }

        .term-report .term-section {
          margin: .2rem 0 .8rem 0;
          border-left: 2px solid var(--grid);
          padding-left: .6rem;
          color: var(--text);
        }

        .term-report .term-title {
          color: var(--text); font-weight: 800; letter-spacing: .3px; margin-bottom: .35rem;
        }

        .term-report .term-panel {
          background: var(--panel);
          border: 1px solid var(--grid);
          border-radius: 12px;
          padding: .75rem .9rem; margin-bottom: .9rem;
          box-shadow: 0 0 0 1px rgba(0,200,5,.04), 0 6px 16px rgba(0,0,0,.35);
        }

        .term-report .term-actions .stDownloadButton>button, 
        .term-report .term-actions .stButton>button {
          border-radius: 10px; border: 1px solid var(--accent);
          background: linear-gradient(180deg, #00D60A 0%, #00B205 100%);
          color: #051207; font-weight: 800; letter-spacing:.3px;
          box-shadow: 0 8px 24px rgba(0,200,5,.18), 0 0 0 1px rgba(0,200,5,.18) inset;
        }

        .term-report .term-kv { display:grid; grid-template-columns: 160px 1fr; gap:.25rem .75rem; }
        .term-report .term-k { color: var(--muted); text-transform: uppercase; font-weight:700; }
        .term-report .term-v { color: var(--text); }

        .term-report .term-meter { display:flex; gap:.25rem; align-items:center; }
        .term-report .term-meter .bar { flex:1; height:8px; border:1px solid var(--grid); border-radius:999px; overflow:hidden; background: rgba(0,200,5,.05); }
        .term-report .term-meter .fill { height:100%; background: var(--accent); }

        /* Narrow tweaks */
        @media (max-width: 1100px) {
          .term-report .term-kv { grid-template-columns: 120px 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


