from __future__ import annotations

import streamlit as st


def setup_page() -> None:
    """Configure Streamlit page (title, icon, layout)."""
    st.set_page_config(page_title="AI Investment Agent", page_icon="📈", layout="wide")


def inject_global_css() -> None:
    """Inject the global CSS used by the app (unchanged from original)."""
    st.markdown(
        """
        <style>
          /* Load IBM Plex Mono for a retro terminal feel */
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

          :root {
            --app-font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace;
          }

          /* Apply mono font globally across the app */
          html, body, h1, h2, h3, h4, h5, h6,
          .stMarkdown, .stTextInput input, .stTextArea textarea, .stSelectbox, .stMultiSelect, .stNumberInput input,
          .stDataFrame, .stColumn, .stTabs, .stMetric, .stCaption,
          .stButton>button, .stDownloadButton>button,
          input, textarea, select, button, code, pre, kbd, samp, table, th, td {
            font-family: var(--app-font-mono) !important;
            letter-spacing: .2px;
          }

          .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px;}
          h1, h2, h3, h4 {letter-spacing: .2px;}

          /* Cards & tables */
          .card {
            background: var(--secondary-background-color, #11181F);
            border: 1px solid rgba(0,200,5,.12);
            border-radius: 14px;
            padding: 16px 16px;
            box-shadow: 0 0 0 1px rgba(0,200,5,.04), 0 10px 30px rgba(0,0,0,.35);
          }
          div[data-testid="stDataFrame"] thead tr th {
            background: rgba(0,200,5,.06);
            border-bottom: 1px solid rgba(0,200,5,.18);
          }

          /* Buttons: sharp, interactive */
          .stButton>button, .stDownloadButton>button {
            border-radius: 12px;
            border: 1px solid #00C805;
            background: linear-gradient(180deg, #00D60A 0%, #00B205 100%);
            color: #051207;
            font-weight: 700; letter-spacing: .2px;
            padding: 0.55rem 0.9rem;
            transition: transform .06s ease, box-shadow .2s ease, filter .2s ease;
            box-shadow: 0 8px 24px rgba(0,200,5,.15), 0 0 0 1px rgba(0,200,5,.18) inset;
            width: 100%;
            min-height: 90px; /* ensure uniform height across wrapped/one-line labels */
            display: inline-flex; align-items: center; justify-content: center; /* vertical centering */
            text-align: center; white-space: normal; line-height: 1.15; /* allow wrapping but keep height consistent */
          }
          .stButton>button:hover, .stDownloadButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 30px rgba(0,200,5,.28), 0 0 0 1px rgba(0,200,5,.2) inset;
            filter: brightness(1.03);
          }
          .stButton>button:active, .stDownloadButton>button:active {
            transform: translateY(0);
            filter: brightness(.98);
          }

          /* Tabs */
          .stTabs [data-baseweb="tab"] {
            text-transform: uppercase; letter-spacing: .4px; font-weight: 700;
          }
          .stTabs [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, rgba(0,200,5,.35), rgba(0,200,5,.08));
          }

          /* Uniform stat bar at top */
          .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 8px 0 10px 0;
          }
          .stat-card {
            background: var(--secondary-background-color, #11181F);
            border: 1px solid rgba(0,200,5,.14);
            border-radius: 14px;
            padding: 14px 16px;
            min-height: 78px;
            display: flex; flex-direction: column; justify-content: center;
            box-shadow: 0 0 0 1px rgba(0,200,5,.05);
          }
          .stat-title { font-weight: 800; letter-spacing: .3px; margin-bottom: 6px; }
          .pills { display:flex; gap:.5rem; flex-wrap:wrap; }
          .pill {
            padding: .12rem .55rem;
            border-radius: 999px;
            border: 1px solid rgba(0,200,5,.25);
            background: rgba(0,200,5,.08);
            font-size: .86rem;
            color: #AEE8B1;
          }
          /* Regime-specific pill accents */
          .pill.neutral { border-color: rgba(180,180,180,.35); background: rgba(180,180,180,.10); color: #D9DEE3; }
          .pill.calm { border-color: rgba(0,200,5,.35); background: rgba(0,200,5,.10); color: #AEE8B1; }
          .pill.stressed { border-color: rgba(255,60,60,.35); background: rgba(255,60,60,.10); color: #FFC1C1; }
          /* Sticky stat bar (removed) */
          /* High-contrast overrides */
          body.high-contrast .card { border-color: rgba(255,255,255,.28); box-shadow: 0 0 0 1px rgba(255,255,255,.12); }
          body.high-contrast div[data-testid="stDataFrame"] thead tr th { border-bottom-color: rgba(255,255,255,.35); }
          body.high-contrast .stat-card { border-color: rgba(255,255,255,.28); box-shadow: 0 0 0 1px rgba(255,255,255,.10); }
          body.high-contrast .pill { border-color: rgba(255,255,255,.35); color: #F3F6F9; }
          /* Skeleton loaders */
          @keyframes shimmer { 0% { background-position: -400px 0 } 100% { background-position: 400px 0 } }
          .skel-table { border: 1px solid rgba(255,255,255,.06); border-radius: 10px; padding: 12px; background: rgba(255,255,255,.02); }
          .skel-row { height: 14px; margin: 10px 0; border-radius: 6px; background: #121820; background-image: linear-gradient(90deg, rgba(255,255,255,0.05) 0, rgba(255,255,255,0.15) 20%, rgba(255,255,255,0.05) 40%); background-size: 800px 100%; animation: shimmer 1.2s infinite; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_high_contrast_from_state() -> None:
    """Apply or remove high-contrast classes based on session state (unchanged)."""
    try:
        if bool(st.session_state.get("ui_high_contrast", False)):
            st.markdown(
                """
                <script>
                document.documentElement.classList.add('high-contrast');
                document.body.classList.add('high-contrast');
                </script>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <script>
                document.documentElement.classList.remove('high-contrast');
                document.body.classList.remove('high-contrast');
                </script>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        pass


