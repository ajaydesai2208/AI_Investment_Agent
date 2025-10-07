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
            /* Button sizing tokens */
            --btn-h-sm: 34px;
            --btn-h-md: 42px;
            --btn-h-lg: 48px;
            /* App background fallback for bleed-guard */
            --app-bg: #0B1217;
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

          /* Buttons: sleek + modern */
          .stButton>button, .stDownloadButton>button,
          .stButton button, .stDownloadButton button {
            min-height: var(--btn-h-sm) !important;
            height: var(--btn-h-sm) !important;
            padding: 0 12px !important;
            border-radius: 10px !important;
            border: 1px solid #00C805 !important;
            background: linear-gradient(180deg, #1fd52a 0%, #10b018 100%) !important;
            color: #051207 !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            letter-spacing: .15px !important;
            width: fit-content !important; /* override Streamlit base full-width */
            max-width: max-content !important;
            flex: 0 0 auto !important;
            transition: transform .06s ease, box-shadow .2s ease, filter .2s ease !important;
            box-shadow: 0 6px 18px rgba(0,200,5,.18), 0 0 0 1px rgba(0,200,5,.18) inset !important;
            display: inline-flex !important; align-items: center; justify-content: center; align-self: flex-start !important;
            text-align: center; white-space: nowrap; line-height: 1;
          }
          /* Streamlit base buttons (data-testid) */
          button[data-testid^="stBaseButton"] {
            min-height: var(--btn-h-sm) !important;
            height: var(--btn-h-sm) !important;
            padding: 0 12px !important;
            width: fit-content !important;
            max-width: max-content !important;
            border-radius: 10px !important;
            border: 1px solid #00C805 !important;
            background: linear-gradient(180deg, #1fd52a 0%, #10b018 100%) !important;
            color: #051207 !important;
            font-weight: 700 !important;
            letter-spacing: .15px !important;
          }
          /* Ensure containers don't force full-width */
          .stButton, .stDownloadButton { width: auto !important; display: inline-flex !important; }
          .stButton > div, .stDownloadButton > div { width: auto !important; display: inline-flex !important; }
          .stButton>button:hover, .stDownloadButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 26px rgba(0,200,5,.24), 0 0 0 1px rgba(0,200,5,.20) inset;
            filter: brightness(1.04);
          }
          .stButton>button:active, .stDownloadButton>button:active { transform: translateY(0); filter: brightness(.98); }
          .stButton>button:focus-visible, .stDownloadButton>button:focus-visible { outline: 2px solid #00C805 !important; outline-offset: 2px; }
          .stButton>button:disabled, .stDownloadButton>button:disabled { opacity: .55; transform: none; filter: none; cursor: not-allowed; }

          /* Button rows for symmetry */
          .btn-row { display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; }
          .btn-row .stButton>button, .btn-row .stDownloadButton>button { flex: 0 0 auto; }
          .btn-row.stretch .stButton>button, .btn-row.stretch .stDownloadButton>button { flex: 1 1 0; }

          /* Actions panel: compact, aligned buttons */
          .term-actions { display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; }
          .term-actions .stButton>button, .term-actions .stDownloadButton>button { width: auto !important; }
          .term-actions { justify-content: flex-start; }

          /* Subtle interactive effect */
          .stButton>button:hover, .stDownloadButton>button:hover { transform: translateY(-1px) scale(1.01); }

          /* Sidebar: full-width, equal-sized buttons */
          [data-testid="stSidebar"] .stButton, [data-testid="stSidebar"] .stDownloadButton {
            width: 100% !important; display: block !important;
          }
          [data-testid="stSidebar"] .stButton > div, [data-testid="stSidebar"] .stDownloadButton > div {
            width: 100% !important; display: block !important;
          }
          [data-testid="stSidebar"] .stButton>button, [data-testid="stSidebar"] .stDownloadButton>button {
            min-height: var(--btn-h-sm) !important;
            height: var(--btn-h-sm) !important;
            padding: 0 12px !important;
            width: 100% !important; /* full-width in sidebar */
          }

          /* Tabs */
          .stTabs [data-baseweb="tab"] {
            text-transform: uppercase; letter-spacing: .4px; font-weight: 700;
          }
          .stTabs [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, rgba(0,200,5,.35), rgba(0,200,5,.08));
          }
          /* Ensure tab panels are opaque to prevent ghost content bleed-through */
          .stTabs [data-baseweb="tab-panel"] {
            background: var(--app-bg, #0B1217);
            position: relative;
          }
          
          /* Shimmer loading animation for Report tab */
          @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
          }

          .report-loading-shimmer {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            padding: 2rem;
            animation: fadeIn 0.3s ease-in;
          }

          .shimmer-block {
            height: 24px;
            background: linear-gradient(
              90deg,
              rgba(0,200,5,0.05) 0%,
              rgba(0,200,5,0.15) 50%,
              rgba(0,200,5,0.05) 100%
            );
            background-size: 1000px 100%;
            animation: shimmer 2s infinite linear;
            border-radius: 8px;
          }

          .shimmer-block.large { height: 120px; }
          .shimmer-block.medium { height: 60px; }
          .shimmer-block.small { height: 24px; }

          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }

          /* Uniform stat bar at top */
          /* Bleed guard block to cap the end of each tab and mask hidden content */
          .tab-bleed-guard { height: 72px; background: var(--app-bg, #0B1217); margin-top: 16px; border-radius: 0 0 8px 8px; }

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

          /* Primary CTA full-width, but sleek */
          .primary-cta .stButton, .primary-cta .stButton > div { width: 100% !important; }
          .primary-cta .stButton>button {
            width: 100% !important;
            min-height: 44px !important; height: 44px !important; /* slightly taller for emphasis */
            padding: 0 16px !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, #22da2f 0%, #13b81a 100%) !important;
            box-shadow: 0 10px 28px rgba(0,200,5,.28), 0 0 0 1px rgba(0,200,5,.18) inset !important;
          }
          .primary-cta .stButton>button:hover { transform: translateY(-1px) scale(1.005); filter: brightness(1.03); }
          .primary-cta .stButton>button:active { transform: translateY(0) scale(1.0); filter: brightness(.98); }
        </style>
        <script>
          try {
            const css = `
              button[data-testid^="stBaseButton"]{
                min-height: 34px !important; height:34px !important; padding:0 12px !important;
                width: fit-content !important; max-width: max-content !important; border-radius:10px !important;
              }
              .stButton, .stDownloadButton{ width:auto !important; display:inline-flex !important; }
              .stButton > div, .stDownloadButton > div{ width:auto !important; display:inline-flex !important; }
              [data-testid="stSidebar"] button[data-testid^="stBaseButton"]{ width:100% !important; }
            `;
            const s = document.createElement('style');
            s.setAttribute('data-ai-injection','buttons');
            s.appendChild(document.createTextNode(css));
            document.head.appendChild(s);

            const tagCTA = () => {
              const btns = Array.from(document.querySelectorAll('button[data-testid^="stBaseButton"]'));
              const cta = btns.find(b => (b.textContent||'').trim().toLowerCase().includes('compare') );
              if (cta) {
                const wrap = cta.closest('[data-testid]') || cta.parentElement;
                if (wrap) wrap.classList.add('primary-cta-wrap');
                cta.style.width = '100%';
                cta.style.minHeight = '44px';
                cta.style.height = '44px';
              }
              // Sidebar buttons full width
              const sBtns = document.querySelectorAll('[data-testid="stSidebar"] button[data-testid^="stBaseButton"]');
              sBtns.forEach(b => { b.style.width = '100%'; b.style.minHeight = '34px'; b.style.height = '34px'; });
            };
            // Run now and after small delay to handle reruns
            tagCTA(); setTimeout(tagCTA, 400); setTimeout(tagCTA, 1200);
          } catch (e) {}
        </script>
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


