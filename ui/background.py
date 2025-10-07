from __future__ import annotations

import threading
import time

import streamlit as st

from ai_agent.catalysts import clear_catalyst_cache
from ai_agent.news import clear_news_cache


def _background_cache_refresher() -> None:
    """Periodically clear news/catalyst caches every 3 hours (unchanged logic)."""
    while True:
        try:
            clear_news_cache()
            clear_catalyst_cache()
        except Exception:
            pass
        time.sleep(3 * 60 * 60)


def ensure_bg_refresher_started() -> None:
    """Start the background refresher thread once per session/process (idempotent)."""
    if not st.session_state.get("bg_refresher_started"):
        try:
            threading.Thread(target=_background_cache_refresher, daemon=True).start()
        except Exception:
            pass
        st.session_state["bg_refresher_started"] = True


