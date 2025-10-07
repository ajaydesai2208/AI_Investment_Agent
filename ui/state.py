from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    """Initialize all session_state keys used by the app with defaults."""
    for key, default in [
        ("report_markdown", None),
        ("export_fname", None),
        ("export_bytes", None),
        ("selected_strategies_md", []),
        ("tickets_extra", []),
        ("ui_high_contrast", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


