from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Project root (ai_agent sits one level below it)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

# Load environment variables from .env explicitly (works from any CWD)
load_dotenv(dotenv_path=ENV_PATH, override=False)

def get_openai_key() -> Tuple[Optional[str], str]:
    """
    Resolve the OpenAI key with this priority:
      1) st.session_state["openai_api_key"]  (typed in UI this session)
      2) Environment / .env  (OPENAI_API_KEY)
      3) st.secrets["OPENAI_API_KEY"]       (.streamlit/secrets.toml)
    Returns: (key or None, source_label)
    """
    # 1) session-typed key
    key = st.session_state.get("openai_api_key")
    if key:
        return key, "session"

    # 2) env/.env
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key, ".env/env"

    # 3) secrets
    secrets_key = None
    try:
        # st.secrets may behave like a Mapping or an object; handle both
        secrets_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st.secrets, "get") else st.secrets["OPENAI_API_KEY"]
    except Exception:
        secrets_key = None

    if secrets_key:
        return secrets_key, "secrets"

    return None, ""
