import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Agno (AI agent + tools)
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

# ---------- App bootstrap ----------

# Load environment variables from a local .env if present
# This lets you keep your own OPENAI_API_KEY without hardcoding
load_dotenv()

st.set_page_config(page_title="AI Investment Agent", page_icon="📈")
st.title("AI Investment Agent")
st.caption(
    "Compare two tickers using Yahoo Finance data and a GPT model. "
    "The app uses a local OPENAI_API_KEY from .env or environment if available, "
    "otherwise you can paste a key below for this session."
)

# ---------- Helper: resolve OpenAI key from env/secrets or UI ----------

def get_openai_key() -> tuple[Optional[str], str]:
    """
    Priority:
      1) st.session_state (what the user typed this session)
      2) Environment / .env (OPENAI_API_KEY)
      3) st.secrets (if a .streamlit/secrets.toml provides it)
    Returns (key, source_label).
    """
    # 1) session
    key = st.session_state.get("openai_api_key")
    if key:
        return key, "session"

    # 2) env or .env
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key, ".env/env"

    # 3) secrets
    secrets_key = None
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st.secrets, "get") else st.secrets["OPENAI_API_KEY"]
    except Exception:
        secrets_key = None

    if secrets_key:
        return secrets_key, "secrets"

    return None, ""


with st.sidebar:
    st.subheader("Authentication")
    current_key, source = get_openai_key()

    if current_key:
        st.success(f"OpenAI key loaded from {source}.")
        # Allow switching to a different key during this session
        if st.button("Use a different key"):
            st.session_state["openai_api_key"] = ""
            current_key = None
            source = ""
            st.experimental_rerun()

    if not current_key:
        typed = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if typed:
            # store only in session memory, never write to disk
            st.session_state["openai_api_key"] = typed
            current_key, source = typed, "session"
            st.success("Key saved for this session.")

# ---------- Inputs ----------

col1, col2 = st.columns(2)
ticker_a = col1.text_input("Ticker A", value="AAPL")
ticker_b = col2.text_input("Ticker B", value="MSFT")
go = st.button("Compare")

# ---------- Run the agent ----------

def build_agent(api_key: str) -> Agent:
    # NOTE: Construct YFinanceTools without keyword args because different agno versions
    # expose different flags; defaults give us price/info/news/etc. where supported.
    return Agent(
        model=OpenAIChat(id="gpt-4o", api_key=api_key),
        tools=[YFinanceTools()],
        show_tool_calls=True,
        instructions=(
            "You are a helpful investment analyst. "
            "When comparing two tickers, gather fundamentals, price history, news, and analyst ratings if available. "
            "Summarize clearly using markdown. Include a small table comparing valuation, growth, "
            "profitability, dividends, risk, and recent performance when possible. "
            "Cite the data source sections you used (e.g., Yahoo Finance). "
            "Avoid financial advice language. Provide a balanced view and a final reasoned comparison."
        ),
    )

def validate_tickers(a: str, b: str) -> Optional[str]:
    if not a or not b:
        return "Please enter both tickers."
    if a.strip().upper() == b.strip().upper():
        return "Please enter two different tickers."
    return None

if go:
    # Validate key
    if not current_key:
        st.error("No OpenAI key found. Add it in the sidebar or create a .env file with OPENAI_API_KEY.")
        st.stop()

    # Validate tickers
    err = validate_tickers(ticker_a, ticker_b)
    if err:
        st.error(err)
        st.stop()

    prompt = (
        f"Compare the investment prospects of {ticker_a.strip().upper()} vs {ticker_b.strip().upper()}. "
        "Use the finance tools to gather facts. "
        "Return a concise, well-structured markdown report with at least one comparison table and a final summary."
    )

    with st.spinner("Analyzing..."):
        try:
            agent = build_agent(current_key)
            result = agent.run(prompt, stream=False)
        except Exception as e:
            st.exception(e)
            st.stop()

    # Agno may return an object with .content or a plain string
    try:
        content = result.content if hasattr(result, "content") else str(result)
    except Exception:
        content = str(result)

    st.markdown(content)
