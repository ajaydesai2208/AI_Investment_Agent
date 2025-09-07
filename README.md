# 📈 AI Investment Agent

A simple Streamlit app that compares two stock tickers using Yahoo Finance data with an AI agent powered by OpenAI.  
It generates a clear markdown report with tables, pros and cons, and a balanced comparison.

**Key behavior:** the app uses a two-way OpenAI key flow.
1. If a local key is available (via `.env` or Streamlit `secrets`), it is used automatically.
2. If no local key is found, the UI shows a password field so anyone can paste a key and run the app.

> This tool is for educational use only. It is not financial advice.

---

## Features

- Compare any two tickers, for example AAPL vs MSFT
- Pulls facts from Yahoo Finance via Agno YFinance tools
- Produces a concise markdown report with at least one comparison table
- Two-way OpenAI key handling  
  - Uses `OPENAI_API_KEY` from `.env` or `st.secrets` if present  
  - Otherwise shows a password field in the sidebar
- No secrets committed to Git by default (`.env` is git-ignored)

---

## Quick Start

### Prerequisites
- Python 3.10 or 3.11
- An OpenAI API key

### 1) Create a virtual environment and install deps

```bash
# from the project root
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Provide an API key

Pick one of the two options below.

**Option A:** local .env (recommended for your own machine)
Create a file named .env in the project root:

```bash
OPENAI_API_KEY=sk-********************************
```

The app will load it automatically on startup.

**Option B:** paste your key in the UI

If no local key is found, run the app and use the password field in the sidebar to paste your OpenAI key for the current session.

### 3) Run the App

```bash
streamlit run investment_agent.py
```

Open the URL shown in the terminal (usually http://localhost:8501). Enter two tickers and click Compare.

## Project Structure
```bash
AI_Investment_Agent/
├─ investment_agent.py      # Streamlit app and agent wiring
├─ requirements.txt         # Dependencies: streamlit, agno, openai, yfinance, python-dotenv
├─ README.md                # This file
├─ .gitignore               # Ignores .env, venv, caches, editor files
└─ .env                     # Local secret (optional, not committed)
```

## Configuration
1. The app looks for an OpenAI key in this order:
2. st.session_state["openai_api_key"] when you paste a key in the UI
3. Environment variables loaded from .env

st.secrets["OPENAI_API_KEY"] when using .streamlit/secrets.toml

### Environment variables:

- OPENAI_API_KEY
Required for automatic auth. If missing, the UI will ask for a key.

Optional Streamlit secrets file:

```markdown
.streamlit/secrets.toml
OPENAI_API_KEY = "sk-********************************"
```
