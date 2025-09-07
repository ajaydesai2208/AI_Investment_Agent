# 📈 AI Investment Agent

A simple Streamlit app that compares two stock tickers using Yahoo Finance data with an AI agent powered by OpenAI.  
It generates a clear markdown report with tables, pros and cons, and a balanced comparison.

**Key behavior:** the app uses a two-way OpenAI key flow.
1. If a local key is available (via `.env` or Streamlit `secrets`), it is used automatically.
2. If no local key is found, the UI shows a password field so anyone can paste a key and run the app.

> This tool is for educational use only. It is not financial advice.

---

## Features

- Compare any two tickers (e.g., AAPL vs MSFT)

- Price charts with optional normalization (index to 100)

- News & filings ingestion with toggles

-- Yahoo Finance RSS, yfinance API

-- Reuters RSS (filtered by ticker/company name)

-- SEC EDGAR filings (8-K, 10-Q, 10-K, S-1, S-3, 424B5, 6-K)

- Options snapshot

-- Nearest expiry (auto 7–45 DTE) or pick a specific expiry

-- ATM call/put mids, ATM strike, straddle debit, implied move, approximate ATM IV

- Position sizing

-- Profile-based risk budget (Conservative / Balanced / Aggressive)

-- Equity share size and debit options contract count using implied move and recent vol

- Time-aware prompt

-- Today’s date is passed to the model; it avoids guessing dates and labels past vs upcoming items

- Download the full report as Markdown (includes metadata, facts, options table, and the model’s analysis)

- Two-way OpenAI key handling and .env/secrets integration

- .env is git-ignored by default

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
# Optional: helps with polite SEC requests (any identifying UA is fine)
SEC_USER_AGENT=AI_Investment_Agent/1.0 (contact@example.com)
```

The app will load it automatically on startup.

**Option B:** paste your key in the UI

If no local key is found, run the app and use the password field in the sidebar to paste your OpenAI key for the current session.

### 3) Run the App

```bash
streamlit run investment_agent.py
```

Open the URL shown in the terminal (usually http://localhost:8501). Enter two tickers and click Compare.

---

## Project Structure
```bash
AI_Investment_Agent/
├─ investment_agent.py           # Streamlit app (single entry point)
├─ ai_agent/
│  ├─ __init__.py                # marks package (can be empty)
│  ├─ settings.py                # key loading (.env, st.secrets, UI)
│  ├─ news.py                    # Yahoo/yfinance + Reuters + SEC sources, toggles, formatting
│  ├─ prices.py                  # price history merge, normalization, chart data
│  ├─ metrics.py                 # snapshots, returns/momentum, compare table, facts pack
│  ├─ options.py                 # expiry list, ATM snapshot, implied move, IV
│  ├─ risk.py                    # risk profiles, stop distance, share/contract sizing
│  ├─ prompts.py                 # time-aware, structured prompt builder (no italics/bold)
│  ├─ agent.py                   # OpenAI agent wrapper
│  └─ export.py                  # sanitize + build Markdown export
├─ requirements.txt              # streamlit, openai, yfinance, pandas, python-dotenv, feedparser, requests
├─ .gitignore                    # ignores .env, venv, caches, editor files
└─ README.md
```

## Configuration
1. The app looks for an OpenAI key in this order:
2. st.session_state["openai_api_key"] when you paste a key in the UI
3. Environment variables loaded from .env

st.secrets["OPENAI_API_KEY"] when using .streamlit/secrets.toml

### Environment variables:

- OPENAI_API_KEY
Required for automatic auth. If missing, the UI will ask for a key.

- SEC_USER_AGENT
Optional but recommended for SEC requests (e.g., AI_Investment_Agent/1.0 (email@domain)).

Optional Streamlit secrets file:

```markdown
.streamlit/secrets.toml
OPENAI_API_KEY = "sk-********************************"
```

## License

This project is provided as-is, for educational use only. It is not financial advice.
