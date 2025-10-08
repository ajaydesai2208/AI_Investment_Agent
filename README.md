# 📈 AI Investment Agent

[![Live Demo – Cloud Run](https://img.shields.io/badge/Live%20Demo-Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)](https://ai.finquant.xyz/)


A Streamlit app that compares two stock tickers using Yahoo Finance data with an AI agent powered by OpenAI.  
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
  - Yahoo Finance RSS, yfinance API
  - Reuters RSS (filtered by ticker/company name)
  - SEC EDGAR filings (8-K, 10-Q, 10-K, S-1, S-3, 424B5, 6-K)
  - Latest snapshot persists after each analysis so you can review what the model saw (clear it with **Refresh caches**)

- Options snapshot
  - Nearest expiry (auto 7–45 DTE) or pick a specific expiry
  - ATM call/put mids, ATM strike, straddle debit, implied move, approximate ATM IV
  - ATM Greeks (per contract)
  - Strategy Picker (debit/credit verticals, CSP, covered/collar) with add-to-report and add-to-tickets

- Position sizing
  - Profile-based risk budget (Conservative / Balanced / Aggressive)
  - Equity share size and debit options contract count using implied move and recent vol
  - Baseline (implied-move anchored) and ATR-based sizing, plus a stock Trade Plan (entry/stop/targets)

- Time-aware prompt
  - Today’s date is passed to the model; it avoids guessing dates and labels past vs upcoming items

- Fundamentals (TTM) with extended valuation/returns: P/E, P/S, EV/EBITDA, ROIC (approx)

- Pair Analyzer (beta-hedged spread) with z-score guidance around ±2.0

- Event Playbook (earnings/SEC/regime-aware tactics) per ticker

- Scenarios & Payoff charts (stock per share and option per contract) using implied move as σ proxy

- Trade Tickets (CSV) built from sizing/options and Strategy Picker; preview in-app

- Download the full report as Markdown and PDF (PDF includes fundamentals, catalysts, pair notes, sizing notes, strategies, playbook, scenarios, tickets)

- Sidebar controls: source toggles, expiry mode, and a “Refresh caches” button (clears stored news + catalysts)

- Strategy comparison: select up to 2 strategies and compare Net (Debit/Credit), Max Loss/Gain, Breakevens, DTE, R:R, and Capital Requirement

- High-contrast mode toggle (Display section) for brighter text and borders on the dark theme

- Position sizing: inline risk budget badge (e.g., “$250 at 1%”) based on profile and account equity

- Inline cache TTL hints: News ~15m (SEC ticker map ~24h), Options ~5m

- Skeleton loaders: lightweight shimmering placeholders during data fetches

- Safety: “Compare & Analyze” is disabled until both tickers are valid

- Two-way OpenAI key handling and .env/secrets integration

- Ticker autocomplete: type a ticker or company name to see suggestions (case-insensitive, prefix+substring, scrollable; SEC company list + Yahoo live search for new IPOs; cached)

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
# if you manage deps manually, ensure this extra UI component is installed:
# pip install streamlit-searchbox
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
├─ investment_agent.py           # Streamlit app orchestrator (thin)
├─ ui/                           # UI package (no behavior changes)
│  ├─ __init__.py
│  ├─ page.py                    # page config, global CSS, high-contrast toggler
│  ├─ background.py              # background cache refresher
│  ├─ state.py                   # session_state initialization
│  ├─ helpers.py                 # shared helpers (formatting, validation, inference)
│  ├─ sidebar.py                 # sidebar controls (auth, display, strategy, sources, expiry)
│  ├─ tickers.py                 # ticker search inputs and expiry pickers
│  ├─ preload.py                 # snapshots, options, catalysts, pair, spark/trend/regime
│  ├─ stat_bar.py                # top stat grid and 3M sparklines
│  ├─ analysis_runner.py         # run_analysis implementation
│  └─ tabs/
│     ├─ overview.py             # chart, snapshot, fundamentals, catalysts, pair analyzer
│     ├─ news.py                 # news/filings with source breakdown
│     ├─ options.py              # options snapshot, greeks, strategy picker and compare
│     ├─ sizing.py               # sizing tables and stock trade plan
│     ├─ scenarios.py            # scenarios table and payoff charts
│     └─ report.py               # report render, PDF/MD export, tickets, playbook
├─ ai_agent/
│  ├─ __init__.py
│  ├─ settings.py
│  ├─ news.py
│  ├─ prices.py
│  ├─ metrics.py
│  ├─ options.py
│  ├─ greeks.py
│  ├─ quant.py
│  ├─ risk.py
│  ├─ scenario.py
│  ├─ strategies.py
│  ├─ tickets.py
│  ├─ playbook.py
│  ├─ prompts.py
│  ├─ agent.py
│  └─ export.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Running

1) Create/activate the virtual environment (first time only), then install deps:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Start the app:
```bash
streamlit run investment_agent.py
```

The UI and behavior are unchanged. The app now imports UI components from `ui/` while keeping all data logic under `ai_agent/`.

## Configuration
The app looks for an OpenAI key in this order:
1. `st.session_state["openai_api_key"]` when you paste a key in the UI
2. Environment variables loaded from `.env`
3. `st.secrets["OPENAI_API_KEY"]` when using `.streamlit/secrets.toml`

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

Optional Streamlit theme/config:

```toml
# .streamlit/config.toml
[theme]
base = "dark"
primaryColor = "#00C805"
backgroundColor = "#0A0F14"
secondaryBackgroundColor = "#11181F"
textColor = "#E6F2E6"
font = "Inter"

[server]
headless = true

[browser]
gatherUsageStats = false
```

## License

This project is provided as-is, for educational use only. It is not financial advice.
