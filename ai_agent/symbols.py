from __future__ import annotations

import json
import re
from typing import List, Dict
from urllib.parse import quote_plus

import requests
import streamlit as st

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_symbols() -> List[Dict[str, str]]:
    """
    Load SEC company tickers index (once per day).
    Returns list of dicts: {ticker, name, cik}
    """
    headers = {"User-Agent": "AI_Investment_Agent/1.0 (https://github.com/)"}
    try:
        resp = requests.get(SEC_COMPANY_TICKERS_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    items: List[Dict[str, str]] = []
    # Data format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    try:
        for _, obj in data.items():
            ticker = (obj.get("ticker") or "").strip().upper()
            name = (obj.get("title") or "").strip()
            cik = str(obj.get("cik_str") or "").strip()
            if ticker and name:
                items.append({"ticker": ticker, "name": name, "cik": cik})
    except Exception:
        return []
    return items


@st.cache_data(ttl=900, show_spinner=False)
def _yahoo_search_symbols(query: str, limit: int = 50) -> List[Dict[str, str]]:
    """
    Yahoo Finance search API (quotes only). This updates faster than SEC list
    and will surface newly listed tickers (e.g., IPOs) quickly.
    """
    q = (query or "").strip()
    if not q:
        return []
    url = (
        "https://query1.finance.yahoo.com/v1/finance/search?"
        f"q={quote_plus(q)}&quotesCount={int(max(1, limit))}&newsCount=0&lang=en-US&region=US"
    )
    headers = {"User-Agent": "AI_Investment_Agent/1.0 (https://github.com/)", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
    except Exception:
        return []

    out: List[Dict[str, str]] = []
    for q in data.get("quotes", []) or []:
        # Filter to equities on major US exchanges
        qtype = (q.get("quoteType") or "").lower()
        exch = (q.get("exchange") or q.get("fullExchangeName") or "").strip()
        if qtype and qtype != "equity":
            continue
        sym = (q.get("symbol") or "").strip().upper()
        nm = (q.get("shortname") or q.get("longname") or q.get("name") or "").strip()
        if sym and nm:
            out.append({"ticker": sym, "name": nm, "exchange": exch})
    return out


def _score_symbol(query: str, ticker: str, name: str) -> int:
    """
    Simple ranking heuristic:
      - exact ticker match: 100
      - ticker startswith: 90
      - name startswith: 80
      - ticker substring: 70
      - name substring: 60
    Lower scores are worse. Non-matches return -1.
    """
    q = query.strip()
    if not q:
        return -1
    ql = q.lower()
    tl = ticker.lower()
    nl = name.lower()

    if q.upper() == ticker:
        return 100  # exact ticker match
    if nl == ql:
        return 95   # exact company name match
    if tl.startswith(ql):
        return 90   # ticker startswith
    if nl.startswith(ql):
        return 85   # name startswith
    if ql in tl:
        return 70   # ticker substring
    if ql in nl:
        return 60   # name substring
    return -1


def search_symbols(query: str, limit: int = 8) -> List[Dict[str, str]]:
    """
    Search symbols from SEC index and Yahoo Finance search.
    - SEC provides broad official mapping (daily refreshed)
    - Yahoo surfaces new listings quickly (near real-time)
    Returns top-N matches sorted by score then ticker length, deduped by ticker.
    """
    query = (query or "").strip()
    if not query:
        return []

    symbols = load_sec_symbols()  # daily cache
    yahoo = _yahoo_search_symbols(query, limit=max(50, limit))  # 15m cache

    # Merge, preferring SEC names when both exist
    merged: Dict[str, Dict[str, str]] = {}
    for it in symbols:
        merged[it["ticker"].upper()] = {"ticker": it["ticker"].upper(), "name": it["name"], "exchange": ""}
    for it in yahoo:
        t = it["ticker"].upper()
        if t not in merged:
            merged[t] = {"ticker": t, "name": it.get("name", t), "exchange": it.get("exchange", "")}

    ranked = []
    for item in merged.values():
        sc = _score_symbol(query, item["ticker"], item["name"])
        if sc >= 0:
            ranked.append((sc, len(item["ticker"]), item))

    ranked.sort(key=lambda x: (-x[0], x[1], x[2]["ticker"]))
    return [it for _, __, it in ranked[: max(1, int(limit))]]


def format_symbol_label(item: Dict[str, str], query: str | None = None) -> str:
    """
    Create a human-friendly label like "AAPL — Apple Inc. (NASDAQ)".
    """
    t = item.get("ticker", "").upper()
    n = item.get("name", "")
    ex = item.get("exchange", "")
    ex_part = f" ({ex})" if ex else ""
    return f"{t} — {n}{ex_part}"


def clear_symbols_cache() -> None:
    try:
        load_sec_symbols.clear()
    except Exception:
        pass


