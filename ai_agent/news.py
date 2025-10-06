from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Public exports
__all__ = [
    "get_recent_news",
    "format_news_digest",
    "format_news_table",
    "clear_news_cache",
]

# ---------- HTTP headers ----------

_DEFAULT_UA = "AI_Investment_Agent/1.0 (+https://github.com/)"
_HEADERS = {
    # You can optionally set SEC_USER_AGENT in your .env for polite SEC access
    "User-Agent": os.getenv("SEC_USER_AGENT", _DEFAULT_UA),
}

_REUTERS_HEADERS = {"User-Agent": "Mozilla/5.0 (AI_Investment_Agent/1.0)"}
_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (AI_Investment_Agent/1.0)"}


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------- Yahoo RSS (existing) ----------

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_yahoo_rss_news(ticker: str) -> List[Dict]:
    """
    Primary source: Yahoo Finance RSS (company headlines).
    Returns list of dicts: {title, publisher, link, published, source}.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"
    try:
        resp = requests.get(url, timeout=10, headers=_RSS_HEADERS)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except Exception:
        return []

    items: List[Dict] = []
    for e in feed.entries:
        pub_dt = None
        if getattr(e, "published_parsed", None):
            pub_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        elif getattr(e, "updated_parsed", None):
            pub_dt = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)
        items.append(
            {
                "title": (getattr(e, "title", "") or "").strip(),
                "publisher": (
                    (getattr(getattr(e, "source", {}), "title", None) or "Yahoo RSS").strip()
                    if hasattr(e, "source") else "Yahoo RSS"
                ),
                "link": (getattr(e, "link", "") or "").strip(),
                "published": pub_dt or _to_utc(datetime.utcnow()),
                "source": "yahoo_rss",
            }
        )
    return items


# ---------- yfinance news (existing) ----------

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_yf_news(ticker: str) -> List[Dict]:
    """
    Secondary source: yfinance .news (can be flaky).
    Normalized to the same structure.
    """
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
    except Exception:
        raw = []

    items: List[Dict] = []
    for it in raw:
        ts = it.get("providerPublishTime")
        if ts:
            try:
                published = datetime.utcfromtimestamp(int(ts)).replace(tzinfo=timezone.utc)
            except Exception:
                published = _to_utc(datetime.utcnow())
        else:
            published = _to_utc(datetime.utcnow())
        items.append(
            {
                "title": (it.get("title") or "").strip(),
                "publisher": (it.get("publisher") or "Yahoo Finance").strip(),
                "link": (it.get("link") or "").strip(),
                "published": published,
                "source": "yfinance",
            }
        )
    return items


# ---------- Reuters RSS (NEW) ----------

_REUTERS_FEEDS = [
    # Good, legal Reuters RSS feeds. We’ll filter by ticker/company name.
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/marketsNews",
    "https://feeds.reuters.com/reuters/technologyNews",
]

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_reuters_rss_filtered(ticker: str, company_hint: Optional[str]) -> List[Dict]:
    """
    Pull general Reuters RSS feeds and filter items that mention the ticker or the company name.
    Note: Reuters company-specific RSS via RIC requires knowing the RIC; we stay on general feeds (legal & simple).
    """
    if not ticker:
        return []

    kw = {ticker.upper(), ticker.lower()}
    if company_hint:
        # Add loose company name tokens as keywords (split on spaces, keep words >2 chars)
        for tok in company_hint.replace(",", " ").split():
            if len(tok) > 2:
                kw.add(tok)
                kw.add(tok.lower())
                kw.add(tok.upper())

    items: List[Dict] = []
    for feed_url in _REUTERS_FEEDS:
        try:
            r = requests.get(feed_url, timeout=10, headers=_REUTERS_HEADERS)
            r.raise_for_status()
            parsed = feedparser.parse(r.text)
        except Exception:
            continue

        for e in getattr(parsed, "entries", []):
            title = (getattr(e, "title", "") or "").strip()
            summary = (getattr(e, "summary", "") or "").strip()
            text = f"{title} {summary}"

            # Filter: any keyword present
            if not any(k in text for k in kw):
                continue

            pub_dt = None
            if getattr(e, "published_parsed", None):
                pub_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            elif getattr(e, "updated_parsed", None):
                pub_dt = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)

            items.append(
                {
                    "title": title,
                    "publisher": "Reuters",
                    "link": (getattr(e, "link", "") or "").strip(),
                    "published": pub_dt or _to_utc(datetime.utcnow()),
                    "source": "reuters_rss",
                }
            )
    return items


# ---------- SEC EDGAR filings (NEW) ----------

@st.cache_data(ttl=86400, show_spinner=False)
def _sec_ticker_map() -> Dict[str, str]:
    """
    Download ticker -> CIK map from the SEC (daily). The file format is: 'aapl\t0000320193' (lower-case ticker).
    """
    url = "https://www.sec.gov/include/ticker.txt"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        mapping = {}
        for line in lines:
            try:
                tkr, cik = line.strip().split("\t")
                mapping[tkr.lower()] = cik.zfill(10)
            except Exception:
                continue
        return mapping
    except Exception:
        return {}

def _sec_cik_for_ticker(ticker: str) -> Optional[str]:
    m = _sec_ticker_map()
    return m.get((ticker or "").lower())

def _sec_doc_url(cik: str, acc_no: str, primary_doc: Optional[str]) -> str:
    """
    Build a direct EDGAR link to the filing document.
    """
    # e.g., CIK '0000320193' -> '320193'
    cik_num = str(int(cik))
    acc_nodash = acc_no.replace("-", "")
    if primary_doc:
        return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/{primary_doc}"
    # fallback to directory
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/"

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_sec_filings(ticker: str) -> List[Dict]:
    """
    Fetch recent filings via SEC submissions endpoint.
    Normalized to: {title, publisher, link, published, source}.
    """
    cik = _sec_cik_for_ticker(ticker)
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    recent = (data.get("filings", {}) or {}).get("recent", {}) or {}
    acc = recent.get("accessionNumber", []) or []
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    prim = recent.get("primaryDocument", []) or []

    items: List[Dict] = []
    for i in range(min(len(acc), len(forms), len(dates))):
        form = (forms[i] or "").strip()
        # Focus on investor-relevant forms; keep 8-K/10-Q/10-K/S-1 etc.
        if form not in ("8-K", "10-Q", "10-K", "S-1", "S-3", "424B5", "6-K"):
            continue
        try:
            dt = pd.to_datetime(dates[i]).to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            dt = _to_utc(datetime.utcnow())
        accno = acc[i]
        link = _sec_doc_url(cik, accno, prim[i] if i < len(prim) else None)
        title = f"{ticker} {form} filing"
        items.append(
            {
                "title": title,
                "publisher": "SEC EDGAR",
                "link": link,
                "published": dt,
                "source": "sec_edgar",
            }
        )
    return items


# ---------- Merge, dedupe, filter ----------

def _dedupe_and_filter(items: List[Dict], days: int, limit: int) -> List[Dict]:
    cutoff = _to_utc(datetime.utcnow()) - timedelta(days=days)
    seen = set()
    cleaned: List[Dict] = []

    # newest first
    items_sorted = sorted(items, key=lambda x: x.get("published", _to_utc(datetime.utcnow())), reverse=True)

    for it in items_sorted:
        title = it.get("title", "")
        link = it.get("link", "")
        published = it.get("published")
        key = (title[:120], link[:180])
        if published and published < cutoff:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(it)
        if len(cleaned) >= limit:
            break
    return cleaned


# ---------- Public API ----------

@st.cache_data(ttl=900, show_spinner=False)
def get_recent_news(
    ticker: str,
    days: int,
    limit: int,
    *,
    use_rss: bool = True,
    use_yf: bool = True,
    use_reuters: bool = True,   # NEW
    use_sec: bool = True,       # NEW
) -> Dict[str, List[Dict]]:
    """
    Fetch+merge news with source toggles.

    Returns:
      {
        "merged": [news dicts within lookback window, deduped & capped],
        "by_source": {
            "yahoo_rss": [...],
            "yfinance":  [...],
            "reuters_rss": [...],
            "sec_edgar": [...]
        }
      }
    """
    # We’ll use company shortName from yfinance as an extra filter hint for Reuters.
    short_name = None
    try:
        t = yf.Ticker(ticker)
        info = {}
        get_info = getattr(t, "get_info", None)
        info = get_info() if callable(get_info) else dict(getattr(t, "info", {}) or {})
        short_name = info.get("shortName") or info.get("longName")
    except Exception:
        short_name = None

    rss = _fetch_yahoo_rss_news(ticker) if use_rss else []
    yf_items = _fetch_yf_news(ticker) if use_yf else []
    reuters = _fetch_reuters_rss_filtered(ticker, short_name) if use_reuters else []
    sec_items = _fetch_sec_filings(ticker) if use_sec else []

    merged = _dedupe_and_filter((rss or []) + (yf_items or []) + (reuters or []) + (sec_items or []), days, limit)

    # Sentiment scoring (headline-only; fast and cached)
    try:
        analyzer = SentimentIntensityAnalyzer()
        for it in merged:
            title = it.get("title") or ""
            vs = analyzer.polarity_scores(title)
            it["sentiment"] = vs.get("compound", 0.0)
    except Exception:
        for it in merged:
            it["sentiment"] = 0.0
    return {
        "merged": merged,
        "by_source": {
            "yahoo_rss": rss,
            "yfinance": yf_items,
            "reuters_rss": reuters,
            "sec_edgar": sec_items,
        },
    }


def format_news_digest(ticker: str, news: List[Dict]) -> str:
    """
    Render a compact markdown digest sent into the LLM prompt.
    """
    if not news:
        return f"No recent news for {ticker} in the chosen window."
    lines: List[str] = []
    for n in news:
        dt = n.get("published")
        try:
            date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
        except Exception:
            date_str = "NA"
        title = n.get("title") or ""
        pub = n.get("publisher") or "Unknown"
        url = n.get("link") or ""
        src = n.get("source") or ""
        tag = "FILING" if src == "sec_edgar" else ("Reuters" if src == "reuters_rss" else pub)
        lines.append(f"- [{date_str}] {ticker} — {tag}: {title} ({url})")
    return "\n".join(lines)


def format_news_table(news: List[Dict]) -> pd.DataFrame:
    """
    Build a compact DataFrame with date, publisher/source, title, sentiment.
    """
    if not news:
        return pd.DataFrame(columns=["Date", "Source", "Title", "Sentiment"])
    rows = []
    for n in news:
        dt = n.get("published")
        try:
            date_str = pd.to_datetime(dt).strftime("%Y-%m-%d")
        except Exception:
            date_str = "NA"
        src = n.get("source") or ""
        tag = "FILING" if src == "sec_edgar" else ("Reuters" if src == "reuters_rss" else (n.get("publisher") or ""))
        rows.append([
            date_str,
            tag,
            n.get("title") or "",
            round(float(n.get("sentiment", 0.0)), 3),
        ])
    return pd.DataFrame(rows, columns=["Date", "Source", "Title", "Sentiment"])


def clear_news_cache() -> None:
    """Helper to clear Streamlit's cache for news (used by UI button)."""
    st.cache_data.clear()
