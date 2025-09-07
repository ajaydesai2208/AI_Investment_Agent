"""
Catalysts: earnings dates (last + next) and recent SEC filings, with
freshness safeguards and multiple fallbacks.

This version strengthens date extraction from news items so that the
fallback correctly recognizes recent earnings headlines (e.g., NVDA Aug 27, 2025).

Key fixes:
- _safe_date_from_item now checks many common keys:
  'published', 'published_at', 'pubDate', 'date', 'time',
  'updated', 'updated_parsed', 'providerPublishTime' (epoch), plus numeric fallbacks.
- Earnings regex slightly broadened (results/EPS/revenue/guidance/“FY” patterns).
- We merge yfinance & news candidates and choose the newest PAST date.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import re

import pandas as pd
import streamlit as st

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

from .news import get_recent_news  # reuse your news collector


# ------------------------ small utils ------------------------

def _utc_today_date() -> datetime.date:
    return datetime.now(timezone.utc).date()

def _safe_date(x) -> Optional[datetime.date]:
    if x is None or x == "":
        return None
    try:
        ts = pd.to_datetime(x, utc=True)
        if isinstance(ts, pd.Timestamp):
            return ts.date()
        return ts.date()  # type: ignore[return-value]
    except Exception:
        return None

def _safe_date_from_item(it: Dict) -> Optional[datetime.date]:
    """
    Try many real-world keys that feeds use for article timestamps.

    Normalized keys we try first: 'published' (ISO), 'published_ts' (epoch seconds).

    Then we probe common provider fields:
      - 'published_at', 'pubDate', 'date', 'time',
      - 'updated', 'updated_parsed' (feedparser),
      - 'providerPublishTime' (yfinance; epoch seconds),
      - Any numeric-looking value that can be interpreted as epoch seconds or ms.
    """
    # First, our normalized fields (if your news module already sets them)
    d = _safe_date(it.get("published"))
    if d:
        return d

    ts = it.get("published_ts")
    try:
        if ts is not None:
            return pd.to_datetime(float(ts), unit="s", utc=True).date()
    except Exception:
        pass

    # Provider-specific keys (ISO-ish)
    for k in ("published_at", "pubDate", "date", "time", "updated"):
        d = _safe_date(it.get(k))
        if d:
            return d

    # feedparser struct_time in 'updated_parsed'
    try:
        up = it.get("updated_parsed")
        if up:
            return pd.to_datetime(up, utc=True).date()
    except Exception:
        pass

    # yfinance: epoch seconds
    try:
        t = it.get("providerPublishTime")
        if t is not None:
            return pd.to_datetime(float(t), unit="s", utc=True).date()
    except Exception:
        pass

    # last resort: any single numeric value that looks like epoch (sec or ms)
    try:
        for k, v in it.items():
            if isinstance(v, (int, float)) and k.lower() not in {"volume", "rank"}:
                # try seconds
                try:
                    return pd.to_datetime(float(v), unit="s", utc=True).date()
                except Exception:
                    pass
                # try milliseconds
                try:
                    return pd.to_datetime(float(v), unit="ms", utc=True).date()
                except Exception:
                    pass
    except Exception:
        pass

    return None

def _age_label(d: Optional[datetime.date]) -> str:
    if d is None:
        return "—"
    today = _utc_today_date()
    delta = (d - today).days
    if delta < 0:
        return f"PAST, {-delta} days ago"
    if delta > 0:
        return f"IN {delta} days"
    return "TODAY"

def _dedupe_keep_latest(items: List[Dict], key: str = "link") -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for it in sorted(items, key=lambda z: z.get("published_ts", 0), reverse=True):
        k = it.get(key)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# ------------------------ earnings via yfinance ------------------------

@st.cache_data(ttl=60 * 60)  # 1 hour
def _yf_earnings_dates_df(ticker: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=20)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()

def _pick_last_next_from_df(df: pd.DataFrame) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    if df is None or df.empty:
        return None, None
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif "Earnings Date" in df.columns:
            dates = pd.to_datetime(df["Earnings Date"], errors="coerce", utc=True)
        else:
            return None, None
        dates = pd.to_datetime(dates, utc=True)
    except Exception:
        return None, None

    today = pd.Timestamp(_utc_today_date(), tz="UTC")
    past = dates[dates <= today]
    future = dates[dates > today]
    last_d = past.max().date() if len(past) else None
    next_d = future.min().date() if len(future) else None
    return last_d, next_d


# ------------------------ fallback: infer from news ------------------------

# Expanded regex: captures “Qx”, “FY”, “results”, “EPS”, “revenue”, “guidance”
_EARNINGS_PAT = re.compile(
    r"\b(earnings|results|quarter|q[1-4]\b|full[- ]year|fy\d{4}|\bfy\b|eps|revenue|guidance)\b",
    re.I,
)

def _infer_last_earnings_from_news(ticker: str, lookback_days: int = 210) -> Optional[datetime.date]:
    try:
        news_all = get_recent_news(
            ticker,
            days=lookback_days,
            limit=180,
            use_rss=True,
            use_yf=True,
            use_reuters=True,
            use_sec=False,
        )
        items = news_all.get("merged", [])
        best_date = None
        for it in items:
            title = (it.get("title") or "").strip()
            if not title:
                continue
            if not _EARNINGS_PAT.search(title):
                continue
            d = _safe_date_from_item(it)
            if d is None:
                continue
            if (best_date is None) or (d > best_date):
                best_date = d
        return best_date
    except Exception:
        return None


# ------------------------ SEC filings via EDGAR feed ------------------------

@st.cache_data(ttl=30 * 60)  # 30 minutes
def _recent_sec_filings(ticker: str, lookback_days: int = 150) -> List[Dict]:
    try:
        news_all = get_recent_news(
            ticker,
            days=lookback_days,
            limit=200,
            use_rss=False,
            use_yf=False,
            use_reuters=False,
            use_sec=True,
        )
        sec_items = news_all.get("by_source", {}).get("sec_edgar", [])
        keep_forms = ("8-K", "10-Q", "10-K", "S-1", "S-3", "S-4", "424B", "13D", "13G")
        out: List[Dict] = []
        for it in sec_items:
            title = (it.get("title") or "").strip()
            link = it.get("link")
            d = _safe_date_from_item(it)
            if not d:
                continue
            if any(form in title.upper() for form in keep_forms):
                out.append(
                    {
                        "title": title,
                        "link": link,
                        "published": d.isoformat(),
                        "published_ts": pd.Timestamp(d).timestamp(),
                    }
                )
        return _dedupe_keep_latest(out)
    except Exception:
        return []


# ------------------------ public API ------------------------

@st.cache_data(ttl=30 * 60)  # 30 minutes
def get_catalysts(ticker: str, *, lookback_days: int = 210) -> Dict:
    """
    Return a dict with:
      - last_earnings: newest PAST date derived from yfinance & news
      - last_earnings_source: 'yfinance' | 'news' | 'yfinance+news' | None
      - next_earnings: earliest FUTURE date from yfinance (if valid)
      - sec_filings: list of recent important EDGAR items
      - asof: today's date (iso)
    """
    today = _utc_today_date()

    # yfinance baseline
    last_yf, next_yf = None, None
    src = None
    try:
        df = _yf_earnings_dates_df(ticker)
        last_yf, next_yf = _pick_last_next_from_df(df)
        if last_yf or next_yf:
            src = "yfinance"
    except Exception:
        pass

    # news inference
    last_news = _infer_last_earnings_from_news(ticker, lookback_days=lookback_days)

    # Choose newest PAST among candidates
    candidates = [d for d in (last_yf, last_news) if d is not None and d <= today]
    if candidates:
        last_final = max(candidates)
        if last_news and last_yf and abs((last_news - last_yf).days) <= 1:
            last_src = "yfinance+news"
        elif last_final == last_news:
            last_src = "news"
        else:
            last_src = "yfinance"
    else:
        last_final = last_yf or last_news
        last_src = "news" if last_final == last_news else ("yfinance" if last_final == last_yf else None)

    # Keep only credible future 'next' from yfinance
    next_final = next_yf if (next_yf and next_yf > today) else None

    # Filings
    sec = _recent_sec_filings(ticker, lookback_days=lookback_days)

    return {
        "last_earnings": last_final.isoformat() if last_final else None,
        "last_earnings_source": last_src,
        "next_earnings": next_final.isoformat() if next_final else None,
        "sec_filings": sec,
        "asof": today.isoformat(),
    }

def clear_catalyst_cache() -> None:
    try:
        _yf_earnings_dates_df.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _recent_sec_filings.clear()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        get_catalysts.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

def format_catalysts_md(ticker: str, cat: Dict) -> str:
    t = ticker.upper()
    last_iso = cat.get("last_earnings")
    next_iso = cat.get("next_earnings")
    last_src = cat.get("last_earnings_source") or "—"
    asof = cat.get("asof") or _utc_today_date().isoformat()

    last_d = _safe_date(last_iso)
    next_d = _safe_date(next_iso)

    lines = [f"### {t} — Catalyst Radar", ""]
    lines.append(f"- Last earnings: {last_iso or '—'} ({_age_label(last_d)}; source: {last_src})")
    lines.append(f"- Next earnings: {next_iso or '—'} ({_age_label(next_d)})")

    filings = cat.get("sec_filings") or []
    if filings:
        lines.append(f"- SEC filings (last {len(filings)} in window):")
        for it in filings[:3]:
            d = _safe_date(it.get("published"))
            title = it.get("title", "").strip()
            link = it.get("link", "")
            datetxt = d.isoformat() if d else "—"
            lines.append(f"  - {datetxt}: [{title}]({link})")
    else:
        lines.append("- SEC filings: none in window")

    lines.append("")
    lines.append(f"As of {asof}")
    lines.append("")
    return "\n".join(lines)
