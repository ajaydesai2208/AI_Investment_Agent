from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import hashlib

import streamlit as st

from ai_agent.news import get_recent_news


NEWS_SNAPSHOT_KEY = "news_snapshot"


def build_news_params(
    *,
    ticker_a: str,
    ticker_b: str,
    lookback_days: int,
    max_news: int,
    use_rss: bool,
    use_yf: bool,
    use_reuters: bool,
    use_sec: bool,
    hf_mode: bool,
) -> Dict[str, Any]:
    """Normalize parameters used to cache news results."""
    return {
        "ticker_a": (ticker_a or "").strip().upper(),
        "ticker_b": (ticker_b or "").strip().upper(),
        "lookback_days": int(lookback_days),
        "max_news": int(max_news),
        "use_rss": bool(use_rss),
        "use_yf": bool(use_yf),
        "use_reuters": bool(use_reuters),
        "use_sec": bool(use_sec),
        "hf_mode": bool(hf_mode),
    }


def clear_news_snapshot() -> None:
    """Remove any cached news snapshot/state."""
    st.session_state.pop(NEWS_SNAPSHOT_KEY, None)
    st.session_state.pop("news_digest_hash", None)


def _cached_snapshot(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    snapshot = st.session_state.get(NEWS_SNAPSHOT_KEY)
    if snapshot and snapshot.get("params") == params:
        snapshot.setdefault("digest_changed", False)
        return snapshot
    return None


def _news_kwargs_from_params(params: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "use_rss": params["use_rss"],
        "use_yf": params["use_yf"],
        "use_reuters": params["use_reuters"],
        "use_sec": params["use_sec"],
    }


def _fetch_snapshot(params: Dict[str, Any]) -> Dict[str, Any]:
    news_kwargs = _news_kwargs_from_params(params)
    news_a_all = get_recent_news(
        params["ticker_a"], params["lookback_days"], params["max_news"], **news_kwargs
    )
    news_b_all = get_recent_news(
        params["ticker_b"], params["lookback_days"], params["max_news"], **news_kwargs
    )
    digest_src = repr(news_a_all.get("merged", [])) + "|" + repr(news_b_all.get("merged", []))
    digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()
    snapshot = {
        "params": dict(params),
        "ticker_a": params["ticker_a"],
        "ticker_b": params["ticker_b"],
        "news_a_all": news_a_all,
        "news_b_all": news_b_all,
        "digest": digest,
        "digest_changed": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return snapshot


def ensure_news_snapshot(
    params: Dict[str, Any],
    *,
    force_refetch: bool = False,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Ensure a news snapshot exists for the given parameters.

    Returns (snapshot, refreshed). Snapshot may be None when hf_mode is disabled
    or tickers are missing. refreshed indicates whether a new fetch occurred.
    """
    if not params.get("hf_mode", True):
        clear_news_snapshot()
        return None, False

    if not params["ticker_a"] or not params["ticker_b"]:
        clear_news_snapshot()
        return None, False

    cached = _cached_snapshot(params)
    if cached and not force_refetch:
        return cached, False

    previous = st.session_state.get(NEWS_SNAPSHOT_KEY)
    snapshot = _fetch_snapshot(params)

    prev_digest = previous.get("digest") if isinstance(previous, dict) else None
    snapshot["digest_changed"] = bool(prev_digest and prev_digest != snapshot["digest"])

    st.session_state[NEWS_SNAPSHOT_KEY] = snapshot
    st.session_state["news_digest_hash"] = snapshot["digest"]
    return snapshot, True


def get_news_snapshot(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return cached snapshot for params without triggering a fetch."""
    return _cached_snapshot(params)
