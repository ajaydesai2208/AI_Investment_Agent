from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import math

import pandas as pd


# -----------------------------
# Small helpers
# -----------------------------

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return v


def _parse_earnings_date(cats: Dict[str, Any]) -> Optional[datetime]:
    """
    Try several common keys/structures to find the next earnings date.
    Works with tolerant inputs produced by your catalysts module.
    """
    if not isinstance(cats, dict):
        return None
    # Common shapes this function will tolerate
    candidates = [
        cats.get("next_earnings"),
        cats.get("earnings_date"),
        cats.get("earnings"),
        cats.get("events", {}).get("earnings"),
    ]
    for c in candidates:
        if c is None:
            continue
        # Could be a string, date-like, or list/tuple (start, end)
        if isinstance(c, (list, tuple)) and c:
            c = c[0]
        try:
            dt = pd.to_datetime(c, utc=True)
            if pd.isna(dt):
                continue
            return dt.to_pydatetime()
        except Exception:
            # Last resort: try naive parse
            try:
                return datetime.fromisoformat(str(c))
            except Exception:
                pass
    return None


def _days_from_now(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    delta = dt - now
    return int(round(delta.total_seconds() / 86400.0))


def _regime_key(regime_label: Optional[str]) -> str:
    if not isinstance(regime_label, str):
        return "normal"
    s = regime_label.strip().lower()
    if s.startswith("calm"):
        return "calm"
    if s.startswith("stress"):
        return "stressed"
    return "normal"


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Play:
    title: str
    thesis: str
    setup: str
    entry: str
    exit: str
    metrics: List[str]  # bullet points like BE, RR, POP-ish
    checklist: List[str]

    def to_markdown(self) -> str:
        out = [f"#### {self.title}", "", f"**Thesis.** {self.thesis}", f"**Setup.** {self.setup}", ""]
        out.append("**Entry.** " + self.entry)
        out.append("**Exit.** " + self.exit)
        if self.metrics:
            out.append("")
            out.append("**Key metrics**")
            for m in self.metrics:
                out.append(f"- {m}")
        if self.checklist:
            out.append("")
            out.append("**Checklist before sending**")
            for c in self.checklist:
                out.append(f"- {c}")
        out.append("\n")
        return "\n".join(out)


# -----------------------------
# Core playbook logic
# -----------------------------

def build_event_playbook_for_ticker(
    *,
    ticker: str,
    catalysts: Dict[str, Any],
    opt_snapshot: Any,                 # duck-typed: .dte, .atm_iv_pct, .implied_move_pct, .spot, .atm_strike, .straddle_debit, .expiry
    trend_label: Optional[str],
    regime_label: Optional[str],
    risk_profile: str,
) -> List[Play]:
    """
    Turn catalysts + options context into a shortlist of tactics.
    Rules are heuristic, retail-friendly, and robust to missing fields.
    """
    plays: List[Play] = []

    # Gather inputs with guardrails
    dte = getattr(opt_snapshot, "dte", None)
    iv_pct = _safe_float(getattr(opt_snapshot, "atm_iv_pct", None))  # %
    imp = _safe_float(getattr(opt_snapshot, "implied_move_pct", None))  # %
    spot = _safe_float(getattr(opt_snapshot, "spot", None))
    strike = _safe_float(getattr(opt_snapshot, "atm_strike", None))
    straddle_debit = _safe_float(getattr(opt_snapshot, "straddle_debit", None))
    expiry = getattr(opt_snapshot, "expiry", None)

    trend = (trend_label or "—")
    regime = _regime_key(regime_label)

    # Earnings timing
    earn_dt = _parse_earnings_date(catalysts)
    days_to_earn = _days_from_now(earn_dt)

    # --- 1) Classic earnings tactics ---
    if days_to_earn is not None:
        if -1 <= days_to_earn <= 1:
            # Today / tomorrow
            if regime == "stressed" or (iv_pct and iv_pct >= 40):
                plays.append(
                    Play(
                        title=f"{ticker}: Iron Fly into Earnings (same expiry)",
                        thesis="Rich IV into the print; mean-reversion in IV post-event; profit from pinning near ATM.",
                        setup=f"Sell ATM straddle; buy protective wings (±5–10%); expiry {expiry or 'nearest weekly'}.",
                        entry="Open near EOD before earnings; center at ATM strike.",
                        exit="Target 40–70% max profit on IV crush or close at open post-print; hard stop at 2× credit.",
                        metrics=[
                            f"ATM IV ≈ {iv_pct:.1f}% (est.)" if iv_pct else "ATM IV high",
                            f"Straddle debit (ref) ≈ ${straddle_debit:,.2f}" if straddle_debit else "Straddle debit: n/a",
                            "Max loss capped by wings; theta positive.",
                        ],
                        checklist=[
                            "Confirm earnings date/time and that options expire after the print.",
                            "Avoid illiquid chains (wide spreads).",
                            "Keep width reasonable so margin fits your account.",
                        ],
                    )
                )
            else:
                plays.append(
                    Play(
                        title=f"{ticker}: Long Straddle into Earnings",
                        thesis="Expect a larger-than-implied move; want convexity through the print.",
                        setup=f"Buy ATM call + ATM put at the nearest expiry after the print.",
                        entry="Open near EOD pre-print; size small (debit risk).",
                        exit="Take gains on a large gap; if small move, cut quickly to avoid IV crush.",
                        metrics=[
                            f"Implied move ≈ {imp:.1f}%" if imp else "Implied move: n/a",
                            f"Straddle debit ≈ ${straddle_debit:,.2f}" if straddle_debit else "Straddle debit: n/a",
                            "Unlimited upside/limited downside (debit).",
                        ],
                        checklist=[
                            "Check spreads & liquidity at ATM.",
                            "Avoid chasing IV if it spikes intraday.",
                            "Have a plan if the gap is opposite to your bias.",
                        ],
                    )
                )
        elif 2 <= days_to_earn <= 14:
            # A few days out → calendars
            plays.append(
                Play(
                    title=f"{ticker}: ATM Call Calendar into Earnings",
                    thesis="Own post-earnings IV (back month) while selling rich near-term IV; theta-balanced.",
                    setup="Buy longer-dated ATM call; sell nearer expiry ATM call (expires before earnings).",
                    entry="Open when earnings are ~1–2 weeks away; ensure short leg expires pre-print.",
                    exit="Close before earnings or roll the short; profit from near-term decay and IV uptick in back month.",
                    metrics=[
                        "Defined risk (net debit).",
                        "Profits if price pins near strike into short expiry.",
                        "Vega positive on back-month.",
                    ],
                    checklist=[
                        "Verify the short leg expires **before** the print.",
                        "Choose back-month with adequate liquidity.",
                        "Avoid very low IV regimes (thin calendars).",
                    ],
                )
            )

    # --- 2) Non-earnings catalysts (SEC, product events) ---
    # If we see recent 8-K / major filing, favor defined-risk directional
    recent_sec = str(catalysts).lower()  # tolerant text check
    if any(k in recent_sec for k in ["8-k", "merger", "guidance", "investor day", "spin-off", "bankruptcy"]):
        if trend.lower().startswith("up"):
            plays.append(
                Play(
                    title=f"{ticker}: Bull Call Debit Spread on news momentum",
                    thesis="Chase confirmed positive momentum with defined risk.",
                    setup="Buy near-ATM call, sell OTM call (width ~5%).",
                    entry="Open on confirmation day; avoid chasing huge gaps.",
                    exit="Take profit at 1.5–2.5R or before IV cools; hard stop at 1R.",
                    metrics=["Defined risk; convex upside; quick time decay if wrong."],
                    checklist=[
                        "Avoid overlapping earnings window unless intended.",
                        "Ensure spreads are reasonably tight.",
                        "Pick width that matches expected follow-through.",
                    ],
                )
            )
        elif trend.lower().startswith("down"):
            plays.append(
                Play(
                    title=f"{ticker}: Bear Put Debit Spread on negative news",
                    thesis="Express downside with limited risk and better carry than naked puts.",
                    setup="Buy near-ATM put, sell further OTM put (width ~5%).",
                    entry="Open on confirmation day; avoid chasing if already extended.",
                    exit="Take profit at 1.5–2.5R; hard stop at 1R.",
                    metrics=["Defined risk; benefits from follow-through; limited theta bleed vs long put."],
                    checklist=[
                        "Watch for reflex rallies; size conservatively.",
                        "Avoid thinly traded chains.",
                    ],
                )
            )

    # --- 3) Regime overlays (IV-aware) ---
    if regime == "stressed":
        plays.append(
            Play(
                title=f"{ticker}: Credit Spread (IV-rich)",
                thesis="Harvest elevated IV with capped risk; align with direction.",
                setup="If bullish → bull put spread OTM; if bearish → bear call spread OTM.",
                entry="Open when IV is clearly above usual and spreads are acceptable.",
                exit="50–70% of max profit or 2× credit stop.",
                metrics=["Positive theta; capped risk; POP >50% when sold sufficiently OTM."],
                checklist=[
                    "Use smaller size in jumpy tape.",
                    "Avoid selling right before earnings unless you intend to hold through.",
                ],
            )
        )
    elif regime == "calm":
        plays.append(
            Play(
                title=f"{ticker}: Debit Vertical (low IV)",
                thesis="Buy convexity when IV is cheaper; defined risk directional bet.",
                setup="Bull call debit (uptrend) or bear put debit (downtrend).",
                entry="Open on pullbacks in trend; avoid chasing.",
                exit="Scale at 1.5R, trail remainder; cut at 1R.",
                metrics=["Limited risk; better bang-for-buck when IV is low."],
                checklist=["Mind earnings date to avoid surprise IV crush."],
            )
        )

    return plays


def playbook_to_markdown(ticker: str, plays: List[Play]) -> str:
    if not plays:
        return f"### {ticker}: Event Playbook\n\n_No specific event-driven tactics at the moment._\n"
    out = [f"### {ticker}: Event Playbook", ""]
    for p in plays:
        out.append(p.to_markdown())
        out.append("---")
    return "\n".join(out)
