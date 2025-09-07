from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Optional, Any, Tuple, Dict, List, Literal

import numpy as np
import pandas as pd
import math

# ATR helper
from ai_agent.quant import atr_value_from_ticker


# ----------------------------
# Risk policy & small helpers
# ----------------------------

@dataclass
class RiskProfile:
    name: str
    risk_per_trade_frac: float  # e.g. 0.01 = 1%
    # default fixed stop (if we have zero signal), used as a fallback
    default_stop_pct: float


PROFILE_MAP: Dict[str, RiskProfile] = {
    "Conservative": RiskProfile("Conservative", risk_per_trade_frac=0.005, default_stop_pct=5.0),
    "Balanced":     RiskProfile("Balanced",     risk_per_trade_frac=0.010, default_stop_pct=8.0),
    "Aggressive":   RiskProfile("Aggressive",   risk_per_trade_frac=0.020, default_stop_pct=12.0),
}


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{float(x):.2f}%"


def _fmt_money(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"${float(x):,.2f}"


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _get_attr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


# -------------------------------------------------
# Core sizing: baseline + ATR-based stop distances
# -------------------------------------------------

def _compute_baseline_stop_pct(
    implied_move_pct: Optional[float],
    profile: RiskProfile,
) -> float:
    """
    Baseline stop as a % of entry.
    If we have an implied move, use ~75% of it (clamped). Otherwise profile default.
    """
    if implied_move_pct is None or np.isnan(implied_move_pct):
        return profile.default_stop_pct
    # e.g., if implied move is 10%, baseline stop ≈ 7.5%, but clamp to sane bounds.
    return _clamp(0.75 * float(implied_move_pct), 3.0, 20.0)


def _compute_atr_stop_pct(
    ticker: str,
    spot: Optional[float],
    profile: RiskProfile,
) -> Optional[float]:
    """
    ATR-based stop as a % of entry.
    - Conservative: 2.0 × ATR
    - Balanced:     1.5 × ATR
    - Aggressive:   1.0 × ATR
    stop% = (ATR_mult * ATR$) / spot * 100
    """
    if spot is None or spot <= 0:
        return None
    atr_mult = {"Conservative": 2.0, "Balanced": 1.5, "Aggressive": 1.0}.get(profile.name, 1.5)
    atr_val = atr_value_from_ticker(ticker, window=14, lookback_days=180, method="wilder")
    if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
        return None
    return (atr_mult * float(atr_val) / float(spot)) * 100.0


def _position_size_shares(
    account_equity: float,
    stop_pct: Optional[float],
    spot: Optional[float],
    risk_per_trade_frac: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (shares, dollar_risk_at_stop).
    We risk `risk_per_trade_frac * account_equity`. Stop distance is stop_pct of entry.
    shares = floor( risk$ / (stop_pct * entry) )
    """
    if (
        account_equity is None
        or account_equity <= 0
        or stop_pct is None
        or spot is None
        or stop_pct <= 0
        or spot <= 0
    ):
        return None, None

    risk_dollars = float(account_equity) * float(risk_per_trade_frac)
    stop_dollars_per_share = float(spot) * (float(stop_pct) / 100.0)
    if stop_dollars_per_share <= 0:
        return None, None

    shares = int(max(1, floor(risk_dollars / stop_dollars_per_share)))
    dollar_risk = shares * stop_dollars_per_share
    return shares, dollar_risk


def _contracts_from_debit(
    account_equity: float,
    risk_per_trade_frac: float,
    est_debit_per_share: Optional[float],
) -> Optional[int]:
    """
    Approximate number of option contracts using a per-share debit (×100 multiplier).
    """
    if est_debit_per_share is None or est_debit_per_share <= 0:
        return None
    risk_dollars = float(account_equity) * float(risk_per_trade_frac)
    c = int(max(1, floor(risk_dollars / (float(est_debit_per_share) * 100.0))))
    return c


# -------------------------------------------------
# Public API used by the app
# -------------------------------------------------

def sizing_summary_table(
    *,
    ticker: str,
    account_equity: float,
    profile_name: str,
    spot: Optional[float],
    vol20_ann_pct: Optional[float],
    opt: Any,
) -> pd.DataFrame:
    """
    Returns a 2-column DataFrame ["Metric", "Value"] summarizing
    baseline and ATR-based sizing suggestions.

    This is defensive: it works if some inputs are missing.
    """
    # Resolve profile
    profile: RiskProfile = PROFILE_MAP.get(profile_name, PROFILE_MAP["Balanced"])
    risk_frac = profile.risk_per_trade_frac

    # Pull option context (defensively)
    implied_move_pct = _get_attr(opt, "implied_move_pct", None)
    straddle_debit = _get_attr(opt, "straddle_debit", None)
    call_mid = _get_attr(opt, "call_mid", None)

    # Prefer a realistic single-leg debit for rough option sizing; fall back to straddle.
    est_debit = None
    if call_mid is not None and call_mid > 0:
        est_debit = float(call_mid)
    elif straddle_debit is not None and straddle_debit > 0:
        # rough: assume half the straddle cost for a directional single-leg reference
        est_debit = float(straddle_debit) * 0.5

    # -------------------------
    # Baseline (implied-move)
    # -------------------------
    base_stop_pct = _compute_baseline_stop_pct(implied_move_pct, profile)
    base_shares, base_dollar_risk = _position_size_shares(
        account_equity=account_equity,
        stop_pct=base_stop_pct,
        spot=spot,
        risk_per_trade_frac=risk_frac,
    )
    base_contracts = _contracts_from_debit(
        account_equity=account_equity,
        risk_per_trade_frac=risk_frac,
        est_debit_per_share=est_debit,
    )

    # -------------------------
    # ATR-based
    # -------------------------
    atr_stop_pct = _compute_atr_stop_pct(ticker=ticker, spot=spot, profile=profile)
    atr_shares, atr_dollar_risk = _position_size_shares(
        account_equity=account_equity,
        stop_pct=atr_stop_pct,
        spot=spot,
        risk_per_trade_frac=risk_frac,
    )

    # -------------------------
    # Assemble rows
    # -------------------------
    rows: List[Tuple[str, str]] = []

    rows.append(("Account equity ($)", _fmt_money(account_equity)))
    rows.append(("Risk per trade (%)", _fmt_pct(risk_frac * 100.0)))
    if spot is not None:
        rows.append(("Spot (approx.)", _fmt_money(spot)))
    if vol20_ann_pct is not None:
        rows.append(("Realized vol (20d, ann.)", _fmt_pct(vol20_ann_pct)))

    # Baseline rows (labels used elsewhere in the app)
    rows.append(("Baseline stop %", _fmt_pct(base_stop_pct)))
    if base_shares is not None:
        rows.append(("Shares (approx.)", f"{base_shares:,d}"))
    if base_dollar_risk is not None:
        rows.append(("Dollar risk at stop", _fmt_money(base_dollar_risk)))
    if base_contracts is not None:
        rows.append(("Options contracts (≈ debit)", f"{base_contracts:,d}"))

    # ATR rows
    rows.append(("ATR stop %", _fmt_pct(atr_stop_pct)))
    if atr_shares is not None:
        rows.append(("ATR shares (approx.)", f"{atr_shares:,d}"))
    if atr_dollar_risk is not None:
        rows.append(("ATR dollar risk at stop", _fmt_money(atr_dollar_risk)))

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return df

# === Trade Plan (entry/stop/targets, R:R) ==========================
from dataclasses import dataclass
from typing import Optional, Literal, Dict
import math

@dataclass
class TradePlan:
    ticker: str
    direction: Literal["long", "short"]
    entry: float                  # assumed entry (uses spot)
    stop: float                   # hard stop
    risk_per_share: float         # |entry - stop|
    target1: float                # ~1.5R
    target2: float                # ~3R
    risk_budget_usd: float        # $ risk per trade from profile
    suggested_shares: int         # floor(risk_budget / risk_per_share)
    rr_at_t1: float               # target1 R multiple
    rr_at_t2: float               # target2 R multiple
    method: str                   # how we computed the stop (ATR / implied move)
    notes: Optional[str] = None

    def to_dataframe(self):
        import pandas as pd
        rows = [
            ("Direction", self.direction.upper()),
            ("Entry", f"${self.entry:,.2f}"),
            ("Stop", f"${self.stop:,.2f}"),
            ("Risk/share", f"${self.risk_per_share:,.2f}"),
            ("Target 1 (~1.5R)", f"${self.target1:,.2f}"),
            ("Target 2 (~3R)", f"${self.target2:,.2f}"),
            ("Risk budget", f"${self.risk_budget_usd:,.2f}"),
            ("Suggested size (shares)", f"{self.suggested_shares:,}"),
            ("R at T1", f"{self.rr_at_t1:.2f}R"),
            ("R at T2", f"{self.rr_at_t2:.2f}R"),
            ("Stop method", self.method),
        ]
        if self.notes:
            rows.append(("Notes", self.notes))
        return pd.DataFrame(rows, columns=["Metric", "Value"])

def _profile_risk_budget(profile: str, account_equity: float) -> float:
    """Risk per trade as a fraction of equity."""
    p = (profile or "Balanced").lower()
    if p.startswith("cons"):
        frac = 0.005    # 0.5%
    elif p.startswith("aggr"):
        frac = 0.02     # 2.0%
    else:
        frac = 0.01     # 1.0%
    return max(0.0, float(account_equity)) * frac

def _approx_atr_pct_from_ann_vol(vol20_ann_pct: Optional[float]) -> Optional[float]:
    """
    If ATR% not available, approximate from annualized vol:
      daily sigma ≈ ann_vol / sqrt(252)
      ATR% ≈ daily sigma * 1.4 (rule-of-thumb)
    """
    try:
        ann = float(vol20_ann_pct)
        if not math.isfinite(ann) or ann <= 0:
            return None
        daily = ann / math.sqrt(252.0)
        return daily * 1.4
    except Exception:
        return None

def build_trade_plan(
    *,
    ticker: str,
    direction: str,
    spot: Optional[float],
    account_equity: float,
    risk_profile: str,
    vol20_ann_pct: Optional[float] = None,
    implied_move_pct: Optional[float] = None,
    atr_pct_hint: Optional[float] = None,
) -> TradePlan:
    """
    Compute a simple, consistent plan:
      - Entry = spot
      - Stop% = max( 1.2×ATR%, 0.5×implied_move% ), fallback to 3%
      - Targets = entry ± {1.5R, 3R}
      - Risk budget = profile % of equity → suggested size
    All '%' values are in percent units (e.g., 20.0 = 20%).
    """
    if spot is None:
        raise ValueError("Spot price required for trade plan.")

    side_long = str(direction).lower().startswith("long")

    # % stops
    atr_pct = atr_pct_hint
    if atr_pct is None:
        atr_pct = _approx_atr_pct_from_ann_vol(vol20_ann_pct)  # may still be None

    imp = None
    try:
        imp = float(implied_move_pct) if implied_move_pct is not None else None
    except Exception:
        pass

    # base stop percent
    candidates = []
    if atr_pct is not None and atr_pct > 0:
        candidates.append(1.2 * atr_pct)
    if imp is not None and imp > 0:
        candidates.append(0.5 * imp)
    stop_pct = max(candidates) if candidates else 3.0  # % fallback

    # clamp sensible range
    stop_pct = min(max(stop_pct, 1.0), 12.0)

    # prices
    if side_long:
        stop = float(spot) * (1.0 - stop_pct / 100.0)
        risk_per_share = float(spot) - stop
        t1 = float(spot) + 1.5 * risk_per_share
        t2 = float(spot) + 3.0 * risk_per_share
    else:
        stop = float(spot) * (1.0 + stop_pct / 100.0)
        risk_per_share = stop - float(spot)
        t1 = float(spot) - 1.5 * risk_per_share
        t2 = float(spot) - 3.0 * risk_per_share

    risk_budget = _profile_risk_budget(risk_profile, float(account_equity))
    shares = int(max(0.0, math.floor(risk_budget / max(risk_per_share, 1e-6))))

    rr_t1 = 1.5
    rr_t2 = 3.0

    method = "max(1.2×ATR%, 0.5×implied move%)"
    notes = None
    if atr_pct is None and imp is None:
        notes = "Fallback stop% used (3%)."

    return TradePlan(
        ticker=ticker,
        direction="long" if side_long else "short",
        entry=float(spot),
        stop=round(stop, 2),
        risk_per_share=round(risk_per_share, 2),
        target1=round(t1, 2),
        target2=round(t2, 2),
        risk_budget_usd=round(risk_budget, 2),
        suggested_shares=shares,
        rr_at_t1=rr_t1,
        rr_at_t2=rr_t2,
        method=method,
        notes=notes,
    )
