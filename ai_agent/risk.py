from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import pandas as pd

from .options import OptionsSnapshot


__all__ = [
    "RiskProfile",
    "profile_params",
    "equity_position_size",
    "options_spread_size",
    "sizing_summary_table",
]


# ---------------- Configuration ----------------

@dataclass(frozen=True)
class RiskProfile:
    name: str
    # % of account equity you are willing to lose if the stop is hit
    risk_per_trade_pct: float
    # Multiplier on baseline stop distance derived from vol/implied move
    stop_mult: float


def profile_params(name: str) -> RiskProfile:
    """
    Return parameters for a named risk profile.
    Profiles are intentionally punchy; tune to your taste.
    """
    key = (name or "").strip().lower()
    if key in ("conservative", "conserv", "c"):
        # ~0.5% risk per trade, tighter stops
        return RiskProfile("Conservative", risk_per_trade_pct=0.005, stop_mult=0.8)
    if key in ("aggressive", "agg", "a"):
        # ~2% risk per trade, wider stops
        return RiskProfile("Aggressive", risk_per_trade_pct=0.02, stop_mult=1.25)
    # default
    return RiskProfile("Balanced", risk_per_trade_pct=0.01, stop_mult=1.0)


# ---------------- Core math ----------------

def _daily_vol_from_ann(vol20_ann_pct: Optional[float]) -> Optional[float]:
    """
    Convert 20-day annualized vol (%) to rough daily vol (%).
    """
    if vol20_ann_pct is None or math.isnan(vol20_ann_pct):
        return None
    return float(vol20_ann_pct) / math.sqrt(252.0)


def _baseline_stop_pct(
    *,
    implied_move_pct: Optional[float],
    vol20_ann_pct: Optional[float],
    dte: Optional[int],
) -> Optional[float]:
    """
    Construct a baseline stop distance (%) from:
      - near-term implied move to expiry (scaled to a 1–2 week horizon), and
      - recent realized daily vol (20d ann).
    Use a blended, conservative estimate.
    """
    dv = _daily_vol_from_ann(vol20_ann_pct)  # %
    # scale implied move down to a 10-trading-day horizon
    im = None
    if implied_move_pct is not None and implied_move_pct > 0:
        if dte and dte > 0:
            # scale like sqrt(time): move_10d ≈ implied_move * sqrt(10 / DTE)
            im = float(implied_move_pct) * math.sqrt(max(1.0, 10.0) / float(dte))
        else:
            # if no DTE, use half of the stated implied move
            im = float(implied_move_pct) * 0.5

    # if neither, bail
    if dv is None and im is None:
        return None

    # conservative blend: take min of (2*daily_vol, 0.75*scaled implied)
    candidates = []
    if dv is not None:
        candidates.append(2.0 * dv)
    if im is not None:
        candidates.append(0.75 * im)

    base = min(candidates) if candidates else None
    # guardrails
    if base is None:
        return None
    # clip to a sane range (1% .. 15%)
    return float(max(1.0, min(15.0, base)))


def equity_position_size(
    *,
    account_equity: float,
    spot: Optional[float],
    stop_distance_pct: Optional[float],
    risk_per_trade_pct: float,
    min_lot: int = 1,
) -> Dict[str, Optional[float]]:
    """
    Position sizing for shares (LONG or SHORT) based on a % risk budget and stop distance.

    Returns dict:
      {
        "shares": int or None,
        "entry_value": float or None,
        "risk_$": float or None,
        "risk_%_equity": float,  # equals risk_per_trade_pct
        "stop_distance_%": float or None
      }
    """
    if spot is None or spot <= 0 or stop_distance_pct is None or stop_distance_pct <= 0:
        return {
            "shares": None,
            "entry_value": None,
            "risk_$": None,
            "risk_%_equity": risk_per_trade_pct * 100.0,
            "stop_distance_%": stop_distance_pct,
        }

    risk_dollars = float(account_equity) * float(risk_per_trade_pct)
    per_share_risk = float(spot) * (float(stop_distance_pct) / 100.0)
    if per_share_risk <= 0:
        return {
            "shares": None,
            "entry_value": None,
            "risk_$": None,
            "risk_%_equity": risk_per_trade_pct * 100.0,
            "stop_distance_%": stop_distance_pct,
        }

    shares = int(max(min_lot, math.floor(risk_dollars / per_share_risk)))
    entry_value = shares * float(spot)
    return {
        "shares": shares,
        "entry_value": entry_value,
        "risk_$": risk_dollars,
        "risk_%_equity": risk_per_trade_pct * 100.0,
        "stop_distance_%": stop_distance_pct,
    }


def options_spread_size(
    *,
    account_equity: float,
    risk_per_trade_pct: float,
    straddle_debit: Optional[float],
    spot: Optional[float],
    contracts_round_to: int = 1,
) -> Dict[str, Optional[float]]:
    """
    Very simple contract sizing for a **debit** options idea:
      - assumes max loss ≈ debit per contract * 100
      - uses the same risk budget in dollars

    Returns dict:
      {
        "contracts": int or None,
        "debit_per_contract": float or None,
        "risk_$": float or None,
        "risk_%_equity": float
      }
    """
    if straddle_debit is None or straddle_debit <= 0 or spot is None or spot <= 0:
        return {
            "contracts": None,
            "debit_per_contract": straddle_debit,
            "risk_$": None,
            "risk_%_equity": risk_per_trade_pct * 100.0,
        }

    risk_dollars = float(account_equity) * float(risk_per_trade_pct)
    max_loss_per_contract = float(straddle_debit) * 100.0
    if max_loss_per_contract <= 0:
        return {
            "contracts": None,
            "debit_per_contract": straddle_debit,
            "risk_$": None,
            "risk_%_equity": risk_per_trade_pct * 100.0,
        }

    contracts = int(max(contracts_round_to, math.floor(risk_dollars / max_loss_per_contract)))
    return {
        "contracts": contracts,
        "debit_per_contract": float(straddle_debit),
        "risk_$": risk_dollars,
        "risk_%_equity": risk_per_trade_pct * 100.0,
    }


# ---------------- Convenience: build a small table ----------------

def sizing_summary_table(
    *,
    ticker: str,
    account_equity: float,
    profile_name: str,
    spot: Optional[float],
    vol20_ann_pct: Optional[float],
    opt: Optional[OptionsSnapshot],
) -> pd.DataFrame:
    """
    Produce a small, readable table with:
      - baseline stop distance (from implied move / realized vol blend)
      - equity share size for that stop
      - option contracts sizing using ATM straddle debit as a rough proxy

    This is deliberately opinionated math to keep it simple and fast.
    """
    prof = profile_params(profile_name)
    implied = opt.implied_move_pct if opt else None
    dte = opt.dte if opt else None

    stop_pct_base = _baseline_stop_pct(implied_move_pct=implied, vol20_ann_pct=vol20_ann_pct, dte=dte)
    stop_pct = None if stop_pct_base is None else float(stop_pct_base) * float(prof.stop_mult)

    eq = equity_position_size(
        account_equity=account_equity,
        spot=spot,
        stop_distance_pct=stop_pct,
        risk_per_trade_pct=prof.risk_per_trade_pct,
    )

    opt_size = options_spread_size(
        account_equity=account_equity,
        risk_per_trade_pct=prof.risk_per_trade_pct,
        straddle_debit=(opt.straddle_debit if opt else None),
        spot=spot,
    )

    def fmt(x, pct=False):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        try:
            v = float(x)
        except Exception:
            return "—"
        return f"{v:.2f}%" if pct else (f"{int(v)}" if abs(v - int(v)) < 1e-9 and not pct else f"{v:.2f}")

    rows = [
        ("Profile", prof.name, ""),
        ("Equity ($)", f"{account_equity:,.0f}", ""),
        ("Baseline stop %", fmt(stop_pct, pct=True), ""),
        ("Shares (approx.)", fmt(eq.get("shares")), ""),
        ("Entry value ($)", fmt(eq.get("entry_value")), ""),
        ("Risk $", fmt(eq.get("risk_$")), ""),
        ("Risk % of equity", fmt(eq.get("risk_%_equity"), pct=True), ""),
        ("Options contracts (≈ debit)", fmt(opt_size.get("contracts")), ""),
        ("Debit / contract ($)", fmt((opt.straddle_debit if opt else None))), "",
    ]

    df = pd.DataFrame(rows, columns=[f"{ticker} sizing", "Value", ""])
    return df
