from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal

import math
import numpy as np
import pandas as pd


# -----------------------------
# Black–Scholes utilities
# -----------------------------

def _norm_cdf(x: float) -> float:
    # Standard normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    r: float,
    option_type: Literal["CALL", "PUT"],
) -> float:
    """
    Black–Scholes theoretical price (no dividends). Robust for edge cases.
    """
    S = float(spot)
    K = float(strike)
    T = max(0.0, float(t_years))
    sigma = max(1e-8, float(vol))
    rr = float(r)

    if T <= 0 or sigma <= 0:
        # expiry or zero-vol edge: intrinsic
        if option_type == "CALL":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    sqrtT = math.sqrt(T)
    d1 = (math.log(max(S, 1e-12) / max(K, 1e-12)) + (rr + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "CALL":
        return S * _norm_cdf(d1) - K * math.exp(-rr * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-rr * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class OptionContext:
    expiry: Optional[str]
    dte: Optional[int]
    atm_strike: Optional[float]
    atm_iv_pct: Optional[float]  # in %
    call_mid: Optional[float]
    put_mid: Optional[float]


@dataclass
class ScenarioRow:
    scenario: str           # e.g., "-2σ", "-1σ", "base", "+1σ", "+2σ"
    spot: float
    stock_pl_per_share: float
    option_pl_per_contract: Optional[float]  # 100x multiplier; None if insufficient inputs


# -----------------------------
# Small helpers
# -----------------------------

def _safe(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _round2(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(round(x, 2))
    except Exception:
        return None


def _default_strike(spot: Optional[float]) -> Optional[float]:
    if spot is None:
        return None
    # nearest 5 or 1 increment (simple, broker-like rounding)
    return float(round(spot / 5.0) * 5.0) if spot > 50 else float(round(spot))


def _years_from_dte(dte: Optional[int]) -> float:
    try:
        d = int(dte) if dte is not None else 30
    except Exception:
        d = 30
    return max(0.0, d) / 365.0


def _vol_decimal(iv_pct: Optional[float]) -> Optional[float]:
    v = _safe(iv_pct)
    if v is None:
        return None
    return max(0.0001, v / 100.0)


def _option_entry_mid(side_long: bool, call_mid: Optional[float], put_mid: Optional[float]) -> Optional[float]:
    # For a long directional view: prefer CALL mid; for short view we still use a long PUT premium.
    if side_long:
        return _safe(call_mid) or _safe(put_mid)
    else:
        return _safe(put_mid) or _safe(call_mid)


# -----------------------------
# Scenario table (5-row ladder)
# -----------------------------

def build_scenarios(
    *,
    ticker: str,
    direction: str,                     # "long" or "short"
    spot: Optional[float],
    implied_move_pct: Optional[float],  # % over the chosen DTE horizon
    opt: OptionContext,
    risk_free_rate: float = 0.04,       # simple default
) -> pd.DataFrame:
    """
    Construct a small table of 5 scenarios for STOCK and a simple ATM OPTION:
    rows: -2σ, -1σ, base, +1σ, +2σ
    - σ is proxied by the implied_move_pct over the option's DTE horizon.
    - Option priced by Black–Scholes using ATM IV and DTE. If mid quotes exist, we
      use them as the entry; otherwise use BS price at base as entry proxy.
    Output columns:
      Scenario | Spot' | Stock P/L per share | Option P/L per contract
    """
    side_long = str(direction).lower().startswith("long")
    S0 = _safe(spot)
    if S0 is None:
        return pd.DataFrame(columns=["Scenario", "Spot'", "Stock P/L/share", "Option P/L/contract"])

    # Sigma proxy: implied move over DTE as a 1σ step (transparent, retail-friendly)
    sigma1_pct = _safe(implied_move_pct)
    if sigma1_pct is None or sigma1_pct <= 0:
        sigma1_pct = 5.0  # fallback 5%

    # Scenario multipliers
    steps = [(-2.0, "-2σ"), (-1.0, "-1σ"), (0.0, "base"), (1.0, "+1σ"), (2.0, "+2σ")]

    # Option context
    dte = getattr(opt, "dte", None)
    t_years = _years_from_dte(dte)
    K = _safe(getattr(opt, "atm_strike", None)) or _default_strike(S0)
    iv_dec = _vol_decimal(getattr(opt, "atm_iv_pct", None))
    if iv_dec is None:
        iv_dec = 0.35  # conservative fallback IV

    entry_call_mid = _safe(getattr(opt, "call_mid", None))
    entry_put_mid = _safe(getattr(opt, "put_mid", None))
    entry_opt_mid = _option_entry_mid(side_long, entry_call_mid, entry_put_mid)

    # If no market mid, use theoretical ATM as entry proxy
    if entry_opt_mid is None and K is not None:
        entry_opt_mid = _bs_price(S0, K, t_years, iv_dec, risk_free_rate, "CALL" if side_long else "PUT")

    rows: List[ScenarioRow] = []

    for mul, label in steps:
        move_pct = mul * sigma1_pct / 100.0
        S1 = float(S0 * (1.0 + move_pct))

        # Stock P/L per share given direction
        stock_pl = (S1 - S0) if side_long else (S0 - S1)

        # Option P/L per contract (100 multiplier), if we have enough context
        opt_pl = None
        if K is not None and entry_opt_mid is not None and iv_dec is not None:
            theo = _bs_price(S1, K, t_years, iv_dec, risk_free_rate, "CALL" if side_long else "PUT")
            opt_pl = (theo - float(entry_opt_mid)) * 100.0  # 1 contract = 100 shares

        rows.append(
            ScenarioRow(
                scenario=label,
                spot=round(S1, 2),
                stock_pl_per_share=round(stock_pl, 2),
                option_pl_per_contract=(round(opt_pl, 2) if opt_pl is not None else None),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.columns = ["Scenario", "Spot'", "Stock P/L/share", "Option P/L/contract"]
    return df


# -----------------------------
# NEW: Dense payoff grid for charts
# -----------------------------

def build_payoff_grid(
    *,
    ticker: str,
    direction: str,                      # "long" or "short"
    spot: Optional[float],
    implied_move_pct: Optional[float],   # % over DTE
    opt: OptionContext,
    risk_free_rate: float = 0.04,
    sigma_span: float = 3.0,             # span in σ on each side (±)
    points: int = 101,                   # grid resolution
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Return a dense payoff grid for STOCK (per share) and ATM OPTION (per contract),
    plus an approximate breakeven for the option payoff (strike ± entry premium).

    Output df columns:
      Spot, Stock P/L/share, Option P/L/contract

    Breakeven:
      CALL  -> K + premium
      PUT   -> K - premium
    If inputs are missing, breakeven is None but the grid will still compute
    (using theoretical mid if necessary).
    """
    side_long = str(direction).lower().startswith("long")
    S0 = _safe(spot)
    if S0 is None:
        return pd.DataFrame(columns=["Spot", "Stock P/L/share", "Option P/L/contract"]), None

    # 1σ proxy (implied move)
    sigma1_pct = _safe(implied_move_pct)
    if sigma1_pct is None or sigma1_pct <= 0:
        sigma1_pct = 5.0

    # Price range
    down = sigma_span * sigma1_pct / 100.0
    up = sigma_span * sigma1_pct / 100.0
    S_min = max(0.01, S0 * (1.0 - down))
    S_max = S0 * (1.0 + up)
    spots = np.linspace(S_min, S_max, int(points))

    # Option context
    dte = getattr(opt, "dte", None)
    t_years = _years_from_dte(dte)
    K = _safe(getattr(opt, "atm_strike", None)) or _default_strike(S0)
    iv_dec = _vol_decimal(getattr(opt, "atm_iv_pct", None)) or 0.35

    entry_call_mid = _safe(getattr(opt, "call_mid", None))
    entry_put_mid = _safe(getattr(opt, "put_mid", None))
    entry_opt_mid = _option_entry_mid(side_long, entry_call_mid, entry_put_mid)
    opt_type: Literal["CALL", "PUT"] = "CALL" if side_long else "PUT"

    # If no market mid, use theoretical ATM at base as entry proxy
    if entry_opt_mid is None and (K is not None):
        entry_opt_mid = _bs_price(S0, K, t_years, iv_dec, risk_free_rate, opt_type)

    # Build payoff arrays
    stock_pl = (spots - S0) if side_long else (S0 - spots)

    if K is not None and entry_opt_mid is not None:
        theo_prices = np.array([_bs_price(float(s), K, t_years, iv_dec, risk_free_rate, opt_type) for s in spots])
        option_pl = (theo_prices - float(entry_opt_mid)) * 100.0  # per contract
        # Breakeven approximation using premium:
        if opt_type == "CALL":
            breakeven = K + float(entry_opt_mid)
        else:
            breakeven = K - float(entry_opt_mid)
    else:
        option_pl = np.full_like(spots, np.nan, dtype=float)
        breakeven = None

    df = pd.DataFrame(
        {
            "Spot": np.round(spots, 2),
            "Stock P/L/share": np.round(stock_pl, 2),
            "Option P/L/contract": np.round(option_pl, 2),
        }
    )
    return df, (round(breakeven, 2) if breakeven is not None else None)
