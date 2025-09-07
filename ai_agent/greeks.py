from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import pandas as pd


# -----------------------------
# Math utils
# -----------------------------

def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _years_from_dte(dte: Optional[int]) -> float:
    try:
        d = int(dte) if dte is not None else 30
    except Exception:
        d = 30
    return max(0.0, d) / 365.0


# -----------------------------
# Black–Scholes pricing & greeks
# -----------------------------

def _d1_d2(S: float, K: float, T: float, sigma: float, r: float) -> Tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        # Avoid math domain issues; return large/small values that collapse greeks
        return (float("inf"), float("inf"))
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S: float, K: float, T: float, sigma: float, r: float, kind: Literal["CALL", "PUT"]) -> float:
    if T <= 0 or sigma <= 0:
        # intrinsic at expiry / zero vol
        return max(0.0, S - K) if kind == "CALL" else max(0.0, K - S)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    if kind == "CALL":
        return S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    else:
        return K * math.exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


def bs_greeks(S: float, K: float, T: float, sigma: float, r: float, kind: Literal["CALL", "PUT"]):
    """
    Returns a dict with greeks:
      delta (per share), gamma (per share per $), theta_per_day (per share per calendar day),
      vega_per_1pct (per share, per +1 vol point), rho (per share per 1% rate).
    Notes:
      - theta returned per *day* (calendar).
      - vega returned per *1 percentage point* change in IV.
    """
    out = {"delta": 0.0, "gamma": 0.0, "theta_per_day": 0.0, "vega_per_1pct": 0.0, "rho": 0.0}
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return out

    d1, d2 = _d1_d2(S, K, T, sigma, r)
    nd1 = _phi(d1)
    sqrtT = math.sqrt(T)

    # Delta
    if kind == "CALL":
        delta = _cdf(d1)
    else:
        delta = _cdf(d1) - 1.0  # = -_cdf(-d1)

    # Gamma
    gamma = nd1 / (S * sigma * sqrtT)

    # Theta (per year) then per day
    if kind == "CALL":
        theta = -(S * nd1 * sigma) / (2 * sqrtT) - r * K * math.exp(-r * T) * _cdf(d2)
    else:
        theta = -(S * nd1 * sigma) / (2 * sqrtT) + r * K * math.exp(-r * T) * _cdf(-d2)
    theta_per_day = theta / 365.0

    # Vega per 1% IV change
    vega = S * nd1 * sqrtT  # per 1.0 of vol (i.e., 100 vol points)
    vega_per_1pct = vega * 0.01  # per 1 vol point

    # Rho (per 1.0 of rate -> 100%); convert to per 1% rate
    if kind == "CALL":
        rho = K * T * math.exp(-r * T) * _cdf(d2)
    else:
        rho = -K * T * math.exp(-r * T) * _cdf(-d2)
    rho_per_1pct = rho * 0.01

    out.update(
        delta=delta,
        gamma=gamma,
        theta_per_day=theta_per_day,
        vega_per_1pct=vega_per_1pct,
        rho=rho_per_1pct,
    )
    return out


# -----------------------------
# ATM Greeks summary for UI
# -----------------------------

def atm_greeks_table(
    *,
    spot: Optional[float],
    atm_strike: Optional[float],
    dte: Optional[int],
    atm_iv_pct: Optional[float],
    call_mid: Optional[float],
    put_mid: Optional[float],
    direction: Literal["long", "short"] = "long",
    risk_free_rate: float = 0.04,
    per_contract: bool = True,
) -> pd.DataFrame:
    """
    Build a small 2-col table with ATM greeks and useful derived metrics.
    If quotes are missing, falls back to theoretical premium as 'entry'.
    Values are per contract if `per_contract=True` (×100).
    """
    S = _safe_float(spot)
    K = _safe_float(atm_strike) or (S if S is not None else None)
    T = _years_from_dte(dte)
    sigma = (_safe_float(atm_iv_pct) or 35.0) / 100.0
    if S is None or K is None or T <= 0 or sigma <= 0:
        return pd.DataFrame({"Metric": [], "Value": []})

    long_view = str(direction).lower().startswith("long")
    kind: Literal["CALL", "PUT"] = "CALL" if long_view else "PUT"

    # Entry premium: prefer market mid for correct kind; fallback to theoretical
    entry_mid = (call_mid if kind == "CALL" else put_mid)
    entry_mid_f = _safe_float(entry_mid)
    theo_price = bs_price(S, K, T, sigma, risk_free_rate, kind)
    premium = entry_mid_f if entry_mid_f is not None else theo_price

    greeks = bs_greeks(S, K, T, sigma, risk_free_rate, kind)

    mult = 100.0 if per_contract else 1.0
    rows = [
        ("Type", kind),
        ("Spot", round(S, 2)),
        ("Strike (ATM)", round(K, 2)),
        ("DTE", int(round(T * 365))),
        ("IV (ATM)", f"{(sigma * 100):.2f}%"),
        ("Premium mid", round(premium * mult, 2)),
        ("Delta", round(greeks["delta"], 4)),
        ("Gamma", round(greeks["gamma"], 6)),
        ("Theta/day", round(greeks["theta_per_day"] * mult, 2)),
        ("Vega/1%", round(greeks["vega_per_1pct"] * mult, 2)),
        ("Rho/1%", round(greeks["rho"] * mult, 2)),
    ]

    # Breakeven (approx) & POP (rough, risk-neutral)
    if kind == "CALL":
        breakeven = K + premium
        # POP: P(S_T > K + prem) is too specific; show N(d2) as classic RN prob ITM
        # Using classic approx: RN P(ITM) ≈ N(d2) for calls
        d1, d2 = _d1_d2(S, K, T, sigma, risk_free_rate)
        pop = _cdf(d2)
    else:
        breakeven = K - premium
        # RN P(ITM) ≈ N(-d2) for puts
        d1, d2 = _d1_d2(S, K, T, sigma, risk_free_rate)
        pop = _cdf(-d2)

    rows.extend(
        [
            ("Breakeven (approx.)", round(breakeven, 2)),
            ("POP (RN, ITM prob)", f"{pop * 100:.1f}%"),
        ]
    )

    return pd.DataFrame(rows, columns=["Metric", "Value"])
