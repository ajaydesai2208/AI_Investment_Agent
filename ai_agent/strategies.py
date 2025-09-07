from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Dict, Tuple
import math

import numpy as np
import pandas as pd

# We reuse BS pricing from the greeks module for consistency
try:
    from ai_agent.greeks import bs_price  # type: ignore
except Exception:
    # Minimal local fallback (should not happen if greeks.py exists)
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(S: float, K: float, T: float, sigma: float, r: float, kind: Literal["CALL", "PUT"]) -> float:
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K) if kind == "CALL" else max(0.0, K - S)
        sqrtT = math.sqrt(T)
        d1 = (math.log(max(S, 1e-12) / max(K, 1e-12)) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        if kind == "CALL":
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class OptionSnapshotLike:
    """Lightweight shape we need from OptionsSnapshot."""
    expiry: Optional[str]
    dte: Optional[int]
    spot: Optional[float]
    atm_strike: Optional[float]
    atm_iv_pct: Optional[float]
    call_mid: Optional[float]
    put_mid: Optional[float]


@dataclass
class StrategyLeg:
    side: Literal["LONG", "SHORT"]
    kind: Literal["CALL", "PUT", "STOCK"]
    strike: Optional[float]  # None for stock
    qty: float               # contracts for options, shares for stock (positive)
    est_price: Optional[float]  # per-share or per-contract premium (positive)
    note: Optional[str] = None


@dataclass
class StrategyPlan:
    name: str
    ticker: str
    direction: Literal["long", "short"]        # underlying directional thesis
    rationale: str
    when_to_use: str
    legs: List[StrategyLeg]
    dte: Optional[int]
    debit_credit: Literal["DEBIT", "CREDIT", "EVEN"]
    est_cost: float           # $ outlay (positive) or 0; for credits, this is 0 (we show margin separately)
    est_credit: float         # $ received (positive) or 0
    max_loss: Optional[float] # $
    max_gain: Optional[float] # $
    breakevens: List[float]   # underlying price BE points if estimable
    rr_ratio: Optional[float] # reward:risk using max_gain/max_loss when defined
    capital_req: Optional[float]  # crude requirement (e.g., stock cost or CSP collateral)
    notes: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for leg in self.legs:
            rows.append({
                "Side": leg.side,
                "Type": leg.kind,
                "Strike": ("" if leg.strike is None else round(leg.strike, 2)),
                "Qty": leg.qty,
                "Est Price": ("" if leg.est_price is None else round(leg.est_price, 2)),
                "Note": leg.note or "",
            })
        df = pd.DataFrame(rows, columns=["Side", "Type", "Strike", "Qty", "Est Price", "Note"])
        return df

    def header_stats(self) -> pd.DataFrame:
        d = {
            "Metric": [
                "DTE", "Net", "Max Loss", "Max Gain", "Breakeven(s)", "R:R", "Capital Requirement"
            ],
            "Value": [
                (self.dte if self.dte is not None else "—"),
                f"{'CREDIT' if self.debit_credit=='CREDIT' else 'DEBIT'} "
                f"${self.est_credit if self.debit_credit=='CREDIT' else self.est_cost:,.2f}",
                ("$"+format(round(self.max_loss,2), ",") if self.max_loss is not None else "—"),
                ("$"+format(round(self.max_gain,2), ",") if self.max_gain is not None else "—"),
                (", ".join(f"{b:.2f}" for b in self.breakevens) if self.breakevens else "—"),
                (f"{self.rr_ratio:.2f}" if self.rr_ratio is not None else "—"),
                ("$"+format(round(self.capital_req,2), ",") if self.capital_req is not None else "—"),
            ]
        }
        return pd.DataFrame(d)


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


def _years_from_dte(dte: Optional[int]) -> float:
    try:
        d = int(dte) if dte is not None else 30
    except Exception:
        d = 30
    return max(0.0, d) / 365.0


def _strike_grid(spot: float, pct: float) -> Tuple[float, float]:
    """
    Build simple OTM/ITM grid around spot: return (lower, upper) rounded to $1 or $5 steps.
    pct is symmetric (% from spot).
    """
    step = 5.0 if spot >= 50 else 1.0
    low = round((spot * (1.0 - pct)) / step) * step
    high = round((spot * (1.0 + pct)) / step) * step
    if low <= 0:
        low = step
    if abs(high - low) < step:
        high = low + step
    return float(low), float(high)


def _theo_mid(S: float, K: float, T: float, iv_pct: Optional[float], kind: Literal["CALL", "PUT"], r: float = 0.04) -> float:
    sigma = (_safe_float(iv_pct) or 35.0) / 100.0
    return float(bs_price(S, K, T, sigma, r, kind))


def _rr(max_gain: Optional[float], max_loss: Optional[float]) -> Optional[float]:
    if max_gain is None or max_loss is None or max_loss <= 0:
        return None
    return max_gain / max_loss


def _breakeven_call(K: float, debit_per_share: float) -> float:
    return K + debit_per_share


def _breakeven_put(K: float, debit_per_share: float) -> float:
    return K - debit_per_share


# -----------------------------
# Strategy builders (single ticker)
# -----------------------------

def build_debit_call_vertical(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], width_pct: float = 0.05
) -> StrategyPlan:
    """Bullish: long ATM call, short OTM call."""
    K_atm = S
    K_low, K_high = _strike_grid(S, width_pct)
    K_long = min(max(K_low, K_atm), K_high)  # nearest to ATM within grid
    K_short = K_high
    c_long = _theo_mid(S, K_long, T, iv_pct, "CALL")
    c_short = _theo_mid(S, K_short, T, iv_pct, "CALL")
    debit = max(0.01, c_long - c_short)  # per share
    width = max(0.01, K_short - K_long)
    max_gain = width * 100.0 - debit * 100.0
    max_loss = debit * 100.0
    be = _breakeven_call(K_long, debit)

    legs = [
        StrategyLeg("LONG", "CALL", K_long, 1, round(c_long * 100, 2), note="Buy ATM-ish"),
        StrategyLeg("SHORT", "CALL", K_short, 1, round(c_short * 100, 2), note="Sell OTM"),
    ]
    return StrategyPlan(
        name="Bull Call Debit Spread",
        ticker=ticker,
        direction="long",
        rationale="Directional bullish with defined risk; better RR than naked long call.",
        when_to_use="Trend Up or improving; regime Normal/Calm; expect moderate rise by expiry.",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit="DEBIT",
        est_cost=round(debit*100.0,2),
        est_credit=0.0,
        max_loss=round(max_loss,2),
        max_gain=round(max_gain,2),
        breakevens=[round(be,2)],
        rr_ratio=_rr(max_gain, max_loss),
        capital_req=round(debit*100.0,2),
        notes=f"Width ≈ ${width:.2f}.",
    )


def build_debit_put_vertical(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], width_pct: float = 0.05
) -> StrategyPlan:
    """Bearish: long ATM put, short further OTM put."""
    K_atm = S
    K_low, K_high = _strike_grid(S, width_pct)
    K_long = min(max(K_low, K_atm), K_high)  # nearest to ATM
    K_short = K_low
    p_long = _theo_mid(S, K_long, T, iv_pct, "PUT")
    p_short = _theo_mid(S, K_short, T, iv_pct, "PUT")
    debit = max(0.01, p_long - p_short)
    width = max(0.01, K_long - K_short)
    max_gain = width * 100.0 - debit * 100.0
    max_loss = debit * 100.0
    be = _breakeven_put(K_long, debit)

    legs = [
        StrategyLeg("LONG", "PUT", K_long, 1, round(p_long * 100, 2), note="Buy ATM-ish"),
        StrategyLeg("SHORT", "PUT", K_short, 1, round(p_short * 100, 2), note="Sell OTM"),
    ]
    return StrategyPlan(
        name="Bear Put Debit Spread",
        ticker=ticker,
        direction="short",
        rationale="Directional bearish with defined risk; cheaper than naked long put.",
        when_to_use="Trend Down or deteriorating; regime Normal/Calm; expect moderate drop by expiry.",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit="DEBIT",
        est_cost=round(debit*100.0,2),
        est_credit=0.0,
        max_loss=round(max_loss,2),
        max_gain=round(max_gain,2),
        breakevens=[round(be,2)],
        rr_ratio=_rr(max_gain, max_loss),
        capital_req=round(debit*100.0,2),
        notes=f"Width ≈ ${width:.2f}.",
    )


def build_credit_put_vertical(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], width_pct: float = 0.05
) -> StrategyPlan:
    """Bullish income: short OTM put, long further OTM put."""
    K_low, K_high = _strike_grid(S, width_pct)
    K_short = K_low  # sell closer to money
    K_long = max(K_low - (K_high - K_low), K_low / 2)  # protect further down; simple gap
    p_short = _theo_mid(S, K_short, T, iv_pct, "PUT")
    p_long = _theo_mid(S, K_long, T, iv_pct, "PUT")
    credit = max(0.01, p_short - p_long)
    width = max(0.01, K_short - K_long)
    max_loss = width * 100.0 - credit * 100.0
    max_gain = credit * 100.0
    be = K_short - credit

    legs = [
        StrategyLeg("SHORT", "PUT", K_short, 1, round(p_short * 100, 2), note="Sell OTM"),
        StrategyLeg("LONG", "PUT", K_long, 1, round(p_long * 100, 2), note="Buy further OTM protection"),
    ]
    return StrategyPlan(
        name="Bull Put Credit Spread",
        ticker=ticker,
        direction="long",
        rationale="Get long with positive theta and capped risk; high POP when sold sufficiently OTM.",
        when_to_use="Trend Up/Sideways; regime Normal/Stressed (elevated IV helps).",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit="CREDIT",
        est_cost=0.0,
        est_credit=round(credit*100.0,2),
        max_loss=round(max_loss,2),
        max_gain=round(max_gain,2),
        breakevens=[round(be,2)],
        rr_ratio=_rr(max_gain, max_loss),
        capital_req=round(max_loss,2),
        notes=f"Width ≈ ${width:.2f}.",
    )


def build_credit_call_vertical(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], width_pct: float = 0.05
) -> StrategyPlan:
    """Bearish income: short OTM call, long further OTM call."""
    K_low, K_high = _strike_grid(S, width_pct)
    K_short = K_high  # sell closer OTM
    K_long = K_high + (K_high - K_low)  # further OTM
    c_short = _theo_mid(S, K_short, T, iv_pct, "CALL")
    c_long = _theo_mid(S, K_long, T, iv_pct, "CALL")
    credit = max(0.01, c_short - c_long)
    width = max(0.01, K_long - K_short)
    max_loss = width * 100.0 - credit * 100.0
    max_gain = credit * 100.0
    be = K_short + credit

    legs = [
        StrategyLeg("SHORT", "CALL", K_short, 1, round(c_short * 100, 2), note="Sell OTM"),
        StrategyLeg("LONG", "CALL", K_long, 1, round(c_long * 100, 2), note="Buy further OTM protection"),
    ]
    return StrategyPlan(
        name="Bear Call Credit Spread",
        ticker=ticker,
        direction="short",
        rationale="Get short with positive theta and capped risk; high POP when sold sufficiently OTM.",
        when_to_use="Trend Down/Sideways; regime Normal/Stressed.",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit="CREDIT",
        est_cost=0.0,
        est_credit=round(credit*100.0,2),
        max_loss=round(max_loss,2),
        max_gain=round(max_gain,2),
        breakevens=[round(be,2)],
        rr_ratio=_rr(max_gain, max_loss),
        capital_req=round(max_loss,2),
        notes=f"Width ≈ ${width:.2f}.",
    )


def build_cash_secured_put(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], otm_pct: float = 0.05
) -> StrategyPlan:
    """Bullish accumulation: short OTM put, fully collateralized."""
    step = 5.0 if S >= 50 else 1.0
    K = round((S * (1.0 - otm_pct)) / step) * step
    p = _theo_mid(S, K, T, iv_pct, "PUT")
    credit = p * 100.0
    max_loss = (K * 100.0) - credit  # assigned at K to 0 (theoretical)
    be = K - p
    legs = [StrategyLeg("SHORT", "PUT", float(K), 1, round(p * 100, 2), note="Cash-secured")]
    return StrategyPlan(
        name="Cash-Secured Put",
        ticker=ticker,
        direction="long",
        rationale="Get paid to potentially buy shares at a discount; high POP when sold OTM.",
        when_to_use="Trend Up/Sideways; regime Normal/Stressed (richer IV = more credit).",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit="CREDIT",
        est_cost=0.0,
        est_credit=round(credit,2),
        max_loss=round(max_loss,2),
        max_gain=round(credit,2),
        breakevens=[round(be,2)],
        rr_ratio=_rr(credit, max_loss),
        capital_req=round(K*100.0 - credit, 2),
        notes="Requires cash collateral roughly Strike×100 minus credit.",
    )


def build_covered_call(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], otm_pct: float = 0.05
) -> StrategyPlan:
    """Own/Buy 100 shares, sell OTM call."""
    step = 5.0 if S >= 50 else 1.0
    K = round((S * (1.0 + otm_pct)) / step) * step
    c = _theo_mid(S, K, T, iv_pct, "CALL")
    credit = c * 100.0
    # If stock owned, downside risk is stock exposure; here show covered profile
    max_loss = None  # not capped; can add stop logic elsewhere
    max_gain = (K - S) * 100.0 + credit
    be = S - c
    legs = [
        StrategyLeg("LONG", "STOCK", None, 100, S, note="Own or buy 100 shares"),
        StrategyLeg("SHORT", "CALL", float(K), 1, round(c * 100, 2), note="Covered"),
    ]
    return StrategyPlan(
        name="Covered Call",
        ticker=ticker,
        direction="long",
        rationale="Income on long stock; lowers breakeven; caps upside at short call strike.",
        when_to_use="Trend Up/Sideways; regime Normal/Stressed (richer IV = more credit).",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit="CREDIT",
        est_cost=round(S*100.0,2),   # stock purchase outlay
        est_credit=round(credit,2),
        max_loss=max_loss,
        max_gain=round(max_gain,2),
        breakevens=[round(be,2)],
        rr_ratio=None,
        capital_req=round(S*100.0 - credit, 2),
        notes="Max gain capped at strike; set a stop on stock per your risk plan.",
    )


def build_collar(
    *, ticker: str, S: float, T: float, iv_pct: Optional[float], put_otm: float = 0.05, call_otm: float = 0.05
) -> StrategyPlan:
    """Protective long put + short OTM call against long stock (cost-reduced)."""
    step = 5.0 if S >= 50 else 1.0
    Kp = round((S * (1.0 - put_otm)) / step) * step
    Kc = round((S * (1.0 + call_otm)) / step) * step
    p = _theo_mid(S, Kp, T, iv_pct, "PUT")
    c = _theo_mid(S, Kc, T, iv_pct, "CALL")
    net = (p - c) * 100.0  # typically small debit or near-even
    max_loss = (S - Kp) * 100.0 + net
    max_gain = (Kc - S) * 100.0 - net
    be_low = S - (net / 100.0)
    legs = [
        StrategyLeg("LONG", "STOCK", None, 100, S, note="Own 100 shares"),
        StrategyLeg("LONG", "PUT", float(Kp), 1, round(p*100, 2), note="Protection"),
        StrategyLeg("SHORT", "CALL", float(Kc), 1, round(c*100, 2), note="Finance protection"),
    ]
    dc = "DEBIT" if net > 0 else ("CREDIT" if net < 0 else "EVEN")
    return StrategyPlan(
        name="Collar (Protected Stock)",
        ticker=ticker,
        direction="long",
        rationale="Cap downside with a put, finance with short call; define a risk box.",
        when_to_use="Holding stock into event or medium horizon; prefer richer IV.",
        legs=legs,
        dte=int(round(T*365)),
        debit_credit=dc,
        est_cost=round(max(net,0.0),2),
        est_credit=round(max(-net,0.0),2),
        max_loss=round(max_loss,2),
        max_gain=round(max_gain,2),
        breakevens=[round(be_low,2)],
        rr_ratio=_rr(max_gain, max_loss),
        capital_req=round(S*100.0 + max(net,0.0), 2),
        notes="Both upside and downside are bounded; great into uncertain catalysts.",
    )


# -----------------------------
# Top-level suggestion API
# -----------------------------

def suggest_strategies(
    *,
    ticker: str,
    direction: Literal["long", "short"],
    opt: OptionSnapshotLike,
    trend_label: Optional[str] = None,   # "Up", "Down", "Sideways" or None
    regime_label: Optional[str] = None,  # "Calm", "Normal", "Stressed" or None
    risk_profile: Literal["Conservative", "Balanced", "Aggressive"] = "Balanced",
    r: float = 0.04,
) -> List[StrategyPlan]:
    """
    Produce a shortlist of 2–3 strategies tailored to direction, trend/regime, and risk profile.
    Uses only the basic snapshot (spot, DTE, ATM IV, ATM mids).
    """
    S = _safe_float(getattr(opt, "spot", None))
    K_atm = _safe_float(getattr(opt, "atm_strike", None)) or S
    T = _years_from_dte(getattr(opt, "dte", None))
    iv_pct = _safe_float(getattr(opt, "atm_iv_pct", None)) or 35.0

    if S is None or T <= 0:
        return []

    suggestions: List[StrategyPlan] = []

    # Simple policy matrix:
    # - Long view: debit call vertical (defined risk), bull put credit (income), CSP (stock acquisition)
    # - Short view: debit put vertical (defined risk), bear call credit (income)
    # - If user likely holds shares (not detectable here), covered call/collar can be added by choice.
    trend = (trend_label or "").lower()
    regime = (regime_label or "").lower()

    # Width choice influenced by risk profile
    width_pct = 0.05 if risk_profile != "Aggressive" else 0.08

    if direction == "long":
        suggestions.append(build_debit_call_vertical(ticker=ticker, S=S, T=T, iv_pct=iv_pct, width_pct=width_pct))
        suggestions.append(build_credit_put_vertical(ticker=ticker, S=S, T=T, iv_pct=iv_pct, width_pct=width_pct))
        # Add CSP for conservative / income preference
        if risk_profile in ("Conservative", "Balanced"):
            suggestions.append(build_cash_secured_put(ticker=ticker, S=S, T=T, iv_pct=iv_pct, otm_pct=0.07 if regime == "stressed" else 0.05))
        else:
            # Covered call only if implicitly holding stock; still include as template
            suggestions.append(build_covered_call(ticker=ticker, S=S, T=T, iv_pct=iv_pct, otm_pct=0.05))
    else:
        suggestions.append(build_debit_put_vertical(ticker=ticker, S=S, T=T, iv_pct=iv_pct, width_pct=width_pct))
        suggestions.append(build_credit_call_vertical(ticker=ticker, S=S, T=T, iv_pct=iv_pct, width_pct=width_pct))

    # Sort by rough “retail friendliness”: defined risk first, then credits with capped risk, then stock-tied plans
    order = {
        "Bull Call Debit Spread": 1,
        "Bear Put Debit Spread": 1,
        "Bull Put Credit Spread": 2,
        "Bear Call Credit Spread": 2,
        "Cash-Secured Put": 3,
        "Covered Call": 3,
        "Collar (Protected Stock)": 4,
    }
    suggestions.sort(key=lambda p: order.get(p.name, 99))

    # Trim to 3 best
    return suggestions[:3]


# -----------------------------
# Formatting helpers for UI
# -----------------------------

def plans_to_markdown(plans: List[StrategyPlan]) -> str:
    """Compact markdown outline for the Report tab."""
    out = []
    for p in plans:
        hdr = f"### {p.ticker}: {p.name}"
        stats = p.header_stats()
        df = p.to_dataframe()
        out.append(hdr)
        out.append("")
        out.append(f"**Rationale:** {p.rationale}")
        out.append(f"**When to use:** {p.when_to_use}")
        out.append("")
        # Render stats
        try:
            stats_md = stats.to_markdown(index=False)
        except Exception:
            stats_md = stats.to_string(index=False)
        out.append(stats_md)
        out.append("")
        out.append("**Legs**")
        try:
            legs_md = df.to_markdown(index=False)
        except Exception:
            legs_md = df.to_string(index=False)
        out.append(legs_md)
        if p.notes:
            out.append("")
            out.append(f"_Notes:_ {p.notes}")
        out.append("\n---\n")
    return "\n".join(out)
