from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Any, Dict
from datetime import datetime
import io
import csv
import math

import pandas as pd


# -----------------------------
# Models
# -----------------------------

@dataclass
class TradeTicket:
    timestamp: str
    ticker: str
    instrument: str          # "Stock" or "Option"
    side: str                # "BUY" or "SELL"
    quantity: int
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    rationale: str

    # Option-only fields (safe to leave None for stock tickets)
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # "CALL" / "PUT"
    notes: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _val_from_sizing(df: pd.DataFrame, metric: str) -> Optional[float]:
    """
    Pull a numeric value from the 2-col sizing table by label.
    Accepts things like '12.34%' or '$123.45' and returns float 12.34 / 123.45.
    """
    if df is None or df.empty:
        return None
    try:
        s = df.set_index("Metric")["Value"]
    except Exception:
        s = df.iloc[:, :2].set_index(df.columns[0])[df.columns[1]]
    raw = s.get(metric)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    txt = str(raw)
    # strip $, %, commas
    txt = txt.replace("$", "").replace(",", "").strip()
    if txt.endswith("%"):
        try:
            return float(txt[:-1])
        except Exception:
            return None
    return _safe_float(txt)


def _choose_stop_pct(sizing_df: pd.DataFrame) -> Optional[float]:
    """
    Prefer ATR stop; fallback to Baseline stop.
    Returned as percent number, e.g., 7.5 means 7.5%.
    """
    atr_stop = _val_from_sizing(sizing_df, "ATR stop %")
    if atr_stop is not None:
        return atr_stop
    return _val_from_sizing(sizing_df, "Baseline stop %")


def _choose_share_qty(sizing_df: pd.DataFrame) -> Optional[int]:
    """
    Prefer ATR shares; fallback to Shares (approx.).
    """
    def _to_int(v) -> Optional[int]:
        try:
            if v is None:
                return None
            if isinstance(v, str):
                v = v.replace(",", "").strip()
            i = int(float(v))
            return i if i > 0 else None
        except Exception:
            return None

    # Try ATR shares first
    try:
        s = sizing_df.set_index("Metric")["Value"]
    except Exception:
        s = sizing_df.iloc[:, :2].set_index(sizing_df.columns[0])[sizing_df.columns[1]]

    v = s.get("ATR shares (approx.)")
    qty = _to_int(v)
    if qty:
        return qty
    v = s.get("Shares (approx.)")
    return _to_int(v)


def _choose_option_contracts(sizing_df: pd.DataFrame) -> Optional[int]:
    """
    Use the “Options contracts (≈ debit)” row if available.
    """
    try:
        s = sizing_df.set_index("Metric")["Value"]
    except Exception:
        s = sizing_df.iloc[:, :2].set_index(sizing_df.columns[0])[sizing_df.columns[1]]

    raw = s.get("Options contracts (≈ debit)")
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = raw.replace(",", "").strip()
        n = int(float(raw))
        return n if n > 0 else None
    except Exception:
        return None


def _compute_stop(entry: Optional[float], stop_pct: Optional[float], direction: str) -> Optional[float]:
    if entry is None or stop_pct is None:
        return None
    if direction.lower().startswith("long"):
        return round(entry * (1.0 - stop_pct / 100.0), 4)
    else:
        # short: stop above entry
        return round(entry * (1.0 + stop_pct / 100.0), 4)


def _compute_target(entry: Optional[float], move_pct: Optional[float], direction: str) -> Optional[float]:
    """
    Target = entry ± (implied_move_pct) as a simple, transparent default.
    """
    if entry is None or move_pct is None:
        return None
    if direction.lower().startswith("long"):
        return round(entry * (1.0 + move_pct / 100.0), 4)
    else:
        return round(entry * (1.0 - move_pct / 100.0), 4)


# -----------------------------
# Public API
# -----------------------------

def build_tickets_for_ticker(
    *,
    ticker: str,
    direction: Optional[str],            # "long" / "short" (case-insensitive); if None we default to "long"
    sizing_df: pd.DataFrame,             # from sizing_summary_table(...)
    opt: Any,                            # OptionsSnapshot (defensive getattr)
    spot: Optional[float],               # latest price
    implied_move_pct: Optional[float],   # from OptionsSnapshot
    profile_name: str,
) -> List[TradeTicket]:
    """
    Create a pair of tickets for one ticker:
      1) Stock ticket using ATR/baseline size and stop
      2) Option ticket (CALL if long, PUT if short) using contracts approximation

    If inputs are missing, we degrade gracefully and still produce a sensible row.
    """
    dir_text = (direction or "long").lower()
    side = "BUY" if dir_text.startswith("long") else "SELL"

    # --- Extract sizing info
    stop_pct = _choose_stop_pct(sizing_df)
    shares = _choose_share_qty(sizing_df)
    contracts = _choose_option_contracts(sizing_df)

    # --- Option context (defensive)
    expiry = getattr(opt, "expiry", None)
    dte = getattr(opt, "dte", None)
    atm_strike = getattr(opt, "atm_strike", None)
    call_mid = getattr(opt, "call_mid", None)
    put_mid = getattr(opt, "put_mid", None)

    entry = _safe_float(spot)
    stop_price = _compute_stop(entry, stop_pct, dir_text)
    target = _compute_target(entry, _safe_float(implied_move_pct), dir_text)

    tickets: List[TradeTicket] = []

    # 1) Stock ticket
    tickets.append(
        TradeTicket(
            timestamp=_now_iso(),
            ticker=ticker,
            instrument="Stock",
            side=side,
            quantity=int(shares) if shares else 0,
            entry_price=entry,
            stop_price=stop_price,
            target_price=target,
            rationale=f"{profile_name} profile; stop≈{stop_pct}% ; implied move≈{implied_move_pct}%",
            notes=None,
        )
    )

    # 2) Option ticket (only if we have minimal data)
    opt_mid = _safe_float(call_mid if side == "BUY" else put_mid)  # for shorts you'd often buy PUTs
    opt_type = "CALL" if side == "BUY" else "PUT"
    if contracts and atm_strike and opt_mid:
        tickets.append(
            TradeTicket(
                timestamp=_now_iso(),
                ticker=ticker,
                instrument="Option",
                side="BUY" if side == "BUY" else "BUY",  # directional: long calls for long view; long puts for short view
                quantity=int(contracts),
                entry_price=opt_mid,
                stop_price=None,            # stops on options are discretionary; left blank
                target_price=None,          # keep simple; premium targets vary
                rationale=f"{opt_type} {atm_strike} {expiry} x{contracts} as directional expression; "
                          f"mid≈{opt_mid}; implied move≈{implied_move_pct}%",
                expiry=str(expiry) if expiry else None,
                strike=_safe_float(atm_strike),
                option_type=opt_type,
                notes=f"DTE={dte}" if dte else None,
            )
        )

    return tickets


def tickets_to_csv_bytes(tickets: List[TradeTicket]) -> Tuple[str, bytes]:
    """
    Return a (filename, bytes) tuple suitable for Streamlit download_button.
    """
    if not tickets:
        # always produce a valid CSV with header
        tickets = [TradeTicket(
            timestamp=_now_iso(), ticker="", instrument="Stock", side="BUY",
            quantity=0, entry_price=None, stop_price=None, target_price=None,
            rationale="(empty)", expiry=None, strike=None, option_type=None, notes=None
        )]

    buf = io.StringIO()
    fieldnames = list(asdict(tickets[0]).keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for t in tickets:
        writer.writerow(asdict(t))
    data = buf.getvalue().encode("utf-8")
    fname = f"tickets_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.csv"
    return fname, data

# === Build tickets from a StrategyPlan =========================================
from typing import Any, Dict, List, Optional

import pandas as _pd

def _lower_cols(df: _pd.DataFrame) -> _pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    return d

def _col(df: _pd.DataFrame, *names) -> Optional[str]:
    """
    Return the first matching column name in df for any of the provided names
    (case-insensitive, whitespace-insensitive). Returns None if not found.
    """
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        key = str(n).strip().lower()
        if key in cols:
            return cols[key]
    return None

def build_tickets_from_strategy(plan: Any, fallback_ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert a StrategyPlan (from ai_agent.strategies) into ticket rows (dicts).
    We intentionally return dicts so callers can merge them with existing tickets,
    and downstream CSV/preview code can handle either dataclasses (.__dict__) or dicts.

    Expected columns in plan.to_dataframe() (tolerant / case-insensitive):
        - action   -> 'BUY' / 'SELL'
        - type     -> 'CALL' / 'PUT' / 'STOCK' / 'EQUITY'
        - expiry   -> 'YYYY-MM-DD' (optional for stock)
        - strike   -> float (options only)
        - qty or quantity -> int (defaults to 1 if missing)
        - ticker   -> overrides fallback_ticker if present

    We also attach: plan name, a short note, and a generic price_note from the plan header stats if available.
    """
    tickets: List[Dict[str, Any]] = []

    # Pull legs as a DataFrame
    try:
        legs_df = plan.to_dataframe()
        if legs_df is None or legs_df.empty:
            return tickets
    except Exception:
        return tickets

    legs_df = _lower_cols(legs_df)

    c_action = _col(legs_df, "Action", "Side")
    c_type = _col(legs_df, "Type", "Leg Type", "Asset")
    c_expiry = _col(legs_df, "Expiry", "Expiration")
    c_strike = _col(legs_df, "Strike")
    c_qty = _col(legs_df, "Qty", "Quantity", "Size")
    c_ticker = _col(legs_df, "Ticker")

    # Optional header stats to annotate price context
    price_note = None
    try:
        if hasattr(plan, "header_stats"):
            hs = plan.header_stats()
            # Look for Net / Max Loss / Max Gain fields to summarize
            if isinstance(hs, _pd.DataFrame) and not hs.empty:
                # Expect 2 columns ["Metric", "Value"] or similar
                if hs.shape[1] >= 2:
                    met_col, val_col = hs.columns[:2]
                    vals = dict(zip(hs[met_col].astype(str), hs[val_col].astype(str)))
                    parts = []
                    for k in ("Net", "Max loss", "Max gain", "Breakeven(s)"):
                        v = vals.get(k)
                        if v:
                            parts.append(f"{k}: {v}")
                    if parts:
                        price_note = "; ".join(parts)
    except Exception:
        pass

    for _, row in legs_df.iterrows():
        action = str(row.get(c_action, "")).strip().upper() if c_action else ""
        leg_type = str(row.get(c_type, "")).strip().upper() if c_type else ""
        expiry = str(row.get(c_expiry, "")).strip() if c_expiry else ""
        strike = row.get(c_strike) if c_strike else None
        qty_raw = row.get(c_qty) if c_qty else 1
        ticker = str(row.get(c_ticker, "")).strip().upper() if c_ticker else ""

        # Basic cleanup
        try:
            qty = int(qty_raw) if _pd.notna(qty_raw) else 1
        except Exception:
            qty = 1

        if not ticker:
            ticker = (fallback_ticker or "").upper()

        asset = "option"
        right = None
        if leg_type in ("CALL", "PUT"):
            right = leg_type
        elif leg_type in ("STOCK", "EQUITY", "SHARES"):
            asset = "stock"

        # Build a generic, CSV-friendly ticket dict
        t: Dict[str, Any] = {
            "plan": getattr(plan, "name", "Strategy"),
            "ticker": ticker,
            "asset": asset,                  # 'stock' or 'option'
            "side": action or "BUY",         # default BUY
            "quantity": qty,
            "expiry": expiry or None,
            "strike": float(strike) if (strike is not None and str(strike).strip() != "") else None,
            "right": right,                  # 'CALL' / 'PUT' or None for stock
            "price_note": price_note,
            "notes": getattr(plan, "rationale", None) or getattr(plan, "when_to_use", None),
        }
        tickets.append(t)

    return tickets
