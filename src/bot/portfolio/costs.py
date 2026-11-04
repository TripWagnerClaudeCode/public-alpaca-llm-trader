"""Honest fills (profile B, stage 2): quotes at execute, spread guard, modeled trading cost at reconcile.

Pure functions only; the journal tables live in bot.journal.store.

Cost model (spec "Honest fills"):
  modeled_cost_pct = 0.5 x spread_pct + IMPACT_COEF x sigma_daily x sqrt(filled_shares / ADV_shares)
  slippage_vs_mid  = side-signed (fill - mid) / mid   (positive = paid more than mid on a buy / got less on a sell)
"""
from __future__ import annotations

import math

import pandas as pd

from bot.models import OrderPlan

IMPACT_COEF = 1.0  # square-root law coefficient


def normalize_quote(q: dict | None) -> dict | None:
    """{"bid","ask","ts"} -> adds mid and spread_pct. IEX quotes on illiquid names can be one-sided or
    crossed: bid<=0, ask<=0 or ask<bid count as no quote (None)."""
    if not q:
        return None
    try:
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask < bid or math.isnan(bid) or math.isnan(ask):
        return None
    mid = (bid + ask) / 2
    return {"bid": bid, "ask": ask, "ts": q.get("ts"), "mid": mid, "spread_pct": (ask - bid) / mid}


def spread_guard(orders: list[OrderPlan], quotes: dict[str, dict | None],
                 max_spread_pct: float | None) -> tuple[list[OrderPlan], list[str], dict[str, str]]:
    """Skip BUY orders whose quoted spread > max_spread_pct of mid, or with no usable quote.
    Sells always go through. max_spread_pct=None disables the guard.
    Returns (kept orders, logged reasons, {symbol: skip reason})."""
    if max_spread_pct is None:
        return orders, [], {}
    out, reasons, skipped = [], [], {}
    for o in orders:
        if o.side != "buy":
            out.append(o)
            continue
        q = quotes.get(o.symbol)
        if q is None:
            why = f"{o.symbol}: buy ${o.notional:,.2f} skipped, no usable quote (spread guard)"
        elif q["spread_pct"] > max_spread_pct + 1e-12:
            why = (f"{o.symbol}: buy ${o.notional:,.2f} skipped, spread {q['spread_pct']:.2%} > "
                   f"max {max_spread_pct:.2%} (bid ${q['bid']:,.2f} / ask ${q['ask']:,.2f})")
        else:
            out.append(o)
            continue
        reasons.append(why)
        skipped[o.symbol] = why
    return out, reasons, skipped


def daily_sigma(bars: pd.DataFrame, n: int = 20) -> float | None:
    """Std of daily close-to-close returns over the last n days."""
    r = bars["close"].pct_change().tail(n).dropna()
    if len(r) < max(2, n // 2):
        return None
    v = float(r.std())
    return v if v > 0 and not math.isnan(v) else None


def adv_shares(bars: pd.DataFrame, n: int = 20) -> float | None:
    """Mean daily share volume over the last n bars."""
    tail = bars.tail(n)
    if len(tail) < n:
        return None
    v = float(tail["volume"].mean())
    return v if v > 0 and not math.isnan(v) else None


def slippage_vs_mid(side: str, fill: float, mid: float | None) -> float | None:
    if not mid or mid <= 0 or not fill:
        return None
    s = (fill - mid) / mid
    return s if side == "buy" else -s


def modeled_cost_pct(spread_pct: float, sigma_daily: float | None, shares: float,
                     adv_sh: float | None) -> tuple[float, float | None]:
    """(cost_pct, impact_pct). Unknown sigma/ADV -> impact None, cost = half spread only."""
    impact = None
    if sigma_daily and adv_sh and adv_sh > 0 and shares > 0:
        impact = IMPACT_COEF * sigma_daily * math.sqrt(shares / adv_sh)
    return 0.5 * spread_pct + (impact or 0.0), impact


def fill_cost(side: str, filled_qty: float, fill_price: float, quote: dict | None,
              sigma_daily: float | None, adv_sh: float | None, default_spread_pct: float) -> dict:
    """Cost record for one fill. quote: {"mid","spread_pct"} as stored at execute, or None."""
    est = quote is None or quote.get("spread_pct") is None
    spread = default_spread_pct if est else float(quote["spread_pct"])
    mid = None if quote is None else quote.get("mid")
    pct, impact = modeled_cost_pct(spread, sigma_daily, filled_qty, adv_sh)
    notional = filled_qty * fill_price
    slip = slippage_vs_mid(side, fill_price, mid)
    flags = []
    if est:
        flags.append("spread_estimated")
    if impact is None:
        flags.append("no_impact_inputs")
    return {"mid": mid, "spread_pct": spread, "spread_estimated": int(est), "sigma_daily": sigma_daily,
            "adv_shares": adv_sh, "impact_pct": impact, "modeled_cost_pct": pct,
            "modeled_cost_usd": pct * notional, "slippage_vs_mid": slip,
            "slippage_usd": None if slip is None else slip * notional, "notional": notional,
            "flags": ",".join(flags)}
